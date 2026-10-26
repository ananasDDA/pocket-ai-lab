#!/usr/bin/env python3
"""Split a DailyUseDeviceTests log into per-scenario JSON files.

Usage:
    python3 Scripts/daily_use_extract.py daily-qwen.log [more.log ...] [--out DIR]

Reads every `DAILY_RESULT_JSON {...}` and `DAILY_SUMMARY_JSON {...}` line
(one JSON object per line, written by local_ai_testTests/Device/
DailyUseDeviceTests.swift) and writes:

    DIR/<model>/<scenario>.json   one file per scenario, full transcript
    DIR/<model>/summary.json      the run summary, when the log has one

then prints a plain-text table of the numbers to stdout: one row per
scenario, followed by the per-generation rates of every `sustained` run
and the per-turn context figures of every `long_session` run.

With --combined it prints one table instead: a row per model, a column
per scenario holding the checks passed (e.g. `3/4`), then the medians of
time to first token and decode rate over every answer, the highest memory
peak and the wall clock. Scripts/daily_use_night.sh ends with it.

DIR defaults to ./daily-use-out. Several logs, or one log holding several
models, are fine; a model that appears twice gets a numbered directory.
"""

import argparse
import json
import os
import sys

RESULT = "DAILY_RESULT_JSON "
SUMMARY = "DAILY_SUMMARY_JSON "


def parse_logs(paths):
    """Return (runs, bad_lines). A run is {"model", "results", "summary"}.

    Results are grouped into a run until the summary line that closes it,
    so two runs of the same model in one log stay apart.
    """
    runs, current, bad = [], {}, 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for marker in (RESULT, SUMMARY):
                    at = line.find(marker)
                    if at < 0:
                        continue
                    try:
                        obj = json.loads(line[at + len(marker):].strip())
                    except json.JSONDecodeError:
                        bad += 1
                        break
                    model = obj.get("model_id", "unknown")
                    run = current.get(model)
                    if run is None:
                        run = {"model": model, "results": [], "summary": None}
                        current[model] = run
                        runs.append(run)
                    if marker == RESULT:
                        run["results"].append(obj)
                    else:
                        run["summary"] = obj
                        del current[model]
                    break
    # A run without the system prompt (DAILY_NO_SYSTEM_PROMPT=1, the A/B
    # "before" case) gets its own label, so it never merges with the
    # normal run of the same model.
    # Likewise a run with DAILY_SAMPLING=global or authors (the sampling
    # A/B); `catalog`, the default, and logs older than the switch get none.
    for run in runs:
        params = [r.get("parameters") or {} for r in run["results"]]
        if params and all(p.get("system_prompt") is None for p in params):
            run["model"] += " [no system prompt]"
        modes = {p.get("sampling") for p in params} - {None, "catalog"}
        if len(modes) == 1:
            run["model"] += f" [sampling {modes.pop()}]"
    return runs, bad


def safe(name):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def write_files(runs, out_dir):
    used = set()
    for run in runs:
        base = safe(run["model"])
        name, n = base, 1
        while name in used:
            n += 1
            name = f"{base}-{n}"
        used.add(name)
        directory = os.path.join(out_dir, name)
        os.makedirs(directory, exist_ok=True)
        for result in run["results"]:
            path = os.path.join(directory, safe(result.get("scenario", "unknown")) + ".json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
        if run["summary"] is not None:
            with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(run["summary"], handle, indent=2, ensure_ascii=False)
        run["dir"] = directory


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def fmt(value, digits=0):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def scenario_row(model, result):
    steps = result.get("steps", [])
    ok = [s for s in steps if s.get("outcome") == "ok"]
    checks = result.get("checks", [])
    passed = sum(1 for c in checks if c.get("passed"))
    score = result.get("score")
    score_text = f"{score}/{result.get('score_max')}" if score is not None else "-"
    prefill = [s.get("prefill_tps") for s in steps if s.get("prefill_tps") is not None]
    return [
        model,
        result.get("scenario", "?"),
        result.get("outcome", "?"),
        f"{len(ok)}/{len(steps)}",
        f"{passed}/{len(checks)}",
        score_text,
        fmt(median([s.get("ttft_ms") for s in ok])),
        fmt(median([s.get("tps") for s in ok]), 1),
        fmt(max(prefill) if prefill else None, 0),
        fmt(max((s.get("peak_mb", 0.0) for s in steps), default=None)),
        fmt(result.get("wall_clock_s")),
        f"{result.get('thermal_before', '?')}>{result.get('thermal_after', '?')}"
        + (" FLAGGED" if result.get("measurement_issues") else ""),
    ]


SHORT = {
    "chat_recall": "recall",
    "summarize_long": "summ",
    "rewrite_email": "email",
    "extract_json": "json",
    "translate": "transl",
    "reasoning_short": "reason",
    "code_small": "code",
    "advice_followup": "advice",
    "doc_qa": "docqa",
    "unknown_entity": "unknown",
    "non_english_chat": "lang",
    "format_follow": "format",
    "long_session": "long",
    "vision_describe": "v_desc",
    "vision_read_text": "v_text",
    "vision_photo_real": "v_photo",
    "sustained": "sust",
}


def print_combined(runs):
    """One row per model: checks passed per scenario, then the numbers."""
    order = list(SHORT)
    seen = {r.get("scenario") for run in runs for r in run["results"]}
    order += sorted(s for s in seen if s and s not in SHORT)
    columns = [s for s in order if s in seen]
    rows = []
    for run in runs:
        by_name = {r.get("scenario"): r for r in run["results"]}
        cells = [run["model"]]
        passed_all = total_all = 0
        ok_steps = []
        peaks = []
        for name in columns:
            result = by_name.get(name)
            if result is None:
                cells.append("")
                continue
            if result.get("outcome") == "skipped":
                cells.append("skip")
                continue
            checks = result.get("checks", [])
            passed = sum(1 for c in checks if c.get("passed"))
            passed_all += passed
            total_all += len(checks)
            mark = "" if result.get("outcome") == "ok" else "!"
            if result.get("measurement_issues"):
                mark += "#"
            cells.append(f"{passed}/{len(checks)}{mark}")
            steps = result.get("steps", [])
            ok_steps += [s for s in steps if s.get("outcome") == "ok"]
            peaks += [s.get("peak_mb") for s in steps if s.get("peak_mb") is not None]
        summary = run["summary"] or {}
        sources = {s.get("token_source") for s in ok_steps if s.get("token_source")}
        estimated = "~" if sources and sources != {"engine"} else ""
        cells += [
            f"{passed_all}/{total_all}",
            fmt(median([s.get("ttft_ms") for s in ok_steps])),
            estimated + fmt(median([s.get("tps") for s in ok_steps]), 1),
            fmt(max(peaks) if peaks else None),
            fmt(summary.get("wall_clock_s")),
            "ok" if summary.get("complete") else (summary.get("error") or "no summary")[:40],
        ]
        rows.append(cells)
    print_table(
        ["model"] + [SHORT.get(c, c) for c in columns]
        + ["checks", "ttft_ms", "tok/s", "peak_mb", "wall_s", "run"],
        rows,
    )
    print("\n'!' = scenario not fully ok (partial/failed/error); '#' = measured under bad "
          "conditions (measurement_issues: a gate ran out, low power mode, background); "
          "'~' = rate from estimated token counts (Apple Intelligence) or stream pieces")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", nargs="+", help="xcodebuild test log(s)")
    parser.add_argument("--out", default="daily-use-out", help="output directory")
    parser.add_argument("--combined", action="store_true",
                        help="print only the one-row-per-model table")
    args = parser.parse_args()

    runs, bad = parse_logs(args.logs)
    if not runs:
        print("no DAILY_RESULT_JSON / DAILY_SUMMARY_JSON lines found", file=sys.stderr)
        return 1
    write_files(runs, args.out)

    if args.combined:
        print_combined(runs)
        if bad:
            print(f"\nwarning: {bad} marker line(s) did not parse as JSON (truncated log?)",
                  file=sys.stderr)
        return 0

    rows = []
    for run in runs:
        rows.extend(scenario_row(run["model"], r) for r in run["results"])
    print_table(
        ["model", "scenario", "outcome", "steps", "checks", "score",
         "ttft_ms", "tok/s", "prefill_tok/s", "peak_mb", "wall_s", "thermal"],
        rows,
    )

    print()
    summary_rows = []
    for run in runs:
        s = run["summary"] or {}
        load = s.get("load") or {}
        summary_rows.append([
            run["model"],
            s.get("backend", "-"),
            "yes" if s.get("complete") else ("no" if s else "missing"),
            fmt(load.get("load_ms")),
            fmt(load.get("peak_mb")),
            fmt(s.get("wall_clock_s")),
            "yes" if s.get("downloaded_here") else "no",
            "yes" if s.get("deleted_after") else "no",
            s.get("error") or "-",
            run["dir"],
        ])
    print_table(
        ["model", "backend", "complete", "load_ms", "load_peak_mb", "wall_s",
         "downloaded", "deleted", "error", "files"],
        summary_rows,
    )

    sustained = [(run["model"], r) for run in runs for r in run["results"]
                 if r.get("scenario") == "sustained" and r.get("steps")]
    if sustained:
        print()
        step_rows = []
        for model, result in sustained:
            for step in result["steps"]:
                step_rows.append([
                    model,
                    step.get("label", "?"),
                    step.get("outcome", "?"),
                    fmt(step.get("tokens")),
                    fmt(step.get("tps"), 1),
                    fmt(step.get("ttft_ms")),
                    fmt(step.get("peak_mb")),
                    step.get("thermal_after", "?"),
                ])
        print_table(["model", "sustained", "outcome", "tokens", "tok/s", "ttft_ms",
                     "peak_mb", "thermal"], step_rows)

    long_runs = [(run["model"], r) for run in runs for r in run["results"]
                 if r.get("scenario") == "long_session" and r.get("steps")]
    if long_runs:
        print()
        step_rows = []
        for model, result in long_runs:
            for step in result["steps"]:
                context = step.get("context") or {}
                kept = context.get("first_turn_kept")
                step_rows.append([
                    model,
                    step.get("label", "?"),
                    step.get("outcome", "?"),
                    fmt(step.get("ttft_ms")),
                    fmt(step.get("tokens")),
                    fmt(context.get("prompt_tokens_estimate")),
                    fmt(step.get("reused_tokens")),
                    fmt(step.get("evaluated_tokens")),
                    fmt(step.get("dropped_turns")),
                    fmt(step.get("dropped_front_tokens")),
                    "-" if kept is None else ("yes" if kept else "no"),
                ])
        print_table(["model", "long_session", "outcome", "ttft_ms", "tokens", "prompt_est",
                     "reused", "evaluated", "dropped_turns", "dropped_front", "first_kept"],
                    step_rows)

    if bad:
        print(f"\nwarning: {bad} marker line(s) did not parse as JSON (truncated log?)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

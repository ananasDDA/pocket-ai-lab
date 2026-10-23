#!/usr/bin/env python3
"""Turn a Pocket AI Lab build test into a Markdown release report.

The input is what a build test leaves on the device - started from the app
(Testing > Test this build) or from `CatalogBenchmarkTests`. Three input
shapes are accepted:

  * a build test report, `Documents/BuildTestReports/build-test-<ts>.json`,
    written by `BuildTestReportStore` (schemaVersion 2: one ordered `entries`
    list, each entry `completed` / `failed` / `skipped` / `crashed`);
  * an old sweep summary, `release-summary-<ts>.json` (schemaVersion 1: three
    parallel `reports` / `failures` / `skips` arrays), so reports from earlier
    releases keep working;
  * a single `BenchmarkReport`, or a directory holding any mixture of the
    above.

Field names match `local_ai_test/Modules/Benchmark/BenchmarkTypes.swift` and
`Modules/Benchmark/BuildTest/BuildTestReport.swift` exactly; nothing here is
guessed from the data.

Usage
-----

    # Report for one run, to stdout
    python3 Scripts/generate_benchmark_report.py build-test-2026-09-03T141233Z.json

    # Same, written to a file
    python3 Scripts/generate_benchmark_report.py \\
        release-summary-2026-09-02T141233Z.json \\
        --out benchmark-1.1.md

    # Compared against the previous release's summary
    python3 Scripts/generate_benchmark_report.py \\
        release-summary-2026-09-02T141233Z.json \\
        --previous releases/1.0/release-summary-2026-04-11T090512Z.json \\
        --out benchmark-1.1.md

    # A folder of individual report files instead of one build test
    python3 Scripts/generate_benchmark_report.py ~/Downloads/BuildTestReports/

Python 3 standard library only - no third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# A model is called out as a regression when it crosses either of these.
GEN_TOKENS_DROP_PCT = 10.0   # generation tok/s, percent slower than before
SCORE_DROP_POINTS = 5.0      # overall score, absolute points lost


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class Run:
    """One sweep: its metadata plus every report, failure and skip in it."""

    def __init__(self, source):
        self.source = source
        self.app_version = ""
        self.app_build = ""
        self.generated_at = ""
        self.stages = ""
        self.device = {}
        self.reports = []
        self.failures = []
        self.skips = []
        self.cancelled = False
        self.interrupted = False
        self.crashes = []
        self.duration_seconds = None

    @property
    def device_line(self):
        device = self.device or {}
        name = device.get("marketingName") or device.get("identifier") or "unknown device"
        parts = [name]
        if device.get("chip"):
            parts.append(str(device["chip"]))
        if device.get("ramGB") is not None:
            parts.append("%.1f GB RAM" % float(device["ramGB"]))
        if device.get("iOSVersion"):
            parts.append("iOS %s" % device["iOSVersion"])
        return ", ".join(parts)

    def reports_by_id(self):
        return {report.get("modelId", ""): report for report in self.reports}


def _is_report(document):
    """A single `BenchmarkReport`, as opposed to a sweep summary."""
    return isinstance(document, dict) and "modelId" in document and "performance" in document


def _is_build_test(document):
    """A `BuildTestReport` (schemaVersion 2): one ordered `entries` list."""
    return (isinstance(document, dict)
            and isinstance(document.get("entries"), list)
            and "device" in document)


def _is_summary(document):
    """A `BenchmarkSweepSummary` (schemaVersion 1) from an older release."""
    return isinstance(document, dict) and "reports" in document


def _is_envelope(document):
    """Either whole-sweep shape, as opposed to a single report."""
    return _is_build_test(document) or _is_summary(document)


def _absorb_envelope(run, document):
    """Fills `run` from either envelope shape."""
    if _is_build_test(document):
        _absorb_build_test(run, document)
    else:
        _absorb_summary(run, document)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run(path):
    """Loads a summary file, a bare report file, or a directory of either."""
    run = Run(path)

    if os.path.isdir(path):
        documents = []
        for name in sorted(os.listdir(path)):
            if not name.endswith(".json"):
                continue
            full = os.path.join(path, name)
            try:
                documents.append(_read_json(full))
            except (ValueError, OSError) as error:
                # One unreadable file must not cost the whole report.
                print("warning: skipping %s (%s)" % (full, error), file=sys.stderr)
        envelopes = [document for document in documents if _is_envelope(document)]
        reports = [document for document in documents if _is_report(document)]

        for envelope in envelopes:
            reports.extend(_envelope_reports(envelope))

        # Envelope data (failures, skips, appVersion, device, date) comes
        # from the NEWEST envelope only. Absorbing it from every sweep found
        # in the folder would inflate the failure/skip counts with stale runs
        # left over from earlier sessions. Per-model reports keep the separate
        # newest-per-model de-dup below.
        if envelopes:
            newest = max(
                envelopes, key=lambda envelope: str(envelope.get("generatedAt") or "")
            )
            _absorb_envelope(run, newest)

        # De-duplicate: a folder can hold both the summary and the individual
        # reports it was built from. Newest report per model wins.
        merged = {}
        for report in reports:
            model_id = report.get("modelId", "")
            existing = merged.get(model_id)
            if existing is None or str(report.get("date", "")) >= str(existing.get("date", "")):
                merged[model_id] = report
        run.reports = sorted(merged.values(), key=lambda report: report.get("modelName", ""))
    else:
        document = _read_json(path)
        if _is_envelope(document):
            _absorb_envelope(run, document)
            run.reports = list(_envelope_reports(document))
        elif _is_report(document):
            run.reports = [document]
        else:
            raise ValueError(
                "%s is neither a build test, a sweep summary, nor a benchmark report" % path)

    _fill_metadata_from_reports(run)
    return run


def _envelope_reports(document):
    """The `BenchmarkReport`s inside either envelope shape."""
    if _is_build_test(document):
        return [entry.get("report") for entry in document.get("entries") or []
                if entry.get("status") == "completed" and entry.get("report")]
    return list(document.get("reports") or [])


def _absorb_build_test(run, document):
    """Splits a build test's ordered `entries` back into the three buckets
    the report sections render.

    The order is lost here on purpose: every section below groups by model
    name, so the timeline the entries preserve is not what this script shows.
    It stays in the JSON for anyone reading the raw artefact.
    """
    run.app_version = document.get("appVersion") or run.app_version
    run.app_build = document.get("appBuild") or run.app_build
    run.generated_at = document.get("generatedAt") or run.generated_at
    run.stages = document.get("stages") or run.stages
    run.device = document.get("device") or run.device
    run.cancelled = bool(document.get("cancelled")) or run.cancelled
    run.interrupted = bool(document.get("interrupted")) or run.interrupted
    if document.get("durationSeconds") is not None:
        try:
            run.duration_seconds = float(document["durationSeconds"])
        except (TypeError, ValueError):
            pass

    for entry in document.get("entries") or []:
        status = entry.get("status")
        if status == "failed":
            run.failures.append(entry)
        elif status == "skipped":
            run.skips.append(entry)
        elif status == "crashed":
            run.crashes.append(entry)


def _absorb_summary(run, summary):
    run.app_version = summary.get("appVersion") or run.app_version
    run.app_build = summary.get("appBuild") or run.app_build
    run.generated_at = summary.get("generatedAt") or run.generated_at
    run.stages = summary.get("stages") or run.stages
    run.device = summary.get("device") or run.device
    run.failures.extend(summary.get("failures") or [])
    run.skips.extend(summary.get("skips") or [])


def _fill_metadata_from_reports(run):
    """A folder of bare reports has no envelope; borrow from the reports."""
    if not run.reports:
        return
    first = run.reports[0]
    run.app_version = run.app_version or first.get("appVersion", "")
    run.app_build = run.app_build or first.get("appBuild", "")
    run.generated_at = run.generated_at or first.get("date", "")
    run.device = run.device or first.get("device", {})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def quality(report):
    return report.get("quality") or None


def category_scores(report):
    """[(category, passed, total)] in suite order, empty for a perf-only run."""
    block = quality(report)
    if not block:
        return []
    result = []
    for category in block.get("categories") or []:
        result.append((
            category.get("category", "?"),
            int(category.get("passed") or 0),
            int(category.get("total") or 0),
        ))
    return result


def chance_sum(report):
    """Sum of the per-task guessing floors (`TaskResult.chance`, absent = 0)."""
    block = quality(report)
    if not block:
        return 0.0
    result = 0.0
    for category in block.get("categories") or []:
        for task in category.get("tasks") or []:
            try:
                result += min(max(float(task.get("chance") or 0.0), 0.0), 1.0)
            except (TypeError, ValueError):
                pass
    return result


def overall_score(report):
    """0-100, the chance-normalised pass rate. None when the quality stage
    did not run.

    Mirrors `BenchmarkReport.overallScore`, which is a computed property and
    therefore never present in the JSON:
    max(0, (passed - sum(chance)) / (graded - sum(chance))) * 100. With no
    multiple-choice tasks (chance 0 everywhere) that is passed / graded.
    """
    categories = category_scores(report)
    if not categories:
        return None
    passed = sum(entry[1] for entry in categories)
    graded = sum(entry[2] for entry in categories)
    if graded <= 0:
        return None
    floor = min(chance_sum(report), float(graded))
    denominator = graded - floor
    if denominator <= 0:
        return 0.0
    return min(max((passed - floor) / denominator, 0.0), 1.0) * 100.0


def truncated(report):
    """True when the quality stage stopped at its time budget."""
    block = quality(report)
    return bool(block and block.get("truncated"))


def score_cell(report):
    """The score as the app prints it, "partial" marker included."""
    score = overall_score(report)
    if score is None:
        return "perf-only"
    if truncated(report):
        block = quality(report)
        return "%s partial (%s of %s tasks)" % (
            number(score, 1), block.get("completed", "?"), block.get("total", "?"))
    return number(score, 1)


def performance(report, key, default=0.0):
    block = report.get("performance") or {}
    value = block.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def performance_optional(report, key):
    """A v2-only performance field: None on old reports, float otherwise."""
    value = (report.get("performance") or {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def throttled(report):
    """True when the run's throughput is not comparable.

    Reads `performance.thermallyConstrained` — the JSON key kept its original
    name for compatibility, but what it records changed: newer reports set it
    when THROTTLING WAS MEASURED (the decode rate fell more than 20% between
    the best and worst repetition, or tg128 landed more than 20% below the
    decode-probe mean), not when `ProcessInfo.thermalState` reached
    `.serious`. Older reports are read exactly as stored.
    """
    return bool((report.get("performance") or {}).get("thermallyConstrained") or False)


def throttle_drop_percent(report):
    """Observed max-to-min decode drop, or None on a report without it.

    Absent on every report written before the measured rule existed, and on
    runs with fewer than two timed repetitions — which is not the same as
    zero, so those rows say `(throttled)` without a number.
    """
    return performance_optional(report, "throttleDropPercent")


def suite_version(report):
    """Version of the quality suite that graded this report, or None.

    Lives inside the quality block (`BenchmarkReport.suiteVersion` is a
    computed property in Swift and never serialized); the top-level key is
    accepted too for hand-written fixtures. Scores are only comparable
    between reports with the SAME suite version.
    """
    block = quality(report)
    version = block.get("suiteVersion") if isinstance(block, dict) else None
    if version is None:
        version = report.get("suiteVersion")
    return version


def metrics_version(report):
    """Version of the throughput measurement methodology, 1 for v1 reports.

    Mirrors `PerformanceMetrics.metricsVersion`, which decodes as 1 when the
    key is absent. Throughput deltas are only comparable between reports with
    the SAME metrics version: v1 numbers were single-shot and unwarmed, so a
    v2-vs-v1 delta shows the methodology change, not the model.
    """
    value = (report.get("performance") or {}).get("metricsVersion")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def standard_protocol(report):
    """The pp512/tg128 block, or None (v1 reports, or the probe was off)."""
    block = (report.get("performance") or {}).get("standardProtocol")
    return block if isinstance(block, dict) else None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def cell(text):
    """Markdown table cells cannot carry a raw pipe."""
    return str(text).replace("|", "\\|")


def table(headers, rows):
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def number(value, digits=1):
    if value is None:
        return "-"
    return ("%%.%df" % digits) % value


def delta(current, previous, digits=1, higher_is_better=True):
    """`12.3 (+1.5, +13.9%)`, or a bare value when there is nothing to compare."""
    if current is None and previous is None:
        return "-"
    if current is None:
        return "dropped"
    if previous is None:
        return "%s (new)" % number(current, digits)

    difference = current - previous
    if abs(previous) > 1e-9:
        percent = difference / abs(previous) * 100.0
        percent_text = ", %+.1f%%" % percent
    else:
        percent_text = ""

    marker = ""
    if abs(difference) > 1e-9:
        improved = difference > 0 if higher_is_better else difference < 0
        marker = " better" if improved else " worse"

    return "%s (%+.*f%s%s)" % (number(current, digits), digits, difference, percent_text, marker)


def gen_tokens_cell(report):
    """`24.6 ± 1.0`, plus `(throttled -46%)` when the run was throttled.

    The drop is dropped from the marker when the report does not carry one
    (older files, single-repetition runs); Old (v1) reports have no standard
    deviation either and render as a plain number, exactly as before.
    """
    text = number(performance(report, "generationTokensPerSecond"), 1)
    stddev = performance_optional(report, "generationTokensPerSecondStdDev")
    if stddev is not None:
        text = "%s ± %s" % (text, number(stddev, 1))
    if throttled(report):
        drop = throttle_drop_percent(report)
        if drop is not None:
            text += " (throttled -%s%%)" % number(drop, 0)
        else:
            text += " (throttled)"
    return text


def standard_protocol_cell(report):
    """`~1180 / 24.6` (pp512 t/s / tg128 t/s), or `-` when the probe did
    not run. The tilde admits the prompt token count is an estimate."""
    block = standard_protocol(report)
    if block is None:
        return "-"
    prefill = block.get("prefillTokensPerSecond")
    generation = block.get("generationTokensPerSecond")
    if prefill is None or generation is None:
        return "-"
    try:
        return "~%.0f / %.1f" % (float(prefill), float(generation))
    except (TypeError, ValueError):
        return "-"


def category_summary(report):
    """`math 5/6, instructions 4/4` - the per-category pass fractions."""
    categories = category_scores(report)
    if not categories:
        return "-"
    return ", ".join("%s %d/%d" % (name, passed, total) for name, passed, total in categories)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def header_section(run, previous):
    lines = ["# Benchmark report", ""]
    lines.append("- App version: %s (build %s)" % (run.app_version or "?", run.app_build or "?"))
    lines.append("- Device: %s" % run.device_line)
    lines.append("- Date: %s" % (run.generated_at or "?"))
    if run.stages:
        lines.append("- Stages: %s" % run.stages)
    lines.append("- Models tested: %d (passed %d, failed %d, crashed %d, skipped %d)" % (
        len(run.reports) + len(run.failures) + len(run.crashes),
        len(run.reports),
        len(run.failures),
        len(run.crashes),
        len(run.skips),
    ))
    if run.duration_seconds is not None:
        lines.append("- Duration: %.0f min" % (run.duration_seconds / 60.0))
    if run.interrupted:
        lines.append("- NOTE: the app CRASHED during this run. The report was rebuilt from "
                     "the run's journal on the next launch: finished models are real "
                     "measurements, the model in flight is listed under Crashed, and the "
                     "models never reached are listed as skipped.")
    elif run.cancelled:
        lines.append("- NOTE: this run was CANCELLED before it finished. The models "
                     "below were measured normally; the rest were never attempted "
                     "and are not listed as skipped.")
    if previous is not None:
        lines.append("- Compared against: %s (app %s, %s)" % (
            os.path.basename(previous.source),
            previous.app_version or "?",
            previous.device_line,
        ))
        if previous.device.get("identifier") != run.device.get("identifier"):
            lines.append("- NOTE: the two runs are from different devices; "
                         "throughput and memory deltas are not comparable.")
    lines.append("")
    return "\n".join(lines)


def results_section(run):
    headers = ["Model", "Backend", "Score", "Gen tok/s", "Prefill tok/s (est.)",
               "pp512/tg128 t/s", "Load s", "Peak MB", "Quality by category"]
    rows = []
    for report in sorted(run.reports, key=lambda item: item.get("modelName", "")):
        rows.append([
            report.get("modelName", report.get("modelId", "?")),
            report.get("backend", "?"),
            score_cell(report),
            gen_tokens_cell(report),
            number(performance(report, "prefillTokensPerSecond"), 1),
            standard_protocol_cell(report),
            number(performance(report, "loadSeconds"), 1),
            number(performance(report, "peakMemoryMB"), 0),
            category_summary(report),
        ])
    body = "## Results\n\n" + table(headers, rows)
    if any(throttled(report) for report in run.reports):
        body += ("\nRuns marked `(throttled)` slowed down while they were being "
                 "measured; their throughput describes how far the device had "
                 "ramped down, not the model, and is not comparable with other "
                 "rows.\n")
    return body


def comparison_section(run, previous):
    headers = ["Model", "Score", "Gen tok/s", "Load s", "Peak MB", "Status"]
    current_by_id = run.reports_by_id()
    previous_by_id = previous.reports_by_id()

    def display_name(model_id):
        report = current_by_id.get(model_id) or previous_by_id.get(model_id) or {}
        return report.get("modelName", model_id)

    rows = []
    # Sorted by display name so this table lines up with Results above.
    for model_id in sorted(set(current_by_id) | set(previous_by_id), key=display_name):
        current = current_by_id.get(model_id)
        earlier = previous_by_id.get(model_id)
        name = display_name(model_id)

        if current is None:
            rows.append([name, "-", "-", "-", "-", "dropped"])
            continue
        if earlier is None:
            rows.append([
                name,
                score_cell(current),
                number(performance(current, "generationTokensPerSecond"), 1),
                number(performance(current, "loadSeconds"), 1),
                number(performance(current, "peakMemoryMB"), 0),
                "new",
            ])
            continue

        # A score is a mark on one specific exam. When the two runs were
        # graded by different suite versions the delta is meaningless, so it
        # is suppressed - tok/s, load and memory deltas stay valid.
        current_suite = suite_version(current)
        earlier_suite = suite_version(earlier)
        current_score = overall_score(current)
        if truncated(current) or truncated(earlier):
            # A partial run was scored on a subset of the suite.
            score_text = "%s (partial run, not comparable)" % score_cell(current)
        elif (current_suite is not None and earlier_suite is not None
                and current_suite != earlier_suite):
            score_text = "%s (suite changed, v%s -> v%s)" % (
                score_cell(current), earlier_suite, current_suite)
        elif current_score is None:
            # A perf-only current run has nothing graded to diff against a
            # graded previous one - "dropped" would misreport this as a
            # regression rather than a difference in what was run.
            score_text = "perf-only"
        else:
            score_text = delta(current_score, overall_score(earlier), 1,
                               higher_is_better=True)

        # Same idea for throughput: a throttled run measures how fast the
        # device was still allowed to go, not the model, so the delta would be
        # meaningless (mirrors the suppression `regressions_section` already
        # does for its own check). v1-vs-v2 metrics-version changes are the
        # other case where the delta compares methodology, not the model.
        current_metrics = metrics_version(current)
        earlier_metrics = metrics_version(earlier)
        if throttled(current) or throttled(earlier):
            gen_cell = "%s (throttled, not comparable)" % (
                number(performance(current, "generationTokensPerSecond"), 1))
        elif current_metrics != earlier_metrics:
            gen_cell = "%s (methodology changed, v%s -> v%s)" % (
                number(performance(current, "generationTokensPerSecond"), 1),
                earlier_metrics, current_metrics)
        else:
            gen_cell = delta(
                performance(current, "generationTokensPerSecond"),
                performance(earlier, "generationTokensPerSecond"), 1,
                higher_is_better=True)

        rows.append([
            name,
            score_text,
            gen_cell,
            delta(performance(current, "loadSeconds"),
                  performance(earlier, "loadSeconds"), 1, higher_is_better=False),
            delta(performance(current, "peakMemoryMB"),
                  performance(earlier, "peakMemoryMB"), 0, higher_is_better=False),
            "-",
        ])

    return "## Compared with the previous run\n\n" + table(headers, rows)


def regressions_section(run, previous):
    """Only real losses: slower generation, or a materially lower score."""
    current_by_id = run.reports_by_id()
    previous_by_id = previous.reports_by_id()

    lines = []
    throttled_excluded = []
    methodology_changed = []
    suite_changed = []
    for model_id in sorted(set(current_by_id) & set(previous_by_id)):
        current = current_by_id[model_id]
        earlier = previous_by_id[model_id]
        name = current.get("modelName", model_id)

        # Throughput check. Skipped - for THIS check only - when either run
        # was throttled (its tok/s measures the device ramping down, not the
        # model) or when the two runs measured throughput under different
        # methodologies (metricsVersion). The score check below still runs in
        # both cases: quality is deterministic and does not depend on clock
        # speed or on how speed was measured.
        if throttled(current) or throttled(earlier):
            throttled_excluded.append(name)
        elif metrics_version(current) != metrics_version(earlier):
            methodology_changed.append(name)
        else:
            current_gen = performance(current, "generationTokensPerSecond")
            earlier_gen = performance(earlier, "generationTokensPerSecond")
            if earlier_gen > 0:
                drop_pct = (earlier_gen - current_gen) / earlier_gen * 100.0
                if drop_pct > GEN_TOKENS_DROP_PCT:
                    lines.append("- %s: generation %.1f -> %.1f tok/s (%.1f%% slower)" % (
                        name, earlier_gen, current_gen, drop_pct))

        # Score check. Scores graded by different suite versions are two
        # different exams; a lower mark on a new exam is not a regression.
        current_suite = suite_version(current)
        earlier_suite = suite_version(earlier)
        if (current_suite is not None and earlier_suite is not None
                and current_suite != earlier_suite):
            suite_changed.append(name)
            continue
        # A partial run (time budget reached) was graded on a subset.
        if truncated(current) or truncated(earlier):
            suite_changed.append(name + " (partial run)")
            continue

        current_score = overall_score(current)
        earlier_score = overall_score(earlier)
        if current_score is not None and earlier_score is not None:
            drop = earlier_score - current_score
            if drop > SCORE_DROP_POINTS:
                lines.append("- %s: score %.1f -> %.1f (%.1f points lost)" % (
                    name, earlier_score, current_score, drop))

    body = "\n".join(lines) if lines else (
        "None. No model lost more than %.0f%% generation throughput or "
        "%.0f score points." % (GEN_TOKENS_DROP_PCT, SCORE_DROP_POINTS)
    )
    if throttled_excluded:
        body += ("\n\nThroughput not compared (throttled run in one of the two "
                 "sweeps, so tok/s is not a regression signal; scores were "
                 "still checked): %s."
                 % ", ".join(throttled_excluded))
    if methodology_changed:
        body += ("\n\nThroughput not compared (measurement methodology "
                 "changed between the runs, so a tok/s delta would show the "
                 "methodology, not the model; scores were still checked): "
                 "%s." % ", ".join(methodology_changed))
    if suite_changed:
        body += ("\n\nScore not compared (quality suite version changed "
                 "between the runs, or a run stopped at its time budget): %s."
                 % ", ".join(suite_changed))
    return "## Regressions\n\n" + body + "\n"


def failures_section(run):
    if not run.failures:
        return ""
    headers = ["Model", "Backend", "Error"]
    rows = [[
        failure.get("modelName", failure.get("modelId", "?")),
        failure.get("backend", "?"),
        failure.get("error", "?"),
    ] for failure in run.failures]
    return "## Failures\n\n" + table(headers, rows)


def crashes_section(run):
    if not run.crashes:
        return ""
    lines = ["## Crashed", ""]
    for crash in run.crashes:
        lines.append("- **%s** [%s]: %s" % (
            crash.get("modelName", crash.get("modelId", "?")),
            crash.get("backend", "?"),
            crash.get("error", "the app was terminated"),
        ))
        tail = crash.get("log") or []
        if tail:
            lines.append("")
            lines.append("  Last %d log lines before the crash:" % len(tail))
            lines.append("")
            lines.append("  ```")
            lines.extend("  " + line for line in tail)
            lines.append("  ```")
        lines.append("")
    return "\n".join(lines)


def skips_section(run):
    if not run.skips:
        return ""
    headers = ["Model", "Backend", "Reason"]
    rows = [[
        skip.get("modelName", skip.get("modelId", "?")),
        skip.get("backend", "?"),
        skip.get("reason", "?"),
    ] for skip in run.skips]
    return "## Skipped\n\n" + table(headers, rows)


def build_report(run, previous):
    sections = [header_section(run, previous), results_section(run)]
    if previous is not None:
        sections.append(comparison_section(run, previous))
        sections.append(regressions_section(run, previous))
    sections.append(failures_section(run))
    sections.append(crashes_section(run))
    sections.append(skips_section(run))
    return "\n".join(section for section in sections if section).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a Markdown benchmark report from a Pocket AI Lab sweep.")
    parser.add_argument(
        "input",
        help="Path to a release summary JSON, a single report JSON, "
             "or a directory containing either.")
    parser.add_argument(
        "--previous",
        help="Path to the previous release's summary, for delta columns.")
    parser.add_argument(
        "--out",
        help="Write the Markdown here instead of stdout.")
    arguments = parser.parse_args(argv)

    try:
        run = load_run(arguments.input)
    except (OSError, ValueError) as error:
        print("error: could not read %s: %s" % (arguments.input, error), file=sys.stderr)
        return 1

    previous = None
    if arguments.previous:
        try:
            previous = load_run(arguments.previous)
        except (OSError, ValueError) as error:
            print("error: could not read %s: %s" % (arguments.previous, error), file=sys.stderr)
            return 1

    markdown = build_report(run, previous)

    if arguments.out:
        with open(arguments.out, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print("wrote %s" % arguments.out, file=sys.stderr)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

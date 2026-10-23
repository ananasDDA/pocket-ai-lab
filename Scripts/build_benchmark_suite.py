#!/usr/bin/env python3
"""Regenerate the dataset-derived half of local_ai_test/Resources/benchmarks.json.

WHY THIS EXISTS
    MLPerf Mobile v6.0 grades on-device LLMs with tasks drawn from TinyMMLU
    and IFEval. Our quality stage only produces comparable numbers if it runs
    the same source items, so the two blocks below are converted straight from
    the published datasets rather than written by hand.

WHAT IT GUARANTEES
    * Deterministic: one fixed seed drives every sample, so re-running the
      script on the same dataset revisions reproduces the same task ids.
    * Additive: hand-written tasks and every existing id survive untouched.
      Ids are the key in stored reports -- renaming one silently breaks
      comparability with reports users already have on their phones.
    * No LLM judge: every generated check is one of the deterministic kinds
      BenchmarkSuite.swift already understands. A source item whose constraint
      none of those kinds can express is dropped, never approximated.

USAGE
    python3 Scripts/build_benchmark_suite.py [--dry-run]

    Needs network access (Hugging Face datasets server) and nothing else --
    no third-party Python packages, so it runs on a stock macOS interpreter.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict
from pathlib import Path

# --- Configuration -----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SUITE_PATH = REPO_ROOT / "local_ai_test" / "Resources" / "benchmarks.json"

# One seed for both blocks. Changing it reshuffles the selection and therefore
# invalidates comparability with every report already collected -- treat it as
# part of the suite version, not as a knob.
SEED = 20260609

SUITE_VERSION = 3
SCHEMA_VERSION = 2

MMLU_TARGET = 50
IFEVAL_TARGET = 24

# A phone pays for the prompt before it generates a token, and MMLU carries a
# long tail of multi-paragraph law and philosophy items. Anything past this
# budget would dominate the runtime of the whole suite for one question.
MMLU_MAX_PROMPT_CHARS = 700
IFEVAL_MAX_PROMPT_CHARS = 700

# Prefixes owned by this script. Tasks carrying them are regenerated; every
# other task in the file is copied through verbatim.
MMLU_ID_PREFIX = "know_mmlu_"
IFEVAL_ID_PREFIX = "instr_ifeval_"

HF_ROWS = "https://datasets-server.huggingface.co/rows"

SOURCES = [
    OrderedDict(
        [
            ("id", "tinyMMLU"),
            ("dataset", "tinyBenchmarks/tinyMMLU"),
            ("url", "https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU"),
            ("project", "https://github.com/felipemaiapolo/tinyBenchmarks"),
            ("license", "MIT"),
            ("upstream", "MMLU (hendrycks/test), MIT"),
            ("idPrefix", MMLU_ID_PREFIX),
        ]
    ),
    OrderedDict(
        [
            ("id", "IFEval"),
            ("dataset", "google/IFEval"),
            ("url", "https://huggingface.co/datasets/google/IFEval"),
            ("project", "https://github.com/google-research/google-research/tree/master/instruction_following_eval"),
            ("license", "Apache-2.0"),
            ("upstream", "Zhou et al., Instruction-Following Eval"),
            ("idPrefix", IFEVAL_ID_PREFIX),
        ]
    ),
]


# --- Download ----------------------------------------------------------------


def fetch_rows(dataset: str, config: str, split: str) -> list[dict]:
    """Every row of a split, paged through the datasets server (100 max/call)."""
    rows: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split,
             "offset": offset, "length": 100}
        )
        payload = None
        # The datasets server answers 502 while it warms a cold split up.
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{HF_ROWS}?{query}", timeout=90) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        assert payload is not None
        batch = payload.get("rows", [])
        rows.extend(item["row"] for item in batch)
        total = payload.get("num_rows_total", len(rows))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return rows


# --- TinyMMLU ----------------------------------------------------------------

LETTERS = ["A", "B", "C", "D"]

MMLU_SYSTEM_PROMPT = (
    "You are taking a multiple choice exam. Reply with a single letter and nothing else."
)


def mmlu_answer_pattern(letter: str) -> str:
    """Accept the letter alone or behind a short 'the answer is' preamble.

    lm-evaluation-harness scores MMLU by log-likelihood over the option
    letters, which a generate-only backend cannot do. This is the closest
    deterministic equivalent: the FIRST letter the model commits to has to be
    the right one, so "B" and "the answer is B" both count against option A.
    """
    return (
        r"(?i)^[^A-Za-z]*"
        r"(?:[A-Za-z' ]{0,40}\b(?:answer|option|choice)\b[^A-Za-z]{0,6}"
        r"(?:is\b[^A-Za-z]{0,6})?)?"
        + letter
        + r"\b"
    )


def mmlu_prompt(row: dict) -> str:
    subject = row["subject"].replace("_", " ")
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"])
    )
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"{row['question'].strip()}\n"
        f"{options}\n\n"
        "Answer with a single letter: A, B, C or D. Write nothing else."
    )


def build_mmlu_tasks(rows: list[dict], rng: random.Random) -> tuple[list[dict], list[str]]:
    dropped: list[str] = []
    eligible: list[tuple[int, dict]] = []
    for index, row in enumerate(rows):
        if len(row["choices"]) != len(LETTERS):
            dropped.append(f"row {index}: {len(row['choices'])} choices, expected 4")
            continue
        size = len(row["question"]) + sum(len(choice) for choice in row["choices"])
        if size > MMLU_MAX_PROMPT_CHARS:
            dropped.append(f"row {index}: prompt {size} chars over the {MMLU_MAX_PROMPT_CHARS} budget")
            continue
        eligible.append((index, row))

    picked = sorted(rng.sample(eligible, min(MMLU_TARGET, len(eligible))), key=lambda pair: pair[0])

    tasks = []
    for index, row in picked:
        letter = LETTERS[row["answer"]]
        tasks.append(
            OrderedDict(
                [
                    ("id", f"{MMLU_ID_PREFIX}{index:03d}"),
                    ("category", "knowledge"),
                    ("prompt", mmlu_prompt(row)),
                    ("systemPrompt", MMLU_SYSTEM_PROMPT),
                    # Enough for "The answer is C." and no room to ramble.
                    ("maxTokens", 12),
                    ("check", OrderedDict([("kind", "regex"), ("pattern", mmlu_answer_pattern(letter))])),
                ]
            )
        )
    return tasks, dropped


# --- IFEval ------------------------------------------------------------------

# Floor used by every regex-shaped constraint. Without it a model that returns
# an empty string would satisfy "no commas", "all lowercase" and friends.
FLOOR = 40

DEFAULT_IFEVAL_TOKENS = 288


def _regex(pattern: str) -> OrderedDict:
    return OrderedDict([("kind", "regex"), ("pattern", pattern)])


def _escape_alternatives(words: list[str]) -> str:
    return "|".join(re.escape(word) for word in words)


def map_forbidden_words(kwargs: dict, _prompt: str):
    words = kwargs.get("forbidden_words") or []
    if not words:
        return None
    # Word boundaries, not substrings: IFEval's own checker allows "fields"
    # when "field" is banned, and `containsNone` (substring, on normalized
    # text) would wrongly fail it. The length floor keeps an empty answer
    # from passing by saying nothing at all.
    return _regex(rf"(?is)^(?!.*\b(?:{_escape_alternatives(words)})\b).{{{FLOOR},}}$"), DEFAULT_IFEVAL_TOKENS


def map_end_checker(kwargs: dict, _prompt: str):
    phrase = kwargs.get("end_phrase")
    if not phrase:
        return None
    return OrderedDict([("kind", "endsWith"), ("value", phrase)]), DEFAULT_IFEVAL_TOKENS


def map_keyword_frequency(kwargs: dict, _prompt: str):
    # Only "at least": a "less than" ceiling is satisfied by saying nothing,
    # and our token cap would hand that pass to any model that runs long.
    if kwargs.get("relation") != "at least":
        return None
    keyword, frequency = kwargs.get("keyword"), kwargs.get("frequency")
    if not keyword or not frequency or len(keyword) < 4:
        return None
    return (
        OrderedDict([("kind", "letterFrequency"), ("letter", keyword.lower()), ("min", int(frequency))]),
        DEFAULT_IFEVAL_TOKENS,
    )


def map_letter_frequency(kwargs: dict, _prompt: str):
    if kwargs.get("let_relation") != "at least":
        return None
    letter, frequency = kwargs.get("letter"), kwargs.get("let_frequency")
    if not letter or not frequency:
        return None
    return (
        OrderedDict([("kind", "letterFrequency"), ("letter", letter.lower()), ("min", int(frequency))]),
        DEFAULT_IFEVAL_TOKENS,
    )


def map_number_words(kwargs: dict, _prompt: str):
    if kwargs.get("relation") != "at least":
        return None
    words = kwargs.get("num_words")
    # 400-word essays are a throughput test, not an instruction test, and one
    # of them costs more phone time than ten other tasks together.
    if not words or words > 150:
        return None
    return OrderedDict([("kind", "wordCount"), ("min", int(words))]), int(words) * 2 + 96


def map_keyword_existence(kwargs: dict, _prompt: str):
    keywords = kwargs.get("keywords") or []
    if not keywords:
        return None
    return (
        OrderedDict([("kind", "containsAll"), ("values", [word.lower() for word in keywords])]),
        DEFAULT_IFEVAL_TOKENS,
    )


def map_json_format(_kwargs: dict, _prompt: str):
    # No required keys: the instruction is "answer in JSON", so the check is
    # simply that a JSON object is there at all.
    return OrderedDict([("kind", "validJSON"), ("requiredKeys", [])]), DEFAULT_IFEVAL_TOKENS


def map_constrained_response(_kwargs: dict, _prompt: str):
    return (
        OrderedDict(
            [
                ("kind", "containsAny"),
                ("values", ["my answer is yes", "my answer is no", "my answer is maybe"]),
            ]
        ),
        64,
    )


def map_placeholders(kwargs: dict, _prompt: str):
    count = kwargs.get("num_placeholders")
    if not count or count > 8:
        return None
    return _regex(rf"(?s)^(?:[^\[]*\[[^\]\n]{{1,60}}\]){{{int(count)}}}"), DEFAULT_IFEVAL_TOKENS


def map_repeat_prompt(kwargs: dict, _prompt: str):
    repeated = kwargs.get("prompt_to_repeat")
    if not repeated:
        return None
    words = len(repeated.split())
    if words > 60:
        return None
    # The check only inspects the beginning, so the budget just has to cover
    # the repetition plus a visible start of the real answer.
    return OrderedDict([("kind", "startsWith"), ("value", repeated)]), words * 3 + 128


def map_highlighted_sections(kwargs: dict, _prompt: str):
    count = kwargs.get("num_highlights")
    if not count or count > 6:
        return None
    return _regex(rf"(?s)^(?:[^*]*\*+[^*\n]{{1,120}}\*+){{{int(count)}}}"), DEFAULT_IFEVAL_TOKENS


def map_quotation(_kwargs: dict, _prompt: str):
    # Curly quotes accepted: a model that wrapped the answer typographically
    # followed the instruction, it just used the other character.
    return _regex(rf'(?s)^\s*["“].{{{FLOOR},}}["”]\s*$'), DEFAULT_IFEVAL_TOKENS


def map_lowercase(_kwargs: dict, _prompt: str):
    return _regex(rf"(?s)^[^A-Z]{{{FLOOR},}}$"), DEFAULT_IFEVAL_TOKENS


def map_capital(_kwargs: dict, _prompt: str):
    return _regex(rf"(?s)^[^a-z]{{{FLOOR},}}$"), DEFAULT_IFEVAL_TOKENS


def map_no_comma(_kwargs: dict, _prompt: str):
    return _regex(rf"(?s)^[^,]{{{FLOOR},}}$"), DEFAULT_IFEVAL_TOKENS


def map_title(_kwargs: dict, _prompt: str):
    return _regex(r"(?s)^(?=.{60,}$).*<<[^<>\n]{1,80}>>"), DEFAULT_IFEVAL_TOKENS


def map_two_responses(_kwargs: dict, _prompt: str):
    return _regex(r"(?s)^.{20,}\*{6}.{20,}$"), DEFAULT_IFEVAL_TOKENS


# Instruction ids we can grade exactly. Everything absent from this table is
# dropped on purpose -- see DROP_NOTES for why.
IFEVAL_MAPPERS = {
    "keywords:forbidden_words": map_forbidden_words,
    "startend:end_checker": map_end_checker,
    "keywords:frequency": map_keyword_frequency,
    "keywords:letter_frequency": map_letter_frequency,
    "length_constraints:number_words": map_number_words,
    "keywords:existence": map_keyword_existence,
    "detectable_format:json_format": map_json_format,
    "detectable_format:constrained_response": map_constrained_response,
    "detectable_content:number_placeholders": map_placeholders,
    "combination:repeat_prompt": map_repeat_prompt,
    "detectable_format:number_highlighted_sections": map_highlighted_sections,
    "startend:quotation": map_quotation,
    "change_case:english_lowercase": map_lowercase,
    "change_case:english_capital": map_capital,
    "punctuation:no_comma": map_no_comma,
    "detectable_format:title": map_title,
    "combination:two_responses": map_two_responses,
}

DROP_NOTES = {
    "language:response_language": "needs a language identifier; no such check kind",
    "length_constraints:number_sentences": "no sentence-count check kind",
    "length_constraints:number_paragraphs": "paragraph splitting differs from lineCount",
    "length_constraints:nth_paragraph_first_word": "needs paragraph indexing",
    "change_case:capital_word_frequency": "needs per-word case counting",
    "detectable_format:number_bullet_lists": "exact bullet counts need line-level counting",
    "detectable_content:postscript": "P.S. marker detection produces false positives",
    "detectable_format:multiple_sections": "needs section-splitter counting",
}


def build_ifeval_tasks(rows: list[dict], rng: random.Random) -> tuple[list[dict], Counter]:
    dropped = Counter()
    by_instruction: dict[str, list[tuple]] = {}

    for row in rows:
        instructions = row["instruction_id_list"]
        # One task carries exactly one check, so a multi-constraint item could
        # only ever be graded on part of what it asks for. Single-instruction
        # items keep our score honest against IFEval's strict accuracy.
        if len(instructions) != 1:
            dropped["multiple instructions in one item"] += 1
            continue
        instruction = instructions[0]
        mapper = IFEVAL_MAPPERS.get(instruction)
        if mapper is None:
            dropped[DROP_NOTES.get(instruction, f"unmapped instruction {instruction}")] += 1
            continue
        if len(row["prompt"]) > IFEVAL_MAX_PROMPT_CHARS:
            dropped["prompt over the phone budget"] += 1
            continue
        kwargs = row["kwargs"][0] if row.get("kwargs") else {}
        mapped = mapper(kwargs, row["prompt"])
        if mapped is None:
            dropped[f"{instruction}: parameters out of range"] += 1
            continue
        check, max_tokens = mapped
        by_instruction.setdefault(instruction, []).append((row["key"], row["prompt"], check, max_tokens))

    # Round-robin over instruction kinds so the block covers as many distinct
    # constraint types as possible instead of 24 variations of one.
    order = sorted(by_instruction)
    rng.shuffle(order)
    for instruction in order:
        by_instruction[instruction].sort(key=lambda item: item[0])
        rng.shuffle(by_instruction[instruction])

    selected: list[tuple] = []
    cursor = 0
    while len(selected) < IFEVAL_TARGET:
        added = False
        for instruction in order:
            bucket = by_instruction[instruction]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) == IFEVAL_TARGET:
                    break
        if not added:
            break
        cursor += 1

    selected.sort(key=lambda item: item[0])
    tasks = [
        OrderedDict(
            [
                ("id", f"{IFEVAL_ID_PREFIX}{key}"),
                ("category", "instructions"),
                ("prompt", prompt),
                ("maxTokens", max_tokens),
                ("check", check),
            ]
        )
        for key, prompt, check, max_tokens in selected
    ]
    return tasks, dropped


# --- Assembly ----------------------------------------------------------------


def merge(existing: list[dict], mmlu: list[dict], ifeval: list[dict]) -> list[dict]:
    """Keep hand-written tasks in place, append each generated block to its own
    category so the file stays grouped the way the UI reads it."""
    merged = [
        task for task in existing
        if not task["id"].startswith((MMLU_ID_PREFIX, IFEVAL_ID_PREFIX))
    ]
    for block, category in ((ifeval, "instructions"), (mmlu, "knowledge")):
        if not block:
            continue
        after = max(i for i, task in enumerate(merged) if task["category"] == category) + 1
        merged = merged[:after] + block + merged[after:]
    return merged


def validate(tasks: list[dict]) -> None:
    ids = [task["id"] for task in tasks]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate task ids: {duplicates}")
    generated = (MMLU_ID_PREFIX, IFEVAL_ID_PREFIX)
    for task in tasks:
        check = task["check"]
        # Only the generated patterns are checked here: the hand-written ones
        # use ICU classes such as \p{L} that Python's `re` cannot compile.
        if check["kind"] == "regex" and task["id"].startswith(generated):
            re.compile(check["pattern"])
        if not task["prompt"].strip():
            raise SystemExit(f"{task['id']}: empty prompt")
        if task["maxTokens"] <= 0:
            raise SystemExit(f"{task['id']}: non-positive maxTokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the summary without writing")
    args = parser.parse_args()

    existing = json.loads(SUITE_PATH.read_text(encoding="utf-8"))

    mmlu_rows = fetch_rows("tinyBenchmarks/tinyMMLU", "all", "test")
    ifeval_rows = fetch_rows("google/IFEval", "default", "train")

    mmlu_tasks, mmlu_dropped = build_mmlu_tasks(mmlu_rows, random.Random(SEED))
    ifeval_tasks, ifeval_dropped = build_ifeval_tasks(ifeval_rows, random.Random(SEED))

    tasks = merge(existing["tasks"], mmlu_tasks, ifeval_tasks)
    validate(tasks)

    document = OrderedDict(
        [
            ("schemaVersion", SCHEMA_VERSION),
            ("suiteVersion", SUITE_VERSION),
            # Provenance rides along in the shipped resource so the bundle
            # carries the dataset notices it has to carry. The Swift decoder
            # ignores keys it does not know, so this costs no code change.
            ("generator", "Scripts/build_benchmark_suite.py"),
            ("samplingSeed", SEED),
            ("sources", SOURCES),
            ("tasks", tasks),
        ]
    )

    print(f"tinyMMLU rows fetched: {len(mmlu_rows)}, converted: {len(mmlu_tasks)}")
    print(f"  source row indices: {[t['id'][len(MMLU_ID_PREFIX):] for t in mmlu_tasks]}")
    print(f"  dropped: {len(mmlu_dropped)} (prompt budget / shape)")
    print(f"IFEval rows fetched: {len(ifeval_rows)}, converted: {len(ifeval_tasks)}")
    print(f"  source keys: {[t['id'][len(IFEVAL_ID_PREFIX):] for t in ifeval_tasks]}")
    for reason, count in ifeval_dropped.most_common():
        print(f"  dropped {count:>4}  {reason}")
    print("category totals:")
    for category, count in Counter(task["category"] for task in tasks).most_common():
        print(f"  {category:<14}{count}")
    print(f"total tasks: {len(tasks)}  suiteVersion: {SUITE_VERSION}")
    print(f"generation budget: {sum(task['maxTokens'] for task in tasks)} tokens")

    if args.dry_run:
        return
    SUITE_PATH.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {SUITE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())

# Benchmark data licenses

`local_ai_test/Resources/benchmarks.json` ships inside the app. Part of it is
derived from public evaluation datasets, so the app binary redistributes their
content and has to carry their notices. This file is that notice.

The same attribution is duplicated inside `benchmarks.json` itself, in the
top-level `sources` array, so a copy travels with the resource even when the
file is served on its own from the GitHub raw URL.

Regenerate the derived tasks with `python3 Scripts/build_benchmark_suite.py`.

## tinyMMLU — task ids `know_mmlu_*`

- Dataset: `tinyBenchmarks/tinyMMLU`
- Source: https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU
- Project: https://github.com/felipemaiapolo/tinyBenchmarks
- License: MIT
- Upstream: MMLU / `hendrycks/test` (Hendrycks et al., *Measuring Massive
  Multitask Language Understanding*), MIT

Modifications: 50 of the 100 curated items are sampled, and each is rewritten
from the dataset's `question` / `choices` / `answer` fields into a lettered
multiple-choice prompt with a regex check on the answer letter. No question or
answer option text is altered.

## IFEval — task ids `instr_ifeval_*`

- Dataset: `google/IFEval`
- Source: https://huggingface.co/datasets/google/IFEval
- Project: https://github.com/google-research/google-research/tree/master/instruction_following_eval
- License: Apache License 2.0
- Upstream: Zhou et al., *Instruction-Following Evaluation for Large Language
  Models*

Modifications: 24 single-instruction items are sampled and their `prompt` field
is used verbatim. The verifier is not the upstream Python checker — each
instruction is re-expressed as one of the deterministic check kinds in
`BenchmarkSuite.swift`. Apache-2.0 section 4(b) requires this note that the
files carry modifications; no upstream `NOTICE` file exists for this dataset.

## Why both are in the app

MLPerf Mobile v6.0 grades on-device LLMs with tasks drawn from TinyMMLU and
IFEval. Running the same source items is what makes the app's quality score
comparable with a published industry number instead of only with itself.

## Open for a human decision

- Neither notice is shown anywhere in the app UI yet. A shipping build should
  surface them next to the existing model-license screen, or link this file
  from the README's Acknowledgements section.
- The MIT and Apache-2.0 texts are referenced by name here, not reproduced.
  Apache-2.0 asks for a copy of the license to accompany the redistribution;
  decide whether to vendor the full text or link to it.

# PersistBench files (vendored)

Copied unmodified from [github.com/ivaxi0s/PersistBench](https://github.com/ivaxi0s/PersistBench)
at commit `d2ded18be6d15ebb14008d081a44bd03c20670e4` (2026-09-15). Paper: Pulipaka et al., ICML 2026,
[arXiv:2602.01146](https://arxiv.org/abs/2602.01146).

The PersistBench repo has no license file, so these copies are for private research use. Keep this
repo private, or get the authors' permission, before publishing it.

| File | Source | Used for |
|---|---|---|
| `generic_prompt.txt` | `prompts/generic_prompt.txt` | The policy's system prompt, in training and evaluation |
| `judges/cross_domain.txt` | `prompts/judges/cross_domain.txt` | Kimi judge system prompt, leakage items (score 1–5, ≥3 = fail) |
| `judges/beneficial_samples.txt` | `prompts/judges/beneficial_samples.txt` | Kimi judge system prompt, beneficial items (rating 1–3, 3 = pass) |
| `eval_subset.jsonl` | rows of `benchmark_samples/*.jsonl` | Fixed evaluation subset (see below) |

The judge prompt files are identical to the constants PersistBench's code actually uses
(`JUDGE_SYSTEM_PROMPT_SINGLE_CROSS_DOMAIN`, `JUDGE_SYSTEM_PROMPT_SINGLE_POSITIVE_MEMORY` in
`src/benchmark/prompts.py`); checked byte for byte.

**`eval_subset.jsonl`**: 100 of the 200 cross-domain items (indices drawn with `random.Random(0).sample`)
plus all 100 beneficial items. Rows are unmodified except for an added `id` field,
`<source file>:<row index>`.

`evaluate.py` reproduces PersistBench's generation and judge call (prompt construction, Kimi-K2-Thinking on
OpenRouter at temperature 0 routed to Google Vertex with high reasoning effort, score parsing) and its
failure-rate definitions, with one generation per item.

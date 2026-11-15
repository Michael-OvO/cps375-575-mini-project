# External data candidates (2026-11-15)

Goal: widen the prompt distribution (helps the held-out sets) without changing the answer format the
graders expect (helps the public benchmarks). Rule for every source: **take prompts/problems only, re-answer
with our teachers in our format, verify, decontaminate, select by what the student fails.** Long R1-style
traces from these datasets are not used as targets: they exceed our 2,048-token training cap and drift from
the graded format.

| Area | Source | License | Size | What it adds | Risk and check |
|---|---|---|---|---|---|
| IF | `allenai/IF_multi_constraints_upto5` (IF-RLVR train) | ODC-BY-1.0 | 95k | 29 IFBench-train constraint types with verifier code, up to 5 constraints per prompt; prompts from Tulu SFT | Our largest IF gap is unseen types (217 of 284 failures). Kept disjoint from IFBench test by the authors. Port the 29 verifiers into our registry; prompts from Tulu, not IFEval examples |
| Code | `KodCode/KodCode-V1` | CC BY-NC 4.0 (fine for coursework) | 487k | Tasks with executed unit tests, 12 subsets; `Complete` style is docstring completion (HumanEval/held-out shape) | Card gives `benchmark_similarity` to HumanEval/MBPP per row: drop high-similarity rows, then our 8-gram filter; re-run its tests |
| Code | `nvidia/OpenCodeInstruct` (suggested by PROJECT.md) | see card | 5M | Broad instruction styles, assert tests with pass rates | n-gram decontaminated per paper; still run our filter |
| Math | `zwhe99/DeepMath-103K` | MIT | 103k | Hard problems with verified final answers, many topics outside MATH style | Mostly harder than MATH (levels 5-9): take the lower part; semantic decontamination only, so run our filter against MATH-500 **and** the full MATH test (our proxy) |
| Math | NuminaMath / OpenR1-Math problems | see cards | 100k+ | Competition problems with answers | Known MATH-500 near-duplicates (numeric variants) in several open sets: filter strictly |

Held-out check for every addition: gain on the proxies (HumanEval, MATH test minus MATH-500, IF unseen and
new types, discovery probes), not only on the public benchmarks. Acceptance rule unchanged.

Sources: arXiv 2507.02833 (Generalizing Verifiable Instruction Following), 2503.02951 (KodCode),
2504.04030 (OpenCodeInstruct), 2504.11456 (DeepMath-103K), 2503.10460 (Light-R1, contamination finding),
2502.03387 (LIMO), and the Hugging Face dataset cards.

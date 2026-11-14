# Weakness discovery, round 1 (v1)

Date: 2026-11-14.
- **Probes:** 2,529 fresh Opus-written probes over 212 cells (62 math, 86 code, 64 IF), about 12 per cell.
  They were verified and checked by the test-overlap filter, and they are never trained on.
- **Cost:** about $113 of Opus plus under $2 on Tinker.
- **Full tables:** `reports/weakness_registry.md` (generated); raw data in `data/discover/`.

The probes target rare combinations, so v1 scores far below dev:

| Task | v1 on probes | v1 on dev | Teacher passes |
|---|---|---|---|
| Math | 48% | 68% | 98% |
| Code | 57% | 74% | 100% |
| IF | 29% | 66% | 69% |

Read the contrasts between facet values, not the absolute rates. One cell (12 probes) is too noisy to act on. The pooled facet values below have 36 to 300 probes each.

## Findings (v1 failure rate, 95% CI)

**Code.** 86% of failures fail an assert shown in the prompt, not only hidden ones. This confirms the v1 error analysis: the model does not check its code against the visible tests.
- **Spec resolved only by the asserts** (indexing, inclusive or exclusive bounds): 60% [54, 66] vs 38% for the other styles. HumanEval-style prompts alone are not the problem (39%).
- **Input regimes:**
  - deeply nested structures: 78% [69, 85];
  - date/time strings: 75% [65, 83];
  - ties and tie-break rules: 54%;
  - empty inputs, for comparison: 23%.
- **Output contracts:** nested lists with a fixed inner order 65% [56, 73]; dictionaries 53%. Booleans and plain lists: about 32%.
- **Topics (12 probes each, weak evidence):** roman numerals, fractions and gcd, regex, named DP problems, bit manipulation, all at 83–92%.

**Math.**
- **Answer types:**
  - complex numbers 81% [65, 90];
  - numerals in another base 78% [62, 88];
  - degrees 69%;
  - radicals 65%;
  - tuples 60%.

  Compare plain expressions at 32% and normalized line or plane equations at 6%.
- **Techniques:** complementary counting 88%, functional equations 88%, CRT and modular inverses 75%, inclusion–exclusion and stars-and-bars 72%, casework 67%.
- **Subjects:** Precalculus 65% and Counting & Probability 64%, vs Prealgebra 18%. Level 3 probes still fail 37% of the time.
- **Not a weakness:** problems with `[asy]` diagrams (42%, below average).

**IF.** Teacher pass is only 69%, so part of the gap is not learnable.
- **Number of rules:** 4 rules 86% and 5 rules 94% vs 48% for 1–2. The teacher also drops (63%, 52%), but stays well above v1.
- **Families:** position 100%, length 92%, chars 89%, letters 83%, tags 80%.
- **Position:** rules placed before the task 77% vs 68% after.
- **Constraint types:** dev-only types 80% vs training types 68% (transfer gap, as in the v1 dev set).
- **Wording and language:** paraphrased wording is only slightly worse (72–74% vs 67%). Language is not a differential weakness.

## Action for v2 (data, not probes)

| Weakness | Generator change |
|---|---|
| Code: does not check visible asserts; spec resolved by asserts | `code-plan` already hand-checks one assert. Add new Opus tasks in the weak cells (ambiguous spec, nested and date inputs, tie-breaks, nested-list and dict outputs, the weak topics) with plan-style solutions. |
| Math: weak answer types and counting techniques | New problems in those cells with long self-checking solutions, verified by two agreeing solves. |
| IF: 3–5 rules, rules before the task, weak families | Shift the `if-count` / `if-new` mix toward 3–5 rules and rules-first or mixed placement. Keep only items the teacher passes. |

Probes stay held out: after v2, rescore with `python -m pipeline.discover.measure score --name v2 ...` to see which weaknesses closed.

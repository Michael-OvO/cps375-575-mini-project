# v1 error analysis

Date: 2026-11-14. Checkpoint: `runs/v1/checkpoint.json`. Per-item outputs for base, smoke and v1 on the
private dev sets are in `runs/{base,smoke,v1}/errors/{math,code,if}.jsonl` (written by
`python -m pipeline.error_analysis collect`; tables by `... summary --names base smoke v1`).

Public test sets were used for aggregate counts only (per instruction type, per level and subject, per
failure category). No public test item was read to design data, so nothing here tunes on the test sets.

## Scores

| | IFEval | MATH-500 | MBPP | dev math | dev code | dev IF prompts | HumanEval | MATH test minus 500 |
|---|---|---|---|---|---|---|---|---|
| base | 47.9 | 15.8 | 65.4 | 17.2 | 65.0 | 18.1 | 70.7 | 18.8 |
| smoke (1k examples) | | | 70.0 | 65.5 | 57.3 | 34.1 | | |
| v1 (35.7k examples) | 72.0 | 70.0 | 75.1 | 67.7 | 74.5 | 44.7 | 67.1 | 70.8 |

## Finding 1. Math: the short reasoning style is the ceiling, not the amount of data

- Smoke (500 math examples) and v1 (21,697) score 65.5 and 67.7 on the 325 dev problems. v1 won 25 and
  lost 18 against smoke. Forty times more math data bought about 2 points.
- Accuracy by level: L1 100, L2 85, L3 80, L4 69, **L5 39**. Weakest subjects: Geometry 44,
  Intermediate Algebra 52, Precalculus 60. Public MATH-500 shows the same shape (L5 42, Intermediate
  Algebra 49, Precalculus 48).
- v1 writes short solutions: median 147 tokens when right, 358 when wrong. Training targets have a median
  of 193 (MATH train) and 239 (new problems), p90 about 400 to 470.
- The base model reasons at length (it emits `<think>` and usually runs into the 4,096-token cap). When
  it does finish, it is right on **18 of 24 level-5 problems**. On 16 dev problems the base model is right
  and v1 is wrong; the base model used 1,100 to 4,000 tokens on them, v1 used 150 to 450.
- About 12 of v1's 105 wrong answers give up: "no solution", "No such k", "NEI", or a multiple-choice
  letter on a problem with a numeric answer. Only 23 of 21,697 training targets look like that (17 are
  multiple-choice MATH-train items, 6 are ill-posed teacher problems), so this is the model bailing out
  when stuck, not something the data taught it. The 6 ill-posed teacher problems should still be removed.
- 6 dev outputs (and 8 on MATH-500) hit the 4,096 cap, mostly repetition loops.
- No scorer false negatives were found among the 105 wrong answers.

Conclusion: SFT replaced the base model's long reasoning with a ~200-token style. That fixed the
answer format (17 to 68) but caps hard problems. More short solutions will not move this.

## Finding 2. Code: v1 helps on dev and MBPP, but writes code without checking it

- Dev code 65.0 (base), 57.3 (smoke), 74.5 (v1). MBPP 65.4, 70.0, 75.1. HumanEval 70.7 to 67.1 (within
  noise on 164 items, but in the wrong direction).
- v1 failures: 75 fail the given tests, 10 fail only hidden tests, 1 wrong function name, no syntax
  errors, no missing code blocks. All 64 MBPP failures are wrong results on the given tests.
- The given tests are in the prompt, yet v1 fails them. Hand-read failures are logic errors that one
  pass over the given asserts would catch: a clock angle wrong for negative minutes that the prompt's
  own assert covers, returning minutes where the assert expects degrees, reversing "-120" into "021-",
  using `math.factorial` without importing `math`.
- Weakest topics: dates and times (2/6), sorting with custom keys (4/9), recursion, simple dynamic
  programming, arithmetic and rounding, fractions and gcd, two-pointer (all 50 to 56%).
- The base model thinks before coding; v1 answers immediately (median 57 tokens). On the hardest dev
  tasks that costs it, as the smoke run already showed (lost 19, won 13 against base).

## Finding 3. Instruction following: trained types improve, new types do not

- Per-constraint pass rate on dev, smoke to v1: library types 68 to 80, our 30 training types 56 to 68,
  **the 6 held-out types 50 to 50**. 6,000 instruction-following examples taught the types we trained
  on and nothing measurable about types the model has not seen. The hidden benchmark uses constraints we
  have not seen, so this is the number that matters most for that part of the grade.
- Held-out types: semicolons 2/32, max words per line 2/22, quote pairs 8/20, open question 27/39,
  hashtags 26/34, min chars 18/19. The model mostly ignores the first two.
- Counting is the main weakness among seen types too: exact words 0/9, exact sentences 1/5, exact keyword
  count 1/5, letter frequency 9/21. Outputs land near the target (42 asked, 54 written; 3 semicolons
  asked, 4 written). Public IFEval agrees: the worst types are keyword frequency 48, letter frequency
  48, number of words 52, keyword existence 56, number of sentences 60.
- More constraints per prompt, lower pass rate: public 77 / 59 / 35% for 1 / 2 / 3 instructions; dev 65 /
  41 / 37 / 38% for 1 to 4.
- **Repetition loops**: 58 of 541 public IFEval outputs (11%) hit the 1,024-token cap; 49 of those end in
  a verbatim loop and only half of the 58 pass. Most common types among them: keyword frequency,
  response language (non-English), number of words. On dev, 10 of 11 capped outputs are loops.
- Teacher keep rate falls with constraint count (97 / 88 / 82 / 72% for 1 to 4) and is lowest for
  nth-paragraph-first-word (53%), exact sentences (62%), paragraph start (64%). The verification filter
  therefore under-represents the hardest combinations in training, but only mildly; exact word counts,
  which the student fails completely, are kept at over 82%, so that weakness is the student's, not a
  data gap.

## Finding 4. Dev set problems

- Some dev instruction-following items cannot be satisfied: "the letter e fewer than 9 times" in a
  150-word essay; "a single comma-separated line" together with "exactly 7 sentences". These deflate the
  dev IF score for every model and add noise to comparisons.
- The `seen`/`unseen` label is per prompt, and an unseen prompt also contains seen types, so the
  prompt-level unseen score (52.9) overstates generalisation. The per-type table above is the clean
  measure.

## What this implies for v2

1. Math: train on longer, verified reasoning for hard problems instead of more short solutions. The
   cheapest source needs no outside teacher: the base model's own long reasoning, rejection-sampled
   against ground truth on train problems and kept only when it fits well under the 4,096-token limit.
2. Code: add a short plan-and-check step before the code block, so the model tests its idea against the
   given asserts. Same source options as math (base model's own verified traces, or teacher traces).
3. Instruction following: more constraint types rather than more examples, explicit counting practice,
   and training examples that end cleanly to suppress repetition loops.
4. Dev set: drop infeasible items and report held-out performance per type.

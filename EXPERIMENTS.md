# Experiment log

The entry point for anyone (human or agent) picking up this project. One entry per version, newest first.
Each entry: what changed from the previous version and why, results, reflection, next step. Keep it terse;
details live in the linked files. Append a new entry before training; fill in results after evaluation.

**Goal.** One LoRA checkpoint of `Qwen/Qwen3.5-9B-Base` (Tinker) for IFEval-style instruction following,
MATH-style math and MBPP-style Python. Hidden held-out sets (15% of grade): Python functions checked by unit
tests; competition math with a single answer; verifiable constraints.

**Rules that always hold.**
- Never train on IFEval, MATH-500 / MATH test, MBPP or HumanEval items in any form. Training data is
  Opus-generated or from train splits, and passes the n-gram test-overlap filter (`pipeline/leakage.py`).
- Public test sets give aggregate scores only. Design data from the private dev sets (`data/*_dev.jsonl`)
  and from discovery probes, never from public test items.
- Ask before paid steps. Tinker budget is ~$250 per account; ~$45 spent on account A as of 2026-11-14.
- The user uploads `evaluation/submission.json` to Gradescope. Agents never submit.

**Where things are.** Run table with costs: `runs/README.md`. Per-run outputs: `runs/<name>/`. Error
analysis: `reports/error_analysis_v1.md`. Designs: `docs/superpowers/specs/`. Weakness registry:
`reports/weakness_registry.md`. Generated data (git-ignored): `data/`.

## Scoreboard

| Version | IFEval | MATH-500 | MBPP | dev math | dev code | dev IF seen / unseen | HumanEval | Tinker $ |
|---|---|---|---|---|---|---|---|---|
| base | 47.9 | 15.8 | 65.4 | 17.2 | 65.0 | 34.7 / 25.1 | 70.7 | 2.1 (eval) |
| smoke | — | — | 70.0 | 65.5 | 57.3 | — | — | 0.8 |
| v1 | 72.0 | 70.0 | 75.1 | 67.7 | 74.5 | 66.4 / 52.9 | 67.1 | ~25 |
| v2 | 72.1 | 71.6 | 70.0 | 70.5 | 70.6 | 71.2 / 47.0 | 31.1 | ~33 |

---

## v2: failure-targeted data plus weakness discovery (done; v1 stays the submission)

**Results (2026-11-14).**
- **Math improved:** MATH-500 70.0 → 71.6; MATH test minus MATH-500 70.8 → 73.2; dev 67.7 → 70.5; dev level 5 39 → 50.5.
- **Code regressed:** MBPP 75.1 → 70.0; dev 74.5 → 70.6; HumanEval 67.1 → 31.1.
- **IF:** public flat (72.1); dev seen 66.4 → 71.2; dev unseen 52.9 → 47.0; new unseen types 41.2 (no v1 number).
- **Probes:** 48/57/29 → 50/63/33 (math / code / IF), so the targeted weaknesses moved.
- **Cost:** training ~$29 plus evals ~$4. About 50 minutes were lost to Mac sleep (see the memory note).

**Reflection.**
- **HumanEval collapse:** a format bug. The code-cells data paired HumanEval-style prompts (signature plus docstring) with plan-first answers. The HumanEval harness appends the output after the signature, so prose breaks the program: all 113 failures start with prose, and 86 contain no code.
- **MBPP and dev code drop:** v2 answered almost every one of these directly; it planned on only 13 of 257 MBPP problems and 18 of 337 dev tasks. The losses are wrong direct code (MBPP: 22 of 26 losses). The direct v1 code examples fell from 6,000 to 3,997 in favour of a mix of two styles; that is the likely cause, but it is not proven.
- **Math:** long solutions work. Level 5 gained 11 points.
- **IF:** more rules per prompt helped seen types and hurt unseen ones.

**Next (v3 proposal).** Keep the v2 math and IF data. Make every code answer start with the code block: answer the plan and cell tasks with their verified code directly, and restore all 6,000 v1 direct examples. No new Opus data; one more training run (~$29).

### v2 build notes

**Change from v1.** Same recipe (rank 32, lr 4.73e-4, batch 32, 1 epoch, from base). The data changes,
each backed by a v1 failure (map in `docs/superpowers/specs/2026-11-14-v2-design.md`):
- Math: long, self-checking Opus solutions (median ~950 student tokens) replace short ones on hard
  problems v1 fails, plus Level 5 problems it gets right. Generator `v2_teacher math-long`, 4,126 items.
- Code: a 2–6 sentence plan that hand-checks one assert, then the code. On v1-failed and weak-topic tasks.
  `code-plan`, 2,166 items.
- IF: weak counting rules with exact word counts capped at 60 (`if-count`, 2,000), and 29 new constraint
  types for breadth (`if-new`, 2,663). Six new types are dev-only to measure transfer (`data/if_dev_new.jsonl`).
- Weakness discovery (new, `pipeline/discover/`, design `docs/superpowers/specs/2026-11-14-weakness-discovery-design.md`):
  facet taxonomy, then a coverage map, an Opus explorer, verified probes per cell, v1 scoring and ranking.
  The top cells get targeted data.

**Status (2026-11-14).**
- Pilots of all four generators passed; the full runs are not started yet.
- Discovery code is built. The coverage map shows these never appear in training:
  - 12 code topics (graphs, grids, encodings, parsing, ...);
  - HumanEval-style and long-spec prompts;
  - 6 IF genres;
  - IF rules placed before or inside the task.
- Explorer proposals have been reviewed: accepted ones are in `data/discover/proposals_*.json`.
- Probe pilot (2 cells per task, about $5 of Opus): every math and code probe verified, with a 24/24 teacher pass on each. The first IF pilot got 0/24 because English-word rules were combined with all-Japanese or all-Hindi answers. After the fix (`probes.english_only` / `valid_if_cell`), IF reached 20/24.
- Round 1 is done: 2,529 probes over 212 cells, about $113 of Opus and under $2 on Tinker. Findings are in `reports/discovery_round1.md`. v1 on the probes: math 48%, code 57%, IF 29%.
  - Code: 86% of failures fail a visible assert. Weakest: specs resolved only by the asserts, nested and date inputs, nested-list and dict outputs.
  - Math: weakest on complex, base-n, degree and radical answers, and on counting techniques.
  - IF: weakest at 4–5 rules and with rules placed before the task.
- Generators now target the discovery findings:
  - `pipeline/generate/v2_cells.py`: Opus writes new math problems and code tasks anchored on each weak value, then answers them in the v2 long and plan formats. Items that overlap the test sets or the probes are dropped.
  - IF generators: 2–4 extra rules per prompt instead of 0–2, and rules placed before the task in 40% of prompts.
- Pilot results:

  | Generator | Kept | Opus cost per kept item |
  |---|---|---|
  | math-cells | 88/96 (median 870 student tokens) | $0.054 |
  | code-cells | 94/96 | $0.036 |
  | if-count, new mix | 84/100 | $0.06 |
  | if-new, new mix | 94/100 | $0.036 |
- Full generation is done: about $555 of Opus. The weekly meter went from 11% to 39%. My first estimate of 6% was wrong: on this plan, about $20 of list-price Opus work is roughly 1% of the weekly limit.

  | Pool | Kept |
  |---|---|
  | math-long | 4,117 (median 845 tokens) |
  | math-cells | 1,134 |
  | code-plan | 2,163 |
  | code-cells | 1,185 |
  | if-count | 2,520 (stopped past its planned range; now capped in code) |
  | if-new | 2,434 |
- Training set (`select --v2 --math 21700 --ifeval 8000 --code 7000`; manifest at `runs/v2/manifest.json`): 38,697 examples, 19.8M tokens, about $29 to train.
  - Math 21,700, the same size as v1: 3,769 long and 1,134 cell-targeted solutions, with the short ones dropped for those problems.
  - Code 7,000: 1,908 plan, 1,095 cells, 3,997 direct.
  - IF 8,000: 4,507 new-mix and 3,493 from v1.
  - General 1,997.
  - The test-overlap filter dropped 2,521 candidates.

**Self-distillation:** on hold. It was not shown to beat Opus data and the budget was questioned.
Pilot data is in `data/sd_*`.

---

## v1: full data, single-solve verified

**Change from smoke.** Scaled from 1k to 35.7k examples / 15.1M tokens:
- math 21,697 (MATH train plus Opus-written problems);
- code 6,000;
- IF 6,000;
- general 1,997.

1,116 steps.

**Results.** Above. All thresholds cleared. Proxies: MATH test minus MATH-500 70.8; HumanEval 67.1 (base 70.7).

**Reflection** (`reports/error_analysis_v1.md`):
- **Math.** Short reasoning is the ceiling, not data volume. Going from 500 to 21.7k math examples gave +2 dev points. Level 5 is at 39%. When the base model finishes its long reasoning, it gets 18/24 level-5 problems right.
- **Code.** v1 writes code without checking it. Most failures miss an assert that is shown in the prompt. HumanEval fell below base, likely because every training prompt was MBPP-style.
- **IF.**
  - Exact counts are the weakest.
  - Unseen constraint types did not improve from smoke to v1 (same-type data does not transfer).
  - 11% of public IFEval outputs hit the cap, mostly in repetition loops.

**Next:** v2 (above).

---

## smoke: pipeline check

1,000 examples, 32 steps.
- It fixed the output format (math ends on `ANSWER:`, a single code block, no IF preamble).
- MBPP went from 65.4 to 70.0, but dev code dropped from 65 to 59: direct-code targets remove the base model's reasoning before code.

## base

Untrained model. The math score is low (15.8) because it rarely stops before the 4,096-token cap. Handout numbers: 47.5 / 17.0 / 63.8.

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

---

## v2 (in progress): failure-targeted data plus weakness discovery

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
- Round 1 is 206 cells. The full probe run is estimated at about $115 of Opus and has not been approved yet.

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

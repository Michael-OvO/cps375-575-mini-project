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
- Ask before paid steps. Tinker budget is ~$250 per account; ~$112 spent on account A as of 2026-11-14 (after v3).
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
| v3 | 69.4 | 72.0 | 73.5 | 70.5 | 75.7 | 70.5 / 50.3 | 67.7 | ~32 |

---

## Status (paused 2026-11-14, weekly Opus usage at 41%; resets 2026-11-20)

**Submission:** v1 (`evaluation/submission.json`, checkpoint published). Best public sum so far.

**Acceptance rule (user, 2026-11-14):** a new version replaces the submission only if no area drops beyond
noise (about 2 stderr) and the targeted area gains. Teacher was Opus only; from 2026-11-15 GPT is a second
teacher to widen the data distribution (below).

**Second teacher: GPT-6.1 Sol (2026-11-15).** Chosen from OpenAI's model page: near GPT-6 Astra on its
reasoning and coding evals at $2/$10 per M tokens (Astra $10/$50). Two backends, same model:
`--teacher codex` (headless `codex exec` on the ChatGPT login, no API cost, ~17k cached agent-prompt tokens
per call) and `--teacher gpt` (OpenAI API). Non-Claude pools go to `data/<pool>.<teacher>.jsonl`.
Pilot results:

| Check | GPT-6.1 Sol | Opus 5.5 |
|---|---|---|
| math-long, first 12 hard items (API, high effort) | 12/12 kept, $0.24 | 12/12 |
| math-long, the 11 items Opus missed (of 4,128) | 4/11 kept, $0.16 | 0/11 |
| math-long, same 12 items via Codex CLI | 12/12, 2 min, no markdown | 12/12 |
| if-count strict pass, 30 prompts, first try (low effort) | 25/30 (83%) | 85% kept with retries |
| GPT-6 Astra + Fast tier via Codex, the same 11 Opus misses | 4/11 kept (the same 4 as Sol), 32 s wall | 0/11 |

The 7 items that all three teachers miss probably have wrong reference answers. On this set Astra adds speed,
not quality. Choose with `OPENAI_TEACHER_MODEL=gpt-6-astra` and `CODEX_TEACHER_TIER=priority` (Fast: about
2x speed, more plan usage per call); the default stays Sol on the standard tier.

Both teachers are near ceiling on math, so GPT's value is a second solution style per problem (more
diverse targets), not a higher kept rate. API answers used markdown headings in 2 of 4 math solutions; the
Codex answers in 0 of 12. Preflight's `math.answer_markdown` drift check covers this before training.

**Next: v4 plan (not started; no code written yet).** From the v3 failure profile (`runs/v3/errors/`):
- **Math (largest expected gain):** 88 of 96 dev failures are wrong answers, not truncations; correct answers
  have a median of 150 tokens. Give every level 4-5 problem without a long solution a long, exploratory,
  self-verifying Opus solution (300-3,000 student tokens): 1,199 MATH-train level 4, then ~5k teacher-new
  level 5 (~20% of weekly usage); ~15k for full coverage. Keep short solutions only for levels 1-3.
- **Code:** 72 of 83 dev failures fail an assert shown in the prompt. Re-answer all tasks in one format: one
  code block, function body opening with a 2-6 line comment block (approach plus a hand-trace of one given
  assert), imports inside the function (survives MBPP and HumanEval). ~5% of weekly usage.
- **IF and code, after v4 SFT:** one RL run with verifiable rewards (constraint checkers, hidden tests),
  ~$50 Tinker; IF fails mostly on unseen constraint types (217 of 284).
- Pilot each generator on ~24 items and report the kept rate and cost before any full run.

**IF-RLVR pilot (2026-11-15, Codex teacher only).** Source: `allenai/IF_multi_constraints_upto5` (train,
ODC-BY), 29 new constraint types with upstream checker code (vendored, `pipeline/constraints/ifevalg/`).
Pool (`python -m pipeline.generate.if_rlvr pool`): 95,373 rows -> **68,873 prompts**, drops
safety/jailbreak sources 10,741, held-out types 12,400, no separable base task 1,462, near-dup 1,255
(MinHash, Jaccard 0.6), exact dup 595, eval overlap 47 (MATH test, IFEval, MBPP, HumanEval), our dev /
probe overlap 0. Held out from training: `counting_composition`, `first_word_sent`, `keyword_specific_position`
(closest to our dev-only types, so the unseen-type dev score stays honest).
Pilot: 200 prompts balanced over the 26 trainable new types. Kept **125/200**: 107 first pass (96 on attempt 1),
18 more from a retry that tells the teacher which instruction it broke (13 of them on the first feedback).
By constraint count: 1 -> 30/32, 2 -> 41/61, 3 -> 26/47, 4 -> 25/46, 5 -> 3/14. Hardest types: bigram
wrapping 1/11, sentence hyphens 2/11, copying 2-3/9-11, letter counting 4/15. Spot check: answers are real
answers; 3 used a meta label (e.g. "Palindrome: level."). Codex at 24 parallel calls gets 403s from the
ChatGPT backend; 8-12 is clean (403 now stops a run).
Next: put the pool into `select.py` behind a flag and run preflight; then the full generation.

**Comparison with a classmate's run (2026-11-15).** Their s1_a (9,600 examples: MATH-train 1,700, GSM8K
1,500, OpenCodeInstruct 3,100, Tulu-3 persona IF 2,600, Tulu-3 no_robots/oasst1 700) vs our v1: IFEval 78.5 vs
72.0, MATH-500 55.2 vs 70.0, MBPP 76.7 vs 75.1 (within the 2.6 stderr); sum 210.4 vs 217.1. Lessons:
- IF: they trained on persona-style prompts with IFEval constraint types (we skipped Tulu's persona IF on
  purpose); IF was 27% of their mix vs 17% of ours. Our long math reasoning also makes the model verbose (11%
  of v1's IFEval answers hit the 1,024-token cap). v4 raises IF volume with the IF-RLVR pool, which comes
  from the same AI2 persona-prompt family but with verified answers and 29 more constraint types.
- Code: OCI alone matched our code score, so v4 adds OCI as a second code source.
- Math: short GSM8K-style data costs 15 points on MATH-500; we keep our long-solution route.

**v4 data (2026-11-15, in progress).** `select --v4` = v3 plus:
- **IF-RLVR full** (`if_rlvr full`): 12,000 prompts, the 200 pilot prompts first, then 7,000 with new types
  (round-robin over types) interleaved with 5,000 standard-type-only prompts (quotas by constraint count
  1/2/3/4/5 = 1,200/2,000/1,300/420/80). Teacher: Sol via Codex, concurrency 8; 3 attempts with feedback.
- **Astra pilot on the same IF items:** 38/60 kept (63%, Sol 62.5%), median 96 words (Sol 68). Throughput
  ~2.8 items/min at concurrency 4; concurrency 10 with the Fast tier got 429s. Astra is not better than Sol on
  IF, so Astra goes to math instead.
- **math-hard** (`v2_teacher math-hard --teacher codex`, Astra + Fast, high effort, concurrency 3): the 39
  MATH-train problems still unsolved, then a second independently written long solution for the 3,128
  v1-failed hard problems (row id `<pid>-long-<model>`, so select keeps both solutions).
- **OpenCodeInstruct** (`python -m pipeline.generate.oci`, shards 0/17/34): rows with OCI test score 1.0 and
  5/5 judge scores, one plain function (classes dropped, usage prints and type hints removed), 15-250 word
  spec, at least 4 asserts calling the function; deduplicated (exact plus MinHash), checked against the eval
  sets, code dev and probes; reference code must pass all tests in our sandbox (3 visible, rest hidden). One-
  shard test: 8,320 candidates of 100,000, 98% survive the harness-safe transform. No teacher calls.

**Budget:** Tinker account A ~$112 spent of ~$250.

---

## v3: v2 math and IF, harness-safe direct code (done; v1 stays the submission for now)

**Results (2026-11-14).** Checkpoint `tinker://eeaf81d9-4d1b-5838-a910-46acab06c548:train:0/sampler_weights/v3`.
- **Math kept v2's gain:** MATH-500 72.0 (v1 70.0); MATH test minus MATH-500 71.8; dev 70.5.
- **Code recovered from v2:** MBPP 73.5 (v1 75.1, v2 70.0); dev 75.7 (best so far); HumanEval 67.7 (v1 67.1).
  - MBPP failures are now all fenced code with wrong logic: no NameError, no prose. The `math.isclose` items
    went from 7/13 to 10/13, so the import fix worked. Against v1: 15 items gained, 19 lost, within noise
    (stderr 2.8 points).
  - HumanEval: 8 failures are unfenced, unindented function bodies (SyntaxError); the rest are wrong answers.
- **IF dipped:** IFEval 69.4 (v1 72.0, v2 72.1; prompt-strict 63.2 vs 66.9). The IF data is identical to
  v2's, so this is probably run-to-run variance, but it is about 2 stderr and not proven.
  Dev seen 70.5, unseen 50.3, new types 42.5.
- **Probes:** 49/59/30 (math / code / IF), v2 50/63/33.
- **Cost:** ~$28 training plus ~$4 evals; 58 minutes of training, no sleep stalls.

**Reflection.** The format bugs are fixed and code is back to v1's level on public sets and above it on dev.
Public sum: v1 217.1, v3 214.9, v2 213.7; dev favours v3 (math +2.8, code +1.2, IF seen +4.1, unseen -2.6).
Not a clear win over v1, so the submission stays v1 until the user decides.

### v3 build notes

**Train-vs-eval audit (2026-11-14), after the v2 HumanEval collapse.** Compared every training format with
the graded harness code (`inspect_evals` mbpp/humaneval, `tinker_cookbook/eval/inspect_utils.py`). Bugs found,
all present since v1:
- **HumanEval drops top-level imports.** `extract_function_body` keeps only what follows the first `def` line.
  About 24% of code targets started with an import, so the model learned a habit that fails with NameError.
  Estimated cost on v1: ~6.7 HumanEval points.
- **MBPP tests can need the model's own imports.** 13/257 MBPP items use `math.isclose`, and the harness adds no
  `test_imports`. No training test ever called a module. Estimated cost: ~2 MBPP points.
- **Trailing whitespace.** The eval adapter strips user content; training prompts ended with `"## Completion:\n"`.
- **Test style.** 38% of MBPP asserts write `f(x)==y`; training never did.
- Not fixed yet: the dev code eval uses our own few-shot and so differs from the graded MBPP prompt.

**Fixes.**
- `pipeline/code_format.py`: imports go inside each function, modules the tests call also stay at top level,
  float asserts become `math.isclose` in half of the tasks that have them, and 38% of tasks drop the spaces
  around `==`. Every transformed row must pass its own tests (`select.harness_safe_code`).
- `formats.to_messages` and `student.Sampler` strip prompts.
- Preflight now FAILs unstripped prompts and code with top-level statements other than functions, classes
  and imports of modules the tests call.

**Change from v2.** `select --v3 --math 21700 --ifeval 8000 --code 6600`: same math and IF as v2. Code is all
direct (no plans): 6,062 v1 tasks plus 516 HumanEval-style and long-spec cell tasks, answered with their
verified code. 38,275 examples, 19.1M tokens, ~$28. Preflight vs v1 passed with three acknowledged shifts
(no-space asserts, math markdown from long solutions, IF rules-first); see `runs/v3/preflight.json`.

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

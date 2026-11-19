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

**v4 generation results (2026-11-15).**
- IF-RLVR: 729 prompts via Codex (Sol, then Sol Fast at 20 parallel with staggered launches), the other
  11,268 via the OpenAI API (Sol, 120 parallel, ~290 prompts/min, no 429s): **7,706 kept (68%)**, 19,578 API
  calls, 10.1M input + 12.2M output tokens, **~$142**. 101 of the first 3,479 answers exceed 900 tokens
  (the 600-word cap misses them); select's 900-token filter drops them.
- math-hard (Astra + Fast via Codex, 20 parallel once IF left Codex, ~45/min): **3,120 kept of 3,167**;
  Astra solutions are shorter than Opus's on the same problem (median 767 vs 965 student tokens, max 1,583)
  and textually different (median similarity 0.05). 1,125 of the first 1,128 are level 5.
- Codex caps concurrent connections at about 22 per account (16 + 6 got 403s); launches are now staggered and
  403/429 calls back off and retry.
- **Qwen3.5-9B (instruct) on our harness**, thinking disabled, greedy: IFEval 35.2, MATH-500 19.6, MBPP 83.7.
  IFEval and math are artifacts: it still writes "Thinking Process:" and hits the caps. Not used as a
  teacher (no better than ours, possible MBPP-style training data, verbose).
- **Final selection** (`select --name v4 --v4 --math 25000 --ifeval 16000 --code 11000`): 53,997 examples,
  30.1M tokens, ~$44 to train; mix math 46%, IF 30%, code 20%, general 4%; sources math_train 7,727,
  math_new 16,139, math_cells 1,134, code 4,112, oci 6,888, IF 8,725, if_rlvr 7,275, general 1,997.
  Preflight passed with three acknowledged intended changes (math share 57% -> 46%, OCI answers longer,
  OCI specs name the function).
- **Data audit before training (2026-11-15).** Re-verified: all 7,275 IF-RLVR rows pass their constraints;
  0 of 300 sampled Astra math answers wrong; 1 of 300 sampled code rows fails a hidden test (it uses `string`
  at test level; hidden tests are not in training). Found in IF-RLVR answers and now filtered in
  `select --v4` (`if_quality_problem`): checker-gaming label lines such as "Palindrome: level." (111),
  answers under 4 words (244), degenerate repetition from absurd constraint mixes (27), more than 5 copies of
  one answer (115). Rebuilt v4: 53,997 examples (IF-RLVR 6,832, other IF 9,168), 30.2M tokens, ~$44;
  preflight passed. Known residue: a few dozen terse or templated IF answers; random constraint mixes
  sometimes force unrelated keywords into an answer (as IFEval's own prompts do).

**Budget:** Tinker account A ~$112 spent of ~$250.

---

## v4: IF-RLVR, OpenCodeInstruct, Astra second solutions on hard math (done; 2026-11-15)

**Results.** Checkpoint `tinker://59ce19da-5ae9-5392-93a1-01e8626fb70f:train:0/sampler_weights/v4`
(53,997 examples, 30.2M tokens, 1,688 steps, rank 32, lr 4.73e-4; ~$44 train + ~$4 evals).

| | IFEval | MATH-500 | MBPP | Sum | Dev math | Dev code | IF seen / unseen / new | MATH rest | HumanEval |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 72.0 | 70.0 | 75.1 | 217.1 | 67.7 | 74.5 | 66.4 / 52.9 / - | 70.8 | 67.1 |
| v3 | 69.4 | 72.0 | 73.5 | 214.9 | 70.5 | 75.7 | 70.5 / 50.3 / 42.5 | 71.8 | 67.7 |
| v4 | **75.9** | 71.8 | 73.2 | **220.9** | **71.4** | 70.3 | 66.6 / 51.0 / 43.4 | 71.4 | **75.6** |

- **IF:** public up 3.9 over v1 (prompt strict 70.1, loose 73.0, inst strict 79.3, loose 81.4), about 2 stderr.
  Dev IF is flat (paired vs v3: won 44, lost 45); unseen types +0.7, new types +0.9. The IF-RLVR data moved the
  public benchmark more than our held-out-style dev types.
- **Math:** flat. MATH-500 71.8 (v3 72.0), dev 71.4 (paired vs v3: won 25, lost 22), level 5 44/99 (v3 47).
  Median output 190 tokens: the Astra second solutions did not make the model reason longer or better.
- **Code:** MBPP 73.2 (v1 75.1, within the 2.8 stderr); dev code 70.3 (paired vs v3: won 25, lost 39, z = -1.75,
  not significant at 2 sigma); HumanEval 75.6 (+7.9, best so far). Dev code tasks come from the same generator
  as our own code tasks, whose share fell from 6,600 to 4,112 as OCI came in; failures are logic slips, not
  format. OCI seems to help general code (HumanEval) and cost in-distribution code.
- **Against my estimates:** IFEval 75-78 predicted, got 75.9; math 72-74 predicted, got 71.8; MBPP 75-77
  predicted, got 73.2.

**Acceptance rule.** Targeted area (IF) gained on public; no area dropped beyond 2 sigma (code dev -1.7 sigma vs
v1, MBPP within noise). Best public sum and best held-out code proxy. v4 passes the rule against v1; switching
the submission is the user's decision.

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

## rl1: bad-case analysis before RL (2026-11-16)

Diagnosis only; public items are never used to choose training prompts. Per-item public results across v1–v4
(inspect logs):

| Benchmark | v4 | v4-wrong but solved by an earlier version | Never right in any version | Ceiling if all once-solved |
|---|---|---|---|---|
| MATH-500 | 71.8 | 61 items (12.2 pts), mostly level 3–4 | 80 (52 at level 5) | 84.0 |
| MBPP | 73.2 | 30 (11.7 pts) | 39 | 84.8 |
| IFEval prompt-strict | 70.1 | 73 (13.5 pts) | 89 | 83.5 |

- IFEval v4 failures are mostly counting constraints: keywords:frequency 24/42, number_words 23/52,
  forbidden_words 17/49, number_sentences 17/52, letter_frequency 16/33, capital_word_frequency 12/25.
- Our IF verifier agrees with the official prompt-strict on 541/541 v4 items.
- Dev flips across v1/v3/v4: math 22%, code 27%, IF 33% of items. These are the 0 < p < 1 items RL
  sharpening can fix.
- An independent review of the RL spec found 6 critical issues (macOS RLIMIT_AS, stub-test wording, IF verifier
  routing, IF gate false positives, per-prompt bands, harness-safe code) plus high/medium ones. All are fixed
  in spec v3 (`docs/superpowers/specs/2026-11-16-rl-design.md`).

### rl1 prompt pools (2026-11-18)

- Built locally ($0) by `pipeline.rl.pools`: math 6,545, IF 9,000 chosen (target reached, bins 2,250/5,400/1,350 =
  25/60/15), code 14,999 kept (stub rule applied last: 32 drops there; earlier drops already removed ~340 stub-weak OCI tasks). 50 trap prompts in `data/rl_traps.jsonl` stay out of every pool.

### rl1 gated-SFT demo coverage (2026-11-18)

- `pipeline.rl.demos` matches pool items to v4 math/code training rows (id dash-prefix plus raw text in the user message; IF never). Coverage on the built pools: math 6,430 of 6,545 (98%), code 8,424 of 14,999 (56%). Demos come only from `runs/v4/train.jsonl`.

### rl1 → Phase 1–3 (2026-11-18)

- Phase 1 replay v4 → v4r: 187 steps (1501–1687), mean loss 0.5049 vs v4 0.5050 over the same steps, train_tokens
  identical (30,209,821); $4.83. Equivalence gate passed: math agreement 0.90 (b 12 / c 20, p 0.22), code 0.99 (2/3,
  p 1.0), IF 0.87 (29/29, p 1.0).
- Step-0 checkpoint eval (v4r, dev): greedy math 0.738, code 0.715, IF seen 0.683 / unseen 0.501 / unseen_new 0.410;
  pass@1→pass@8 math 0.69→0.91, code 0.56→0.87, IF 0.35→0.64; trap hack rate 0.0.
- Phase 2 calibration: 900 prompts × (8 + greedy), $3.20, 0 score errors. Baseline S/L: math 0.416/538, IF 0.335/300,
  code 0.431/70; trunc math 0.3%, IF 4.3%; IF gate 2.0%; code vphf 2.7%. Every reward branch fired.
- Hand review of 90 rollouts (+ sample 1 rescored): math 30/30 and code 30/30 rewards agree. IF 24 agree / 3 disagree /
  3 unsure. Known IF reward weaknesses, left as is for the pilot and watched in the pilot review:
  the `too_short` gate zeroes concise answer-only replies; plural "Keywords:" label lines slip past the gate;
  `keywords:palindrome` passes on any one-letter word; langdetect checks are unreliable on very short text;
  `new:copy_span_idx` is effectively unlearnable (constant groups, so no gradient).
- Schedule: task weights IF 0.42 / math 0.25 / code 0.33 → 27/16/21 groups per step; $0.771 per step; 128 steps
  ($98.63 projected).
- Phase 3 smoke: 3 steps, exit 0, datums 57/56/34, gated SFT fired (code 1 group, math 2), kl_sample_train 0 to 4
  decimals (probe confirmed the trainer returns its own logprobs and they match the sampler's); resume exits at once.

### rl1 → Pilot (40 steps, 2026-11-18)

- Paused once at step 20 on `code/len 148 > 2x baseline 70`. Not a blowup: code length was 149 at step 0 and flat
  after. Cause: calibration's stratum-uniform pick gave 286 `ct` (65 tok) vs 14 `oci` (178 tok) code prompts, while
  the schedule is 71% `oci`. Pause baselines (len/trunc/gate/vphf) were reweighted to the schedule's stratum mix
  (code len 70 → 151, vphf 0.027 → 0.095; math/IF moved < 3%) and the run resumed from step 21. EMA S/L init unchanged.
- Training curves, 10-step means (steps 0–9 → 30–39): reward math 0.58 → 0.57, IF 0.78 → 0.82, code 0.75 → 0.82;
  informative share math ~0.5, IF 0.88 → 0.79, code 0.56 → 0.40; IF gate 1.5–2.3%; code vphf 3.4–5.9%; IF trunc 2–3%;
  math gated-SFT 2.6–3.5 groups/step, math length 521–583 (no rise after demos). kl_sample_train 0 at every step;
  KL to v4 under the 0.05 warning at steps 10–40 (value not logged yet).
- Dev checkpoint evals (greedy; step 0 → 20 → 40): math 73.8 → 73.8 → 73.5; code 71.5 → 72.7 → 74.2;
  IF seen 68.3 → 70.1 → 71.2, unseen 50.1 → 53.4 → 53.9, unseen_new 41.0 → 45.2 → 45.4.
  pass@1 math 0.69 → 0.70 → 0.74, code 0.56 → 0.64 → 0.63, IF 0.35 → 0.39 → 0.41. Trap hack rate 0.0 throughout.
- Step 40 vs v4 (paired, dev): math +2.2 ± 1.8, code +3.0 ± 1.5, IF +4.3 ± 1.6 points.
- Spend: ledger $29.28 (pilot ≈ $19.5, ~$0.49/step vs the $0.77 estimate). Tinker billing sync still reports $0 for
  today (API lag); the ledger is the working total.

### rl1 → Pre-4b changes (2026-11-18)
- IF quality gate: the label-line rule now also catches plural "Keywords:" lines (a keyword-stuffing path found
  in the hand review). Phase 0 rerun still passes; gate firing on good teacher answers 0.836% → 0.842%, and the new
  pattern fires on none of them. Applies from step 41, so steps 1–40 used the old gate.
- Left as is (reviewed): the <4-word `too_short` gate (rare; brevity constraints are exempt), the trivially-true
  palindrome checker (vendored IFEvalG behaviour) and langdetect on short text (IFEval's own grader uses it).
- KL-to-v4 is now written to `runs/rl1/kl_v4.jsonl` every 20 steps (it was only checked against the 0.05 warning).

### rl1 → Main run to step 128, then extension (2026-11-18)
- Steps 41–128 ran clean: no pauses; warnings only (math/IF train reward dips over 30-step windows while dev rose).
  Spend $70.60 at step 128 incl. evals (~$0.46/step). KL to v4: 0.0047 (60), 0.0084 (80), 0.0079 (100), 0.0127 (120).
- Dev greedy, math / code / IF seen / unseen / unseen_new:
  step 0 73.8 / 71.5 / 68.3 / 50.1 / 41.0; 40 73.5 / 74.2 / 71.2 / 53.9 / 45.4; 60 72.6 / 74.5 / 74.4 / 59.6 / 48.1;
  80 74.5 / 75.1 / 74.9 / 59.3 / 49.6; 100 72.6 / 75.4 / 76.2 / 60.2 / 52.5; 120 73.5 / 76.3 / 76.8 / 61.4 / 53.3;
  128 75.4 / 76.6 / 76.0 / 59.7 / 56.5. Math greedy is noise around v4 (SE ~2.4); IF and code still rising.
  Trap hack rate 0–0.04 (pause line baseline + 0.10).
- Extension (the improvement round): `schedule.py --extend 72 --usd_per_step 0.47` adds steps 129–200 from prompts
  not yet scheduled; steps 1–128, the smoke tail and the config (pause baselines included) are unchanged apart from
  `steps`. Projected +$33.84, leaving the $27 eval/Phase 5 reserve under the $135 cap. Pre-extension copies:
  `runs/rl1/schedule.step128.json`, `runs/rl1/config.step128.json`.

### rl1 → Extension paused at step 160 (2026-11-19)
- Pause: math/trunc 0.025 over steps 150–159 > 2 x baseline 0.002 + 0.02 (sampled rollouts, T=1). Ten-step means
  rose 0.2% (steps 0–9) → 1.2% (140–149) → 2.5% (150–159); math length 521 → 666 tokens.
- Greedy is not affected: math dev outputs at ≥4,000 tokens 13/325 (step 0), 9 (140), 6 (160); mean 461 → 530 tokens.
- Dev greedy, math / code / IF seen / unseen / unseen_new: step 140 76.3 / 76.6 / 80.9 / 63.7 / 55.3;
  step 160 78.2 / 76.9 / 79.7 / 63.3 / 58.9. pass@1 160: math 0.76, code 0.70, IF 0.51. Trap hack 0.02. KL to v4 0.0144.
- Spend $84.06 at the pause. Renamed `PAUSED` to `PAUSED.step160` after review.

### rl1 → Graded math truncation from step 161 (2026-11-19)
- Why: the step-160 pause was sampled math truncation. Greedy dev showed long answers now pay off (3,072–4,000 tokens:
  3/5 correct at step 160 vs none that long at v4), while cut-off answers always score 0 and about half are
  repetition loops (3/6 at step 160, 8/13 at v4). Under group centering, a cut-off in an all-fail group got no
  signal at all. A linear/DAPO near-cap ramp was rejected: it would dock correct answers the grader accepts.
- Reward (math only; `graded_trunc` in config): correct finished 1; wrong finished 0 (no length penalty); cut off
  ending in a correct final ANSWER line 0.75 (the grader reads an ANSWER line only at the end of the text, `\Z`);
  other cut-offs -0.25; repetition loops (a 12-word chunk repeated >= 4 times, cut-off text only) get an extra
  -0.25 on the loop tokens only, not group-centered, so loops are pushed down even in all-fail groups.
  λ = 0.25 keeps long attempts worth trying: at step 160 long attempts had p(correct) ≈ p(cut off), and p - λq > 0
  with a 4x margin. Order correct > wrong > cut off holds in every case.
- Gated SFT now keys on the grader's verdict (no sample correct), not on rewards == 0. `math/acc` (unshaped) and
  `math/loops` are logged; the truncation pause line is unchanged.
- Checks: loop detector flagged 0/599 finished calibration answers; rescoring 239 finished calibration samples
  changed none; the 6 cut-off greedy answers at step 160 split 3 loops / 3 not.
- Paused again at step 162 after 2 graded steps (math trunc 0.0%, 3.9%): the 10-step truncation window was 8/10
  pre-change steps. With approval, the math truncation window now restarts at the reward change (`graded_trunc.
  from_step`): the windowed rule uses only steps from 160 (0-based) once 5 exist; before that any single step
  above 6% pauses. All other pause rules unchanged. `PAUSED` renamed to `PAUSED.step162`.

### rl1 → Paused at step 170; truncation reached greedy (2026-11-19)
- Pause: math trunc averaged 2.7% over the 10 graded steps (161–170), 0 loops in every step: all cut-offs were
  productive reasoning running out of room; math length 634 → 682–894 tokens. The -0.25 penalty did not bend it.
- Step 170 dev greedy, math / code / IF seen / unseen / unseen_new: 75.7 / 78.0 / 81.0 / 63.8 / 55.5 (IF avg 66.8).
  Greedy math cut-offs (>= 4,000 tokens) 13 (step 0) → 9 (140) → 6 (160) → **14 (170)**; math greedy 78.2 → 75.7, about
  the 8 extra cut-offs. Code and IF unchanged within noise. Trap hack 0.0.
- Reading: steps 161–170 hurt math through truncation; step 160 is the last checkpoint before greedy was affected.
  Spend $89.22. `PAUSED` renamed to `PAUSED.step170`; training not resumed.

### rl1 → Branch b: resume from step 160 with a 3,584-token math training cap (2026-11-19)
- Why: steps 161–170 (graded truncation only) let sampled math length keep rising and greedy cut-offs went 6 → 14.
  Training math now samples with max_tokens 3,584 while every eval keeps 4,096 (`ckpt_eval` hardcodes it), so the
  model learns to finish with 512 tokens of slack at the grader's cap. Graded truncation (cut off -0.25, loops,
  0.75 for a final correct ANSWER line) applies at the training cap.
- Checks: the math cap is read only by training sampling, the KL-to-v4 monitor (same sampling params) and the
  spend reservation; SFT demos are all <= 1,794 tokens; greedy answers in 3,584–4,000 tokens that the cap puts at
  risk: correct 1 (step 140), 2 (160), 1 (170), against 8 extra cut-offs at step 170.
- Resume: step-160 state (with optimizer) from checkpoints.jsonl; EMA lengths reversed exactly from step 170 using
  the logged per-step lengths (math 698 → 650); EMA spreads and rng from step 170. Old step-170 resume kept as
  `runs/rl1/resume.step170.json`. Branch checkpoints are named `rl1-stepNNNNb`. Steps 160–169 reuse the same
  prompts as the abandoned graded-only branch.
- Monitor for this branch (approved): math sampled truncation warns on the usual line and pauses only above 15%
  (window from step 160); greedy cut-offs at 4,096 are the gate: stop if > 9/325 or greedy math < 76.2 at the
  step-170b or 180b eval. Run length: 20 steps (to 180), then review.

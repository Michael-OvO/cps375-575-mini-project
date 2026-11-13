# Data and training strategy: design

Date: 2026-11-13. Course: CPS 375. Team of two. Final submission due 2026-12-11, 11:59 PM.

## Goal

One LoRA checkpoint fine-tuned from `Qwen/Qwen3.5-9B-Base` that

1. clears all three public thresholds (IFEval 58.0, MATH-500 45.0, MBPP 70.0), which is 60% of the grade, and
2. ranks as high as possible on the hidden held-out benchmarks (15% of the grade).

Staff have said the hidden benchmarks are: Python function generation from a short description checked by unit tests; competition-style math with a single final answer; explicit output constraints checked by code. Response quality is not judged.

## Hard rules

- Train only on training splits.
- No IFEval, MATH-500 or MBPP example may appear in training in any form, including as a prompt for generation or RL. We avoid every MBPP split, not only test.
- The cookbook math RL recipe mixes MATH test problems in by default; its data loader is never used as is.
- The report is written by the team by hand. All AI use is disclosed.
- The final checkpoint is published with no expiry.

## Open questions for staff (posted on Ed)

1. Are teachers outside Tinker (Claude, GPT) allowed for generating training data? PROJECT.md says "other models on Tinker"; the announcement does not.
2. Does loading the test sets only to run an overlap filter count as using them?

3. The MBPP harness prompt contains three few-shot examples taken from MBPP itself. May training prompts reproduce that header? Until answered, we assume no: training prompts use the same layout with few-shot examples we wrote ourselves, and no MBPP item appears in any training file.

Generation with Claude and GPT starts immediately, but no run trains on that data until question 1 has a written yes. If the answer is no, the teacher becomes a Tinker-hosted model (Qwen3.5-397B-A17B), paid from the Tinker budget. If the answer to question 2 is no, the leakage check compares against nothing from the test sets and relies on source-level exclusion only.

## Principles

1. **Every training example passes the real scorer.** Math: the final answer matches ground truth under `evaluation/math_scorer.py`. Code: the solution runs and passes its asserts. Instruction following: the response passes the pinned `instruction_following_eval` checkers in strict mode.
2. **Train in the harness prompt format, plus paraphrases.** Rendered prompts are dumped from the harness in Stage 0. About 70% of examples use the harness wording and 30% use paraphrased wording.
3. **Outputs contain only what was asked.** No preamble, no sign-off, a proper end-of-turn token.
4. **Teacher tokens are free; Tinker training tokens are scarce.** We generate a pool much larger than we train on and select from it.
5. **Decisions are made on a private dev set.** Public scores are a sanity check. Differences under 3 points on the public sets are treated as noise.

## Data

### Math

- Prompts: MATH train split; other competition-style train sets; teacher-written new problems.
- Targets: step-by-step solution ending in a line `ANSWER: <answer>`, answer in the canonical form the scorer accepts.
- Verification: against ground truth where it exists. For teacher-written problems with no ground truth, Claude and GPT solve independently. Agreement is accepted. Disagreement is not discarded (strict agreement filters remove the hard problems and lowered accuracy in published ablations); a third high-effort solve decides, and the problem is dropped only if all three differ.
- Unique problems over extra solutions: one concise solution per problem by default. Problems the student fails get up to three distinct correct solutions.
- Concise solutions: where several correct solutions exist, keep the shortest one that does not skip steps (target roughly 200 to 400 tokens).
- Difficulty: sampling is weighted toward hard problems, and every hard problem in the pool has at least one solution.
- Teacher-written problems are checked for being paraphrases of test problems (one published pipeline found about 9% were, despite train-only seeds). See the leakage guard.

### Code

- Prompts: teacher-written tasks in MBPP style (short description plus three asserts that fix the function name and signature); OpenCodeInstruct items reformatted the same way.
- Targets: a single fenced Python block containing only the function and needed standard-library imports.
- Verification: execute in a sandboxed subprocess with a timeout; all asserts pass. For teacher-written tasks, a second teacher writes extra hidden asserts and the solution must pass those too.

### Instruction following

- Prompts: a fresh base task (varied topics and personas) plus one to four constraints sampled from a registry, most often two or three.
- Registry: the constraint types covered by the pinned checker library, with paraphrased wording, plus at least 30 new types of our own, each with its own verifier. Published results show models trained only on the standard types score about 81 on IFEval and about 25 on unseen constraints, so breadth of types matters more than volume per type. Numeric parameters (word counts, sentence counts, repetitions) are drawn from wide ranges. Length and keyword constraints are always well represented.
- Targets: teacher responses, kept only if every constraint passes strict checking.
- Tulu-3's instruction-following subset is included after the same strict filter.

### General

A small sample of Tulu-3 general chat to keep ordinary behaviour intact.

### Starting mixture

The mixture is not a fixed percentage. It is the core plus the math block defined under Run sizes, so math's share grows with run size (about 40% of tokens at S, about 85% at L). Math scales because it has the largest headroom and keeps improving with more unique problems; published MBPP gains from fine-tuning at this size are only 2 to 7 points, so code stays at coverage size. Core sizes are a best guess and change only if v1 shows a task falling short.

### Run sizes

Sizes are our own, derived from what each task needs and what the budget buys. They are defined in training tokens, because both cost ($1.46 per million) and LoRA capacity are in tokens; example counts are estimates until real data is measured. Every run is a single epoch: more unique data is preferred to repeated passes.

A run is a fixed core plus a variable math block.

**Core (sized by coverage, about 6M tokens, ~$9).** These slices stop improving once every category is covered, so they stay the same in every run.

| Slice | Sizing rule | Examples | Tokens |
|---|---|---|---|
| Instruction following | ~55 constraint types, each appearing ~250 times, ~2.5 constraints per prompt | ~6k | ~2.6M |
| Code | ~40 task categories, ~150 tasks each | ~6k | ~2.6M |
| General | enough to keep ordinary behaviour | ~2k | ~0.8M |

**Math block (the part that grows between runs).**

| Run | Math problems | Total tokens | Cost | Purpose |
|---|---|---|---|---|
| Smoke | 500, with 500 core examples | ~0.5M | ~$1 | catch pipeline and format bugs before spending |
| v1 | 30k | ~19.5M | ~$29 | first best guess |
| v2 | 90k, including failure-mined data | ~46M | ~$68 | one improvement round |
| v3 (only if needed) | up to ~170k | ~85M | ~$125 | only if v2 clearly beat v1 on math and time allows |

There are no ablation sweeps and no separate scaling curve. v1 against v2 gives two points on the scaling question for free. If instruction following or code drop from v1 to v2 as their share shrinks, the core is enlarged for any later run.

Extra data goes mostly to math, where published results keep improving with more unique problems. Instruction following gets more constraint types, not more examples per type. Code grows least. The teacher pool is generated well beyond the final size (for example 200k math problems) so selection has room to work.

### Training defaults

Best-guess settings, not swept: LoRA on all linear layers, rank 32, learning rate from the cookbook's recommended-rate helper, one epoch, loss on assistant tokens only. A setting is revisited only if a run shows a specific problem (for example training loss diverging or not falling). Stage 0 confirms the model stops correctly under the chat template (the end-of-turn token is supervised and matches what the sampler stops on).

### Selection from the pool

v1 takes a diverse sample of the pool. After v1 there is a student checkpoint; for v2, pool examples are ranked by whether the student currently fails that problem (failures first), then by diversity (topic, constraint type, difficulty level), with at most two examples per unique prompt.

## Stages

| Stage | What | Gate to replace current best |
|---|---|---|
| 0 | Install environment; base eval; dump rendered prompts; save base failures; test whether one account can continue training from the other's published checkpoint | none |
| 1 | Build generators, verifiers, dev set and leakage check; generate the pool with Claude and GPT | none |
| 2 | Smoke run, then v1: core plus 30k math, best-guess settings. Score on public, private dev and proxies; submit to Gradescope if it clears the thresholds | clears all three thresholds |
| 3 | v2: one improvement round. The student's failures on held-back train problems drive new teacher data and verified self-distillation (student's own verified solutions); retrain from the base model on v1 data plus the new data, with math at 90k | wins on private dev set and proxies |
| 4 | Only if time and budget remain: a larger v3, or a short mixed-task RL stage with verifier rewards (published gains about 1 to 2 points at this size) | wins on private dev set, no task regresses by more than 1 point |
| 5 | Publish the best checkpoint with no expiry, full `submission.json` | none |

If the staff answer on outside teachers has not arrived when the pool is ready, v1 is trained on the same recipe with the pools regenerated by the Tinker-hosted teacher (paid from the Tinker budget, with the math block halved) so there is a valid submission on file, and the Claude and GPT data enters at v2.

## Private dev set

Never trained on. About 500 items per task:

- Math: a held-out slice of the MATH train split.
- Code: teacher-written MBPP-style tasks with hidden asserts.
- Instruction following: fresh prompts, half using constraint types excluded from training.

### Public proxy benchmarks (evaluation only, never trained on)

Stand-ins for the hidden sets, chosen to match what staff described:

- Math: the MATH test problems that are not in MATH-500.
- Code: HumanEval (Python functions checked by unit tests, different prompt style from MBPP).
- Instruction following: IFBench (constraints outside IFEval's types). Our own new training constraint types must not be copied from its test constraints.

Each candidate checkpoint gets public, private dev and proxy scores, about $2 in total. A checkpoint whose public score rises while its proxy scores stay flat or fall is treated as overfit.

### Vault split

A third of the private dev set and of each proxy is set aside and scored only for the final two or three candidates, so the final choice is not tuned on the same items used during development.

### Comparing runs

Runs are compared on the same items with a paired test. Differences under about 3 points on a 500-item set are treated as noise.
## Leakage guard

- Source-level: a manifest lists every dataset, split and count; a test fails if any source is a non-train split or an excluded dataset.
- Example-level (if staff allow): normalised n-gram overlap against the three test sets; any hit is dropped and logged.

## Tinker budget ($500 across two accounts)

Rates for the student: train $1.46, sample $0.81, prefill $0.30 per million tokens.

| Bucket | Amount | Account |
|---|---|---|
| Setup, base evals, smoke tests | $10 | A |
| Evals, including private dev set | $50 | both |
| Smoke run and v1 | $30 | A |
| Failure and self-distillation sampling for v2 | $40 | B |
| v2 | $70 | A |
| Unallocated: a rerun if something breaks, then v3 or RL | $300 | both |

Account A holds the mainline and the final checkpoint. Account B runs sampling-heavy jobs whose output is data files. If cross-account training does not work, the RL stage needs its own SFT run on B (about $22 from the reserve), and the final checkpoint is whichever account produced the winner.

Teacher API spend is outside this budget and is not a constraint.

## Code layout

```
pipeline/
  prompts/      harness templates and paraphrases
  generate/     math.py, code.py, ifeval.py   (teacher calls, resumable)
  verify/       math.py, code.py, ifeval.py   (also used as RL rewards)
  select.py     pool -> training file, writes manifest.json
  leakage.py    source and overlap checks
  train_sft.py  adapted from evaluation/train_and_publish.py
  sample.py     student sampling for self-distillation and failure mining
  train_rl.py   verifier-reward stage
  dev_eval.py   private dev set evaluation
data/           generated pools, manifests (large files git-ignored)
runs/           one folder per run: config, cost, public and dev scores
```

The `evaluation/` folder is left untouched so results match what staff rerun.

## Risks

- **External teachers disallowed.** Stages 1, 3 (self-distillation part) and 4 need no outside teacher and still stand.
- **RL degrades other tasks.** Mixed batches, early stopping, and the replace-only-if-better gate.
- **Overfitting to public benchmarks.** Selection on the private dev set; limited number of full public evals per checkpoint.
- **Eval templates differ from what was assumed.** Stage 0 dumps the real rendered prompts before any data is formatted.
- **Budget overrun.** Every run logs token counts and cost, and the smoke run confirms the per-token cost before v1.

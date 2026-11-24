# Final round: probe, merge SFT and RL, confirm once

Date: 2026-11-24. Deadline: final submission December 11. Current submission: rl1 step 150.

## 1. Goal

All three public thresholds are passed (IFEval 81.6 vs 58, MATH-500 74.6 vs 45, MBPP 74.7 vs 70), so public score
gains earn nothing. What is left: the held-out relative ranking (15%) and the report. The target is therefore gains
that transfer to unseen tasks, measured on our dev set and confirmed once on the vault. Public results never choose
data, weights or checkpoints.

Only the biggest levers are pursued: one training probe, one evaluation check, one final round.

## 2. Evidence (dev, rl1 step 150)

| | greedy | pass@1 sampled | pass@8 | main failure |
|---|---|---|---|---|
| Math | 76.0 | 76.1 | 92 | wrong reasoning; 9 of 78 failures cut off at 4,096 tokens (no ANSWER line) |
| Code | 77.2 | 68.4 | 90 | all 77 failures are logic errors in well-formed code; no reasoning before the code |
| IF | seen 78.3 / unseen 65.2 / new 56.0 | 47.9 | 74 | unseen-new constraint types (vault: +1.8 over v4 vs +14.8 seen) |

The pass@1 to pass@8 gap is the main lever: the model already solves most failures on some sample.

## 3. Work split

- **Teammate (SFT):** v5 data mixture. Owns `pipeline/select.py` (`select --v5`) and the SFT run.
- **User (RL):** probe, distilled data, rl2. Owns `pipeline/rl/` and `pipeline/self_distill.py`.
- Files barely overlap, so both branches merge into `main` without conflicts.

### 3.1 Merge contract (both sides)

1. Output formats are unchanged: a final `ANSWER:` line for math; exactly one ```python block for code, no prose
   outside it; IF answers as plain responses. RL rewards and the course scorers depend on these
   (`pipeline/formats.py`).
2. Training data passes `pipeline/leakage.py` and the project rules: no IFEval, MATH-500, MATH test, MBPP or HumanEval
   items, no dev or vault items, no IFBench test prompts.
3. One yardstick: `pipeline.dev_eval` on the dev split, paired against the reference with `equiv.compare`. The vault
   is used once, at the end.
4. v5 passes `pipeline.preflight` before any RL runs on it.

### 3.2 How the halves merge

```
rl1 step 150 ──samples──> distilled data ──┐
                                           ├──> v5 SFT (teammate) ──gate──> rl2 on v5 ──┐
teammate mixture work ─────────────────────┘                                            ├──> pick on dev ──> confirm
                                         (gate fails or late) ──> rl2 on rl1 step 150 ───┘
```

- rl1's gains flow into SFT as data (§4.1), so rl2 on v5 starts from a stronger point.
- RL is re-run on v5 by changing the start checkpoint in the config; rewards, pools, pauses and evals carry over.
- **v5 gate (December 2):** v5's dev sum beats v4's (math 71.4, code 70.3, pooled IF) by more than 1 paired SE, and
  no area falls more than 1 SE. If v5 fails the gate or is not ready by December 2, rl2 starts from rl1 step 150.

## 4. Work items

### 4.1 Probe: shortest-correct distillation (~$8, pre-approved)

- Sample rl1 step 150 eight times (temperature 1) on ~2k training prompts: math (MATH train and agreed-target teacher
  problems), code (our verified tasks with hidden tests), IF (training-mode constraints).
- Keep, per prompt, the shortest output that the task's verifier passes. Math traces must finish under ~3,000 tokens
  and be loop-free; code may carry a short plan as comments inside the block.
- Probe: a short LoRA SFT continuation from step 150 on this data; dev eval against step 150.
- **Pass:** sampled pass@1 (8 samples) rises more than 1 SE in at least one area; no area falls more than 1 SE;
  greedy cut-offs do not rise.
- **Pass →** the data goes to the teammate as a v5 source. **Fail →** it is not used; record why.

### 4.2 Check: prompt robustness (~$3, pre-approved, evaluation only)

- Evaluate step 150 on dev under 3 rewordings of the prompt (no system prompt vs a generic one; a reworded
  instruction; a math request for `\boxed{}` alongside the ANSWER line).
- **Flag** any area that drops more than 2 SE. A flagged area gets varied prompt wording in v5 and rl2 data.

### 4.3 Final round: rl2 (needs the user's OK, ~$90)

- Start: v5 if it passed the gate, else rl1 step 150.
- Same pipeline and safety rules as rl1, plus new verifiable IF constraint families (IFBench-train style, training
  data only) in the IF pool, aimed at the unseen-new split.
- `caps.math` back to 4,096 (it is 3,584 from branch b).
- Checkpoint choice: the rl1 rule (best 3-checkpoint moving average of dev sum), frozen before confirmation.

### 4.4 Choice and confirmation (~$12)

- Candidates: the rl2 choice vs rl1 step 150, compared on dev only.
- Confirm the winner once on vault, public and proxies, paired against rl1 step 150, with the rl1 acceptance rule:
  no area more than 2 SE below, at least one area above on both vault and public. Fail → step 150 stays.

## 5. Budget ($250 extra, outside the rl1 cap)

| Item | USD | Approval |
|---|---|---|
| Robustness check + distillation probe | ~11 | pre-approved (each probe < $10, probes < $60 total) |
| v5 SFT (teammate) | ~45 | user |
| rl2 | ~90 | user |
| Confirmation | ~12 | user |
| Reserve | ~92 | user |

Every paid call goes through the ledger.

## 6. Timeline

| Dates | Work |
|---|---|
| Nov 25–26 | Robustness check; distillation data and probe; hand data to teammate |
| Nov 25 – Dec 2 | Teammate builds and trains v5; v5 gate on December 2 |
| Dec 3–6 | rl2 |
| Dec 7–8 | Choice on dev; confirmation |
| Dec 9–11 | Report; submission decision (user uploads) |

## 7. Traceability

- One card per probe in `runs/probes/<id>/card.md` (hypothesis, evidence, cost, metric, pass and kill bars,
  written before running; result and decision after), results beside it.
- One line per probe or round in `EXPERIMENTS.md`; a row per model in `runs/README.md`.
- No new framework: the probe reuses `self_distill`, `train_sft` and `dev_eval`; rl2 reuses `train_rl`.

## 8. Out of scope

SRPO-style self-distillation in RL, a provisional early ANSWER line, a separate code-plan probe, checkpoint merging
(upload not enabled), mixture sweeps.

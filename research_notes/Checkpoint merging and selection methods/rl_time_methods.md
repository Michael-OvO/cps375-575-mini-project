# In-training checkpoint methods for RL of LLMs (anchor resets, EMA anchors, online merging, elastic reset, rollback), 2023-2026

Scope note: our setting is Qwen3.5-9B-Base LoRA r32, SFT to v4, then about 200 steps of clip-free IS-REINFORCE (G=8, 64 groups/step, lr 2e-5), on mixed IF/math/code, with greedy eval. Checkpoints every 10 steps and KL-to-v4 every 20 steps. For each method the notes mark whether it is **FREE** (uses checkpoints we already save, or a cheap change to the update) or **EXTRA-COMPUTE** (needs more RL runs).

Research budget: about 15 tool calls. Several primary PDFs (Elastic Reset, Skywork-OR1) could only be read at abstract level, and those gaps are flagged.

## Q1. WARP, WARM, Online Merging Optimizers, Elastic Reset, reference resets (ProRL), merging during RL: gains and costs

### Takeaway
The methods with real effect sizes split into two groups:
- **Multi-run methods.** WARP (EMA anchor plus SLERP of M runs plus interpolation toward init) gives about +5 to +10 points of side-by-side win rate and +5 to 11 points on GSM8K/MATH/MBPP for Gemma-7B. It needs M≥2 full RL runs per iteration, so it is EXTRA-COMPUTE.
- **Free per-step modifications.** An EMA anchor in place of a frozen reference (WARP ablation; EMA-PG 2026: +3.1 on OlympiadBench) and the online merging optimizer (+0.5 to 1.3 benchmark average, +0.1 to 0.3 MT-Bench, DPO only, not LoRA-compatible) are both FREE in compute.

ProRL's reference-plus-optimizer reset is a fix for KL domination in runs of more than 2k steps. It has no ablation and is mostly irrelevant at 200 steps.

### Cited Findings

**WARP (Ramé et al., Google DeepMind, 2024)**
- Three stages:
  1. RL with the policy's EMA as a dynamic KL anchor.
  2. SLERP merge of M independently RL-fine-tuned policies.
  3. Linear interpolation of the merged model back toward the init to "recover features from pre-training".
  The procedure is then iterated, using the result as the new init. — [arXiv 2406.16768](https://arxiv.org/abs/2406.16768)
- Hyperparameters:
  - EMA rate μ = 0.01. The text also says 0.1 in one place, an internal inconsistency.
  - M = 2 by default, up to 5.
  - T = 9k steps in iteration 1, 7k in iterations 2-3, 5k later.
  - LITI η = 0.3, I = 5 iterations, β = 0.1, SLERP λ = 0.5, SLERP applied per layer.
  - Training sampling temperature 0.9. Evaluation decoding temperature not found in the parts read. — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)
- EMA anchor vs SFT anchor (ablation):
  - β = 0: KL explodes.
  - SFT anchor, β = 0.1: reward saturates around −0.62.
  - SFT anchor, β = 0.01: reward saturates around −0.46.
  - EMA anchor gives the best KL-reward Pareto front. — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)
- SLERP reaches higher reward than LERP at slightly higher KL. Larger M (up to 5) gives Pareto fronts above any single RL trajectory. — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)
- Side-by-side win rates (WARP vs single-run REINFORCE with EMA anchor):

  | Opponent | WARP iter 3 | Single-run REINFORCE + EMA anchor | Gemma 1.1 |
  |---|---|---|---|
  | Mistral-7B-v1 | 0.45 | 0.37 | 0.37 |
  | Mistral-7B-v2 | 0.26 | 0.16 | 0.16 |
  | Mixtral-8x7B | 0.18 | 0.07 | 0.08 |

  Iteration 1 already gives 0.42 / 0.23 / 0.13. Gains plateau after iteration 3 (iteration 5 gives 0.45 / 0.24 / 0.17). — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)
- Zero-shot benchmarks, WARP iter 3 vs Gemma 1.1:

  | Benchmark | WARP iter 3 | Gemma 1.1 |
  |---|---|---|
  | MBPP | 45.4 | 39.0 |
  | GSM8K | 66.8 | 55.6 |
  | MATH | 31.0 | 25.6 |
  | HumanEval | 50.0 | 46.9 |
  | MMLU | 57.6 | 56.4 |
  | BBH | 58.8 | 53.1 |

  — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)
- Cost: training is expensive because of multiple RL runs per iteration. There is no inference overhead. — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)

**EMA anchor in GRPO-style RLVR (EMA-PG, Zhang and Ba, ICML 2026)**
- EMA-PG replaces the frozen KL reference with an EMA of the policy's own weights, analogous to a DQN target network. It adds a Top-k KL estimator: exact KL on the top-k tokens plus a sampled tail estimate. — [arXiv 2602.04417](https://arxiv.org/abs/2602.04417)
- Results:
  - R1-Distill-Qwen-1.5B: 53.9% on OlympiadBench vs 50.8% for GRPO. — [arXiv 2602.04417](https://arxiv.org/abs/2602.04417)
  - Qwen-3B base on search-QA: +33.3% average relative over GRPO across 7 datasets. This figure comes from a secondary write-up and was not checked against the paper. — [ResearchGate listing](https://www.researchgate.net/publication/400459520_EMA_Policy_Gradient_Taming_Reinforcement_Learning_for_LLMs_with_EMA_Anchor_and_Top-k_KL)
- The authors argue that EMA decay, learning rate and KL coefficient jointly predict when training becomes oscillatory or unstable. — [ResearchGate listing](https://www.researchgate.net/publication/400459520_EMA_Policy_Gradient_Taming_Reinforcement_Learning_for_LLMs_with_EMA_Anchor_and_Top-k_KL)
- M-GRPO uses a slowly evolving momentum model as an anchor for self-supervised RL, plus an IQR filter because of rapid entropy collapse. — [arXiv 2512.13070](https://arxiv.org/abs/2512.13070)

**Online Merging Optimizers (Lu et al., Qwen, 2024)**
- At every optimizer step, the update is merged with the SFT delta (θ_SFT − θ_base):
  - OnDARE: random sparsification. Update is θ ← θ + (1−α)·F(Δθ) + α·F(τ).
  - OnTIES: top-magnitude sparsification plus sign consensus.
  - Reserve rate p is robust down to 5e-4. α is about 5e-6 to 5e-7, and large α destabilizes training. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)
- Results (benchmark average / MT-Bench / AlpacaEval-LC):

  | Model | AdamW | OnDARE | Offline DARE |
  |---|---|---|---|
  | Qwen1.5-7B-Chat | 58.4 / 7.18 / 11.66 | 59.5 / 7.30 / 11.94 | 58.6 / 7.05 / 10.07 |
  | LLaMA-3-8B-Instruct | 58.0 / 8.04 / 24.10 | 59.3 / 8.23 / 24.94 | — |
  | Qwen1.5-1.8B-Chat | 41.8 / 4.59 / 4.79 | 42.3 / 4.83 / 4.84 | — |

  Offline merging raised some benchmarks but lowered preference scores. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)
- Tested only with DPO, IPO and KTO, not PPO or RLVR. Under IPO the benchmark average fell: OnDARE 57.0 vs AdamW 58.1. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)
- Merging every K steps instead of every step pulls MT-Bench back toward AdamW, so "onlineness" matters. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)
- Cost:
  - Memory for cached SFT deltas.
  - Not usable with LoRA unless the reference delta is also LoRA. The authors suggest GaLore instead.
  - Compute overhead not quantified. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)
- The paper cites Lin et al. (2024): simple offline averaging of RLHF and SFT weights reduces alignment tax. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)

**Elastic Reset (Noukhovitch et al., NeurIPS 2023)**
- Mechanism: the online model is periodically reset to its own EMA, and the EMA is then reset to the initial model. There is no KL term in the objective.
- Reported results: higher reward with less drift than PPO+KL at equal steps, and SOTA on pivot translation. On IMDB mock sentiment it beats all baselines, and on StackLLaMA-style technical QA (LLaMA-7B) it gives a better and more aligned chatbot. — [arXiv 2312.07551](https://arxiv.org/abs/2312.07551)
- Numerical hyperparameters (EMA decay, reset interval) and effect sizes could not be extracted (PDF unreadable). See Gaps.

**ProRL reference reset (NVIDIA, NeurIPS 2025)**
- Mechanism: the reference policy is hard-reset to a recent snapshot of the online policy and optimizer states are reinitialized. This is done periodically, when validation stagnates or degrades, or when the KL term comes to dominate the loss.
- Run details: R1-Distill-Qwen-1.5B, more than 2k steps, about 16k H100-hours. Rollout temperature 1.2, clip-higher ε_low = 0.2 / ε_high = 0.4, and dynamic filtering of prompts with accuracy 0 or 1.
- No ablation of resets or KL is reported, and the β value is not stated. — [arXiv HTML 2505.24864](https://arxiv.org/html/2505.24864v1)
- ProRL keeps KL because it starts from a model that already produces coherent CoT. It notes that "remove KL" arguments apply mainly to base models before SFT. — [arXiv 2505.24864 via search summary](https://arxiv.org/pdf/2505.24864)
- ProRL v2 (NVIDIA blog): reset the reference every 200-500 steps, or on KL spikes or stalled validation. The optimizer is *not* cleared in v2, which differs from v1. — [NVIDIA blog](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2)

**Merging RL vs SFT models (2026)**
- Pairwise merges of task experts, 5-task average:

  | Merge method | GRPO models | SFT models |
  |---|---|---|
  | Unmerged | 72.2 | 61.5 |
  | Uniform average | 60.1 (−17%) | 47.9 (−22%) |
  | TIES | 67.1 (−7%) | 49.9 (−19%) |

  — [arXiv 2607.22039](https://arxiv.org/html/2607.22039v1)
- On instruction following under TIES, RL models drop 0.3% (GRPO), 2.5% (PPO) and 2.6% (REINFORCE++), vs 28.7% for SFT. — [arXiv 2607.22039](https://arxiv.org/html/2607.22039v1)
- Update norms (Math/Code/IF) are RL 0.78/0.71/0.64 vs SFT 6.50/7.75/4.83, so RL deltas are about 10x smaller. Training on negative samples improves mergeability. — [arXiv 2607.22039](https://arxiv.org/html/2607.22039v1)

**Merged checkpoint as a free teacher**
- GTR-Turbo (agentic VLM RL) uses a merged checkpoint from the run's own history as a free teacher. Only the title and abstract were seen. — [arXiv 2512.13043](https://arxiv.org/pdf/2512.13043)

**WARM** (weight-averaged reward models) concerns reward models. Our rewards are verifiers, so it is not applicable and was not researched further.

### Inferences
- At 200 steps, ProRL-style reference resets add nothing. They address KL domination after hundreds to thousands of steps, and ProRL v2's interval is 200-500 steps, which is our whole run. Our KL-to-v4 is a check, not a loss term, so there is nothing to reset.
- The cheapest transferable idea is an **EMA of the policy as an evaluation candidate**, not as a KL anchor. That means a running EMA of LoRA weights (for example decay 0.9 to 0.95 per step, or averaging the last 3-5 saved checkpoints), evaluated alongside the last checkpoint. This is FREE given our saved checkpoints.
  - Uniform averaging of adapters works in weight space only if we average the merged ΔW = BA, not A and B separately, since BA is bilinear. Averaging A and B separately for checkpoints from one trajectory is close but not exact.
  - This inference is our own; we did not find a paper testing EMA-of-checkpoints for LoRA RLVR.
- Elastic Reset and WARP's LITI step both amount to "interpolate partway back toward init/SFT". A post-hoc version is FREE: evaluate θ_v4 + η·(θ_RL − θ_v4) for η ∈ {0.5, 0.75, 1.0}, which is also cheap extrapolation if η > 1.
  - Given the user's no-sweep preference, a single η = 0.7 check is a reasonable best guess if regressions on held-out appear.
- Online Merging Optimizer is not directly usable: we train LoRA, and it has no RLVR evidence.

### Gaps
- Elastic Reset's EMA decay, reset interval and quantitative KL/reward numbers (the PDF was not machine-readable here).
- No paper found that applies WARP-style SLERP to RLVR with verifiable rewards on reasoning benchmarks for models under 10B.
- WARM, DAPO and VAPO specifics were not fetched. From background knowledge (unverified in this session), DAPO drops the KL term entirely, so it has no reference to reset.

## Q2. Rollback and early-stopping rules for RLVR: collapse detection and monitors

### Takeaway
Published RLVR recipes monitor validation score, response length, KL and entropy. They intervene *forward*, by resetting the reference, raising the temperature, using clip-higher, or adding entropy bonuses, rather than rolling back. No primary source found describes an automatic "revert to checkpoint k" rule. The closest thing to a quantitative trigger is a 2026 entropy-ratio monitor.

### Cited Findings
- **ProRL**
  - Monitors: a blended validation set (AIME24, Codeforces, GPQA-D, IFEval, graph_color) at pass@1 and pass@16, average response length, and KL.
  - Interventions are triggered by validation stagnation or degradation. No rollbacks are described.
  - Higher rollout temperature only *delays* entropy collapse. — [arXiv HTML 2505.24864](https://arxiv.org/html/2505.24864v1)
- **Skywork-OR1**: "mitigating premature entropy collapse is critical for improved test performance".
  - Gains: 7B average 43.6 → 57.5 and 32B 57.8 → 72.8 on AIME24/25 plus LCB.
  - Monitor and rollback details not read (abstract only). — [arXiv 2505.22312](https://arxiv.org/abs/2505.22312)
- **GRPO entropy timing**
  - GRPO policy entropy "largely depletes" after about 175 steps (collapse time) in one study. — [search result, arXiv 2510.10959](https://arxiv.org/pdf/2510.10959)
  - Entropy drops early and declines toward zero. — [arXiv 2605.11491](https://arxiv.org/html/2605.11491v1)
- **Concrete monitor**: a 2026 paper proposes an early GRPO entropy check at step 150. It compares worst-token entropy against its value at step 10 with threshold 0.50, and links SFT overtraining to rank inversion under RLVR. Seen only via a search snippet. — [arXiv 2606.18487](https://arxiv.org/html/2606.18487v1)
- **Entropy and length move together.** For answer-only reward, entropy collapse mirrors response-length collapse. Conversely, entropy regularization produced an entropy surge plus length blow-up after about 400 steps. So monitor both. — [search summaries of arXiv 2510.10959 / 2509.03493 / 2511.05993](https://arxiv.org/pdf/2509.03493)

### Inferences
- Our 200-step run sits right around the reported GRPO entropy-collapse time (about 150-175 steps), so collapse is plausible late in our run.
- A practical FREE rule given our cadence:
  1. Log mean token entropy, mean response length per domain, format/parse-failure rate per domain, and KL-to-v4.
  2. Mark a checkpoint "suspect" if any of these hold:
     - entropy falls below 0.5x its step-10 value (mirrors the 2606.18487 threshold);
     - mean length moves more than ±50% from v4 in any domain;
     - IF format-failure or code-parse-failure rises by more than 2 points;
     - held-out greedy score drops more than one standard error below its best so far.
  3. At the end, select the best non-suspect checkpoint on a held-out dev split (not the final test sets), instead of defaulting to step 200.

  The thresholds are our proposal, not from literature, except the entropy ratio.

### Gaps
- No primary source found with an explicit automated rollback-to-earlier-checkpoint policy in RLVR. Tulu 3 and OLMo 3 checkpoint-selection details were not fetched.

## Q3. SLERP/averaging of several short independent RL runs vs one longer run at fixed compute

### Takeaway
The only direct evidence is WARP. Merging M = 2 runs beat a single run, and more runs push the Pareto front further. However, WARP compares M runs of T steps against one run of T steps, not equal total compute. We found no fixed-compute comparison for RLVR.

### Cited Findings
- WARP: merged policies (M up to 5) lie on Pareto fronts above any individual RL trajectory. The win rate vs Mixtral is 0.13 after iteration 1 (M = 2) vs 0.07 for a single REINFORCE+EMA run of the same length. — [arXiv HTML 2406.16768](https://arxiv.org/html/2406.16768)
- RL-trained experts merge with small losses (TIES −7% vs SFT −19%). This suggests RL deltas from a shared init are close to linearly mode-connected. That evidence is for different-task experts, not seeds. — [arXiv 2607.22039](https://arxiv.org/html/2607.22039v1)
- OLMo 3 could not afford multi-seed runs, so it merged checkpoints from one run, and two learning-rate variants for SFT, instead. — [Cameron Wolfe on OLMo 3](https://cameronrwolfe.substack.com/p/olmo-3)
- Souper-Model cautions against souping checkpoints from different training stages. — [arXiv 2511.13254](https://arxiv.org/html/2511.13254v1)

### Inferences
- For our budget, two 100-step seeds SLERPed would likely be worse than one 200-step run: WARP used about 9k steps per run, and our runs are short and early in the reward curve. This is unverified.
- Averaging *adjacent checkpoints of one run* (for example steps 170-200) is the FREE proxy and is what OLMo 3 did for SFT.

### Gaps
- No equal-compute study of seed-merging vs a longer single run for RLVR.

## Q4. Greedy decoding vs sampled pass@1 under averaging and merging

### Takeaway
No source found directly measures whether weight averaging or anchor methods change greedy accuracy differently from sampled pass@1 in RLVR. This remains open.

### Cited Findings
- RLVR raises pass@1 while shrinking pass@k coverage (pass@256 falls as training progresses), meaning RL sharpens the distribution toward already-likely paths. — [Yue et al., arXiv 2504.13837](https://arxiv.org/html/2504.13837v4)
- ProRL tracks pass@1 and pass@16 separately. Its rollout temperature of 1.2 is used to keep entropy up. — [arXiv HTML 2505.24864](https://arxiv.org/html/2505.24864v1)

### Inferences
- Because RL sharpens the distribution, interpolating back toward v4 (LITI/elastic) or averaging checkpoints should flatten it slightly. Greedy accuracy is usually the least sensitive to mild flattening when the argmax is preserved, and sampled pass@1 at T = 1 benefits more from sharpening.
- Expect greedy gains from RL to be smaller than sampled-pass@1 gains, and interpolation to cost less in greedy than in sampled metrics. Evaluate candidate merges with greedy decoding, since that is our metric. This is an inference, not a cited finding.

### Gaps
- No direct measurement found.

## Q5. LoRA-specific results on RL merging and averaging

### Takeaway
"LoRA Without Regret" shows LoRA (even rank 1) matches full fine-tuning in policy-gradient RL at about 10x LR, but it says nothing about merging or averaging. The only LoRA-specific checkpoint technique found is trajectory extrapolation (NExt, about 37.5% compute saving), and Online Merging is not LoRA-compatible.

### Cited Findings
- **LoRA Without Regret**
  - LoRA matches full fine-tuning in policy-gradient RL "even with ranks as low as 1", because RL delivers about 1 bit per episode.
  - Optimal LoRA LR is about 10x full fine-tuning (fit 9.8 across 14 models), and about 15x for runs under 100 steps.
  - LoRA is less tolerant of large batches.
  - It does not discuss adapter or checkpoint merging. — [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)
- **NExt (2026)**: in LoRA RLVR, the rank-1 subspace of parameter differences across checkpoints grows dominant and evolves nonlinearly. A predictor trained on saved checkpoints extrapolates future weights, saving about 37.5% of compute. Accuracy vs full training is not shown in the abstract. — [arXiv 2604.11446](https://arxiv.org/abs/2604.11446)
- Online Merging Optimizer cannot be used with LoRA unless the reference delta is LoRA. — [arXiv HTML 2405.17931](https://arxiv.org/html/2405.17931v1)

### Inferences
- With r32 (far more capacity than RL needs, per LoRA Without Regret), the RL deltas are low-rank and small. This makes averaging or extrapolating saved adapter checkpoints in ΔW space well-posed and cheap.
- The NExt rank-1 dominance suggests linear extrapolation beyond the final checkpoint (θ_200 + c·(θ_200 − θ_150)) is a FREE candidate, but risky: the paper says the trajectory is nonlinear.
- LR check: 2e-5 for LoRA corresponds to about 2e-6 full fine-tuning-equivalent, in the normal range.

### Gaps
- No paper on averaging LoRA RL checkpoints or seeds specifically. NExt's accuracy numbers were not retrieved.

**Bottom line for our loop.**
- FREE options, ranked by expected value:
  1. Choose the final model by held-out-dev greedy evaluation over saved checkpoints, with collapse monitors (entropy ratio, per-domain length, format-failure).
  2. Evaluate a tail average or EMA of the last 3-4 checkpoints in ΔW space.
  3. Evaluate an interpolation toward v4 (η ≈ 0.7) only if held-out regressions appear.
- EXTRA-COMPUTE options: WARP-style multi-seed SLERP has the best published effect sizes, but those came from about 9k-step runs and have no fixed-compute evidence. Not recommended at our budget.

# Policy-optimization choices for short-horizon LoRA RLVR (Qwen3.5-9B-Base, ~100-120 steps, 64x8, greedy eval)

Source-verification note: items marked [verified] were read in this session (fetched page or search abstract). Items marked [prior] come from the paper's arXiv abstract as known before this session and were NOT re-fetched; the report writer should treat their exact numbers as "to double-check". No number below was invented; where I was unsure, it is in Gaps.

## 1. GRPO variants: what each fixes, under which conditions, and whether it matters for ~100 sync on-policy steps

### Takeaway
Almost every 2025 GRPO variant fixes a problem that is driven by off-policyness (many minibatch updates per rollout, async rollouts), long generations (8k-32k), or MoE routing noise. In a strictly synchronous, one-gradient-step-per-batch setup the importance ratio is ~1 (only sampler/trainer numerical mismatch), so clipping variants (DAPO clip-higher, GSPO, CISPO, GMPO) barely activate; the components that still matter are the advantage estimator, dropping zero-variance groups, loss aggregation, and truncation handling.

### Cited Findings
- Tinker losses: `importance_sampling` = REINFORCE with p/q ratio (no clipping); `ppo` clips the ratio; `cispo` "clip the ratio but use it as a gradient coefficient"; `dro` adds a quadratic penalty on policy divergence. "For all objectives we sum the token-level losses over the sequence length unlike some other loss implementations"; other aggregation must be folded into the advantage tensor. REINFORCE and PPO "do not use an additional KL term". RL losses accept only `target_tokens`, `logprobs`, `advantages`. [verified] — [Tinker losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)
- Tinker guidance: start with `importance_sampling` for RL; switch to `ppo` or `cispo` if you see instability from large policy updates. CISPO default caps only the upper side (threshold 4.0, no lower bound). [verified, via search snippet of docs] — [Tinker loss tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/); [Tinker CISPO](https://tinker-docs.thinkingmachines.ai/tinker/losses/cispo/)
- DAPO (arXiv 2503.14476): clip-higher (eps_low 0.2, eps_high 0.28), dynamic sampling (drop all-correct/all-wrong groups and refill), token-level policy-gradient loss, overlong reward shaping; no KL. Qwen2.5-32B base reaches 50 on AIME24 using ~50% of the steps of DeepSeek-R1-Zero-Qwen-32B. Max length 20k, 16 samples per prompt, 512 prompts per batch. [prior] — [DAPO arXiv](https://arxiv.org/abs/2503.14476)
- Dr. GRPO (2503.20783): GRPO's per-response 1/|o| normalization and per-question std normalization bias optimization (long wrong answers under-penalized, response length inflates); removing both gives an unbiased PPO-style objective, shorter incorrect responses, better token efficiency. Minimal recipe: Qwen2.5-Math-7B reaches 43.3% AIME24. [verified] — [Dr. GRPO arXiv](https://arxiv.org/html/2503.20783v2)
- GSPO (2507.18071, Qwen team): sequence-level (length-normalized) importance ratio and sequence-level clipping; motivated by token-level ratio noise that accumulates over long responses and destabilizes MoE training (removes need for "routing replay"); used for Qwen3 RL. [prior] — [GSPO arXiv](https://arxiv.org/abs/2507.18071)
- CISPO (MiniMax-M1, 2506.13585): clip and stop-gradient the IS weight instead of clipping the token update, so low-probability "fork" tokens (e.g., "However", "Recheck") keep gradients; reported to match DAPO with about half the steps on Qwen2.5-32B base in their comparison; designed for a 456B hybrid MoE with long (40k-80k) generations and off-policy minibatches. [prior] — [MiniMax-M1 arXiv](https://arxiv.org/abs/2506.13585)
- ScaleRL (Meta, 2510.13786, ICLR 2026 oral): >400k GPU-hours, ablations on an 8B dense model (3.5k-4k GPU-hours per run), largest 100k GPU-hours; GSPO and CISPO clearly beat DAPO on asymptotic pass rate, CISPO marginally better late, GSPO sometimes diverged; in leave-one-out at scale, reverting to DAPO gave a similar asymptote but lower efficiency (B = 1.77 vs 2.01). Common tweaks (loss aggregation, advantage normalization, curriculum, length penalty) mainly change efficiency, not the ceiling; loss type, FP32 logits, batch size, and generation length move the asymptote. [verified] — [ScaleRL arXiv html](https://arxiv.org/html/2510.13786v1); [ICLR 2026 oral page](https://iclr.cc/virtual/2026/oral/10010608)
- ScaleRL setting is asynchronous PipelineRL (off-policy); final recipe: CISPO, prompt-level loss averaging, batch-level advantage normalization, FP32 logits, zero-variance filtering, No-Positive-Resampling (drop prompts with pass rate >= 0.9 permanently). [verified] — [ScaleRL arXiv html](https://arxiv.org/html/2510.13786v1)
- "Tricks or Traps?" (2508.08221, reported ICLR 2026): on Qwen3-4B/8B **base** models, clip-higher had minor effect on entropy and sometimes hurt (base-model clip rate ~0.003); it helped **aligned** models (4B-aligned best at 0.32; 8B-aligned best at 0.28). Lite PPO = group-mean + batch-std advantages + token-level aggregation + plain PPO loss, no critic, beat GRPO and DAPO (GRPO/DAPO peaked then collapsed on 4B base). Setting: 500 steps, 128 prompts x 8, lr 1e-6 full FT, 8k max length. [verified] — [Tricks or Traps arXiv v2](https://arxiv.org/html/2508.08221v2)
- GMPO (2507.20673): geometric mean of token-level ratios instead of arithmetic mean, to reduce sensitivity to outlier ratios and allow a wider clip range; reported a few-point average gain over GRPO on math benchmarks at 7B. [prior] — [GMPO arXiv](https://arxiv.org/abs/2507.20673)
- REINFORCE++ (2501.03262): critic-free REINFORCE with global (batch) advantage normalization and PPO-style clipping; argues group-level normalization overfits per prompt. [prior] — [REINFORCE++ arXiv](https://arxiv.org/abs/2501.03262)
- VAPO (2504.05118): value-model-based PPO (value pretraining, decoupled GAE, length-adaptive GAE); Qwen2.5-32B reaches 60.4 AIME24 within 5,000 steps. Requires a learned critic. [prior] — [VAPO arXiv](https://arxiv.org/abs/2504.05118)
- "stable_reinforce": no paper or library loss by this name was found in search; nearest matches were REAL (rewards-as-labels BCE) and ΔL normalization. [verified absence in search] — [search results incl. ΔL Normalization](https://www.microsoft.com/en-us/research/?p=1149997)

### Inferences
- With one optimizer step per freshly sampled batch, p/q differs from 1 only by sampler/trainer numerical mismatch, so PPO/CISPO/GSPO clipping rarely fires; on base models even DAPO's 1e-6-lr full-FT setting clipped ~0.3% of tokens (Tricks or Traps). The choice among IS/PPO/CISPO is mainly a guard against mismatch spikes, not a performance lever at 100 steps.
- GSPO's main benefit (MoE routing stability, long sequences) does not apply to a dense 9B with <=4k tokens. VAPO needs a critic, which Tinker's loss interface (advantages only) does not train for you: skip.
- The project's draft (mean-centered, no std, token-sum, no KL, drop constant groups) is essentially Dr. GRPO + DAPO dynamic-sampling-by-filtering, run with the Tinker cookbook default; this matches the Thinking Machines LoRA RL experiments (see section 5).

### Gaps
- No paper found that ablates these variants specifically at ~100 steps with LoRA on a 7-14B dense model; all head-to-heads are at 500+ steps, full FT, and often 8k-32k tokens.
- Exact CISPO-vs-DAPO speedup number and GMPO gain were not re-fetched.
- Tinker's per-loss pages (exact PPO clip defaults) were not readable (404 / summary only).

## 2. Advantage normalization by std; leave-one-out vs mean baseline

### Takeaway
Evidence is mixed but converges on "it matters little for the ceiling": Dr. GRPO argues std normalization adds a difficulty bias; Tricks-or-Traps finds std division destabilizes on easy data and recommends group-mean + batch-std; ScaleRL finds prompt-level, batch-level, and no normalization perform similarly. Leave-one-out vs mean baseline is a constant rescaling for fixed group size.

### Cited Findings
- Dr. GRPO: per-question std normalization up-weights questions that are too easy or too hard (low std), a "question-level difficulty bias"; recommends removing it. [verified] — [Dr. GRPO arXiv](https://arxiv.org/html/2503.20783v2)
- Tricks or Traps: on easy data, reward std shrinks and dividing by it inflates advantages, causing instability/gradient explosions; removing std stabilized. On hard data, mean-only vs std-normalized performed similarly. Group-level mean + batch-level (global) std beat local std on 4B and 8B base. [verified] — [Tricks or Traps](https://arxiv.org/html/2508.08221v2)
- ScaleRL: prompt-level, batch-level, and no normalization all similar; batch-level chosen as "theoretically sound and marginally better"; normalization affects efficiency, not asymptote. [verified] — [ScaleRL](https://arxiv.org/html/2510.13786v1)
- RLOO (Ahmadian et al., 2402.14740, ACL 2024) uses the leave-one-out mean of the other k-1 samples as baseline. [prior] — [RLOO arXiv](https://arxiv.org/abs/2402.14740)

### Inferences
- Algebra: r_i - mean_{j≠i} r_j = (G/(G-1)) (r_i - mean_all). With G=8 this is mean-centering times 8/7: identical direction, absorbed by the learning rate. RLOO vs group-mean is therefore a non-decision for G fixed.
- For binary rewards, group std = sqrt(p(1-p)). With G=8: p=1/8 gives std ~0.33, p=1/2 gives 0.5, so std normalization up-weights 1-of-8 groups by ~1.5x relative to 4-of-8. That is a mild "focus on hard-but-solvable prompts" effect; the instability case (std -> tiny) is ruled out anyway by dropping constant groups, but at p=1/8 an advantage magnitude of 0.875/0.33 ~ 2.6 per token is possible.
- Mixing domains matters more here: math/code binary rewards and IFEval partial rewards (fraction of constraints satisfied) have different scales. Without any std normalization, a partial-reward group with small spread gets small advantages; batch-level std (per domain) is the cleanest way to equalize domains without the per-prompt difficulty bias. This is the strongest reason to deviate from the draft.

### Gaps
- No direct evidence on mixed-domain RLVR with heterogeneous reward scales at small step counts.

## 3. Entropy collapse and exploration in short runs; helpful or harmful for greedy eval?

### Takeaway
Entropy falls fastest early in training and performance gains track entropy consumption; interventions (Clip-Cov/KL-Cov, clip-higher, forking-token-only updates) mostly pay off over long horizons and on pass@k. For greedy pass@1 in ~100 steps, moderate entropy reduction (sharpening) is the mechanism of the gain and is not something to fight, as long as it does not cause repetition/degeneration.

### Cited Findings
- Entropy Mechanism (Cui et al., 2505.22617): without intervention, downstream performance is predictable from policy entropy via a simple exponential fit, implying a ceiling once entropy is exhausted; entropy change is driven by covariance between log-prob and advantage; Clip-Cov (detach a small fraction of high-covariance tokens) and KL-Cov (KL on top-covariance tokens) keep entropy higher than GRPO and GRPO+clip-higher; up to 6.4% downstream gains reported by one summary. [verified via abstract/summaries] — [arXiv 2505.22617](https://arxiv.org/html/2505.22617v1); [HF papers](https://huggingface.co/papers/2505.22617)
- 80/20 forking tokens (Wang et al., 2506.01939, NeurIPS 2025): ~20% of CoT tokens are high-entropy "forks"; updating only those matches full-gradient updates on Qwen3-8B base and beats it on Qwen3-32B (+11.04 AIME25, +7.71 AIME24) and Qwen3-14B (+4.79 AIME25, +5.21 AIME24); training on the bottom 80% only hurts. Gains scale with model size; effectively nil at 8B. [verified via abstract/summaries] — [arXiv 2506.01939](https://arxiv.org/html/2506.01939v2); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115123)
- Clip-higher has minor entropy effect on base models (clip rate ~0.003) and sometimes hurts; helps aligned models. [verified] — [Tricks or Traps](https://arxiv.org/html/2508.08221v2)
- A follow-up paper argues clip-low increases entropy and clip-high decreases it (2509.26114). [verified title only] — [HF papers 2509.26114](https://huggingface.co/papers/2509.26114.md)
- NSR-only training keeps entropy near the base model; PSR-only causes rapid entropy drop (see section 6). [verified via abstract/summary] — [arXiv 2506.01347](https://arxiv.org/pdf/2506.01347)
- Yue et al. "Does RL really incentivize reasoning capacity beyond the base model?" (2504.13837, NeurIPS 2025): RLVR raises pass@1 but base models win at large k; RL mostly sharpens toward already-sampled solutions. [prior] — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)

### Inferences
- Greedy decoding is the zero-temperature limit; collapse of probability mass onto the correct mode directly improves greedy pass@1. The harmful side of collapse (lost pass@k, stalled learning at long horizons) is largely irrelevant for a 100-step run evaluated greedily.
- Real risk at short horizon is the opposite failure: temperature-1 training with a policy that is still diffuse, while greedy eval reveals repetition loops. Monitor greedy eval length/truncation rate, not entropy alone.
- Clip-higher does nothing under the IS loss (no clipping) and little under sync PPO; entropy bonus is not supported by Tinker's RL loss inputs (advantages only) without custom forward-backward. Skip both.

### Gaps
- Exact fitted constants of R = -a·exp(H) + b and the "fraction of gain in early steps" figure were not verified.
- No study found that measures greedy (not sampled) accuracy versus entropy trajectory.

## 4. KL penalty to the reference

### Takeaway
The dominant 2025 practice for RLVR from base models (DAPO, Dr. GRPO, Tinker defaults) drops the KL term; KL+reference resets (ProRL) are a tool for 2,000+ step runs to keep stability, not a short-horizon gain.

### Cited Findings
- DAPO removes the KL penalty, arguing the policy should diverge substantially from the initial model in long-CoT RL. [prior] — [DAPO arXiv](https://arxiv.org/abs/2503.14476)
- Tinker REINFORCE/PPO losses do not include a KL term. [verified] — [Tinker losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)
- ProRL (2505.24864): keeps a small KL to a reference plus periodic hard resets of the reference (and optimizer state) to a recent snapshot, combined with DAPO clip-higher and dynamic sampling, over >2k steps on DeepSeek-R1-Distill-Qwen-1.5B with 136K problems; argues KL helps prevent entropy collapse while a static reference would hold back learning. [verified via abstract/summaries] — [ProRL arXiv](https://arxiv.org/html/2505.24864v1); [follow-up 2507.12507](https://arxiv.org/html/2507.12507v1)

### Inferences
- ProRL starts from a distilled reasoning model (already aligned, low entropy) and trains thousands of steps; neither condition holds here. LoRA rank-32 at ~100 steps already limits drift.
- If a KL is wanted to protect non-RL skills (e.g., IFEval from math/code RL), it can be approximated in Tinker by subtracting beta·(logp_sampler − logp_ref) from per-token advantages (the cookbook has a KL-penalty-in-advantage option per my prior knowledge, not verified this session). Treat as optional.

### Gaps
- No ablation found of KL vs no-KL specifically at ~100 steps with LoRA.

## 5. LoRA for RL

### Takeaway
Thinking Machines' "LoRA Without Regret" shows LoRA matches full FT for policy-gradient RL even at rank 1, with optimal LR ~10x the full-FT LR and LoRA applied to MLP layers (attention-only underperforms). Rank 32 is far above need; capacity is not the bottleneck at this budget.

### Cited Findings
- LoRA "fully matches the learning performance of FullFT when running policy gradient algorithms", down to rank 1; on MATH/GSM it has a wider range of good LRs and the same peak. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- Information argument: policy gradient learns ~O(1) bits per episode; MATH 10k problems x 32 samples ~ 320k bits vs ~3M params in rank-1 LoRA on Llama-3.1-8B. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- Optimal LoRA LR ~10x FullFT LR (fitted 9.8x), in both SFT and RL. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- Attention-only LoRA underperforms MLP-only in RL on MATH; adding attention to MLP gave no further benefit. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- RL algorithm used: policy gradient with importance sampling correction plus GRPO-style group-mean centering (i.e., the same as the project's draft). Models: Llama-3.1-8B base (MATH/GSM; Qwen avoided due to math in pretraining), Qwen3-8B-base on DeepMath-103K at 8,192 tokens; LoRA ranks and FullFT trained near-identically, similar AIME24/25. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- Large-batch LoRA penalty was shown only for SFT, not RL. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- "Defeating the Training-Inference Mismatch via FP16" reports (via secondary summary) BF16 LoRA RL collapsing after ~600 steps while FP16 LoRA stayed stable. [verified only as secondary summary] — [arXiv 2510.26788](https://arxiv.org/html/2510.26788v1)

### Inferences
- Full-FT RLVR papers use lr ~1e-6 (Tricks or Traps: 1e-6); 10x gives ~1e-5 for LoRA; the planned 2e-5 is within ~2x of that and reasonable, slightly aggressive. Since gains at 100 steps are LR-limited, LR is likely the single most consequential hyperparameter; keep it.
- Qwen-family pretraining contains math data (Thinking Machines avoided Qwen for that reason), so early math gains may reflect format/elicitation more than learning; this argues for reporting a no-RL baseline with identical eval.

### Gaps
- No public LoRA-RL ablation on Qwen3.5 specifically; no LoRA-RL evidence on code or IF tasks.

## 6. Positive vs negative sample reinforcement; pass@k-aware objectives; effect on pass@1/greedy

### Takeaway
Negative-only reinforcement preserves diversity and pass@k; positive-only maximizes pass@1 but kills pass@k. For a greedy pass@1 objective, the standard balanced (mean-centered) signal or mild up-weighting of negatives is fine; pass@k-oriented objectives trade off exactly the metric we evaluate.

### Cited Findings
- Zhu et al. (2506.01347, NeurIPS 2025): PSR-only raises pass@1 but hurts pass@k at larger k (diversity loss); NSR-only improves over base across pass@k up to k=256 and often matches/beats PPO and GRPO; NSR keeps entropy near base; W-REINFORCE = λ·L_PSR + L_NSR with λ=0.1 competitive or better than PPO/GRPO on MATH, AIME25, AMC23. Models: Qwen2.5-Math-7B, Qwen3-4B, Llama-3.1-8B-Instruct. [verified via abstract/NeurIPS page; λ from secondary summary] — [arXiv 2506.01347](https://arxiv.org/pdf/2506.01347); [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b8026fe01f5f8f7a80a1a9bf01d08cad-Abstract-Conference.html)
- RLVR mainly sharpens toward existing solutions (pass@1 up, large-k pass@k down vs base). [prior] — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- Pass@k Training (Chen et al., 2508.10751) optimizes pass@k as reward to keep exploration; [prior, numbers not verified] — [arXiv 2508.10751](https://arxiv.org/abs/2508.10751)

### Inferences
- With group-mean centering, a 1-of-8 group gives the single correct sample +0.875 and each wrong sample −0.125 (positive-dominated per sample, balanced in sum); a 7-of-8 group is negative-dominated. Mean-centering is already a balanced PSR/NSR mix; W-REINFORCE's λ=0.1 would push toward pass@k, which is not our metric.
- Mild NSR emphasis could still help greedy eval if it suppresses degenerate modes (repetition, format errors) that greedy decoding would otherwise land in; untested.

### Gaps
- No paper found evaluating NSR/W-REINFORCE with greedy decoding specifically.

## 7. Group size, batch size, number of steps vs gains (1.5B-14B)

### Takeaway
Published small-model recipes use 8-16 samples per prompt and 128-512 prompts per step for 500-2,000+ steps; a 64x8 batch for ~100 steps is ~5-10x less compute than the smallest standard ablation, so expect gains mostly in format/elicitation and the first steep part of the curve. ScaleRL shows small batches look better early and are overtaken later, which favors small batches at our horizon.

### Cited Findings
- Tricks or Traps: 128 prompts x 8, 500 steps, Qwen3-4B/8B base, 8k tokens. [verified] — [Tricks or Traps](https://arxiv.org/html/2508.08221v2)
- ScaleRL: larger batches reach higher asymptote; smaller batches look better early but are overtaken; longer generation (14k->32k) slows early progress but raises the asymptote. [verified] — [ScaleRL](https://arxiv.org/html/2510.13786v1)
- DAPO: 512 prompts x 16 samples, 32B. [prior] — [DAPO arXiv](https://arxiv.org/abs/2503.14476)
- ProRL: >2k steps on 1.5B. [verified via summary] — [ProRL](https://arxiv.org/html/2505.24864v1)
- Thinking Machines LoRA RL: MATH ~10k problems x 32 samples. [verified] — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- 1-shot RLVR (Wang et al., 2504.20571): a single training example lifted Qwen2.5-Math-1.5B MATH500 from ~36% to ~73.6%, showing much of early RLVR gain on Qwen math models is elicitation. [prior] — [arXiv 2504.20571](https://arxiv.org/abs/2504.20571)
- Spurious-reward results (2506.10947): Qwen2.5-Math gains even with random/incorrect rewards, not reproduced on Llama/OLMo. [prior] — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)

### Inferences
- G=8 with binary rewards: a prompt with true pass rate p yields a zero-variance group with prob p^8 + (1-p)^8 (e.g., p=0.9 -> 43%, p=0.5 -> 0.8%). After SFT, many math/IF prompts will be near-saturated, so filtering prompts to pass-rate ~0.1-0.9 (ScaleRL's No-Positive-Resampling idea) is likely the most compute-efficient lever under an $80 budget.
- Because Qwen-family math gains can be elicitation-dominated, the post-SFT model may already capture much of it; expect smaller math gains than published base-model RL papers.

### Gaps
- No published LoRA-RL step-vs-gain curve at 9B with a 64x8 batch.

## 8. Training-inference mismatch (sampler vs trainer logprobs) and fixes

### Takeaway
Separate sampler and trainer kernels make "on-policy" RL slightly off-policy; the effect is largest for MoE, long sequences, and low-precision rollouts, and fixes are truncated IS (cap the p/q ratio), masked IS, FP16, or batch-invariant kernels. With Tinker's `importance_sampling` loss the p/q ratio is already applied but uncapped, so the cheap safeguard is to log the per-token ratio / sampler-trainer KL and switch to `cispo` or `ppo` if spikes appear.

### Cited Findings
- FP16 paper (Qi, Liu et al., 2510.26788, Sea AI Lab/NUS): BF16 rounding is the main cause; switching to FP16 nearly eliminates the mismatch; under FP16 differences between RL algorithms nearly disappear; TIS/MIS described as patches needing an extra forward pass and leaving a deployment gap; secondary summary reports BF16 LoRA collapse at ~600 steps vs stable FP16 LoRA. [verified via abstract/summaries] — [arXiv 2510.26788](https://arxiv.org/html/2510.26788v1)
- Truncated IS (Yao et al., Aug 2025 blog "Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"): weight the policy gradient by min(p_train/p_sampler, C), with a small cap (C=2 in their examples). [prior; page not readable in this session] — [Feng Yao blog](https://fengyao.notion.site/off-policy-rl)
- ScaleRL: FP32 LM-head logits raised the 8B asymptote from ~0.52 to ~0.61 but made little difference in the 8B leave-one-out; mattered more for the MoE. [verified] — [ScaleRL](https://arxiv.org/html/2510.13786v1)
- Thinking Machines "Defeating Nondeterminism in LLM Inference" (Sep 2025): batch-invariant kernels give bitwise-identical sampler/trainer logprobs and truly on-policy RL (zero KL between sampler and trainer). [prior] — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- Tinker docs define p_theta as the trainer and q as the sampling distribution; logprobs passed in are "from the rollout policy". No mention of truncation. [verified] — [Tinker losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

### Inferences
- The project cannot change Tinker's precision or kernels; it can (a) log mean/max p/q and KL(q||p) per step, (b) use `cispo` (upper cap 4.0 by default) as a TIS-like guard, or (c) clip the ratio implicitly via `ppo`. At 100 steps on a dense 9B with <=4k tokens, mismatch-driven collapse (reported at hundreds of steps, mostly MoE/long) is unlikely but cheap to guard.

### Gaps
- No public measurement of Tinker's sampler/trainer mismatch magnitude was found.

## 9. Ranked adopt / consider / skip for our case

### Takeaway
Keep the draft's core (group-mean advantages, token-sum, no KL, drop constant groups, sync IS loss). The only changes with good evidence-per-dollar are prompt filtering by pass rate, per-domain reward scaling, truncation handling, and logging mismatch; everything else targets long-horizon, off-policy, or MoE problems we do not have.

### Cited Findings
- Supporting evidence is in sections 1-8 above: ScaleRL (zero-variance filtering, No-Positive-Resampling, normalization only affects efficiency) — [ScaleRL](https://arxiv.org/html/2510.13786v1); Tricks or Traps (std unstable on easy data; clip-higher minor for base; overlong filtering helps at 8k) — [Tricks or Traps](https://arxiv.org/html/2508.08221v2); LoRA Without Regret (IS + group-mean with LoRA matches FullFT) — [LoRA blog](https://thinkingmachines.ai/blog/lora/); Tinker guidance (start IS, move to ppo/cispo on instability) — [Tinker tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)

### Inferences
Adopt
1. Group-mean-centered advantages, no per-group std (Dr. GRPO; ScaleRL says normalization barely moves the ceiling; avoids small-std blow-ups).
2. Drop zero-variance groups and pre-filter/resample prompts to post-SFT pass rate in ~[0.1, 0.9] (ScaleRL zero-variance filtering + No-Positive-Resampling; largest compute saving at G=8).
3. Token-sum loss, no length normalization (Tinker default; Dr. GRPO-equivalent up to a constant; avoids length bias).
4. No KL penalty (DAPO, Dr. GRPO, Tinker defaults; KL+reset only shown useful for 2k+ step runs).
5. Sync on-policy `importance_sampling` with LoRA lr ~1-2e-5 (LoRA Without Regret: 10x full-FT LR, IS + group mean is their exact setup).
6. Log per-step p/q ratio stats, sampler-trainer KL, entropy, and greedy-eval truncation rate (cheap early warning for mismatch and degeneration).

Consider
7. Per-domain (or batch-level) reward scale normalization so IFEval partial rewards and binary math/code rewards contribute comparably (Tricks-or-Traps group-mean + batch-std; ScaleRL batch-level).
8. Truncation handling: mask or zero the loss for responses hitting the cap rather than giving reward 0 (Tricks or Traps: overlong filtering helps at 8k; relevant at 1k IF cap).
9. Switch to `cispo` (or `ppo`) only if ratio spikes or loss instability appear (Tinker guidance; ScaleRL's preferred loss, robust to clip setting).
10. LoRA rank: 32 is fine; lowering it saves nothing meaningful on Tinker and rank is not the bottleneck (rank-1 suffices for RL).

Skip
11. GSPO (fixes MoE/long-sequence token-ratio noise; dense 9B, <=4k tokens).
12. Clip-higher / DAPO decoupled clip (no-op under IS loss; minor or harmful on base-derived models per Tricks or Traps).
13. VAPO / any critic-based method (needs value model; Tinker loss takes only advantages).
14. Entropy bonus, Clip-Cov/KL-Cov, 80/20 forking-token masking (long-horizon pass@k tools; 80/20 gives ~no gain at 8B; greedy eval benefits from sharpening).
15. W-REINFORCE / NSR-heavy or pass@k objectives (optimize pass@k diversity, not greedy pass@1).
16. RLOO leave-one-out baseline (equals mean-centering x G/(G-1); no change).
17. GMPO, REINFORCE++ global normalization as separate algorithms (marginal at sync 100 steps; batch-std idea is captured by item 7).
18. Overlong reward shaping with soft penalty buffer (DAPO tuned for 20k contexts; filtering in item 8 is simpler).

### Gaps
- No direct evidence for any of these choices at the exact regime (LoRA, 9B, ~100 steps, mixed math/code/IF, greedy eval); the ranking extrapolates from 500+-step full-FT studies and should be treated as best-guess priors.

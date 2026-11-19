# Hybrid SFT+RL post-training for a small LoRA model on a tight budget

Scope: methods that mix supervised (demo / teacher) signal with RL (verifiable reward) signal, 2024-2026, judged against our setup: Qwen3.5-9B-Base, LoRA r=32 on Tinker (cross_entropy + importance_sampling losses, custom per-token weights/advantages), v4 SFT checkpoint (IFEval 75.9 / MATH-500 71.8 / MBPP 73.2 greedy), planned ~200-step clip-free IS-REINFORCE RLVR (G=8, 64 groups/step) on IF+math+code, large pools of verified GPT-teacher answers, Qwen 397B on Tinker (use as teacher pending approval), ~$135 total.

Main caveat across the whole literature: almost every hybrid paper is **math only**, starts from a **base or weak instruct model with no prior SFT on the target distribution** (mostly Qwen2.5-Math-7B "zero-RL"), uses **full fine-tuning**, and trains on **DeepSeek-R1 long-CoT traces**. None of the off-policy-guidance papers I found reports IFEval or MBPP. Our v4 is already a strong SFT model, so the headline gains (+5 to +10 avg) overstate what we should expect.

## 1. Off-policy guidance / mixing expert demos into RL (LUFFY, ReLIFT, SRFT, UFT, CHORD, HPT, BREAD, Prefix-RFT)

### Takeaway
At the 1.5B-8B scale on math, mixing expert demos into RL beats pure SFT, pure RL, and SFT->RL by about 2-9 points on in-distribution averages, and the gain is biggest where on-policy RL gets no signal (prompts where every rollout fails, or weak models like Llama-3.1-8B). The cheapest version to build, and the one with the best reported numbers, is HPT's per-prompt gate: use RL when the group has any correct rollout, and SFT on a demo when all rollouts fail. Simply adding a fixed SFT loss to RL was the *worst* baseline in LUFFY.

### Cited Findings
**LUFFY: "Learning to Reason under Off-Policy Guidance" (Yan et al., 2025, arXiv 2504.14945)**
- Method: Mixed-Policy GRPO. Each group has 8 rollouts: 1 off-policy R1 trace + 7 on-policy. "Policy shaping" uses regularized importance sampling f(x)=x/(x+γ), γ=0.1, which upweights low-probability tokens in the expert trace and counters entropy collapse — [arXiv HTML](https://arxiv.org/html/2504.14945)
- Data: about 45k prompts from OpenR1-Math-220k, keeping correct R1 traces of 8192 tokens or fewer. LR 1e-6, 500 steps, 8xA100 — [arXiv HTML](https://arxiv.org/html/2504.14945)
- Qwen2.5-Math-7B, 6-math-benchmark avg / OOD avg (ARC-c, GPQA*, MMLU-Pro): SFT 44.1/47.5; on-policy RL 45.5/57.3; SFT->RL 48.2/44.8; **RL w/ SFT loss 40.1/46.0**; LUFFY 50.1/57.8 — [arXiv HTML](https://arxiv.org/html/2504.14945)
- Compute (GPU-hours x8): SFT 24, LUFFY 77, SFT->RL 130, RL w/ SFT loss 133. The paper says the SFT-loss variant's extra cost comes from "excessively long generations induced by rigid imitation" — [arXiv HTML](https://arxiv.org/html/2504.14945)
- Without shaping, Mixed-Policy GRPO plateaus and converges back to on-policy RL, and its entropy collapses faster than on-policy RL — [arXiv HTML](https://arxiv.org/html/2504.14945)
- Other models (avg, LUFFY vs on-policy RL): Qwen2.5-Math-1.5B 38.0 vs 30.0; Qwen2.5-Instruct-7B 40.7 vs 35.2; Llama-3.1-8B 13.2 vs 9.6 (SFT 5.9). On-policy RL collapses on the Hard subset for Llama-8B — [arXiv HTML](https://arxiv.org/html/2504.14945)
- Abstract claims: +6.4 avg over prior RLVR methods, +6.2 OOD — [arXiv abs](https://arxiv.org/abs/2504.14945)

**HPT: "Towards a Unified View of LLM Post-Training" (Lv et al., 2025, arXiv 2509.04419)**
- Gate: per question, sample n=8 rollouts and score them with a binary verifier, giving mean P. If P > γ, use the RL loss. Otherwise use the SFT loss on a demo. γ=0 for Qwen (meaning SFT only when all 8 fail) and 2/8 for Llama. 1.5B ablation: γ=0 gives 41.9, γ=1/8 gives 38.7, γ=2/8 gives 39.0 — [arXiv HTML v2](https://arxiv.org/html/2509.04419v2)
- Qwen2.5-Math-7B, ID/OOD avg: SFT 44.5/46.4; GRPO 43.1/58.8; SFT->GRPO 46.5/49.3; LUFFY 49.8/60.1; SRFT 44.2/58.7; **HPT 52.7/62.3**. On AIME24, HPT is +6.9 over LUFFY — [arXiv HTML v2](https://arxiv.org/html/2509.04419v2)
- Qwen2.5-Math-1.5B: SFT 36.9, GRPO 34.0, LUFFY 34.7, HPT 41.9. Llama-3.1-8B: SFT 5.9, GRPO 9.6, LUFFY 13.2, HPT 18.2 — [arXiv HTML v2](https://arxiv.org/html/2509.04419v2)
- Methods that include SFT get higher pass@k at large k (up to 1024) than pure GRPO, and HPT is highest — [arXiv HTML v2](https://arxiv.org/html/2509.04419v2)
- Framework: a "Unified Policy Gradient Estimator" in which SFT and RL differ only in four parts: stabilization mask, reference-policy denominator, advantage, and likelihood gradient — [arXiv abs](https://arxiv.org/abs/2509.04419)

**CHORD (Zhang et al., Alibaba, 2025, arXiv 2508.11408)**
- Loss L = (1−μ)·L_GRPO + μ·L_SFT. CHORD-μ decays μ from 0.9 to 0.05 over 200 steps. CHORD-φ fixes μ=0.1 and weights each SFT token by φ(p)=p(1−p), so tokens the model already knows (p near 1) and tokens far off-distribution (p near 0) are both downweighted — [arXiv HTML v3](https://arxiv.org/html/2508.11408v3)
- Qwen2.5-7B-Instruct with R1 traces (5k SFT + 20k RL, non-overlapping). AMC / AIME24 / AIME25 / MMLU-Pro: original 43.8/11.7/6.7/24.7; SFT-best 55.9/15.8/15.2/38.4; GRPO 52.1/13.2/8.5/45.8; SFT->RL 58.4/17.1/16.3/51.3; LUFFY 52.8/16.6/14.3/44.0; CHORD-μ 60.8/18.1/17.9/43.3; **CHORD-φ 62.5/18.2/17.2/56.2** — [arXiv HTML v3](https://arxiv.org/html/2508.11408v3)
- No IFEval. SFT-light dropped tool-use BFCL from 46.2 to 34.0 (LLaMA3.2-3B-Instruct), which is evidence that SFT on narrow expert data hurts other skills — [arXiv HTML v3](https://arxiv.org/html/2508.11408v3)
- Code: modelscope Trinity-RFT — [arXiv abs](https://arxiv.org/abs/2508.11408)

**SRFT (Fu et al., 2025, arXiv 2506.19767)**: single-stage SFT+RL with entropy-aware weighting. 59.1% avg on 5 math benchmarks, +9.0 over zero-RL, +10.9 on 3 OOD benchmarks — [arXiv abs](https://arxiv.org/abs/2506.19767). In HPT's re-run on Qwen2.5-Math-7B it scored only 44.2 ID (below SFT->GRPO 46.5) — [HPT](https://arxiv.org/html/2509.04419v2). The two papers' numbers conflict.

**ReLIFT (Ma et al., ICLR 2026, arXiv 2506.07527)**: mostly RL. Collects high-quality solutions for the hardest questions (ones RL fails on) and interleaves SFT steps on them. +5.2 avg over zero-RL models on 5 competition benchmarks + 1 OOD, using only 13% of the demo data, and beats both RL and SFT — [arXiv abs](https://arxiv.org/abs/2506.07527)

**BREAD (Zhang et al., NeurIPS 2025, arXiv 2506.17211)**: GRPO variant aimed at *small* LMs. When self-generated traces fail, it inserts short expert prefixes ("anchors") and branches rollouts from them. Uses fewer than 40% of ground-truth traces, beats GRPO, and trains about 3x faster — [arXiv](https://arxiv.org/html/2506.17211v1), [NeurIPS](https://neurips.cc/virtual/2025/poster/118358)

**UFT (Liu, Farina, Ozdaglar, NeurIPS 2025, arXiv 2505.16984)**: RFT rollouts start from a hint (a solution prefix) whose length is cosine-annealed toward 0, plus a log-likelihood term on the hint tokens. Claims to beat SFT and RFT at all model sizes — [arXiv PDF](https://arxiv.org/pdf/2505.16984), [NeurIPS](https://neurips.cc/virtual/2025/poster/115460)

**Prefix-RFT (Huang et al., 2025, arXiv 2507.01679)**: samples continuations from demo prefixes. Beats standalone SFT, RFT and parallel mixed-policy RFT on math, and is robust to demo quality and quantity — [arXiv abs](https://arxiv.org/abs/2507.01679)

### Inferences
- The shared mechanism is to inject demo signal only where on-policy RL has zero gradient: groups where all G rollouts fail (zero variance, so centered advantage = 0) or where reward is sparse. HPT (γ=0), ReLIFT, and BREAD all do this. For us, that means "SFT on our verified teacher answer for any prompt where 0/8 v4 rollouts pass." It has a clean Tinker mapping: a cross_entropy datum for that prompt, mixed into the same step as the IS datums.
- A fixed, large SFT term during RL (LUFFY's "RL w/ SFT loss", 40.1 vs 50.1) can be worse than either SFT->RL or pure RL. CHORD shows a *small* μ (0.05-0.1) with p(1−p) token weighting works. If we add any always-on SFT term, keep μ at or below 0.1 and weight it by p(1−p).
- LUFFY-style in-group off-policy traces need a behavior logprob for the demo. We don't have GPT-teacher logprobs, and LUFFY effectively treats π_off as about 1 and relies on custom shaping f(x)=x/(x+0.1). That needs a custom loss rather than Tinker's built-in importance_sampling. HPT/ReLIFT-style gated SFT needs no custom loss.
- Expected gain for us is smaller than reported: v4 already absorbed the teacher pool through SFT, so further SFT on the same answers mostly re-teaches known material. The marginal value is concentrated on the hard tail, where v4 gets 0/8.
- Pass@k evidence (HPT) suggests the demo-mixed variants keep diversity better than pure GRPO. That matters more for best-of-n than for our greedy grading.

### Gaps
- No hybrid off-policy-guidance paper I found reports IFEval, MBPP/HumanEval, or LoRA results. All are math (plus ARC/GPQA/MMLU-Pro OOD; CHORD adds BFCL tool use).
- None starts from a strong SFT checkpoint of the same family and data, which is our case. CHORD's SFT->RL baseline (58.4 AMC) is the closest analogue, and CHORD-φ beat it by about 4 AMC, +5 MMLU-Pro.
- Prefix-RFT, UFT and BREAD numeric tables were not retrieved. SRFT numbers conflict between its own paper and HPT's re-run.
- Whether Tinker supports a fully custom per-token loss function (needed for LUFFY shaping), beyond per-token advantages/weights, was not verified.

## 2. On-policy distillation (OPD) vs RLVR at small scale; mixing a teacher reverse-KL term with verifiable rewards

### Takeaway
OPD (the student samples, the teacher scores each token, and the loss is per-token reverse KL used as the advantage in the IS loss) is the strongest per-dollar method in the literature for 8B students. Qwen3-8B reached AIME'24 74.4 with OPD vs 67.6 with RL at about 1/10 the GPU hours. It also restored IFEval from 45% to 83% after forgetting. Thinking Machines' Tinker cookbook now uses Qwen3.5-9B-Base as the student, which is our exact base. KDRL shows that adding a reverse-KL teacher term on top of a GRPO reward beats either one alone (+0.7 to +1.1 avg at 1.5B). This depends entirely on approval to use the 397B as teacher.

### Cited Findings
**Thinking Machines, "On-Policy Distillation" (Lu et al., Oct 2025)** — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Loss: per-token reverse KL(π_student ‖ π_teacher) on student samples, discount 0, advantage = −(per-token reverse KL), fed into Tinker's importance_sampling loss. The teacher is called via `compute_logprobs` (prefill only). It is a "one-line change" from the Tinker RL script (swap the KL-regularizer reference model for the teacher). Recipe: `tinker-cookbook/recipes/distillation`. Default batch 64 prompts x 4 samples.
- Qwen3 report numbers quoted there (Qwen3-8B-Base student, Qwen3-32B teacher): off-policy distillation (SFT) AIME'24 55.0 / GPQA-D 55.6; +RL 67.6 / 61.3 at **17,920 GPU-h**; +OPD **74.4 / 63.3 at 1,800 GPU-h**.
- Own replication: SFT on 400K OpenThoughts-3 gives 60% AIME'24. Extrapolated SFT to 2M prompts would give about 70%. OPD from the 400K checkpoint reaches 70% in about 150 steps (~77K prompts x 4 samples). RL reaches 68%. Cost efficiency vs SFT-2M: 9x (SFT data amortized), about 18x in GPU-h, about 30x including teacher sampling.
- **LoRA:** at rank 32, LoRA trails full FT by 13% after SFT but only 6% after OPD. LoRA lags on high-volume SFT.
- **IFEval forgetting recovery** (Qwen3-8B, internal-docs midtraining): base 85% IFEval; 100% docs gives 45%; 70/30 docs/chat gives 79%; 70/30 + OPD on Tulu3 prompts (teacher = the earlier Qwen3-8B) gives **83%**, while knowledge QA stays at 41% (vs 36%).
- SFT on the model's *own* samples (Qwen3-32B, T=1) degrades IFEval at any practical LR > 0. This warns against naive self-SFT (RFT) for IF.
- Self-distillation from an RL-trained LoRA (rank 128) teacher recovers AIME in under 10 steps vs 70 for RL. They estimate 50-100x compute efficiency.
- RL gives about O(1) bits per episode; OPD gives about O(N) bits (N = tokens).
- Update (June 2026): the Qwen3 models were retired from Tinker; the cookbook now uses **Qwen3.5-9B-Base (student) and Qwen3.5-9B (teacher)**.

**KDRL (Xu et al., 2025, arXiv 2506.02208)** — [arXiv HTML](https://arxiv.org/html/2506.02208v1)
- J = J_GRPO − β·KL_k2(π_θ ‖ π_T), with token reward log(π_T/π_θ) and the k2 estimator ½R². Default β=2e-3 fixed, or annealed from 5e-3 to 1e-3. Reward shaping (adding the KL into the scalar reward) *collapsed early*. Using it as a separate loss term worked. Optional masking applies the KD term only to responses with r=0 (response-level) or groups where all fail (group-level). This keeps performance and shortens outputs.
- Student DeepScaleR-1.5B, teacher Skywork-OR1-Math-7B. 6-benchmark avg: SFT 51.5, GRPO 54.6, KD-RKL (pure OPD) 56.1, KDRL 56.8, KDRL-annealing 57.2. 280 KDRL steps cost about the same as 200 KD-RKL steps or 460 GRPO steps (about 80h each).
- Abstract-level claims: +5.7 over SFT, +2.6 over GRPO — [arXiv abs](https://arxiv.org/abs/2506.02208) (secondary sources quote +4.7 vs SFT. Note the conflict.)

**GKD (Agarwal et al., ICLR 2024)**: the original on-policy distillation formulation (student-generated sequences, generalized JSD/reverse KL). I did not re-fetch it this session; see the TM blog for the lineage.

### Inferences
- For us, OPD adds a teacher prefill on every sampled sequence. Our RLVR plan samples 200x64x8 ≈ 102K sequences. Scoring all of them with the 397B roughly adds a large-model prefill per token on top of 9B sampling+training. Prefill on 397B-A? MoE should cost a few times the 9B train cost per token, not 40x, if it is MoE-priced by active params. Check the Tinker price sheet in the ledger. A cheaper variant is KDRL's **group-level masking**: apply the teacher term only to all-fail groups, which cuts teacher calls to the hard tail (likely 10-30% of groups).
- OPD needs no reward, so it covers IF constraints our checkers can't verify, and it directly targets the IFEval-type forgetting that RL or extra SFT can cause. The TM IFEval 45→83 recovery is the most IF-relevant evidence in this whole area.
- In our IS-REINFORCE loss, the change is advantage_t = A_group + β·(log π_T(x_t) − log π_θ(x_t)), with β about 1e-3 to 5e-3 per KDRL. The teacher term needs the 397B and the 9B to share a tokenizer. Both are Qwen3.5, which presumably share one (needs verification).
- If the teacher is not approved, the fallback is a teacher "frozen v4 or best RL checkpoint" for self-distillation. That only regularizes and does not add new capability.

### Gaps
- No public OPD result on IFEval/MBPP *gains* (as opposed to forgetting recovery) at 9B, and no Qwen3.5-397B-teacher result.
- Tinker cost of `compute_logprobs` on the 397B per token was not verified in this research. It must come from the project budget ledger or the Tinker pricing page.
- Qwen3 tech report distillation details were taken via the TM blog rather than fetched from the report directly.

## 3. Rejection-sampling FT / RAFT / ReST / expert iteration as a cheap RL substitute

### Takeaway
On math, RAFT++ (SFT on verified-correct own samples, with IS/clipping) matches GRPO almost exactly at 3B-7B (56.1 vs 56.3 avg) and learns *faster* early. It plateaus around iteration 100 from entropy collapse, and GRPO overtakes it later. With about 200 steps we are near that crossover, so RAFT is a viable but not superior substitute. For IF there is direct evidence that SFT on own samples can erode IFEval.

### Cited Findings
- Xiong et al. 2025, "A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce" (arXiv 2504.11343). Numina-Math prompts (~860k). Avg@16 on MATH500/Minerva/OlympiadBench: Qwen2.5-Math-7B base 23.6, **RAFT++ 56.1**, GRPO 56.3, Reinforce-Rej 56.4, PPO 52.5. LLaMA-3.2-3B-Instruct: 13.1, RAFT++ 27.6, GRPO 28.4, Reinforce-Rej 28.5, PPO 26.9 — [arXiv HTML](https://arxiv.org/html/2504.11343)
- RAFT++ learns faster early but slows around iteration 100 and is overtaken by GRPO. The cause is faster entropy collapse from positive-only training, and its KL from init also rises faster early — [arXiv HTML](https://arxiv.org/html/2504.11343)
- GRPO's main benefit comes from discarding prompts whose groups are all-correct or all-wrong, not from reward normalization. Reinforce-Rej (filter both) is recommended — [arXiv abs](https://arxiv.org/abs/2504.11343)
- SFT on own samples degrades IFEval at any practical LR — [TM blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

### Inferences
- Our planned clip-free IS-REINFORCE with centered group advantages already gets the main Reinforce-Rej benefit, since zero-variance groups give zero advantage. Replacing it with RAFT would save little compute (sampling dominates in both). RAFT only saves the negative-sample backward pass, at the cost of faster entropy collapse.
- RAFT's real use for us is offline, as a cheap pre-RL step: sample v4 at G=8 once, keep verified-correct answers, and SFT. That is essentially another SFT round, and our v4 SFT pool already covers it with stronger teacher answers. Low expected marginal value.

### Gaps
- No 2025 head-to-head RAFT vs GRPO on IFEval or code at 7-9B was found. No LoRA RAFT comparisons were found.
- ReST-EM / expert-iteration papers (Singh et al. 2023) were not re-fetched.

## 4. SFT/NLL auxiliary loss during RL to prevent forgetting

### Takeaway
A naive always-on SFT/NLL term during RL hurt badly in LUFFY: 40.1 vs 45.5 for pure RL ID, 46.0 vs 57.3 OOD, and it was the costliest variant because of long imitative generations. A small decaying or p(1−p)-weighted term (CHORD, μ ≤ 0.1) helped. On-policy RL itself forgets less than SFT (RL's Razor), so the forgetting problem an NLL term solves is smaller in RLVR than in SFT. A teacher reverse-KL or OPD term is the better-evidenced anti-forgetting regularizer.

### Cited Findings
- LUFFY "RL w/ SFT Loss": 40.1 ID / 46.0 OOD vs on-policy RL 45.5/57.3 and LUFFY 50.1/57.8. Cost 133x8 GPU-h vs 77x8 — [LUFFY HTML](https://arxiv.org/html/2504.14945)
- CHORD-φ (μ=0.1, φ=p(1−p)) is the best in its table, including MMLU-Pro 56.2 vs SFT->RL 51.3 and GRPO 45.8 — [CHORD HTML](https://arxiv.org/html/2508.11408v3)
- RL's Razor (Shenfeld, Pari, Agrawal, MIT, NeurIPS 2025 workshop, arXiv 2509.04259): at matched new-task accuracy, RL forgets less than SFT. Forgetting tracks forward KL from the base on the new task, and on-policy updates find KL-minimal solutions. "Oracle SFT" on a KL-minimal distribution forgets even less — [project page](https://jyopari.github.io/posts/rl_razor), [arXiv](https://arxiv.org/html/2509.04259v1)
- Mixing 30% chat data into SFT partially preserved IFEval (79 vs 45), but no mixing ratio fully preserved it. OPD restored it to 83 — [TM blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

### Inferences
- For our ~200-step RLVR from v4, IFEval forgetting most likely comes from optimizing math/code reward on non-IF prompts. The cheapest guard is keeping IF prompts in the RL mix (already planned) plus checkpoint selection. If an aux term is used, prefer (a) gated SFT on all-fail groups (HPT) or (b) μ ≤ 0.1 with p(1−p) token weighting on v4-pool IF data, over plain NLL.
- With LoRA r=32 and a small LR, total KL drift over 200 steps is limited. That further reduces the need for a replay term.

### Gaps
- I did not retrieve quantified Tulu 3 / Llama 3 "ptx"-style NLL-during-RL ablations for this session, so the effect size of PPO-ptx in modern RLVR is unsourced here.
- No LoRA-specific forgetting comparison (aux-NLL vs none) was found.

## 5. Recommendation: top 2 options for our setup, with cost

### Takeaway
(1) **RLVR + HPT-style gated SFT on all-fail groups** (optionally CHORD-φ weighting): near-zero extra cost, no teacher approval needed, fits Tinker's existing losses. (2) **If the 397B teacher is approved: KDRL-style RLVR + reverse-KL teacher term** (group-masked to save cost), or a short pure-OPD phase on IF prompts. This is the best-evidenced route for IF and for LoRA, but adds teacher prefill cost. RAFT and always-on NLL are not recommended.

### Cited Findings
- HPT γ=0 gate is the best variant at 1.5B (41.9 vs 38.7/39.0) and beats LUFFY/SRFT/SFT->GRPO at 7B (52.7 vs 49.8/44.2/46.5) — [HPT](https://arxiv.org/html/2509.04419v2)
- ReLIFT uses 13% of demos and BREAD uses fewer than 40% of traces with about 3x speedup. Demo injection on hard prompts is data-cheap — [ReLIFT](https://arxiv.org/abs/2506.07527), [BREAD](https://neurips.cc/virtual/2025/poster/118358)
- OPD: LoRA r32 gap shrinks from 13% to 6% vs full FT. 1/10 the GPU-h of RL for higher AIME. IFEval recovered from 45 to 83. Cookbook student is Qwen3.5-9B-Base — [TM blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- KDRL > KD-RKL > GRPO (57.2 / 56.1 / 54.6). Group-level masking keeps accuracy — [KDRL](https://arxiv.org/html/2506.02208v1)

### Inferences (estimated, relative to the planned RLVR run cost C)
- **Option 1: RLVR + gated SFT (HPT γ=0 / ReLIFT-style).** After sampling each group, if 0/8 pass and a verified teacher answer exists in the SFT pool, emit a cross_entropy datum for (prompt, teacher answer) in the same forward_backward step, optionally scaled by μ about 0.1-0.3 relative to the IS datums. Optionally apply per-token weights p(1−p) using v4 logprobs from the forward pass (CHORD-φ). Extra cost: one extra SFT sequence per all-fail group, at most 64 sequences/step vs 512 sampled, so about **+3-10% of C**, with no extra sampling. Expected effect: recovers signal on the hard tail (MATH level 5, hard code). Literature magnitude on math is +2 to +9 avg from weak starts. For us I would guess +0.5 to +2 on MATH-500/MBPP. That is a guess with no direct evidence at our starting point. IF effect is likely neutral. Risk: re-imitating teacher style lengthens outputs (LUFFY's observation), so monitor response length.
- **Option 2: RLVR + 397B reverse-KL term (KDRL), or pure OPD on IF prompts.** Advantage_t = A_group + β(log π_T − log π_θ), β about 2e-3 (annealed from 5e-3). Use group-level masking (teacher term only on all-fail groups) to cap 397B calls at the hard-tail fraction. Or run about 50-100 steps of pure OPD (64x4) on IF prompts after RLVR to fix IF regressions. Extra cost: 397B `compute_logprobs` (prefill only) on the masked fraction (about 10-30%) of sampled tokens. Without masking, on all tokens, this could plausibly be +50-150% of C, depending on Tinker's per-token price for the 397B. **This must be priced from the ledger before committing.** Blocked on staff approval of the teacher.
- Not recommended: RAFT/expert iteration as the main RL. It matches GRPO only up to about 100 iterations, then collapses entropy, and self-SFT is known to hurt IFEval. Also not recommended: always-on plain NLL aux loss (LUFFY: −5 ID, −11 OOD vs pure RL). LUFFY in-group off-policy traces are also out: they need custom shaping loss and teacher behavior logprobs we don't have for GPT answers.
- This fits the project's "one best guess + one improvement round" preference: run Option 1 as the RL1 best guess (it is a small code delta to the planned pipeline). If the teacher is approved and budget remains, use Option 2 (group-masked KDRL term or a short OPD-on-IF phase) as the single improvement round.

### Gaps
- Tinker per-token prices for 9B training/sampling and 397B prefill were not checked here, so dollar figures are relative only.
- No evidence at our exact starting point (strong SFT model, LoRA, mixed IF/math/code, greedy eval). All effect sizes are extrapolated from math-only, mostly full-FT, zero-RL studies.

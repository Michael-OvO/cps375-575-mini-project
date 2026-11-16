# Lab post-training reports (2025–2026) and on-policy distillation for a 9B LoRA student on a tiny Tinker budget

Scope note: researched 2026-11-16. Sources marked **[fetched]** were read in this session. Sources marked **[recall]** are cited from model knowledge (arXiv IDs and dates are believed correct) but were not re-opened in this session. Treat their exact numbers as needing a check before they go into the report. Local code facts come from the installed `tinker_cookbook` at `/Users/michael/.cache/uv/archive-v0/2tuLD5brMjxkId2p/tinker_cookbook/` and the project spec `/Users/michael/Documents/Github/cps375-575-mini-project/docs/superpowers/specs/2026-11-16-rl-design.md`.

## Q1. Thinking Machines: On-Policy Distillation, LoRA Without Regret, Defeating Nondeterminism, and the Tinker distillation recipe

### Takeaway
OPD samples from the student and has the teacher score every sampled token. The per-token advantage is the negative reverse KL, −(log p_student − log p_teacher), with discount 0. The loss is the same importance-sampling loss used for RL. Thinking Machines reports it matching or beating RL at about 1/10 of the compute, and it recovered IFEval after mid-training damage (79% → 83%) in about 100 steps. In the cookbook, the teacher scores **the student's exact token sequence**, so a teacher with a different chat template needs custom re-rendering.

### Cited Findings
- **On-Policy Distillation blog** (Kevin Lu et al., Thinking Machines, 2025-10-27) **[fetched]** — [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
  - Method: per-token reverse KL KL(π_θ‖π_teacher). The advantage is −reverse KL, plugged into the importance-sampling RL loss. Discount factor 0, because "higher discount factors did not help in practice". The teacher needs one forward pass (`compute_logprobs`) per student sample. No reward model is needed. The post calls it about a one-line change to RL code that already has a KL regularizer.
  - Math result: Qwen3-8B-Base was SFT'd on OpenThoughts-3 (400k prompts) to 60% on AIME'24. OPD then reached about 70% in about 150 steps (about 77k prompts × 4 samples). Extrapolated SFT would need about 2M prompts for 70%. RL reached 68%.
  - Compute: OPD used 8.4e19 teacher + 8.2e19 student FLOPs, versus 3.4e21 + 1.5e21 for SFT-2M. That is 9× cheaper if the SFT data already exists, about 30× if teacher sampling for SFT is counted, and about 18× in GPU-hours.
  - Quoted from the Qwen3 report (Qwen3-8B, starting from an off-policy-distilled checkpoint): off-policy distillation 55.0 AIME'24 / 55.6 GPQA-D; +RL 67.6 / 61.3 for 17,920 GPU-h; +OPD 74.4 / 63.3 for 1,800 GPU-h.
  - LoRA rank 32 trailed full fine-tuning by 13 points after SFT but only by 6 points after OPD. OPD closes part of the LoRA gap.
  - Personalization: Qwen3-8B baseline was 18% internal QA / 85% IFEval. Mid-training on 70% docs + 30% chat gave 36% / 79%. Adding OPD with the *original Qwen3-8B as teacher* on Tulu3 prompts gave 41% / 83%. IF was mostly recovered and knowledge was not lost.
  - Self-distillation (teacher = the same base after RL on DeepMath): distillation matched the teacher in 7–10× fewer gradient steps. AIME recovered in under 10 steps versus 70 for RL. The authors estimate a 50–100× compute saving, partly because distillation works at shorter context lengths.
  - Default batch: 64 prompts × 4 samples = 256 samples per step. When the teacher's behavior lies outside the student's support, "a significantly larger batch size is required." A strong SFT initialization allows small batches.
  - Training on the model's own temperature-1 samples degraded IFEval at every practical learning rate. OPD with a fixed teacher avoided that drift.
  - The post does not discuss tokenizer or chat-template requirements.
  - Update (June 2026): Qwen3-32B and Qwen3-8B-Base were retired from Tinker. **The cookbook now uses student Qwen3.5-9B-Base and teacher Qwen3.5-9B**, which is exactly our base model.
- **Tinker cookbook distillation recipe** **[fetched]** — [tinker-docs: recipes/distillation](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
  - Modules: `tinker_cookbook.recipes.distillation.on_policy_distillation` (`dataset=deepmath` or `tulu3`), plus `on_policy_multi_teacher` and a Harbor multi-turn variant.
  - Reasoning OPD settings: LoRA rank 128, lr 1e-4 (LoRA) or 5e-5 (full FT), `groups_per_batch=512`, max_tokens 16,384, 200 steps. Result: AIME'24 about 76.7%, versus about 65% after 3,000 SFT steps.
  - Personalization OPD: `groups_per_batch=64`, lr 1e-4. IFEval recovered within about 100 steps.
  - Single-turn OPD uses an environment with no correctness reward; the KL is the only signal.
- **`train_on_policy.Config`** **[fetched]** — [tinker-docs: train_on_policy_config](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/distillation/train_on_policy_config/)
  - Defaults: `lora_rank=32`, `kl_penalty_coef=1.0`, `kl_discount_factor=0.0`, `temperature=1.0`, `loss_fn='importance_sampling'`. A teacher is set per dataset config (`base_model`, optional `load_checkpoint_path`).
- **Local cookbook source** (`distillation/train_on_policy.py`, lines 55–125) **[read locally]**
  - `teacher_client.compute_logprobs(sequence_input)` runs on the **same `full_sequence_inputs` the student produced**. The code comment says: "if your teacher has a different renderer than the student, you may want to modify the full_sequence_inputs_D to match the teacher's renderer."
  - The advantage is `existing_advantages + (−kl_penalty_coef · mask · (logp_student − logp_teacher))`. The KL term is **added to the environment advantage**, and rollouts use `do_remove_constant_reward_groups=False`. A real reward environment therefore turns this trainer into RLVR + OPD with no new code.
  - The RL trainer (`rl/train.py`) also has `kl_penalty_coef` and `KLReferenceConfig(base_model, load_checkpoint_path)`. That penalty is centered (`β·(mean Δ − Δ_t)`, `rl/metrics.py` line 218), so it acts as a regularizer toward any reference, including a teacher.
- **Tinker prices** **[fetched]** — [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)
  - Qwen3.5-397B-A17B (64K): prefill $1.65/M, cached prefill $0.33/M, sample $7.50/M, train $6.60/M.
  - Qwen3.5-9B (hybrid) and 9B-Base: prefill $0.30, sample $0.807, train $1.463.
  - Qwen3.6-35B-A3B: prefill $0.45, sample $1.335.
  - Qwen3.6-27B (retiring): prefill $1.86.
  - The 397B is "Hybrid + Vision", meaning it reasons by default and thinking is optional.
- **LoRA Without Regret** (John Schulman et al., Thinking Machines, 2025-09-29) **[recall]** — [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)
  - In RL, even rank-1 LoRA matched full fine-tuning, because policy-gradient RL absorbs about 1 bit per episode.
  - The optimal LoRA learning rate is about 10× the full-FT rate.
  - LoRA should be applied to all layers, including the MLP, not only attention.
  - LoRA degrades at large batch sizes and on high-volume SFT.
- **Defeating Nondeterminism in LLM Inference** (Horace He et al., 2025-09-10) **[recall]** — [thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  - Temperature-0 outputs vary because kernels are not batch-invariant: Qwen3-235B gave 80 distinct completions out of 1,000 identical requests.
  - With batch-invariant kernels, sampler and trainer logprobs match, so RL is truly on-policy (KL 0).
  - Without importance correction, the sampler/trainer mismatch hurt RL.

### Inferences
- Reverse KL is mode-seeking, which suits our **greedy** eval: OPD sharpens the student's top choice toward the teacher's mode.
- The cookbook's own OPD default is the pairing we would use (Qwen3.5-9B-Base student). The only change would be a teacher of 397B instead of 9B.
- Because KL advantages add onto environment advantages, a hybrid is roughly a config change. The only real engineering is **teacher re-rendering** (see Q4).
- Greedy-eval nondeterminism on Tinker means 1-standard-error checkpoint comparisons should account for sampler noise. That is not specific to OPD.

### Gaps
- Whether Tinker's prefix cache applies to `compute_logprobs` (a cached prompt costs $0.33/M instead of $1.65/M) is not documented on the pages read.
- The recipe page does not state whether teacher and student must share a renderer; the local code shows they implicitly do.

## Q2. Strong-to-weak distillation and RL in frontier and open lab reports

### Takeaway
Every major lab that ships small models (Qwen3, DeepSeek-R1 distills, Llama-Nemotron Nano, MiMo-V2-Flash, DeepSeek-V3.2) trains them mainly by distilling from a stronger model. RL on small models is used mostly as a cheaper add-on or as RL from scratch with very large problem sets (MiMo-7B, OLMo 3). The 2025–2026 trend is **dense teacher signal plus sparse outcome reward together** (MiMo-V2-Flash MOPD; the Sparse-to-Dense principle).

### Cited Findings
- **Qwen3 Technical Report**, arXiv [2505.09388](https://arxiv.org/abs/2505.09388) (2025-05-14) **[partially fetched via search summary and the TM blog]**
  - Small models (0.6B–14B, 30B-A3B) use "strong-to-weak distillation": off-policy distillation from teacher outputs first, then on-policy distillation that aligns student logits with the teacher.
  - Versus direct RL from the same checkpoint: better scores at about 1/10 the GPU hours (Qwen3-8B AIME'24 74.4 vs 67.6; 1,800 vs 17,920 GPU-h).
  - Distillation from teacher logits widened exploration, measured by higher pass@64 on AIME. RL did not raise pass@64. — [Qwen3 report](https://arxiv.org/pdf/2505.09388)
- **Qwen3.5**: no technical report found. Only a Feb-2026 blog titled "Towards Native Multimodal Agents" is referenced second-hand, with no post-training details. — [search result referencing it](https://friendli.ai/models/lzy510016411/qwen3.5-9b-distill-agent-instruct). Treat Qwen3.5's post-training recipe as unknown.
- **DeepSeek-R1**, arXiv [2501.12948](https://arxiv.org/abs/2501.12948) (2025-01-22) **[recall]**
  - Large-scale RL on Qwen2.5-32B-Base (R1-Zero-Qwen-32B) reached about 47.0 AIME'24.
  - SFT distillation from R1 (R1-Distill-Qwen-32B) reached 72.6 AIME'24, with MATH-500 94.3 vs about 91.6.
  - The authors conclude that distilling into small models beats RL on them, which "may not even achieve the performance of distillation".
- **DeepSeek-V3.2**, arXiv [2512.02556](https://arxiv.org/abs/2512.02556) (2025-12-02) **[abstract fetched; RL details recall]**
  - Specialist distillation: per-domain RL specialists distilled into one model.
  - GRPO changes: an unbiased KL estimate (IS-corrected k3), off-policy sequence masking (drop negative-advantage sequences whose sampler/policy divergence is high), "Keep Routing" for MoE, and "Keep Sampling Mask" (reuse the top-p mask in training).
  - Post-training compute reported above 10% of pretraining.
- **MiMo-V2-Flash Technical Report** (Xiaomi), arXiv [2601.02780](https://arxiv.org/abs/2601.02780) (2026-01) **[fetched]**
  - **MOPD**: domain teachers (RL, SFT, or the student itself) give a token-level signal. Advantage A_t = sg[log π_teacher/π_θ] + α·A_ORM, where A_ORM is a GRPO outcome advantage; α is not stated.
  - The truncated IS weight is zeroed outside [ε_low, ε_high], which drops tokens with high train/inference mismatch.
  - Result: the student matched or beat the best teacher on most benchmarks. AIME'25 went 89.3 → 94.1 (best teacher 93.9) and LiveCodeBench 77.5 → 83.2 (teacher 82.6). There were regressions on Arena-Hard Creative (−3.9) and BrowseComp (−6.3 vs teacher). — [MiMo-V2-Flash HTML](https://arxiv.org/html/2601.02780v1)
  - Generalized as a standalone paper: [MOPD, arXiv 2606.30406](https://arxiv.org/abs/2606.30406) (2026-06), with small regressions on IFBench and SWE-Bench Verified. **[search snippet only]**
- **MiMo-7B**, arXiv [2505.07608](https://arxiv.org/abs/2505.07608) (2025-05-12) **[abstract fetched; details recall]**
  - RL used 130k verifiable math and code problems.
  - Test-difficulty-driven code reward: partial credit by test-difficulty tier, which fixes sparse code reward. [fetched]
  - Strategic data resampling for stability [fetched]. Per recall: about 10% of samples from an "easy" pool, and dropping problems the model always solves.
  - A Seamless Rollout Engine (recall: about 2.3× faster training).
- **Kimi k1.5**, arXiv [2501.12599](https://arxiv.org/abs/2501.12599) (2025-01-22) **[recall]**
  - Length penalty λ = 0.5 − (len − min_len)/(max_len − min_len) within a group, applied to correct answers (and min(0, λ) to wrong ones), warmed up gradually.
  - Curriculum plus prioritized sampling ∝ (1 − success rate).
  - Partial rollouts for long contexts.
  - long2short: model merging, shortest-rejection sampling, DPO, and long2short RL.
- **Kimi K2**, arXiv [2507.20534](https://arxiv.org/abs/2507.20534) (2025-07) **[recall]**
  - Per-task token budgets, with responses over the budget truncated and penalized.
  - A PTX (pretraining/SFT) auxiliary loss against forgetting.
  - Temperature decay during RL (high early for exploration, lower later).
- **MiniMax-M1**, arXiv [2506.13585](https://arxiv.org/abs/2506.13585) (2025-06-16) **[recall]**
  - CISPO clips the importance-sampling *weights* (with stop-gradient) instead of dropping clipped tokens' gradients, so rare "fork" tokens keep a gradient.
  - Reported about 2× faster than DAPO on Qwen2.5-32B.
  - The full RL run cost about $535k (512 H800 for 3 weeks).
- **Phi-4-reasoning**, arXiv [2504.21318](https://arxiv.org/abs/2504.21318) (2025-04-30) **[recall]**
  - SFT on o3-mini traces, then a short GRPO stage (Phi-4-reasoning-plus): about 6k math problems, on the order of 90 steps, a length-aware reward, and roughly +10 AIME points. The cost was longer outputs.
- **Llama-Nemotron**, arXiv [2505.00949](https://arxiv.org/abs/2505.00949) (2025-05) **[recall]**
  - Nano/Super are trained mainly by SFT distillation with a "detailed thinking on/off" toggle. Large-scale reasoning RL was applied to Ultra (253B), not Nano. Small models were followed by short RL/RPO for IF and chat.
- **OLMo 3**, arXiv [2512.13961](https://arxiv.org/abs/2512.13961) (2025-12) **[secondary summaries]**
  - Think pipeline: SFT → DPO with **Delta Learning** pairs (chosen from Qwen3-32B, rejected from Qwen3-0.6B) → RLVR (OlmoRL/GRPO with truncated IS, clip-higher, zero-gradient filtering, token-level loss).
  - An RL-Zero variant is also released. — [Interconnects](https://interconnects.ai/p/olmo-3-americas-truly-open-reasoning); [emergentmind topic](https://www.emergentmind.com/topics/olmo-3)
  - Delta Learning paper: arXiv [2507.06187](https://arxiv.org/abs/2507.06187).
- **Tulu 3**, arXiv [2411.15124](https://arxiv.org/abs/2411.15124) (Nov 2024, just before the window; it is the RLVR origin paper) **[recall]**
  - RLVR uses binary verifiable rewards (GSM8K/MATH answer match, IFEval-style constraint checkers) with PPO, starting from the DPO model.
  - Gains were a few points on GSM8K, MATH and IFEval, largest on IFEval-type constraints.
- **Gemma 3**, arXiv [2503.19786](https://arxiv.org/abs/2503.19786) (2025-03) **[recall]**
  - Post-training uses improved knowledge distillation from a large instruct teacher, then RL with BOND/WARM/WARP-style reward models plus verifiable math and code rewards. Small sizes get the same distillation-first pipeline.
- **2026 OPD academic follow-ups** **[fetched]**
  - *Sparse-to-Dense Reward Principle*, arXiv [2605.12483](https://arxiv.org/abs/2605.12483) (2026-05-12, v5 2026-10-12)
    - Proposed pipeline: teacher RL → forward-KL warm-up → OPD → optional student sparse RL.
    - Qwen3-1.7B student with an RL-improved 8B teacher vs direct GRPO: MATH 79.3 vs 75.9, AIME'24 25.2 vs 19.8.
    - Stage gains on MATH: RL-improved teacher +7.8, forward-KL warm-up +1.7, OPD +3.3.
    - Ordering: raw-teacher transfer < direct GRPO < RL-teacher transfer. **A raw (non-RL'd) teacher did worse than GRPO.**
    - Replicated with a Llama-3.3-70B → Llama-3.1-8B-Instruct pair and on code with a Qwen2.5-32B → 7B pair.
  - *Does On-Policy Distillation Really Distill?*, arXiv [2608.31046](https://arxiv.org/abs/2608.31046) (2026-08-31)
    - Teacher supervision on student trajectories is substantially noisy, and the noise grows with teacher scale. Removing the noisy supervision does not change results.
    - A fixed negative advantage on low-probability tokens matched OPD.
    - Their teacher-free OPSA beat OPD by 16.8 Avg@32 on AIME24 (Qwen3-1.7B).
  - *Simple-OPD*, arXiv [2608.06802](https://arxiv.org/abs/2608.06802) (2026-08-07)
    - Before OPD, warm up the student with LoRA on **teacher-compatible** chain-of-thought.
    - Incorrect teacher rollouts help about as much as correct ones; the warm-up transfers the thinking *pattern*.
    - LoRA at near-saturation beats full SFT for the warm-up.
  - Also relevant: *KL for a KL* (control-variate baseline for OPD), arXiv [2605.07865](https://arxiv.org/abs/2605.07865) (2026-05) **[title only]**.

### Inferences
- The lab consensus favors distillation for small models, but in the lab setups the teacher (a) is in the student's own family and post-training style, and (b) has usually had RL itself. Our SFT data came from other teachers (not Qwen3.5-397B), so (a) does not fully hold. Simple-OPD and Sparse-to-Dense both say a teacher-compatible warm-up matters.
- The 2608.31046 result is a warning: part of OPD's measured gain may be generic sharpening (suppressing low-probability tokens) that RLVR or low-temperature effects also give. The marginal value of a 397B teacher over RLVR alone is therefore not guaranteed.

### Gaps
- Exact Qwen3 OPD hyperparameters and Qwen3.5's post-training recipe were not found.
- Exact DeepSeek-V3.2, Kimi and MiniMax numbers above are from recall, not fetched.

## Q3. Concrete techniques transferable to our setup (9B-Base + LoRA r32, ~120 steps, ~$80, greedy eval)

### Takeaway
The cheap, transferable tricks are about data selection and estimator hygiene, not scale:
- Filter prompts to mid-difficulty (pass rate between 0 and 1).
- Drop constant-reward groups (or keep them only when a dense KL signal is present).
- Use token-level loss with no std normalization.
- Give partial credit for code tests.
- Set per-task token caps and penalize truncation.
- Use a modest temperature.
- Use a 10× LoRA learning rate relative to full FT.
- Correct sampler/trainer mismatch with truncated importance sampling.

### Cited Findings
- Difficulty filtering and resampling: Kimi k1.5 prioritized sampling ∝ (1 − success) [recall](https://arxiv.org/abs/2501.12599). MiMo-7B data resampling [fetched](https://arxiv.org/abs/2505.07608). OLMo 3 zero-gradient filtering [summary](https://interconnects.ai/p/olmo-3-americas-truly-open-reasoning).
- Partial credit for code: MiMo-7B test-difficulty-driven reward — [arXiv 2505.07608](https://arxiv.org/abs/2505.07608).
- Length control: Kimi k1.5 length penalty; Kimi K2 per-task budgets with truncation penalty — [2501.12599](https://arxiv.org/abs/2501.12599), [2507.20534](https://arxiv.org/abs/2507.20534).
- Off-policy correction: truncated IS (OLMo 3; MiMo-V2-Flash zeroes IS weights outside [ε_low, ε_high]); CISPO weight clipping (MiniMax-M1) — [2601.02780](https://arxiv.org/html/2601.02780v1), [2506.13585](https://arxiv.org/abs/2506.13585).
- LoRA for RL: low rank is enough; use about 10× the full-FT learning rate; apply to all layers — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).
- Anti-forgetting: OPD against the pre-RL model or a strong teacher fixes IF drift [fetched](https://thinkingmachines.ai/blog/on-policy-distillation/). Kimi K2 uses a PTX loss [recall](https://arxiv.org/abs/2507.20534).
- Preference warm-up before RL: Delta-learning DPO (strong vs weak model responses, no human labels) — [2507.06187](https://arxiv.org/abs/2507.06187).

### Inferences
- The existing rl1 spec already covers most of these: G=8, 64 groups, constant-group removal, per-task caps, no std normalization, prefiltered pools.
- What the spec lacks from this list: code partial credit, a hard truncation penalty (it scores truncated text as-is), and a teacher-dense signal.
- **Interaction to watch:** with a teacher KL term, constant-reward groups still carry signal. Removing them throws away OPD signal, and keeping them raises training cost by about 30%.

### Gaps
- No lab report gives a recipe for "100 steps of RL on a LoRA-SFT 9B" specifically. Phi-4-reasoning-plus (about 90 GRPO steps) is the closest public analogue.

## Q4. OPD on Tinker for our case: prompt format, cost per token, steps, expected gains, and risks

### Takeaway
OPD on Tinker is feasible with existing code. The teacher must be fed a **re-rendered** sequence: the teacher's own chat template for the prompt, the same student completion tokens, and logprobs sliced back onto the completion positions. The tokenizer is shared, so the completion tokens map one-to-one.

Teacher scoring costs about $1.65/M student tokens at the uncached prefill rate. On the spec's batch shape that is about $0.5 per step, which roughly doubles the RL step cost. The main risks are template and format mismatch (thinking vs non-thinking, our answer conventions) and unverified teacher quality on our exact formats.

### Cited Findings
- The teacher scores the exact student token sequence unless the code is modified (comment in `train_on_policy.py`; local source, lines 73–84).
- 397B-A17B prefill is $1.65/M ($0.33 cached) and sampling is $7.50/M — [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).
- Project spec per-step estimate for RLVR: about 300k training tokens ($0.31 after 30% constant-group removal) and about 170k sampled tokens ($0.14), so about $0.45 per step. Phase 4 has a $55 guard. (`/Users/michael/Documents/Github/cps375-575-mini-project/docs/superpowers/specs/2026-11-16-rl-design.md`, lines 240–260)
- Step counts in the TM/cookbook experiments: about 100 steps to recover IFEval (64 groups); 150–200 steps for reasoning OPD (256–512 groups × long context) — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/), [recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/).
- OPD gains are larger when the student starts from a strong SFT checkpoint. A teacher outside the student's support needs much bigger batches — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/).
- Teacher choice: in the TM personalization run, sampling from and distilling against Qwen3-8B worked better than Qwen3-32B. For the math OPD they used the 8B as teacher "because it performed slightly better". **The bigger teacher was not automatically better** — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/).
- A raw (non-RL'd) teacher transfer underperformed direct GRPO. An RL-improved teacher beat it — [2605.12483](https://arxiv.org/abs/2605.12483).
- Teacher noise grows with teacher scale — [2608.31046](https://arxiv.org/abs/2608.31046).

### Inferences: cost model (our numbers, list prices)
- Hybrid step (RLVR reward + 397B KL, same 64×8 batch):
  - Teacher prefill: about 300k tokens × $1.65/M ≈ **$0.50**. If the prompt prefix is cached, about $0.40.
  - Training: constant groups are kept because they now carry KL signal, so about 300k × $1.46/M ≈ **$0.44** instead of $0.31.
  - Sampling: unchanged at **$0.14**.
  - Total ≈ **$1.05–1.10 per step**, versus $0.45 for RLVR alone. With the $55 Phase-4 guard that is about 50 hybrid steps instead of about 120 RLVR steps.
- Cheaper variants:
  1. **IF-only teacher signal**: score only IF groups, which are capped at 1,024 tokens. If IF is about a third of the batch with short outputs, the teacher cost is roughly $0.08–0.12 per step. The overall step becomes about $0.6, so about 90 steps fit.
  2. **Half batch** (32 groups) with the teacher on everything: about $0.55 per step, but noisier.
  3. **A 9B-hybrid teacher** (the cookbook default, $0.30/M prefill, about $0.09 per step). This is likely **not allowed** by the course rule, since the teacher must be larger. Qwen3.6-35B-A3B ($0.45/M prefill, about $0.14 per step) *is* larger, if it scores well on our formats.
- Expected gains: no source measures OPD from a 397B teacher on a 9B LoRA student with greedy IFEval/MATH-500/MBPP. Extrapolating from TM, Qwen3, and the 2026 papers:
  - Largest and most reliable for **IF retention/recovery**: a few IFEval points, mostly by preventing the drift RL causes.
  - Uncertain for MATH-500 at 71.8 greedy: the 397B is much stronger, but our 4,096-token non-thinking format differs from its default thinking mode.
  - Smallest for MBPP, where the unit-test reward is already precise.
  - Under a fixed $55, my estimate of the net difference vs RLVR alone ranges from −1 to +2 points average. This is an inference with wide uncertainty, not a sourced number.

### Inferences: risks to verify first (cheap pre-flight, about $1–3)
1. **Template mismatch.** Our student uses the `role_colon` base renderer. Feeding those tokens unmodified to a chat/hybrid model puts it out of distribution: the per-token KL is dominated by "this isn't my chat format" rather than content. Re-render the prompt with the Qwen3.5 chat template in **non-thinking mode** (empty think block), append the student's completion tokens, and mask the final stop token(s). The student ends on a role_colon terminator; the teacher expects `<|im_end|>`.
2. **Format and convention conflicts.** Our evaluator needs specific answer formats: answer label lines, code-block conventions, IF constraint formatting. A teacher that doesn't know them penalizes them. Mitigation: the teacher prompt may differ from the student prompt, so add a short system instruction that states the required output format. This is "privileged context" for the teacher, as in prompt distillation.
3. **Thinking vs non-thinking.** In thinking mode the teacher expects a `<think>` block. Scoring a direct answer under thinking mode inflates KL on the first tokens. Use non-thinking rendering and check per-position KL on the first ~10 tokens.
4. **Teacher quality on our formats.** Diagnostic: on calibration rollouts we already pay for, compute mean (log p_teacher − log p_student) per sample and test whether it **separates reward-1 from reward-0 samples** within a group, per task. If it doesn't separate on math or code, the teacher signal there is mostly style noise, consistent with 2608.31046; restrict it to IF or drop it. A greedy-sampling check of teacher accuracy is expensive at $7.50/M; only do it on about 50 prompts if needed.
5. **Scale of KL vs task advantage.** The spec's KL-to-v4 is negligible at β=0.02. For OPD the coefficient must be chosen so that the per-token −KL (which starts at about 0.1–1 nat on mismatched tokens) does not swamp |A| ≈ 0.3–0.5. Calibrate α so the KL term is about 0.3–1× the task advantage magnitude on average.
6. **Course rule.** Confirm that "teacher" includes logprob supervision, not just generated data. Confirm that a 397B hybrid (or 35B-A3B) counts as "a larger model".

### Gaps
- Tinker's `compute_logprobs` cost semantics (caching, whether it bills at the prefill rate) need confirming from a usage log of one test call.
- No published per-token KL statistics exist for cross-template OPD.

## Q5. Hybrid approaches: RL + distillation reward, and KL-to-teacher regularization

### Takeaway
The strongest 2026 evidence (MiMo-V2-Flash MOPD; Sparse-to-Dense) supports **outcome reward plus a dense teacher term**, with the student matching or beating teachers. In the cookbook this is the default behavior of the OPD trainer when the environment returns real rewards. KL-to-teacher as a *regularizer* (centered penalty) is a weaker form: it constrains drift rather than adding signal.

### Cited Findings
- MOPD advantage: sg[log π_teacher/π_θ] + α·A_GRPO, with truncated-IS token masking. The student reaches or exceeds the RL-trained teachers on math and code — [arXiv 2601.02780](https://arxiv.org/html/2601.02780v1).
- Sparse-to-Dense: dense transfer first, then optional sparse student RL. RL-teacher transfer beats direct GRPO; a raw teacher does not — [2605.12483](https://arxiv.org/abs/2605.12483).
- KDRL (unified KD + RL with reverse-KL plus GRPO), arXiv [2506.02208](https://arxiv.org/abs/2506.02208) (2025-06) **[recall]**: reports the combination beating either alone on math reasoning for a 1.5B-scale student.
- Cookbook code: KL advantage = env advantage + (−coef · reverse KL). The RL trainer also supports a centered KL penalty toward any `KLReferenceConfig` model or checkpoint (local source).
- OPD against the *pre-RL* checkpoint (self-teacher) prevented IF regression where self-sample training failed — [TM blog](https://thinkingmachines.ai/blog/on-policy-distillation/).

### Inferences
- **Recommendation: a combination, gated by a cheap teacher-quality pre-flight.**
  - **Plan A (default if the pre-flight shows separation on IF only):** RLVR on all three tasks as in the rl1 spec, plus the 397B reverse-KL term **on IF groups only**, using the non-thinking chat re-render and a format-instruction system prompt.
    - Estimated cost about $0.6 per step, so about 90 steps within the $55 guard. Total project spend stays about $72–78.
    - Expected: RLVR's math and code gains are kept, with lower IF regression risk, and possibly +1–3 IFEval (inference).
  - **Plan B (if the teacher separates correct from incorrect on math too):** the KL term on IF + math (math capped at 4,096 tokens). About $0.85 per step, so about 65 steps. Use α small enough that the outcome advantage dominates.
  - **Plan C (pre-flight fails or template issues can't be fixed in a day):** RLVR only, as specced (about 120 steps, about $55). Add a centered KL toward v4 at a *meaningful* β only if IF drifts. This is the safe default.
  - **OPD-only is not recommended.** Our evaluation is outcome-scored, the teacher has not been checked on our formats, and a non-RL'd raw-teacher transfer underperformed GRPO in 2605.12483. OPD-only also forfeits the precise MBPP and math verifiers already built.
  - **Self-teacher alternative (cheapest, no rule risk):** use **v4 itself** as the KL reference or teacher on IF groups, at $0.30/M prefill (about $0.03 per step on IF). This is the TM "recover IF after training" trick. It guards against IF regression but cannot add capability.

### Gaps
- KDRL numbers and α schedules were not verified in this session.
- No source tests reverse-KL hybrid with LoRA r32 at batch 512 for ≤100 steps. The step-count and gain estimates here are inferences.

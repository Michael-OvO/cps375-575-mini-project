# What RLVR does to small (1.5B–14B) models, and how to choose prompt data and curriculum (2025–2026)

Scope note: results are tagged **[ZERO-RL]** (RL directly on a base model), **[SFT/distilled→RL]** (RL on a model that was already SFT'd or distilled), and **[Qwen2.5-Math only]** where the evidence comes only from that model family. Where a claim comes from my background knowledge and was not re-verified in this session, it is marked *(unverified this session)*. Project context: Qwen3.5-9B-Base, LoRA r=32, SFT on about 54k examples; planned RL is about 100–120 steps × 64 prompts × 8 samples at T=1; eval is greedy on MATH-500 (71.8), MBPP (73.2) and IFEval (75.9); budget about $80.

---

## Q1. Does RLVR expand capability or only sharpen the distribution? What does that mean for pass@1 and greedy?

### Takeaway
The consensus from 2025–2026 is that short RLVR mostly **sharpens**: it moves probability mass onto correct paths the model can already sample. That raises pass@1 and greedy accuracy and usually lowers or holds pass@k at large k. Real expansion has only been shown with very long training (thousands of steps, ProRL) or very wide rollouts (N=512, BroRL), and mostly on tasks where the base model is weak. For greedy eval, sharpening is exactly the effect we want. A 100-step run should be thought of as "convert pass@8 into pass@1", not "teach new skills".

### Cited Findings
- Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", arXiv 2504.13837, **NeurIPS 2025 oral**. RLVR models beat their base models at small k, but base models reach higher pass@k at large k. The study covers several model families, six RL algorithms, and math, code and visual reasoning. Coverage and perplexity analyses show the RLVR models' reasoning paths already lie inside the base model's sampling distribution, so RL "biases the output distribution toward rewarded paths" and narrows the reasoning boundary. Distillation, by contrast, can bring in genuinely new knowledge. [NeurIPS oral page](https://neurips.cc/virtual/2025/oral/119945); [papers.cool summary](https://papers.cool/arxiv/2504.13837); [arxiviq summary](https://arxiviq.substack.com/p/neurips-2025-does-reinforcement-learning). (A third-party blog and the slides call it a Best Paper Runner-Up; that is not verified on the official site.)
- ProRL, arXiv 2505.24864, **NeurIPS 2025** (NVIDIA). Prolonged RL with KL control and periodic resets of the reference policy can reach strategies the base model does not reach even under heavy sampling. Gains are uneven: on math, pass@1 rises while pass@128 often falls, while on harder or weaker domains such as code, pass@k keeps improving with longer training. The model is Nemotron-Research-Reasoning-Qwen-1.5B, which starts from DeepSeek-R1-Distill-Qwen-1.5B, so it is **[SFT/distilled→RL]**. [arXiv](https://arxiv.org/abs/2505.24864); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html)
- BroRL, arXiv 2510.01180. With ProRL's rollout size of 16 per prompt, training plateaued and then degraded. Raising the rollout size to 512 kept improving and passed ProRL's final results about 35 hours sooner (about 98 h vs about 133 h). Their argument is that wider exploration per prompt, not more steps, breaks the plateau. [arXiv PDF](https://arxiv.org/pdf/2510.01180)
- "RL's Razor: Why Online Reinforcement Learning Forgets Less" (Shenfeld, Pari, Agrawal), arXiv 2509.04259, **ICLR 2026**. On-policy RL is implicitly biased toward the solution closest to the starting policy in KL among all solutions that solve the task. SFT can move arbitrarily far. Forgetting tracks the forward KL on the new task. An "oracle" SFT target with minimal KL forgets even less than RL. LLM experiments used Qwen2.5-3B-Instruct. [arXiv](https://arxiv.org/pdf/2509.04259); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011312); [author blog](https://jyopari.github.io/posts/rl_razor)
- "RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs" (Matsutani et al.), arXiv 2509.21128, **ICLR 2026 poster**. RL reduces the number of distinct *incorrect* reasoning paths, and SFT increases the number of distinct *correct* ones. At the step level, RL steepens the decay rates of node-visitation, degree and centrality distributions in the reasoning graph by about 2.5×, while SFT flattens them to about 1/3. The authors use this to justify SFT-then-RL as the order that balances diversity against error suppression. [arXiv](https://arxiv.org/abs/2509.21128); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009898)
- Thinking Machines, "LoRA Without Regret" (2025). In GRPO-style RL on MATH and GSM, LoRA matched full fine-tuning even at rank 1. Their information-theoretic argument is that policy gradients give about O(1) bits per episode, compared with O(tokens) for SFT, so RL needs very little capacity. Caveats: rank 1 needed a lower learning rate, a practitioner replication on Qwen3-0.6B found full fine-tuning pulling ahead later, and almost all of the evidence is on math. [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/); [replication notes](https://leonericsson.github.io/blog/2025-10-15-lorawithoutregrets)

### Inferences
- Greedy accuracy is roughly "the mode of the policy is correct", so it is the metric sharpening helps most. Prompts that are greedy-wrong but sometimes right at T=1 (pass@8 > 0) are where RL can flip the mode. This supports the planned tilt toward greedy-wrong prompts with 0.1 ≤ p ≤ 0.9.
- The ceiling for a short run is set by the SFT model's pass@k on the eval distribution. A cheap upper-bound diagnostic is the SFT model's pass@8 at T=1 on MATH-500, MBPP and IFEval (or on held-out lookalikes). If greedy is 71.8 and pass@8 is about 85, the gap available for sharpening is the room RL can plausibly capture in about 100 steps, and in practice only a fraction of it.
- Rank-32 LoRA is more than enough capacity for RL (RL's Razor plus the LoRA result). Learning rate matters much more than rank. The Thinking Machines/Tinker guidance is an LR about 10× the full-FT LR for LoRA *(the "10×" figure is from the same blog; unverified this session)*.
- RL's KL-minimal bias means RL on top of SFT should cause little regression in other skills. Most of the regression risk comes from reward hacking on the IF constraints and from length drift, not from forgetting.

### Gaps
- No paper found that measures pass@1 vs pass@k after exactly about 100 steps of RL on a 7–9B *SFT'd* model with greedy eval. Most "sharpening" analyses use base models or long runs.
- Whether Yue et al.'s conclusion holds for SFT-initialized models was not directly checked in this session.

---

## Q2. Qwen-specific artifacts: spurious rewards and contamination. What this means for Qwen3.5, and which controls to run

### Takeaway
On Qwen2.5-Math, RL with random or even incorrect rewards gives most of the MATH-500 gain that true rewards give. The effect does not transfer to Llama or OLMo. It is explained by GRPO clipping amplifying high-prior pretraining behaviors and by MATH-500 contamination in Qwen. Qwen3.5 is a newer family, but the same caution applies: **any MATH-500 gain on a Qwen model needs a spurious-reward control and a clean or perturbed eval set before you believe it.**

### Cited Findings
- Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR", arXiv 2506.10947, **ICML 2026 (PMLR v306)**. **[ZERO-RL, mostly Qwen2.5-Math]** On Qwen2.5-Math-7B, random rewards raised MATH-500 by **21.4 points**, against **29.1** for ground-truth rewards. The same random reward *degrades* Llama3.1-8B-Instruct by 6.4 points. Spurious rewards that work for Qwen often fail for Llama3 and OLMo2. Mechanism: GRPO's clipping term has a bias that amplifies high-prior behaviors; disabling clipping removes the random-reward gain. The amplified behavior is "code reasoning" (writing Python-style reasoning without executing it), which rose from 65% to over 90% of responses. The authors recommend validating on several model families and using spurious-reward baselines as diagnostic controls. [PMLR](https://proceedings.mlr.press/v306/shao26d.html); [ICML 2026](https://icml.cc/virtual/2026/poster/61082)
- Wu et al., "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination", arXiv 2507.10532, **AAAI 2026**. Given the first 60% of each MATH-500 problem, Qwen2.5-Math-7B regenerates the remaining 40% with a **54.6% exact-match rate** and answers 53.6% of these incomplete problems correctly. Llama is far lower. On RandomCalculation, a generated arithmetic benchmark with no leakage, only accurate rewards give steady gains over the base model, and spurious-reward gains disappear. Pre- and post-RL responses overlap much more lexically on MATH-500 than on the clean set. [arXiv](https://arxiv.org/html/2507.10532v2); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40687)
- One-shot RLVR (Wang et al., NeurIPS 2025) found that **entropy loss alone, with no outcome reward, raised Qwen2.5-Math-1.5B MATH-500 by 27.4 points**. This is another sign that format and entropy effects, not reasoning, can explain large Qwen-Math gains. **[ZERO-RL, Qwen2.5-Math]** [arXiv](https://arxiv.org/abs/2504.20571v1)

### Inferences
- The spurious-reward effect is largest on a *zero-RL base model* whose output format is not yet settled. Our model is already SFT'd, with format fixed and greedy 71.8 on MATH-500, so the "free format gain" left to unlock should be much smaller. A spurious control is still cheap insurance because the model is a Qwen base.
- Contamination raises the absolute score but does not by itself make an RL *delta* fake. The danger is that RL mainly helps recall memorized answers. The defense is to also report results on a perturbed or clean set: MATH-500 with numbers changed, recently written problems, or generated arithmetic in the style of RandomCalculation.
- **Recommended control (one cheap extra run).** Run the same RL config, same prompts and seed, for about 30–40 steps with a **random reward** (Bernoulli 0.5 per sample, independent of correctness), or alternatively a format-only reward. Evaluate greedy on all three benchmarks. If the random-reward run recovers a large share of the true-reward gain at the same step, the gain is not evidence of learning from the verifier. Spurious effects in Shao et al. appear within the first tens to hundreds of steps *(exact curves unverified this session)*, so a short control at roughly 25–35% of the main run's cost is enough to detect them.
- Noise floor, my calculation: the binomial standard error at about 72% on 500 problems is sqrt(0.72·0.28/500) ≈ 2.0 pp. For IFEval (about 541 prompts, about 76%) it is about 1.8 pp. For MBPP (500 test problems, about 73%) it is about 2.0 pp, and larger if the 257-problem sanitized split is used. A paired comparison on the same items has lower variance, but deltas under about 2–3 pp should not be claimed without a paired test (McNemar, or a bootstrap over items).

### Gaps
- No published spurious-reward test on Qwen3 or Qwen3.5 was found. Whether the clipping-bias effect persists in newer Qwen bases is unknown.
- No contamination study specific to Qwen3.5 on MATH-500, MBPP or IFEval was found.

---

## Q3. One-shot and few-shot RLVR, data efficiency, LIMR, learnability-based selection

### Takeaway
RLVR is extremely data-efficient on Qwen-Math. One example, or about 1.4k selected examples, matches the full dataset. Most of that evidence is **[ZERO-RL, Qwen2.5-Math]**, so it supports "a small, well-chosen pool is enough" but not specific gain sizes. The most general selection principle is learnability, p(1−p): favor prompts the model sometimes solves.

### Cited Findings
- Wang et al., "Reinforcement Learning for Reasoning in Large Language Models with One Training Example", arXiv 2504.20571, **NeurIPS 2025**. **[ZERO-RL, mainly Qwen2.5-Math-1.5B]** One example raised MATH-500 from **36.0 to 73.6**, and the six-benchmark average from 17.6 to about 35.7. This matches the 1.2k DeepScaleR subset (73.6 / 35.9). Two examples gave 74.8 / 36.6. Similar effects were reported for Qwen2.5-Math-7B, Llama3.2-3B-Instruct and DeepSeek-R1-Distill-Qwen-1.5B (the last is distilled, i.e. **[SFT→RL]**). Test accuracy kept improving after training accuracy saturated ("post-saturation generalization"). An entropy bonus was important. [NeurIPS proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/b1ea3f93167c55f3fc2999b68a170ff3-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2504.20571v1)
- LIMR (Li, Zou, Liu), arXiv 2502.11886. **[ZERO-RL, Qwen2.5-Math-7B]** "Learning Impact Measurement" scores each sample by how closely its reward curve tracks the model's average learning trajectory. At threshold 0.6 it keeps **1,389 of 8,523 MATH problems** and matches or beats the full set while clearly beating a random subset of the same size. A secondary source gives MATH-500 78.0 vs 78.4 for the full set; verify against the paper. [arXiv](https://arxiv.org/pdf/2502.11886)
- Foster et al., "Learning to Reason at the Frontier of Learnability", arXiv 2502.12272. Many questions are always or never solved and contribute no gradient. Sampling questions by learnability p(1−p), with a buffer of high-learnability questions mixed with random ones, consistently improved training across PPO and VinePPO and across datasets. [arXiv](https://arxiv.org/abs/2502.12272)

### Inferences
- A pool of a few thousand calibrated prompts per domain is plenty for 100 steps. 100 × 64 = 6,400 prompt draws, so a pool of about 2–4k means each prompt is seen about 2–3 times, which the post-saturation result suggests is fine.
- p(1−p) is maximal at p=0.5, and with 8 samples the expected group-advantage signal is near zero outside about 0.1–0.9. The planned filter matches this principle. Weighting draws by p(1−p), instead of sampling uniformly inside the band, is a cheap improvement.
- Do not expect one-shot-style jumps of 30+ points. Those come from fixing format and eliciting latent skill in a zero-RL Qwen-Math base.

### Gaps
- No one-shot or LIMR-style experiments on an SFT'd general model (not a Math model) at 7–9B were found.
- Selection evidence for code and instruction-following prompts is much thinner than for math.

---

## Q4. Difficulty filtering and curricula: Polaris, DeepScaleR, Skywork-OR1, online filtering, zero-variance groups, pass@k training

### Takeaway
Every strong 2025 recipe filters out all-correct and all-wrong prompts, either offline before training or online during it. The target difficulty distribution is model-relative and slightly skewed toward hard prompts. Difficulty drifts easier as training goes on, so prompts should be re-filtered or re-weighted. Zero-variance groups are wasted compute and should be skipped or refilled.

### Cited Findings
- POLARIS (HKU NLP and ByteDance Seed, 2025 blog). **[SFT/distilled→RL]** Base models were Qwen3-4B and DeepSeek-R1-Distill-Qwen-7B. Difficulty was measured with 8 rollouts per prompt on DeepScaleR-40K: most prompts sat at 0/8 or 8/8, skewed hard for 1.5B and easy for 7B. They built a **mirrored-J distribution slightly skewed toward hard problems**. During training the mix drifted easier, and they re-filtered. The 4B run used batch 128 × rollout 8, 700 steps in three stages with length growing from 40K to 52K, and reached AIME24 81.2 and AIME25 79.4. [HKU blog](https://hkunlp.github.io/blog/2025/Polaris); [Seed blog](https://seed.bytedance.com/en/blog/seed-s-open-source-rl-recipe-makes-4b-model-rival-235b-on-math-reasoning); [model card](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview)
- DAPO, arXiv 2503.14476. **[ZERO-RL, Qwen2.5-32B]** Dynamic sampling over-samples and drops groups with accuracy 0 or 1 so every batch has gradient signal. Overlong handling: penalizing truncated samples adds reward noise; masking truncated samples from the loss ("overlong filtering") stabilizes training, and a soft length penalty near the cap ("Soft Overlong Punishment") is the final choice. Result: 50 on AIME24 with half the steps of DeepSeek-R1-Zero-Qwen-32B (47). [arXiv](https://arxiv.org/pdf/2503.14476)
- GRESO, "Act Only When It Pays", arXiv 2506.02177, **NeurIPS 2025**. Over 90% of zero-variance prompts in an epoch were also zero-variance earlier, but about 20% of previously zero-variance prompts become useful again. GRESO skips prompts *before* rollout with a probability that rises with consecutive zero-variance rounds. Reported savings: up to 3.35× fewer rollouts and 2.0× faster total training than dynamic sampling at matched accuracy (speedups via secondary summaries). [project page](https://infini-ai-lab.github.io/GRESO); [arXiv](https://arxiv.org/html/2506.02177v1)
- Pass@k Training, arXiv 2508.10751 (RUC and ByteDance Seed). Using pass@k as the reward, with an analytically derived advantage, improves exploration *without hurting pass@1*. Their argument is that pass@1 rewards push toward conservative local optima. [arXiv](https://arxiv.org/abs/2508.10751)
- DeepScaleR (Agentica, Feb 2025 blog) *(unverified this session)*. DeepSeek-R1-Distill-Qwen-1.5B (**[distilled→RL]**) was trained on about 40k math problems with context growing 8K→16K→24K. AIME24 went from 28.8 to 43.1. Starting at a short context forced concise reasoning before lengthening. Polaris's analysis of the same dataset found it too hard for 1.5B and too easy for 7B.
- Skywork-OR1, arXiv 2505.22312 *(details unverified this session)*. Uses offline and online difficulty filtering and identifies premature entropy collapse as the main failure mode, mitigated with adaptive entropy control.

### Inferences
- With 64 prompts × 8 samples, a prompt calibrated at p≈0.5 still has a probability of 0.5⁸·2 ≈ 0.8% of a zero-variance group in a given step. A prompt at p=0.1 has about 0.9⁸ ≈ 43% probability of all-wrong. So for prompts near the band edges, many groups will still give no signal. Options: (a) narrow the band to about 0.2–0.8, (b) drop zero-variance groups and log the effective batch size, (c) GRESO-style down-weighting on later draws. Refilling the batch (DAPO) costs extra sampling, so with a fixed budget it is better to just drop them and accept a smaller effective batch.
- Difficulty drifts down during training (Polaris). Re-calibrating the pool halfway through (step about 50–60) is the cheapest curriculum step.
- Hard-skew: Polaris targets slightly hard. Our "tilt toward greedy-wrong" achieves the same effect.
- Pass@k reward is an interesting option, but it targets pass@k/exploration, while our eval is greedy. Skip it for a 100-step run.

### Gaps
- No direct ablation of the "0.1–0.9 vs 0.2–0.8" band at small scale was found.
- No difficulty-curriculum evidence specifically for IFEval-style constraint rewards was found.

---

## Q5. SFT→RL interaction for small models; distillation vs RL

### Takeaway
For small models, distilling from a stronger teacher gives far more than RL from scratch. RL *after* a good SFT/distill stage still adds gains, and a stronger SFT start gives a better final model, but the gap between SFT starting points shrinks during RL. On-policy distillation can match RL at a small fraction of the compute when a teacher is available.

### Cited Findings
- DeepSeek-R1, arXiv 2501.12948 (published in *Nature*, 2025) *(numbers unverified this session)*. Large-scale RL on Qwen2.5-32B-Base reached roughly QwQ-32B-Preview level, while the R1-distilled 32B was far better. The paper concludes that distilling a strong model into small ones beats RL on the small models, and that small models trained with RL alone fall short of distillation.
- Qwen3 Technical Report, arXiv 2505.09388 *(unverified this session)*. Qwen3's small models were trained with strong-to-weak distillation (off-policy, then on-policy logit distillation) instead of the full four-stage RL pipeline. The report says this gives better results than RL at much lower GPU cost.
- Thinking Machines, "On-Policy Distillation" (Oct 2025 blog), run on Tinker. The setup samples from the student and has the teacher grade every token. It reproduced Qwen3's finding that on-policy distillation matches RL at a fraction of the cost: AIME'24 about 70% with about 1,800 GPU-hours instead of about 17,920 for RL, a 9–30× compute saving. The teacher was Qwen3-32B and the student Qwen3-8B. Figures are via secondary coverage. [Thinking Machines blog](https://www.thinkingmachines.ai/blog/on-policy-distillation/); [Maginative coverage](https://www.maginative.com/article/thinking-machines-claims-30x-cost-cut-for-training-ai-models/)
- AceReason-Nemotron 1.1, arXiv 2506.13284, **ICLR 2026**. **[SFT→RL, Qwen2.5-Math-7B lineage]** Stronger SFT starts generally give better results after RL, but the performance gap between SFT starts narrows a lot during RL. Math-only RL improves both math and code benchmarks. RL temperature was set so temperature-adjusted entropy stayed around 0.3. [arXiv](https://arxiv.org/abs/2506.13284); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10010287)
- "RL Squeezes, SFT Expands" (ICLR 2026) frames SFT→RL as SFT adding correct paths and RL pruning incorrect ones. [arXiv](https://arxiv.org/abs/2509.21128)

### Inferences
- For our case, SFT on 54k teacher examples has already done the "expand" part. RL's job is to "squeeze": remove wrong modes, especially on greedy-wrong prompts.
- If a teacher is available on Tinker at reasonable cost, on-policy distillation is the strongest alternative to RLVR at this budget. It gives dense per-token signal instead of O(1) bits per episode. It is a different method and would need its own budget, so it is listed as "consider".

### Gaps
- No head-to-head of short RLVR vs on-policy distillation at about $80 or 100 steps on a 9B LoRA was found.

---

## Q6. Length: concise reasoning, length penalties, token caps, truncation handling

### Takeaway
Length behaves like a reward-hackable side channel. Token caps act as an implicit length penalty, and how truncated samples are scored matters a lot. Masking truncated samples or using a soft penalty near the cap is more stable than giving them a hard zero. Explicit length-control RL (L1/LCPO) works but is a separate objective.

### Cited Findings
- DAPO (arXiv 2503.14476): hard-penalizing truncated samples injects reward noise. Masking their loss significantly stabilizes training and improves performance. The soft penalty near the cap is the final design. [arXiv](https://arxiv.org/pdf/2503.14476)
- L1 / LCPO (Aggarwal & Welleck), arXiv 2503.04697, **COLM 2025**. **[distilled→RL, R1-Distill-Qwen-1.5B]** RL rewards correctness plus adherence to a length target given in the prompt. It beats S1-style truncation by up to 100% relative / 20 points absolute on math at matched budgets, and gives "Short Reasoning Models" with non-reasoning-model CoT lengths. [arXiv](https://arxiv.org/abs/2503.04697)
- ShorterBetter, arXiv 2504.21370 *(unverified this session)*. Rewards the shortest correct response in the group and reports large length cuts on distilled reasoning models without accuracy loss.
- DeepScaleR's short-to-long context schedule (8K→16K→24K) is evidence that a short cap early on can help, by forcing efficient reasoning, before lengthening *(unverified this session)*.

### Inferences
- Our model is SFT'd, not a long-CoT reasoner. The main length risk at T=1 is runaway or repetitive generation hitting the cap. Recommended: set the cap at about the 99th percentile of SFT output length per domain, mask truncated samples (reward 0 but excluded from loss, or a soft penalty), and log the truncation rate. If the truncation rate rises, stop or lower the LR.
- Greedy decoding is more prone to repetition loops than T=1 sampling. Track greedy truncation/repeat rate on the eval sets at checkpoints, not just accuracy.
- Do not add an explicit length penalty unless output length is actually growing. It is a second objective, and with only about 100 steps it competes with the accuracy signal.

### Gaps
- No study of token-cap effects specifically on short RLVR of SFT'd (non-long-CoT) 7–9B models was found.

---

## Q7. Expected size of gains from about 100 RL steps on an already-SFT'd 7–9B model

### Takeaway
On already-tuned models, expect **single-digit point gains**, and often only 1–3 points on math, which is close to our noise floor of about 2 pp. IFEval-style constraints are where verifiable-reward RL has given the clearest gains on SFT'd models. Large jumps (20–37 points) appear only in zero-RL Qwen-Math settings and should not be used as expectations.

### Cited Findings
- Tülu 3 (AI2), arXiv 2411.15124. **[SFT→DPO→RLVR, Llama-3.1-8B]** RLVR after DPO: GSM8K 84.3 → **87.6 (+3.3)**, MATH 42.0 → **43.7 (+1.7)**. Most of the gain came from the earlier SFT stage (MATH SFT 31.5 → DPO 42.0). Note this is a late-2024 paper, used here as the standard reference for RLVR on an SFT'd 8B. [model card](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B); [arXiv](https://arxiv.org/pdf/2411.15124). Tülu 3 also reported RLVR gains on IFEval *(exact 8B delta not found this session)*.
- Contrast with zero-RL Qwen-Math gains: +29.1 points on MATH-500 with true rewards (and +21.4 with random) for Qwen2.5-Math-7B [PMLR](https://proceedings.mlr.press/v306/shao26d.html); 36.0 → 73.6 with one example on Qwen2.5-Math-1.5B [arXiv](https://arxiv.org/abs/2504.20571v1). These are mostly format and elicitation effects.
- AceReason-Nemotron 1.1: RL narrows the gap between SFT starting points, so a weaker SFT has more room to gain from RL than a strong one [arXiv](https://arxiv.org/abs/2506.13284).

### Inferences
- Rough expectation for our setup (my estimate, not a published number): MATH-500 +0 to +3; MBPP +0 to +3; IFEval +2 to +6 if the constraint verifier matches IFEval's checks and greedy-wrong prompts are targeted. Any single delta under about 2–3 pp needs a paired significance test.
- The compute budget is small in RL terms: 100 × 512 = about 51k rollouts, roughly the scale of one ProRL/Polaris stage's first tens of steps. That argues for sharpening-oriented choices: greedy-wrong prompts, band filter, no exploration bonuses.

### Gaps
- No 2025–2026 paper reports a curve for exactly about 100 steps × 512 rollouts on a 7–9B SFT'd model with greedy eval across math, code and IF. The estimate above is extrapolated.

---

## Recommendations for our case (adopt / consider / skip)

**Adopt**
1. **Pass-rate filter on 8-sample calibration with 0.1 ≤ p ≤ 0.9, weighted by p(1−p), tilted to greedy-wrong prompts.** Supported by learnability (2502.12272), DAPO dynamic sampling, and Polaris's slightly-hard target. Consider narrowing to 0.15–0.85 so fewer groups have zero variance.
2. **Drop zero-variance groups and log the effective batch size each step.** Do not pay to refill. Optionally down-weight prompts that were zero-variance last time (simplified GRESO, 2506.02177).
3. **Mid-run re-calibration at about step 50–60.** Drop prompts that have become always-solved, since difficulty drifts easier (Polaris).
4. **Truncation handling: mask truncated samples from the loss, or use a soft penalty near the cap. Cap at about the p99 of SFT length per domain. Log truncation and repetition rates** (DAPO).
5. **Spurious-reward control (required).** Same config, same prompts and seed, random Bernoulli(0.5) reward, about 30–40 steps, greedy eval on all three benchmarks. Compare its gain at the same step to the real run. Claim an RL effect only for the part of the real gain above the control. Also report on at least one **perturbed or clean set** (MATH-500 with changed numbers or recent problems, held-out MBPP-style tasks) (2506.10947, 2507.10532).
6. **Paired significance testing.** McNemar or a bootstrap over items for every reported delta; noise floor about ±2 pp at n≈500.
7. **Track pass@1 at T=1 and pass@8 alongside greedy** on a dev subset. That separates sharpening (greedy and pass@1 up, pass@8 flat or down) from expansion, and tells you when the sharpening room is used up.

**Consider**
- **Small KL penalty or an early-stop rule based on drift from the SFT policy.** RL's Razor suggests drift is naturally small. Use early stopping on dev greedy accuracy as the main guard.
- **On-policy distillation from a teacher** as an alternative or follow-up (Thinking Machines blog; Qwen3). Much denser signal per dollar if a teacher is accessible on Tinker.
- **Domain-balanced batches** (math, code and IF mixed in each batch). AceReason found math RL transfers to code, so math may deserve a larger share if budget is tight.

**Skip** (for a 100-step, $80 run)
- Pass@k reward training (2508.10751): it targets exploration and pass@k, not greedy.
- Very wide rollouts per prompt (BroRL N=512) and prolonged RL (ProRL): out of budget, and they aim at capability expansion.
- Entropy bonuses or exploration tricks (one-shot RLVR, Skywork-OR1): helpful for long runs, but they work against sharpening in a short run judged by greedy eval.
- Explicit length-control RL (L1, ShorterBetter), unless output length actually drifts.
- Using one-shot or LIMR gain sizes as expectations: they are zero-RL Qwen2.5-Math results.

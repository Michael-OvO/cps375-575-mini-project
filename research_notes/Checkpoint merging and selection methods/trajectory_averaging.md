# Weight averaging along a single training trajectory for LLM post-training (SFT and RL), 2022-2026

Scope note: our case is Qwen3.5-9B-Base, LoRA r=32, SFT for ~1,688 steps with linear-decay LR, then RLVR for ~200 steps at a constant 2e-5 (clip-free IS-REINFORCE, group advantages), with checkpoints every 10 RL steps and greedy-decoded evals on IFEval / MATH-500 / MBPP. Findings below come from web searches and the papers' own text where I could fetch it. Several summaries came from search-result abstracts rather than full reads, and those are flagged.

## Q1. EMA / SWA / LAWA / last-k averaging: reported gains on LLM SFT, instruction tuning, reasoning and RL

### Takeaway
The large, well-documented gains come from **pretraining at constant or high LR**: +5 to +7 HumanEval points (ByteDance PMA), and LAWA beating EMA and SWA on GPT-2/Pythia loss. For **post-training** the evidence is thinner and the gains are small and mixed. The best systematic within-trajectory study I found (DPO, 4 models) reports averaging all saves as the most robust recipe, about +0.3 z over honest checkpoint selection. It still loses to the best single checkpoint on some axes, and it hurt in 1 of 4 models. I found no paper that reports a gain or loss in points from averaging RLVR checkpoints of a math/code/IF LLM.

### Cited Findings
**Pretraining (strongest evidence)**
- ByteDance Seed, "Model Merging in Pre-training of LLMs" (PMA, NeurIPS 2025). In the constant-LR (WSD stable) phase, averaging checkpoints raised HumanEval from 31.1 to 36.6 on Seed-MoE-1.3B/13B and from 54.3 to 61.6 on Seed-MoE-10B/100B. Gains on near-saturated benchmarks such as BBH were smaller. — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- PMA: SMA, EMA and WMA (linear weights) were compared. Early in training WMA was best and EMA α=0.2 beat α=0.1, i.e. weighting recent checkpoints helps while weights are still moving fast. Later in training the methods converged, and the authors default to SMA (uniform). — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- PMA: averaging taken early in the cosine-decay phase was "comparable to or even superior" to the fully annealed checkpoint, and better for larger models. — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- PMA-init, i.e. starting SFT from an averaged pretrained model: "gains are not consistently observed" but it "does not adversely affect" SFT. CT-stage models converge to parity. — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- Sanyal et al., "Early Weight Averaging meets High Learning Rates for LLM Pre-training" (LAWA, NeurIPS 2023 WANT workshop). LAWA averages the latest k checkpoints from a FIFO queue sampled at widely spaced intervals. It beat the original run, EMA and SWA on GPT-2 small/medium/large validation loss. Tested on Pythia 1B-12B, with up to ~4,200 GPU-hours saved for 6.9B. The gain is largest at high LR. — [arXiv 2306.03241](https://arxiv.org/abs/2306.03241v2)
- Foundational, older: Kaddour 2022, "Stop Wasting My Time!", introduced LAWA for BERT/ImageNet. I did not fetch it; it is cited only as the origin of LAWA in Sanyal et al. — [arXiv 2306.03241](https://arxiv.org/abs/2306.03241v2)
- Llama 3: the final pretrained model is the average of checkpoints taken during annealing. For post-training, Meta averages models "obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage". That is cross-run souping, not within-trajectory averaging. No ablation numbers are given. — [Llama 3 Herd of Models, arXiv 2407.21783](https://arxiv.org/pdf/2407.21783); summary in [Medium explainer](https://ritvik19.medium.com/papers-explained-187b-llama-3-1-f0fb06898c59)
- Nemotron 3 Super uses sliding-window weighted averaging of recent checkpoints, mainly during pretraining. This comes from a search snippet only. — [arXiv 2604.12374](https://arxiv.org/pdf/2604.12374)

**SFT / instruction tuning / preference tuning**
- OpenEuroLLM DPO checkpoint-averaging study (GitHub issue, not peer-reviewed). Models were Qwen3 4B/8B/14B and Llama 3.1 8B, with DPO saves every 200 steps over ~2,000 steps (~10 saves) and 282 mergekit linear averages. Evaluation used 10 held-out benchmarks, including GSM8K and IFEval. — [OpenEuroLLM Taskboard #434](https://github.com/OpenEuroLLM/Taskboard/issues/434)
  - Averaging all saves was the best recipe overall: +0.30 z vs honest (held-out) checkpoint selection and +0.11 z vs the last checkpoint. Last-3, last-5 and EMA (decay 0.7 per save) came "close behind". Top-k soups and peak-to-peak interpolation added nothing.
  - Per model, soup vs honest selection: Qwen3-8B +0.43 z, 14B +0.35, Llama-3.1-8B +0.23 (but −0.31 on capability), Qwen3-4B −0.22. Soup vs the last checkpoint: −0.03, +0.29, +0.07 and −0.23 respectively.
  - In 87% of cases some single checkpoint matched or beat the merge on both axes. Averaging is described as a robust default, not a way to beat the best checkpoint in a sweep.
  - The 14B gain over the last checkpoint came mostly from runs that declined in the second half of training. Averaging helps most when the end of the trajectory regresses.
  - Preference peaked before capability in 81% of runs (e.g. Qwen3-8B: step 400 vs 1,600).
  - LoRA vs full fine-tuning is not stated.
- Dang et al., "Weight Ensembling Improves Reasoning in Language Models" (2025). During reasoning SFT, Pass@1 keeps rising but Pass@k drops (diversity collapse). WiSE-FT, a 50/50 average of the latest and an early SFT checkpoint, recovers Pass@k and also improves Pass@1. It beat early stopping and higher temperature, and gave better results when RL was run afterwards. The main figure uses Gemma-2-2B on GSM8K. I did not extract exact point deltas. — [arXiv 2504.10478](https://arxiv.org/html/2504.10478v4)
- EMA during SFT: I found no study that directly measures benchmark gains from weight-EMA in LLM SFT. A LLaVA-7B continual-instruction-tuning experiment found results sensitive to the EMA decay even in 0.990-0.992. — [arXiv 2410.10868](https://arxiv.org/pdf/2410.10868)
- LightOnOCR (1B VLM) averages the last 5 SFT checkpoints before RLVR and reports the average beats every single checkpoint. This comes from a search snippet; OCR is not reasoning. — [arXiv 2601.14251](https://arxiv.org/html/2601.14251v2)

**RL / RLHF**
- Lin et al., "Mitigating the Alignment Tax of RLHF". Interpolating pre-RLHF (SFT) and post-RLHF weights gave the strongest alignment-vs-forgetting Pareto front among the methods tested. Layer-wise ratios (HMA) did better still. Models were OpenLLaMA-3B and Mistral-7B. This is endpoint interpolation, not trajectory averaging. — [arXiv 2309.06256](https://arxiv.org/abs/2309.06256v4)
- WARP (Ramé et al., Google, 2024). Uses an EMA of the policy as the KL anchor instead of the fixed SFT model, which gives sustained reward improvement at moderate KL. Independent EMA-anchored runs are then SLERP-merged, and the result is interpolated back toward the init (LITI), iterated. Tested on Gemma-7B with REINFORCE RLHF. The reported plateau after ~3 iterations comes from secondary summaries. Here EMA is a training-time regularizer, not post-hoc checkpoint averaging. — [arXiv 2406.16768](https://arxiv.org/html/2406.16768v1)
- EMA-PG (2025/26). An EMA anchor plus a top-k KL term on top of GRPO. R1-distill-Qwen-1.5B reaches 53.9% on OlympiadBench vs 50.8% for GRPO. This is again a training-time anchor, and the number comes from a ResearchGate copy. — [ResearchGate](https://www.researchgate.net/publication/400459520_EMA_Policy_Gradient_Taming_Reinforcement_Learning_for_LLMs_with_EMA_Anchor_and_Top-k_KL)
- M-GRPO uses a slowly evolving momentum (EMA) model as the target to stop policy collapse in self-supervised RLVR. — [arXiv 2512.13070](https://arxiv.org/html/2512.13070)
- GTR-Turbo (agentic VLM RL) merges the RL checkpoints from every update (TIES, with a linear-average comparison). The merged model beats the current training agent and is used as a free teacher. This is not a text LLM on math/code/IF. — [arXiv 2512.13043](https://arxiv.org/html/2512.13043v1)
- "Synergy Dilemma of Long-CoT SFT and RL" (reasoning VLMs) interpolates SFT and RL models. Linear merging at SFT ratio 0.25 gave the best average; TIES had compatibility problems. — [arXiv 2507.07562](https://arxiv.org/pdf/2507.07562)
- OLMo 3: the only merge I could confirm is model souping at 32B midtraining. I found no confirmed SFT- or RL-checkpoint merge in the Think/Instruct pipelines. — [Emergent Mind topic summary](https://www.emergentmind.com/topics/olmo-3); [Cameron Wolfe on OLMo 3](https://cameronrwolfe.substack.com/p/olmo-3)

### Inferences
- Since PMA, LAWA and the DPO study all agree that SMA and EMA come out close once weights are stable, uniform averaging of the last k checkpoints is a reasonable default. Recency weighting (WMA/EMA) matters only while weights are still moving fast.
- Realistic expectations for our 200-step RLVR: a within-noise to roughly +1-point change on aggregate, with the largest benefit if the final ~50 steps regress. For a short, low-LR LoRA RL run I would not expect the multi-point pretraining-style gains.
- WARP, EMA-PG and M-GRPO show that RL practitioners use EMA mostly as a regularizing anchor during training. That is a different intervention from post-hoc averaging, and it is not available to us without changing the training loop.

### Gaps
- I found no paper reporting points gained or lost from post-hoc averaging of RLVR (GRPO/REINFORCE) checkpoints on MATH-500, IFEval or MBPP. Searches for "checkpoint averaging GRPO" returned only score averaging over checkpoints and checkpoint selection methods.
- I could not confirm whether Tulu 3, OLMo 2/3, DeepSeek-R1 or Qwen3 merge RL checkpoints from one trajectory. I found no such statement.
- I found no quantitative LLM-SFT EMA/SWA study (full FT or LoRA).

## Q2. Interaction with the LR schedule: does averaging substitute for LR decay, and does it matter more under constant-LR RL?

### Takeaway
Averaging is a partial substitute for LR decay. It recovers a large part, but not all, of the cooldown benefit (Hägele et al.). PMA finds stable-phase averages comparable to annealed models, and the gain is largest at high or constant LR (LAWA, PMA). Our SFT already had linear decay to zero-ish LR, so averaging late SFT checkpoints should add little. Our RL runs at a constant 2e-5, which is exactly the regime where averaging acts as an implicit cooldown. The caveat is that 2e-5 is already a low LR for LoRA, so iterate noise may be small.

### Cited Findings
- Hägele et al. 2024, "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations". SWA on a constant LR gives a "significant performance boost" but "does not reach the loss values of explicit cooldowns". On cosine, SWA "reduces the gap without the need for a separate decay phase". The window was h=500 steps, and windows of 2,500 steps (256M tokens) or less were best. No downstream SWA numbers. Cooldown and cosine matched downstream (aggregate 46.23 vs 46.26 at 1B/100B tokens). — [arXiv 2405.18392](https://arxiv.org/html/2405.18392v3)
- PMA: an averaged constant-LR model "significantly outperformed" both the constant-LR and annealed continuations early on, and was later comparable to the annealed model. The authors use PMA as a cheap simulator of annealing. — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- Sanyal et al.: LAWA's main benefit is allowing high LR without hurting generalization. Gains shrink as the LR decays. — [arXiv 2306.03241](https://arxiv.org/abs/2306.03241v2)
- LoRA's optimal LR is about 10x the full-FT optimum and roughly independent of rank (1/r scaling). LoRA matches full FT in RL even at very low rank. — [Thinking Machines, "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/); [TRL docs](https://huggingface.co/docs/trl/main/en/lora_without_regret)

### Inferences
- Our RL LR (2e-5 with LoRA) is roughly 20x below our SFT peak (4.7e-4). If the LoRA-optimal RL LR is around 10x full-FT, 2e-5 is on the low side, so per-step iterate noise is small and so is the potential averaging gain. Averaging matters more if RL reward or eval curves look noisy step to step.
- Averaging across SFT checkpoints under linear decay, where the last checkpoint is already annealed, is expected to add little. The Hägele/PMA finding that averaging and decay are near-equivalent suggests combining them is redundant. Only WiSE-FT-style early+late interpolation, which serves diversity and Pass@k rather than greedy accuracy, has evidence in decayed SFT.

### Gaps
- I found no study of averaging under constant-LR RLVR specifically, and no study of short (<500-step) RL runs.
- I found no study of averaging on top of schedule-free optimizers in post-training. I did not search this further.

## Q3. Averaging during RL specifically

### Takeaway
Within RL, trajectory averaging appears mainly as (a) an EMA anchor or teacher during training (WARP, EMA-PG, M-GRPO, GTR-Turbo) and (b) post-hoc interpolation between SFT and RL endpoints (alignment-tax papers, Synergy Dilemma). Both report gains. There is no direct published evidence for or against post-hoc averaging of last-k RLVR checkpoints on math/code/IF with greedy decoding.

### Cited Findings
- See the Q1 "RL / RLHF" bullets: WARP ([arXiv 2406.16768](https://arxiv.org/html/2406.16768v1)), EMA-PG (+3.1 points OlympiadBench vs GRPO, [ResearchGate](https://www.researchgate.net/publication/400459520_EMA_Policy_Gradient_Taming_Reinforcement_Learning_for_LLMs_with_EMA_Anchor_and_Top-k_KL)), M-GRPO ([arXiv 2512.13070](https://arxiv.org/html/2512.13070)), GTR-Turbo ([arXiv 2512.13043](https://arxiv.org/html/2512.13043v1)), alignment-tax averaging ([arXiv 2309.06256](https://arxiv.org/abs/2309.06256v4)), Synergy Dilemma ([arXiv 2507.07562](https://arxiv.org/pdf/2507.07562)).
- RLVR papers commonly report the mean score of their top-3 checkpoints (evaluated every 16 steps) to cope with RL instability. That averages scores, not weights, and it shows checkpoint-to-checkpoint eval noise is large enough that authors average over it. — [arXiv 2508.05928](https://arxiv.org/pdf/2508.05928)
- Uncertainty-based checkpoint selection for GRPO has been proposed as an alternative to picking the last checkpoint. — [OpenReview WrkArtWf3u](https://openreview.net/pdf?id=WrkArtWf3u)

### Inferences
- Endpoint interpolation (WiSE-FT between our SFT and RL models, e.g. λ≈0.5-0.75 toward RL) is the RL-adjacent averaging with the most support. It mainly protects against forgetting and alignment tax, e.g. IFEval regressions caused by math-heavy RL, and is cheap to test.
- Within-RL last-k averaging is cheap with 20 saved checkpoints. Expect it to smooth noise rather than add capability.

### Gaps
- No public per-benchmark deltas (points) for last-k RLVR checkpoint averaging on Qwen-family models.

## Q4. Does averaging LoRA factors (A and B separately) approximate averaging the merged delta B·A?

### Takeaway
Not exactly. Factor-wise averaging yields mean(B)·mean(A) rather than mean(B·A). The error is a second-order covariance term. It is small when one factor, typically A, barely moves along a single trajectory. For independently trained adapters it is large, and that is where the published failures come from. The exact alternative is to average the merged deltas, or to concatenate factors into a rank-k·r adapter scaled by 1/k.

### Cited Findings
- Linear factor averaging (A_merged = mean A_i, B_merged = mean B_i) is not equal to averaging deltas. A rescaled concatenation (rank N·r, 1/N scaling) gives exactly the mean of the deltas. Plain concatenation without rescaling sums the deltas, an "N× over-perturbation". — [arXiv 2604.16826, "Crowded in B-Space"](https://arxiv.org/html/2604.16826)
- In multi-adapter merging, interference concentrates in the output-side matrix B. — [arXiv 2604.16826](https://arxiv.org/html/2604.16826)
- Zhu et al. (ICML 2024), "Asymmetry in Low-Rank Adapters": A acts as a feature extractor and B creates the output. A random, untrained A performs nearly as well as a tuned one, so B carries most of the learning. — [arXiv 2402.16842](https://arxiv.org/abs/2402.16842)
- Delta-based methods such as TIES must operate on B·A, not on the raw factors (practitioner note). — [The Neural Base](https://theneuralbase.com/lora-qlora/learn/advanced/ties-merging-task-informed-merging/)

### Inferences (own derivation, not from a source)
- Write B_i = B̄ + δB_i and A_i = Ā + δA_i over the k checkpoints. Then mean(B_i A_i) = B̄Ā + mean(δB_i δA_i), so the error of factor averaging is exactly the cross-covariance term (1/k)Σ δB_i δA_i. It is second order in within-window drift.
- Within one trajectory, LoRA starts with B=0 and A random, and in practice A changes much less than B (per the asymmetry result). Over a 10-50-step RL window at LR 2e-5, δA should be small, making the error small relative to ‖B̄Ā‖. This should be checked numerically on our checkpoints by computing ‖mean(BA) − mean(B)mean(A)‖_F / ‖mean(BA)‖_F per layer. That takes a few seconds on CPU.
- Factor averaging fails badly when (a) the adapters come from different runs or seeds (A's are unrelated, and LoRA's rotation symmetry B→BR, A→R⁻¹A means the factors are not aligned), or (b) the window spans the start of training, where B goes from 0 to large and A is still moving.
- The safest way to stay a valid rank-32 LoRA file is to average factors only within one run's late window. To be exact, average the merged deltas: either merge into the base, or SVD-truncate the averaged delta back to rank 32. The truncation error is negligible if the deltas share a subspace.

### Gaps
- I found no published measurement of factor-averaging error for checkpoints from a single LoRA trajectory.
- I could not confirm whether Tinker can load an externally averaged LoRA (upload or custom weights) for sampling and eval. This needs checking in the Tinker docs.

## Q5. Recommended k, spacing, which checkpoints, and failure modes

### Takeaway
Average only stable, late checkpoints that do not straddle a regression. Uniform weights are fine. PMA and LAWA say that including early, unstable weights hurts and that more checkpoints help only once training has settled. The DPO study found averaging all ~10 saves best overall, with last-3/5 close behind. Validate on a held-out dev split, because averaging can lose to the best single checkpoint on specific axes.

### Cited Findings
- PMA interval ablation (N=10). Early in training, large intervals (16B/32B tokens) underperformed the un-merged baseline because they pulled in unstable early weights. Late in training, interval differences shrank. The optimal interval grows with model size. — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- PMA count ablation (N=3/6/10/15). Early, more models hurt (unstable weights). After training completed, more models gave "significant" improvements. — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)
- Hägele: SWA windows of 2,500 steps or less were best. — [arXiv 2405.18392](https://arxiv.org/html/2405.18392v3)
- DPO study: all ~10 saves (every 200 of ~2,000 steps) were best. Last-3, last-5 and EMA(0.7) were close. Top-k soups added nothing, and interpolating two peaks traded one metric for another. Averaging lost to a judge-picked checkpoint on preference by 0.2-0.4 z in 3 of 4 models. — [OpenEuroLLM #434](https://github.com/OpenEuroLLM/Taskboard/issues/434)
- Different capabilities peak at different steps (preference before capability in 81% of runs), so an average is a compromise across them. — [OpenEuroLLM #434](https://github.com/OpenEuroLLM/Taskboard/issues/434)
- Pass@k (diversity) collapses during SFT while Pass@1 rises. Averaging with an earlier checkpoint restores diversity. — [arXiv 2504.10478](https://arxiv.org/html/2504.10478v4)
- Early-interval WMA beats SMA (recency weighting helps when weights move fast). — [arXiv 2505.12082](https://arxiv.org/abs/2505.12082v1)

### Inferences (applied to our run, 20 RL checkpoints at steps 10-200)
- Suggested first candidate: uniform average of the last 5 RL checkpoints (steps 160-200), or the last 10 (110-200) if per-checkpoint dev scores are flat over that window. Exclude the first ~50 steps, where the policy moves fastest from SFT. Spacing of 10 steps is fine. With LR 2e-5 and 200 steps, all checkpoints are close in weight space, so linear mode connectivity is very likely.
- Second candidate: WiSE-FT between the SFT final and the RL final (λ≈0.5-0.75 toward RL). Use this if RL improved MATH but regressed IFEval or MBPP.
- Failure modes to check:
  - (1) Averaging across a regression or reward-hacking phase, for example a format or length drift starting mid-run. The average then inherits a partial version of it. Pick a window after or before the drift.
  - (2) Format drift interacts badly with greedy decoding. A blend of two answer-format conventions (e.g. `\boxed{}` vs plain answers, or IFEval constraint styles) can tip greedy outputs into neither. Inspect format-compliance rates on the averaged model, not only accuracy.
  - (3) Capabilities peaking at different steps: the average can be worse than the best checkpoint on each individual benchmark while better on the aggregate.
  - (4) Factor-wise LoRA averaging across a window that includes early checkpoints (see Q4).
  - (5) Small-sample selection noise: MATH-500 has ±~2 pts standard error at 500 items, so differences of 1 point between the average and a single checkpoint are noise. Use a held-out dev split for the choice and keep the hidden test untouched.
- Budget fit: averaging costs nothing in training. The only spend is extra eval runs (one per candidate), which fits the "one best guess plus one improvement" approach. Candidates: last-5 average, and possibly the SFT↔RL interpolation.

### Gaps
- I found no evidence on how greedy decoding specifically interacts with weight averaging. The format-blend failure mode above is my inference.
- I found no recommended k for RL with ~20 checkpoints. The values above are extrapolated from pretraining and DPO studies.

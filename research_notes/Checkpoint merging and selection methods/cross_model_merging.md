# Cross-model / cross-run merging and extrapolation for LLM post-training (2022-2026)

Scope note: our setting is Qwen3.5-9B-Base + LoRA r=32, SFT v1..v4 (v4 best: IFEval 75.9, MATH-500 71.8, MBPP 73.2), then ~200 steps of RLVR on v4 (rl1), greedy eval. About 18 tool calls were used; several primary tables (Olmo 3 soups, KnOTS 7B, DELLA per-task scores, long-to-short per-size tables, WiSE-FT reasoning numbers) were not reached and are listed under Gaps. Everything marked "Inference" is my reasoning, not a source claim.

## Q1. WiSE-FT-style interpolation between pre-RL (SFT) and post-RL weights

### Takeaway
No source shows plain linear interpolation of SFT and RL(VR) weights giving a clear gain over the RL endpoint on reasoning benchmarks. Where it was tested, it traces a smooth trade-off between the endpoints. The strongest positive WiSE-FT evidence is for interpolating an early and a late *SFT* checkpoint, which mostly recovers pass@k diversity rather than greedy pass@1. RL changes the weights little and forgets less than SFT, so the RL endpoint is usually already the "safe" point.

### Cited Findings
- "Weight Ensembling Improves Reasoning in Language Models" (Dang, Baek, Wen, Kolter, Raghunathan; arXiv 2504.10478, v4 Oct 2025): during reasoning SFT, pass@1 keeps rising while pass@k falls quickly ("diversity collapse"). Interpolating the latest SFT checkpoint with an early one (WiSE-FT) "almost completely recovers Pass@k while also improving Pass@1". The gains cannot be matched by early stopping or by temperature/diversity-aware decoding. Recipe: run SFT while pass@1 improves, then WiSE-FT. The WiSE-FT model also scales better at test time (Best@k, majority vote) and gets better results with less data when RL is run on top. Models: Gemma-2-2B and Qwen-2.5 on GSM8K/MATH/AIME (per secondary review). No independent replication found. — [arXiv 2504.10478](https://arxiv.org/abs/2504.10478v3); [review](https://www.themoonlight.io/en/review/weight-ensembling-improves-reasoning-in-language-models); [SOTAVerified lists it as unreproduced](https://sotaverified.org/papers/weight-ensembling-improves-reasoning-in)
- "The Synergy Dilemma of Long-CoT SFT and RL" (arXiv 2507.07562, reasoning VLMs): merging SFT and RL models with linear and SLERP gives trade-offs between the two endpoints. TIES runs into compatibility problems. Two-stage training, interleaving, progressive schedules, data mixing and merging all show a "persistent synergy dilemma" (a trade-off, not complementarity). — [arXiv 2507.07562](https://arxiv.org/html/2507.07562v1)
- DoTS, "Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors" (arXiv 2605.00610, preprint): a naive linear merge of independently trained SFT and RLVR task vectors lets the SFT update dominate and drowns out the weaker RLVR signal. Their conflict-aware composition reportedly matches training-based SFT→RLVR on math at about 3% of the compute. Not independently checked. — [arXiv 2605.00610](https://arxiv.org/html/2605.00610v1)
- "RL's Razor: Why Online RL Forgets Less" (arXiv 2509.04259; NeurIPS 2025 / ICLR 2026): at matched new-task accuracy, on-policy RL forgets less than SFT. Forgetting is predicted by the KL between the fine-tuned and base policy, measured on the new task (R² ≈ 0.96 in the toy ParityMNIST setting; LLM validation on Qwen2.5-3B-Instruct). An "oracle SFT" built to minimise KL forgets even less than RL. — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259); [blog](https://jyopari.github.io/posts/rl_razor)
- RLVR weight updates are close to rank-1 and nearly linear over training. A rank-1 SVD reconstruction keeps nearly all of the MATH gain, and the rank-1 coefficient fits a line in time with R² > 0.98 for most tensors (Qwen2.5-Math-1.5B, Qwen3-4B/8B-Base, GRPO, 500 steps). — [RELEX, arXiv 2605.21468](https://arxiv.org/html/2605.21468v1)

### Inferences
- For rl1, interpolating θ_v4 + λ(θ_rl1 − θ_v4) with λ < 1 is effectively "less RL". Since RLVR updates are small and almost linear in time, λ≈0.5 should behave much like an earlier RL checkpoint (≈ step 100). It is useful mainly as a hedge if rl1 over-optimises (for example IFEval or MBPP regress while MATH rises), not as a source of new gains. Picking the best saved RL checkpoint does the same job and costs nothing extra.
- The positive WiSE-FT result targets pass@k and diversity. Our grading is greedy pass@1, so expect small effects, at most around ±1 pt. That magnitude is my inference, not a sourced number.

### Gaps
- No found paper reports WiSE-FT(SFT↔RLVR) on IFEval or MBPP for a 7-9B text model with numbers. I did not reach Tulu 3 or Olmo 3 SFT/RL merge ablations: a secondary summary says Olmo 3 uses "model souping", but without numbers or stage details ([Olmo 3 summary](https://www.emergentmind.com/topics/olmo-3); primary arXiv 2512.13961 not fetched).
- Dang et al.'s exact pass@1 deltas were not extracted.

## Q2. ExPO (model extrapolation) and follow-ups; applicability to RLVR

### Takeaway
ExPO (θ = θ_aligned + α(θ_aligned − θ_SFT)) gives consistent but modest gains on chat-preference metrics after DPO/RLHF at 7-8B: +0.8 to +4.4 LC win-rate points at α = 0.1-0.5. Its effect on reasoning is mixed. GSM8K rose sharply for undertrained zephyr models but fell 7.7 pts for llama3-8b-iter. On RLVR, 2026 work finds that naive two-point ExPO is clearly worse than continuing RLVR or than trajectory-fitted extrapolation (RELEX). Extrapolating past a *finished* RLVR run gives at best ≈ +0.1-0.5 on MATH and can collapse.

### Cited Findings
- ExPO, "Weak-to-Strong Extrapolation Expedites Alignment" (Zheng et al., arXiv 2404.16792, ICML 2024 listing). Results on 12 public models, AlpacaEval 2.0 LC win rate before → after (α): zephyr-7b-alpha 10.0→13.6 (0.3); zephyr-7b-beta 13.2→14.0 (0.1); starling-7b-alpha 18.3→19.5 (0.2); starling-7b-beta (RLHF) 25.8→26.4 (0.5); snorkel-7b 24.0→26.4 (0.3); llama3-8b-iter 36.0→37.8 (0.3); internlm2-7b 18.3→22.7 (0.5); tulu-2-dpo-7b 10.2→11.7, -13b 15.5→17.6, -70b 21.2→25.7 (all 0.5). MT-Bench gains range from +0.02 to +0.66. — [arXiv 2404.16792 v2](https://arxiv.org/html/2404.16792v2)
- ExPO helps undertrained models most. In a controlled zephyr-7b-sft DPO run, LC WR before → after ExPO was: 5% data 9.1→14.7, 10% 10.4→16.3, 20% 12.9→21.3, 40% 14.6→16.6, 100% 17.3→20.2. The 20%-data model with ExPO beat the fully trained one. — [arXiv 2404.16792](https://arxiv.org/html/2404.16792v2)
- ExPO effects on reasoning (Open LLM Leaderboard appendix): GSM8K zephyr-7b-alpha 14.0→28.3, zephyr-7b-beta 11.4→27.3, internlm2-7b 27.1→30.5; but llama3-8b-iter 67.3→59.6, tulu-2-dpo-70b 62.6→59.4, starling-7b-alpha 62.3→61.6. MMLU moves slightly, sometimes down. No MATH, code or IFEval results. — [arXiv 2404.16792](https://arxiv.org/html/2404.16792v2)
- ExPO failure modes reported by the authors: (a) extrapolating Base→SFT degrades the model; (b) pairs of separately trained SFT/RLHF models do not benefit, because alignment may not increase monotonically along that path; (c) models from different bases can collapse; (d) overtrained models (higher LR or more epochs) gain less, attributed to spurious features such as length bias; (e) length grows with α, and the best reward was at about 500-600-token outputs. α was chosen by grid search with a reward model on a dev set. — [arXiv 2404.16792](https://arxiv.org/html/2404.16792v2)
- RELEX (arXiv 2605.21468; GRPO on MATH, lr 1e-6, KL 0.001, 500 steps): fit a rank-1 direction plus a linear time coefficient from the first 15-20% of steps, then extrapolate. MATH base/RLVR(500)/RELEX: Qwen2.5-Math-1.5B 48.2/71.5/71.6; Qwen3-4B-Base 64.0/85.5/85.6; Qwen3-8B-Base 73.9/88.5/87.4. At matched cost, two-point ExPO was much worse on MATH: 67.7 / 82.7 / 85.5 vs RELEX 71.6 / 85.6 / 87.4. OOD averages were ExPO 27.2/39.8/41.7 vs RELEX 30.0/43.0/46.2. — [arXiv 2605.21468](https://arxiv.org/html/2605.21468v1); [code](https://github.com/weizhepei/RELEX)
- Extrapolating beyond the end of RLVR (RELEX Table 3, MATH): Qwen2.5-1.5B 71.7 at "step 750" and 71.6 at "1000" vs 71.5 at the real step 500 (flat). Qwen3-8B 88.0 at 750 (−0.5 vs the RLVR endpoint of 88.5), then 85.6 at 1000. Qwen3-4B degraded: 82.7 at 750, then 73.6 or 50.9 at 1000 depending on the fit window. With raw-weight extrapolation, direction similarity to the true update fell from 0.72 to 0.35 and the magnitude ratio grew from 1.26 to 2.70. Cubic fits collapsed (17.8 at step 200). — [arXiv 2605.21468](https://arxiv.org/html/2605.21468v1)
- Related 2026 work: "Linear Dynamics in the RLVR Training of LLMs" (arXiv 2601.04537) forecasts future logits and weights from earlier DeepScaleR checkpoints. NExt (arXiv 2604.11446) extrapolates *LoRA*-RLVR trajectories with a learned nonlinear predictor and notes that "not all parameters in LLM RLVR satisfy linearity". AlphaRL is another rank-1 extrapolation baseline (85.0 MATH on Qwen3-8B in RELEX's table). — [2601.04537](https://arxiv.org/pdf/2601.04537); [NExt 2604.11446](https://arxiv.org/html/2604.11446); [RELEX](https://arxiv.org/html/2605.21468v1)

### Inferences
- For rl1 (~200 steps, small LR), ExPO past the final RL checkpoint is a cheap lottery ticket. Evidence suggests ≈0 to +1 pt on the RL-trained domain when RL is undertrained, and a real risk of regressions elsewhere: GSM8K −7.7 at 8B in ExPO; collapse at long horizons in RELEX. If tried, use small α (0.1-0.3), check every benchmark, and select α on our own held-out dev split, not on the public test sets.
- ExPO helps most when the RL run is clearly undertrained, i.e. reward still rising at step 200. If the rl1 reward curve has plateaued, expect nothing.
- Never extrapolate along v_k→v4 (SFT→SFT) directions. The ExPO authors report that Base→SFT and separately trained SFT pairs do not benefit.

### Gaps
- No found study applies ExPO to an RLVR run that is LoRA-trained with mixed IF+math+code rewards, or reports IFEval/MBPP effects.

## Q3. Merging specialized adapters (IF / math / code) vs one multi-task model; LoRA-specific issues

### Takeaway
Merging separately trained experts from one base is a real alternative to data mixing only in some settings: large instruction-tuned bases, many experts, or objectives that conflict (safety vs helpfulness). Gains are typically a few points, and a 2025-26 TMLR study finds that on modern LLMs only plain Task Arithmetic reliably beats the base. For LoRA, combining A and B separately creates cross-terms. Merge in ΔW space or concatenate (rank grows). With more than two skills, data mixing tends to win.

### Cited Findings
- DARE, "Language Models are Super Mario" (Yu et al., arXiv 2311.03099; full-FT 13B WizardLM/WizardMath/WizardCoder on Llama-2): merging WizardLM + WizardMath raised WizardLM GSM8K from 2.2 to 66.3 (WizardMath alone: 64.2). Task Arithmetic + DARE gave +3.10% (LM&Math&Code vs LM) on AlpacaEval, +3.18% (LM&Math vs Math) on GSM8K, and +19.57% (LM&Code vs Code) on MBPP (v3 text; numbers differ between versions). A 7B merge failed because WizardLM-7B and WizardMath-7B had different backbones (llama-1 vs llama-2). — [arXiv 2311.03099](https://arxiv.org/html/2311.03099v3); [GitHub](https://github.com/yule-BUAA/MergeLM/blob/main/README.md)
- DELLA (arXiv 2406.11617; same LM/Math/Code experts; AlpacaEval/GSM8K/MBPP): on average +2.4 pts over pruning baselines (≈ +1.2 over DARE, +3.6 over TIES) and +11.1 over plain TA. It wins in 3 of 4 merges. Random dropping can hurt precise math relative to TIES top-k. — [arXiv 2406.11617](https://arxiv.org/abs/2406.11617v1)
- "A Systematic Study of In-the-Wild Model Merging for LLMs" (Hitit, Girrbach, Akata; arXiv 2511.21437; TMLR 03/2026): 4 open-weight bases × 12 fine-tuned checkpoints with overlapping or conflicting objectives, 6 methods, 16 benchmarks. "Task Arithmetic is the only method" that reliably gave gains. Interference-aware and subspace methods (TIES-like, etc.) generally showed no notable improvement over the base. — [arXiv 2511.21437](https://www.arxiv.org/abs/2511.21437)
- "What Matters for Model Merging at Scale?" (Yadav et al., arXiv 2410.03617; PaLM-2 1B-64B, full FT): merging works better from instruction-tuned bases than from pretrained ones, and better at larger scale and with more experts. At scale, the choice of merge method matters little. Per a secondary summary, merging more than 6 large IT experts often generalised better than multitask training (not confirmed in primary text). — [arXiv 2410.03617](https://arxiv.org/html/2410.03617v1); [summary](https://www.emergentmind.com/papers/2410.03617)
- "Mix Data or Merge Models?" (Aakanksha et al., Cohere, arXiv 2410.10801, NeurIPS 2024 workshop): objective-based merging (safety vs general) beat data mixing by up to 8% on general performance and 10% on safety. Language-based merging gave +4% general and −7% harm on top of data mixing. SLERP was best (+7% general, −3.1% harm). Not all merge algorithms helped. — [arXiv 2410.10801](https://arxiv.org/html/2410.10801v1); [Cohere](https://cohere.com/research/papers/mix-data-or-merge-models-optimizing-for-diverse-multi-task-learning-2024-10-15)
- LoRA Soups (Prabhakar et al., arXiv 2410.13025; Llama-7B LoRAs, GSM-Hard): base 5.91, code LoRA 8.04, math LoRA 14.18, DATA-MIX 18.8, CAT (learned concatenation of LoRAs, layer-wise weights) 21.11 (21.63 ± 1.07 over 3 seeds). TIES and DARE fell below DATA-MIX (figure only). The abstract claims CAT beats merging by 43% and data mixing by 12% on average. With 3 prompt formats, DATA-MIX beat all merging methods, and the authors conclude that data mixing is likely better as the number of skills grows. — [arXiv 2410.13025](https://arxiv.org/html/2410.13025v2)
- KnOTS (arXiv 2410.19735): SVD-aligns the concatenated LoRA updates per layer, then applies TIES/DARE. Up to +4.3% across vision and language. On Llama3-8B NLI LoRAs, KnOTS-TIES gives +2.9% average normalised accuracy over baseline merging. — [arXiv 2410.19735](https://arxiv.org/pdf/2410.19735)
- LoRA cross-terms: averaging A and B separately, then multiplying, introduces cross-terms A_s·B_t (s≠t). Concatenated merging reduces interference better than linear merging (LoRI, arXiv 2504.07448). "Crowded in B-Space" (arXiv 2604.16826) finds cross-task overlap much larger in B than in A, so B is the main source of merge interference, and proposes data-free calibration of B. AdapterHub offers a ΔW merge followed by SVD back to A,B as the alternative to factor averaging. — [LoRI](https://arxiv.org/html/2504.07448v2); [Crowded in B-Space](https://arxiv.org/html/2604.16826); [AdapterHub docs](https://docs.adapterhub.ml/merging_adapters.html)

### Inferences
- Our adapters are LoRA r=32 on one base, so any merge should be done on ΔW_i = (α/r)·B_i A_i, summed in full-weight space, then served merged or SVD-truncated back to rank ≤ 64-96. Do NOT average A and B factor-wise. Interpolating the factors of v4 and rl1 is not a linear interpolation of ΔW: it is quadratic in λ and adds cross-terms. If rl1 continues the v4 adapter (same A,B init), the factors are close and the error is small, but merging ΔW is still the clean way. Concatenation is exact: rank grows to the sum of ranks, e.g. 64 for two adapters.
- We do not currently have IF-, math- and code-specialist adapters. Training them would cost three extra SFT runs, and the evidence (LoRA Soups with 3 formats, the TMLR in-the-wild study) says a well-mixed v4 is hard to beat by merging at 9B. Expected gain at best ≈ +1-3 pts on one domain, with losses elsewhere. Low priority given our budget.

### Gaps
- No found paper merges IF + math + code LoRA experts at 7-9B and reports IFEval/MATH/MBPP against a multi-task LoRA. DELLA/DARE use full-FT 13B. KnOTS 7-8B results are NLI only. Model Stock has been validated mainly on CLIP ([arXiv 2403.19522](https://arxiv.org/abs/2403.19522)). I found no LLM-post-training numbers for it beyond its use as a baseline.

## Q4. Soups across different data-mix SFT runs from the same base (v1..v4)

### Takeaway
Uniform soups need a shared init and runs that stay in one basin. For different data mixes, positive evidence is mostly "merge beats mix" for distinct objectives, plus Task Arithmetic being the only reliable method in the wild. Expect soups of v1..v4 to land near the average of their scores unless the versions are complementary. Extrapolating between separately trained SFT versions is explicitly reported not to help.

### Cited Findings
- ExPO authors: two separately trained SFT or RLHF models (same or different base) do not benefit from extrapolation, because alignment is not monotonic along the path between them. Models from different bases can collapse. — [arXiv 2404.16792](https://arxiv.org/html/2404.16792v2)
- Task Arithmetic is the only method that reliably gives gains across 12 overlapping or conflicting fine-tunes per base. TIES and subspace methods generally show no notable gain over base. — [arXiv 2511.21437](https://www.arxiv.org/abs/2511.21437)
- Merging works better with instruction-tuned starting points and at larger scale. Small models benefit less. — [arXiv 2410.03617](https://arxiv.org/html/2410.03617v1)
- Merging separately trained objective-specific models beat a single mixed-data run by 4-10% in the Cohere multilingual and safety study. — [arXiv 2410.10801](https://arxiv.org/html/2410.10801v1)
- Model Stock: with only 2 fine-tuned models plus the pretrained anchor, layer-wise interpolation toward the anchor (ratio set by the angle between the two fine-tuned deltas) matched or beat a 48-model uniform soup on CLIP. It is available in mergekit as `model_stock`. — [arXiv 2403.19522](https://arxiv.org/html/2403.19522v2)

### Inferences
- A cheap test that fits our budget: Task Arithmetic or Model Stock of ΔW_v4 with the next-best ΔW_v3 (e.g. 0.7/0.3), evaluated once. It helps only if v3 is better than v4 on some benchmark. If v4 dominates v1-v3 everywhere, a soup will most likely just dilute v4.
- Including ΔW_v1 or v2 (worse data mixes) is unlikely to help and adds risk.

### Gaps
- I found no numbers for soups of different data-mix SFT LoRAs at 7-9B. The Olmo 3 "souping" details are unverified (see Q1 Gaps).

## Q5. Long-to-short / reasoning-length merging (eval cap 4,096 tokens)

### Takeaway
Merging a long-CoT model with a short-CoT model from the same base (task vectors, TA/TIES) cuts response length by about 50-55% at 7B with accuracy parity or marginal gains. It works poorly at 1.5B and better at larger scale. This matters only if rl1 makes outputs grow toward the 4,096 cap.

### Cited Findings
- "Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging" (Wu, Yao et al., arXiv 2503.20641): task-vector methods (TA, TIES) give ≈50% length reduction with accuracy parity or marginal gains on 7B (Qwen2.5-Math-7B + DeepSeek-R1-Distill-Qwen-7B). SVD-based methods are moderate. Small (1.5B) models struggle to gain long-CoT ability through merging. Efficacy correlates strongly with scale. Headline: "up to 55%" length reduction (another version says 53%). Tested at 1.5B/7B/14B/32B. — [arXiv 2503.20641](https://arxiv.org/abs/2503.20641v2); [HF](https://huggingface.co/papers/2503.20641); [code](https://github.com/hahahawu/Long-to-Short-via-Model-Merging)
- ExPO raises output length with α, a length-bias failure mode. — [arXiv 2404.16792](https://arxiv.org/html/2404.16792v2)

### Inferences
- If rl1 lengthens math outputs so that truncation at 4,096 tokens starts costing accuracy, interpolating rl1 back toward v4 (λ≈0.5-0.8) is the cheap long-to-short lever, since v4 is our "short" model. Conversely, ExPO past rl1 may lengthen outputs and increase truncation. Measure the truncation rate for every merge candidate.

### Gaps
- Per-size accuracy and length tables were not extracted. No found study covers length-merging for RLVR'd LoRA models.

## Q6. Practical recipe and typical gain magnitudes (summary for decision)

### Takeaway
Typical gains from merging or extrapolation in the reviewed papers are small. ExPO adds 1-4 LC-WR points on chat metrics and has mixed effects on GSM8K (−7.7 to +16). Expert merges add about 2-3 points over baselines (DELLA vs DARE/TIES), and up to about 10% in specific conflicting-objective settings. Merging after RLVR on the RL domain ranges from about −0.5 to +0.1 (RELEX/ExPO tables). For our budget, the defensible order is: (1) pick the best rl1 checkpoint on a held-out dev split; (2) optionally evaluate one ΔW interpolation rl1↔v4 and one small-α ExPO; (3) skip training specialist adapters.

### Cited Findings
- Magnitudes: ExPO 7-8B LC-WR +0.8 to +4.4 at α 0.1-0.5 ([2404.16792](https://arxiv.org/html/2404.16792v2)). DELLA +2.4 avg over DARE/TIES on AlpacaEval/GSM8K/MBPP ([2406.11617](https://arxiv.org/abs/2406.11617v1)). DARE TA merges +3.1-3.2% on AlpacaEval/GSM8K ([2311.03099](https://arxiv.org/html/2311.03099v3)). KnOTS +2.9% at 8B NLI ([2410.19735](https://arxiv.org/pdf/2410.19735)). Long-to-short −50% length at parity, 7B ([2503.20641](https://arxiv.org/abs/2503.20641v2)). Two-point ExPO on RLVR Qwen3-8B MATH 85.5 vs real RLVR 88.5 ([2605.21468](https://arxiv.org/html/2605.21468v1)).
- Tooling: mergekit implements linear, SLERP, task_arithmetic, TIES, DARE(-TIES), DELLA and model_stock. Merges are computed on full weights, so LoRAs should be merged into the base first. Methods named per the papers above. I did not fetch the mergekit docs this session.

### Inferences
- Concrete minimal plan, consistent with the "one best guess plus one improvement round" constraint:
  1. Pick the rl1 checkpoint (e.g. every 25-50 steps) by a held-out dev mix of IF/math/code, not by the public test sets.
  2. One interpolation: θ = θ_base + ΔW_v4 + λ(ΔW_rl1 − ΔW_v4) with λ ∈ {0.5}. This is a hedge if IFEval/MBPP regress or lengths approach 4,096.
  3. One extrapolation: λ = 1.2-1.3 (ExPO α 0.2-0.3), only if the rl1 reward is still rising at the end. Keep it only if no benchmark drops by more than about 1 pt.
  4. All merges in ΔW space (merge each LoRA into the base weights, then combine), never factor-wise A/B averaging.
- Expected outcome: most likely within ±1 pt of the best rl1 checkpoint on each benchmark. The upside is a small hedge against RL over-optimisation, not a major gain. Greedy decoding removes the pass@k benefit that motivated WiSE-FT for reasoning.

### Gaps
- No source gives IFEval or MBPP deltas for any SFT↔RL interpolation or extrapolation at 7-9B. All numbers above are on adjacent metrics (AlpacaEval, MT-Bench, GSM8K, MATH, GSM-Hard, NLI), so transfer to our benchmarks is uncertain.

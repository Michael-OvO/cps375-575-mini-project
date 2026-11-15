# Teaching 7-9B open models competition math (MATH / MATH-500) via SFT on teacher-generated data

Conventions: "MATH" = Hendrycks MATH test accuracy (5,000 problems unless stated), zero-shot, greedy, text-only, unless marked otherwise. "val" = a held-out slice of the MATH *training* split used for ablations. Numbers are as extracted from the papers' HTML text via a fetch tool; values that exist only in figures are flagged as not retrieved. Contamination notes are marked **[CONTAM]**.

## 1. OpenMathInstruct-1 and OpenMathInstruct-2 (NVIDIA): what the ablations found

### Takeaway
For an 8B student, OpenMathInstruct-2 found that (a) a concise text CoT format beats a verbose one (+3.9 pts with 40% fewer tokens), (b) teacher strength matters a lot (+7.8 pts for 405B vs 8B teacher at equal size), (c) number of *unique questions* matters far more than more solutions for the same questions (>10 pts lost at 1K vs 6.5K unique questions at fixed 256K pairs), and (d) filtering wrong solutions with LLM judges gave no measurable gain; the final Llama3.1-8B reached 67.8% MATH with 14M pairs. OpenMathInstruct-1 (code-interpreter format, Mixtral teacher) reached only ~44% MATH on 7B models, and showed that balanced ("fair") per-problem sampling and hiding the reference answer from the teacher both help.

### Cited Findings

**OpenMathInstruct-2 (Llama3.1-405B-Instruct teacher, Llama3.1-8B-Base student, text CoT, no code)**
- Solution format ablation (MATH val, same size): verbose "Llama CoT" 40.6 +/- 0.6 at 331.3 mean tokens vs concise "OpenMath CoT" 44.5 +/- 0.8 at 237.0 mean tokens. Shorter (about 40% fewer tokens) and +3.9 pts. — [OpenMathInstruct-2, Table 1](https://arxiv.org/html/2410.01560)
- The concise format is described as "detailed as well but less verbose"; it was elicited with few-shot "Question: ... / My solution: ..." prompts. With the instruct chat template about 57% of generations reverted to the verbose native format; using the base (no role tokens) template cut that to 0.1%. Post-processing truncates after the first sentence containing `\boxed` and drops solutions with multiple `\boxed` entries. — [OpenMathInstruct-2, Sec 2 / App A.2](https://arxiv.org/html/2410.01560)
- Teacher strength (same pipeline, MATH val): data from Llama3.1-8B-Base 30.1 +/- 0.6 (mean length 205.7) vs Llama3.1-405B-Instruct 37.9 +/- 0.6 (mean length 180.2). 45-67% of the 8B teacher's data was flagged low quality by the filters. — [OpenMathInstruct-2, Table 2](https://arxiv.org/html/2410.01560)
- Question diversity: at a fixed 256K question-solution pairs, varying unique questions over 1K / 2K / 4K / 6.5K; accuracy rises with unique questions, with a ">10 point" drop at 1K and a quoted 10.5% gain from 1K to 6.5K. (Implied solutions per question about 256 / 128 / 64 / 39; that arithmetic is not stated in the paper.) — [OpenMathInstruct-2, Sec 2.2.4](https://arxiv.org/html/2410.01560)
- Filtering wrong solutions (128K scale, MATH val): unfiltered 43.6 +/- 1.7; LLM-judge prompt 1 (113K kept) 43.6 +/- 0.1; LLM-judge prompt 2 (116K) 43.0 +/- 0.8; Nemotron-4-340B-Reward helpfulness>=3 (118K) 43.8 +/- 0.4; correctness>=3 (120K) 43.1 +/- 0.4. Filters removed 6-12% of data and none beat unfiltered; manual check of 20 flagged items found only about 60% truly wrong. — [OpenMathInstruct-2, Table 3](https://arxiv.org/html/2410.01560)
- Noise robustness: deliberately adding wrong-answer solutions or mismatched question-solution pairs caused little to no degradation up to 20% incorrect data at >=256K scale; robust even at 40% for mismatched pairing. — [OpenMathInstruct-2, Sec 2.2.3](https://arxiv.org/html/2410.01560)
- Majority-vote agreement threshold for synthetic questions (32 samples at T=0.7, majority answer used as pseudo ground truth): min-votes 0 -> 381K pairs, 50.1; 8 -> 339K, 49.2; 16 -> 254K, 44.4; 24 -> 160K, 42.0. Stricter agreement lowered accuracy (it removes data, and likely the harder questions); threshold 0 was chosen. — [OpenMathInstruct-2, Table 9](https://arxiv.org/html/2410.01560)
- Final dataset: 607.3K unique questions, 13.97M pairs. MATH solution augmentation 7.4K questions / 2.46M pairs; MATH question+solution augmentation 519.1K questions / 8.94M pairs; GSM8K solution augmentation 7.4K / 0.46M; GSM8K question augmentation 73.6K / 2.11M. — [OpenMathInstruct-2, Table 5](https://arxiv.org/html/2410.01560)
- Size scaling: 8B trained on 1M / 2M / 5M / 14M fair-downsampled subsets; consistent gains, "no sign of saturation" at 14M. — [OpenMathInstruct-2, Sec 3](https://arxiv.org/html/2410.01560)
- Final results (greedy): OpenMath2-Llama3.1-8B GSM8K 91.7 / MATH 67.8 (maj@256: 76.1); Llama3.1-8B-Instruct 84.2 / 51.8; NuminaMath-7B-CoT 75.4 / 55.2; Qwen2.5-Math-7B-Instruct 95.2 / 83.6; OpenMath2-Llama3.1-70B 94.9 / 71.9 vs Llama3.1-70B-Instruct 67.9. — [OpenMathInstruct-2, Table 4](https://arxiv.org/html/2410.01560)
- 70B student gained little; authors suspect the data blend / solution format was tuned for the weaker 8B student. — [OpenMathInstruct-2, Limitations](https://arxiv.org/html/2410.01560)
- Training recipe: ablations 4 epochs, batch 256, AdamW, constant LR 5e-6, 4 runs averaged. Final: 2 epochs, batch 512, LR 2e-5 (1e-5 for 70B), weight decay 1e-2, 6 equally spaced checkpoints averaged (checkpoint averaging gave >2% in one ablation). — [OpenMathInstruct-2, Sec 3 / App](https://arxiv.org/html/2410.01560)
- **[CONTAM]** Seeds are MATH/GSM8K *training* questions only, yet 50K of 569K (about 9%) teacher-synthesized MATH-style questions had to be removed as near-duplicates/paraphrases of test problems (embedding top-5 retrieval + LLM paraphrase judge against GSM8K, MATH, AMC 2023, AIME 2024). About 1.4% of Omni-MATH test questions appear in training data (released after training). — [OpenMathInstruct-2, Sec 2.3](https://arxiv.org/html/2410.01560)

**OpenMathInstruct-1 (Mixtral-8x7B teacher, code-interpreter solutions)**
- 1.8M problem-solution pairs from GSM8K + MATH train (plus 6.6M incorrect samples released separately). Coverage (>=1 correct solution): MATH 93.0%, GSM8K 99.9%. 787K unique solutions for 6,978 of 7,500 MATH problems. — [OpenMathInstruct-1](https://arxiv.org/html/2402.10176)
- Masked-text prompting (few-shot reference solutions with intermediate numbers masked, so the teacher cannot copy the answer): MATH coverage 80.1% (default) -> 85.9% (masked) -> 87.5% (masked + subject-specific few-shots); unique solutions 177K -> 192K -> 227K. — [OpenMathInstruct-1, Table 2](https://arxiv.org/html/2402.10176)
- Fine-tuning ablations (Mistral-7B, 1K val subsets, GSM8K / MATH): naive downsampling 74.3 / 35.0 vs fair (per-problem balanced) 75.3 / 37.0; default-prompt data 73.8 / 36.9 vs masked-prompt data 77.7 / 37.4; size 128K 75.3 / 37.0 -> 256K 79.0 / 38.6 -> 512K 81.0 / 41.6. — [OpenMathInstruct-1, Sec 4 ablations](https://arxiv.org/html/2402.10176)
- Code vs text: preferring code solutions ("Any-Code") improved greedy MATH 37.4 -> 39.4 but lowered self-consistency (45.2 -> 42.6). On val, problems the model answered with text only scored 32.0% vs 45.3% with code+text (overall 41.6%). — [OpenMathInstruct-1](https://arxiv.org/html/2402.10176)
- Final greedy (with code execution, so not comparable to a no-tools setting): OpenMath-Mistral-7B GSM8K 80.2 / MATH 44.5; OpenMath-CodeLlama-7B 75.9 / 43.6; CodeLlama-70B 84.6 / 50.7. MATH Level 1 72.4% vs Level 5 16.3%. — [OpenMathInstruct-1, Table 3](https://arxiv.org/html/2402.10176)

### Inferences
- With a strong teacher and a limited token budget, the OpenMathInstruct-2 evidence points to spending tokens on *new questions with one concise solution each* before adding many solutions per original problem, and to not spending budget on LLM-judge filtering of solutions (answer-match against ground truth for training problems is free and was always applied).
- The majority-vote threshold result means high teacher self-agreement is not a useful inclusion criterion for synthetic questions; it mostly discards hard items. Majority answer as label with no minimum agreement was best.
- The concise format helping the 8B student but not the 70B student is consistent with the "learnability gap" literature (Section 3): format should be tuned to the student.
- OpenMathInstruct-1's lesson that transfers to a text-only setting is the sampling one: balance solutions across problems ("fair") and do not let the teacher see the final answer or copyable intermediate numbers.

### Gaps
- Exact MATH scores at 1M / 2M / 5M for the 8B model and per-point values of the question-diversity curve are only in figures; not retrieved.
- OpenMathInstruct-2 has no code-vs-text ablation (text only); OpenMathInstruct-1 has no pure-text-trained model, so there is no clean same-teacher code-vs-text comparison for a no-tools student.
- No ablation in either paper isolates "k solutions per problem" at fixed unique-question count for MATH.

## 2. Rejection-sampling fine-tuning (RFT): value of multiple distinct correct reasoning paths

### Takeaway
On GSM8K (the RFT paper has no MATH experiments), adding distinct correct reasoning paths per training question gave +5-6 pts for 7B/13B models with one sampler and up to +13 pts when pooling paths from several samplers, but returns diminish quickly per doubling of k and vanish for stronger students. On MATH, DART-Math shows vanilla rejection sampling is biased to easy problems and that the fix (many responses for hard problems) is worth +5-7 pts on 7-8B general models.

### Cited Findings
- SFT vs RFT (k=100 samples per question, dedup by equation list), GSM8K greedy: LLaMA-7B 35.9 -> 41.7; LLaMA2-7B 41.6 -> 47.5; LLaMA-13B 43.0 -> 49.1; LLaMA2-13B 50.0 -> 54.8; LLaMA-33B 54.6 -> 54.5. — [Yuan et al. 2023, Tables 1/6](https://ar5iv.labs.arxiv.org/html/2308.01825)
- Effect of k (LLaMA-7B accuracy / distinct paths per question): k=1 37.6 / 1.17; k=3 39.0 / 1.44; k=6 39.5 / 1.74; k=12 41.6 / 2.20; k=25 40.9 / 2.93; k=50 40.7 / 3.94; k=100 41.7 / 5.25. Accuracy is flat/noisy beyond k about 12 (about 2.2 distinct paths per question). — [Yuan et al. 2023, Tables 2/5](https://ar5iv.labs.arxiv.org/html/2308.01825)
- Dedup ablation: k=100 without dedup (about 400K samples) vs with dedup (about 47K): 7B 43.6 vs 41.7; 7B-2 46.7 vs 47.5; 13B 46.9 vs 49.1; 13B-2 53.7 vs 54.8. Similar accuracy with about 8.5x fewer samples; near-duplicate paths add essentially nothing. — [Yuan et al. 2023, Table 5](https://ar5iv.labs.arxiv.org/html/2308.01825)
- Pooling paths from multiple sampler models (RFT-U13B: 104K samples, 12.84 distinct paths per question): LLaMA-7B 35.9 -> 49.3; LLaMA2-7B 41.6 -> 50.3; LLaMA-13B 43.0 -> 52.1; LLaMA2-13B 50.0 -> 55.4; LLaMA-33B 54.6 -> 56.5; LLaMA-65B 59.3 -> 59.0; LLaMA2-70B 63.2 -> 62.3. Adding a 33B sampler raised distinct paths only to 13.65. — [Yuan et al. 2023, Tables 2/5](https://ar5iv.labs.arxiv.org/html/2308.01825)
- Log-linear data relation: LLaMA-7B GSM8K at 1/32, 1/16, 1/8, 1/4, 1/2, full data = 7.8, 12.7, 16.5, 22.7, 30.9, 35.9. — [Yuan et al. 2023, Sec 3.2](https://ar5iv.labs.arxiv.org/html/2308.01825)
- MuggleMath (GSM8K): for 7B students, performance stops improving once responses reach about 97K (about 3 per query); the 13B student kept improving 37K -> 157K. Query augmentation follows log-linear fits (LLaMA-7B: y = 10.7 log(x) + 13.2; LLaMA2-7B: y = 9.8 log(x) + 21.3), i.e. about +7 pts per doubling of queries. Majority-vote filtering of responses generally *lowered* accuracy at equal data scale. — [MuggleMath](https://arxiv.org/html/2310.05506)
- MetaMath: adding 20K more answer-augmentation (more solutions to the same questions) gave about +0.1% vs +0.4% (rephrasing), +2.3% (FOBAR), +2.6% (self-verification) for 20K new bootstrapped questions; diversity-gain vs accuracy correlation 0.972. Mixing in the external RFT dataset (about 47K) *hurt* at all MetaMathQA sizes 20K-100K. — [MetaMath, Sec 4.5/4.8](https://arxiv.org/html/2309.12284)
- Bias of vanilla rejection sampling on MATH: in MetaMathQA-MATH-AnsAug, 51.1% of the hardest (Level 5) training queries received zero new responses and Level 5 responses are only 10.5% of samples. Level-5 query coverage: ToRA-Corpus-16k 68.0%, MetaMath-AnsAug 48.9%, vanilla rejection tuning (VRT) 62.9%, DART-Math 99.1%. — [DART-Math, Table 4 / Sec 1](https://arxiv.org/html/2407.13690)
- DART-Math vs VRT at equal about 590K size, MATH greedy: Llama3-8B VRT 39.7 / Uniform 45.3 / Prop2Diff 46.6; Mistral-7B 38.7 / 43.5 / 45.5; DeepSeekMath-7B 53.0 / 52.9 / 53.6; Llama3-70B 53.1 / 54.9 / 56.1. — [DART-Math, Table 2](https://arxiv.org/html/2407.13690)

### Inferences
- "Distinct" matters, not count: 2-3 genuinely different correct solutions per problem capture most of the per-problem benefit for a 7B student; beyond that, new or harder questions give more per token (RFT k-curve, MuggleMath 3-per-query saturation, MetaMath diversity result, OpenMathInstruct-2 unique-question result all agree).
- The RFT gain shrinks as the student gets stronger (7B +13.4, 13B +9.1, 33B +1.9, 65B -0.3 with U13B data). A modern 8-9B model is far stronger on GSM8K than LLaMA-1/2, so expect the "more paths for the same easy problems" lever to be small; the DART result says the remaining value is concentrated in hard problems.
- All RFT numbers are GSM8K; applying the saturation point to MATH is extrapolation. DART-Math is the closest MATH evidence and suggests hard MATH problems can absorb far more responses (it averages about 107 responses per Level 5 query in the Hard set) without saturating at about 590K samples.

### Gaps
- No MATH experiment in the RFT paper; no study found that sweeps k = 1, 2, 4, 8 distinct solutions per MATH training problem at fixed total tokens with a strong teacher.
- Whether multiple samplers (teacher diversity) help on MATH as they did on GSM8K was not found.

## 3. Rationale style: length, granularity, teacher-student gap, and difficulty weighting

### Takeaway
For 7-8B students the evidence favours solutions that are complete but not padded: a 40%-shorter concise CoT beat verbose CoT on Llama3.1-8B (OpenMathInstruct-2), while very long "reasoning-model" traces only start to pay off at about 7B and clearly hurt at <=3B (Small Model Learnability Gap). A stronger teacher is better for 7-8B students (unlike <=3B), and biasing the response budget toward hard problems (DART Prop2Diff) adds about +1-2 MATH pts over uniform and +5-7 over vanilla rejection sampling on general 7-8B models, at a small cost on easy benchmarks (GSM8K -1.4).

### Cited Findings

**Length / verbosity**
- Llama3.1-8B-Base student: concise OpenMath CoT (237 tokens mean) 44.5 vs verbose Llama CoT (331 tokens) 40.6 on MATH val. — [OpenMathInstruct-2, Table 1](https://arxiv.org/html/2410.01560)
- Long CoT (QwQ-32B-Preview style, mean 3,384.7 tokens) vs short CoT (Qwen2.5-32B-Instruct style, mean 575.7 tokens), students fine-tuned on 7.5K MATH train problems, MATH accuracy long / short: Qwen2.5-0.5B 23.0 / 31.5; 1.5B 41.6 / 52.3; 3B 56.2 / 61.0; **7B 68.2 / 67.8**; 14B 78.3 / 76.2; 32B 84.8 / 82.3. Llama-3.2-3B 48.7 / 50.9; **Llama-3.1-8B 50.0 / 44.6** (+5.4 for long). — [Small Models Struggle to Learn from Strong Reasoners, Tables 1/8](https://arxiv.org/html/2502.12143)
- Mix-Long (long:short = 1:4, alpha = 0.2; mean 1,248.9 tokens) on Qwen2.5-3B: MATH 64.7 vs 56.2 (long only) vs 61.0 (short only); Llama3.2-3B: 53.0 vs 48.7 / 50.9. Reported only for 3B students. — [Small Models Struggle..., Table 3](https://arxiv.org/html/2502.12143)
- Hyperparameter check (Qwen2.5-1.5B): long CoT underperformed short CoT at every LR (1e-4 to 5e-6) and epoch (2-5) setting tried; best long 0.286 vs short 0.341 average. — [Small Models Struggle..., Table 12](https://arxiv.org/html/2502.12143)
- Perplexity under the student: Qwen-0.5B assigns PPL 2.237 to long CoT vs 1.278 to short CoT; gap shrinks with student size. — [Small Models Struggle..., Table 10](https://arxiv.org/html/2502.12143)
- Xwin-Math models per-step accuracy as Acc_final = Acc_step^s (s = number of CoT steps): estimated single-step accuracy on GSM8K rises 78.9% (7.5K data) -> 89.7% (120K) -> 94.2% (960K). Resampling the 960K data toward longer-CoT (harder) problems raised PassRatio@256 from 71.1 to 72.8, improving hard problems while leaving easy ones unchanged. — [Xwin-Math, Table 4 / Sec 4](https://arxiv.org/html/2403.04706)

**Teacher strength / capability gap**
- Large (Qwen2.5-72B) vs small (Qwen2.5-3B) teacher, both short CoT, MATH accuracy large / small: student 0.5B 30.0 / 31.0; 1.5B 50.3 / 50.7; 3B 57.5 / 60.3; **7B 71.3 / 63.6 (+7.7)**; 14B 76.4 / 72.8; 32B 80.5 / 76.8. Llama 8B student: average over 5 benchmarks 25.6 vs 25.1 (+0.5). — [Small Models Struggle..., Tables 2/7](https://arxiv.org/html/2502.12143)
- Mix-Large (large:small teacher = 1:4) on Qwen2.5-3B: MATH 65.8 vs 57.5 (strong only) / 60.3 (weak only). — [Small Models Struggle..., Table 3](https://arxiv.org/html/2502.12143)
- Same-pipeline teacher comparison for an 8B student: 405B teacher 37.9 vs 8B teacher 30.1 on MATH val. — [OpenMathInstruct-2, Table 2](https://arxiv.org/html/2410.01560)
- MuggleMath (GSM8K, LLaMA-2-7B): augmented queries answered by GPT-4 give 57.0-58.0 vs 46.7 when responses come from GPT-3.5. — [MuggleMath, Table 2](https://arxiv.org/html/2310.05506)
- Counter-evidence that correctness is not everything: MathScale found 100% GPT-3.5 solutions (69% correct on a 100-sample manual check) scored 11.5 macro vs 10.9 for 100% GPT-4 solutions (87% correct) at 5K examples on LLaMA-2-7B, and dropped validation. Very small scale (5K), differences about 0.6 pts. — [MathScale, Table 7](https://arxiv.org/html/2403.02884)
- A 2026 re-examination reports that in most configurations on the five math benchmarks, CoT distillation *degraded* students relative to their own pre-distillation baseline (instruct students, same prompt), and that when teachers differ substantially in accuracy one should prefer the stronger teacher. Per-benchmark values are only in figures; its teacher-selection guidance comes from BBH, not math. — [Revisiting the Capacity Gap (arXiv 2604.08880)](https://arxiv.org/html/2604.08880v1)
- BRIDGE (curriculum + structure-aware masking) claims +11.29% accuracy and 27.4% shorter outputs on GSM8K for Qwen2.5-3B, arguing plain SFT on verbose CoT hurts small models. Search-snippet level only; not verified in the paper. — [BRIDGE (arXiv 2602.17686)](https://arxiv.org/abs/2602.17686)

**Difficulty-aware data (DART-Math)**
- Setup: only MATH + GSM8K *training* queries (no new questions); responses sampled from DeepSeekMath-7B-RL (T=1.6, top-p 0.95, max 2048 tokens), kept if the answer matches ground truth. Difficulty = fail rate of the sampler. Uniform = fixed number of correct responses per query (k=40); Prop2Diff = number of correct responses proportional to difficulty (up to 192). Sizes 591K / 585K. — [DART-Math](https://arxiv.org/html/2407.13690)
- MATH / GSM8K greedy: Llama3-8B Uniform 45.3 / 82.5, Prop2Diff 46.6 / 81.1, VRT 39.7 / 81.7; Mistral-7B 43.5 / 82.6, 45.5 / 81.1, 38.7 / 82.3. Six-benchmark average gain over VRT for Llama3-8B: +4.5 (Prop2Diff), +3.5 (Uniform). For the math-pretrained DeepSeekMath-7B and Llama3-70B the gain is about 1 pt. — [DART-Math, Table 2](https://arxiv.org/html/2407.13690)
- Same-base comparisons on Llama3-8B: MetaMathQA (0.40M) 32.5 MATH; MMIQC (2.3M) 39.5; DART-Hard (0.59M) 46.6. Mistral-7B: MetaMathQA 29.8, MMIQC 37.4, KPMath-Plus (1.6M) 46.8, DART-Hard 45.5. — [DART-Math, Table 2](https://arxiv.org/html/2407.13690)
- Coverage ablation: forcing at least one correct response per query matters most at small data sizes (about +8 pts on GSM8K) and fades as data grows; adding that constraint to VRT did not significantly help MATH. — [DART-Math, Fig 4](https://arxiv.org/html/2407.13690)
- Cost: reaching the per-query target for >90% of queries took about 5M raw samples for Uniform and >15M for Prop2Diff; full pipeline about 150M raw samples (about 160 A100 GPU-days) with a 7B sampler. — [DART-Math, Sec 4](https://arxiv.org/html/2407.13690)
- Scaling: MATH accuracy rises steadily with log data size from thousands to about 1M; >20-pt total gain for Mistral-7B / Llama3-8B but only about 3 pts for DeepSeekMath-7B, which already exceeds 50% with thousands of samples. — [DART-Math, Fig 3](https://arxiv.org/html/2407.13690)
- Training/eval format: Alpaca-style "### Instruction / ### Response" template, zero-shot, greedy, >=3 seeds; grading by regex extraction + SymPy comparison. — [DART-Math, App](https://arxiv.org/html/2407.13690)

**Distilling step-by-step and Fine-tune-CoT**
- Distilling Step-by-Step uses teacher rationales as an auxiliary multi-task target (predict label and rationale as separate tasks); a 770M T5 beat few-shot 540B PaLM using 80% of the data on one benchmark. Abstract-level only; no MATH-style results. — [Hsieh et al. 2023](https://arxiv.org/abs/2305.02301)
- Fine-tune-CoT ("Large Language Models Are Reasoning Teachers") reports that "diverse reasoning" (several distinct teacher rationales per question) gives a substantial boost across datasets even for very small students. Exact numbers not retrieved. — [Ho et al. 2023](https://arxiv.org/abs/2212.10071)

**NuminaMath**
- NuminaMath-CoT: 859,608 problems with CoT solutions. Sources: cn_k12 276,591; synthetic_math 167,895; orca_math 153,334; olympiads 150,581; synthetic_amc 62,111; aops_forum 30,201; math 7,478; gsm8k 7,345; amc_aime 4,072. Pipeline: OCR of exam PDFs -> segmentation -> translation to English -> "realignment to produce a CoT reasoning format" -> final answer formatting. Apache 2.0. — [NuminaMath-CoT dataset card](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- NuminaMath-7B (DeepSeekMath-7B-base): stage 1 CoT fine-tune, stage 2 tool-integrated reasoning. MATH zero-shot: 55.8% CoT (no tools) vs 68.1% TIR (with code). Hyperparameters listed: LR 2e-5, total batch 32, cosine schedule with 0.1 warmup, 4 epochs. Card notes greedy decoding handles about AMC-12 level and fails on harder AIME/olympiad problems; weak on geometry. — [NuminaMath-7B-TIR model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
- Independent re-evaluation: NuminaMath-7B-CoT GSM8K 75.4 / MATH 55.2 vs OpenMath2-Llama3.1-8B 67.8 MATH. — [OpenMathInstruct-2, Table 4](https://arxiv.org/html/2410.01560)

### Inferences
- For a 7-9B student the two "gap" results point the same way: use the strongest teacher, and have it write normal-length worked solutions (a few hundred tokens), not multi-thousand-token reasoning traces. At 7-8B long traces are roughly neutral (Qwen 7B +0.5) to positive (Llama 8B +5.4) on accuracy, but cost about 6x the training tokens (3,385 vs 576), so under a token budget short/concise is the better trade. This token-efficiency conclusion is an inference; no paper ran an equal-token comparison.
- The Llama-3.1-8B result favouring long CoT (50.0 vs 44.6) and the OpenMathInstruct-2 result favouring concise CoT (44.5 vs 40.6) are not contradictory: the former compares about 3.4K vs 0.6K tokens from different teachers on 7.5K examples; the latter compares 331 vs 237 tokens from the same teacher. Together they suggest an interior optimum (complete steps, no filler), and that results with only 7.5K examples may not hold at larger scale.
- "Teacher too strong" is a <=3B phenomenon in the cited data; at 7B the strong teacher wins by 7.7 MATH pts. A cheap hedge is the mix result (majority concise, minority detailed), although it was only tested on 3B students.
- Difficulty weighting has to be bought with samples when the sampler is a weak 7B model (15M+ raw samples). With a much stronger teacher, most hard training problems are solved in one or a few tries, so Prop2Diff-style allocation (more distinct solutions for Level 4-5 or for problems the *student* fails) is cheap. Using the student's own fail rate as the difficulty signal is an extrapolation; DART used the sampler's fail rate.
- DART's benefit is concentrated in models without math-specific pretraining; expect smaller gains if the student base is already strong on MATH (as with the 2026 "revisit" finding that distillation can even degrade a strong instruct student). Always measure against the un-fine-tuned student with the same prompt.
- The Xwin-Math step-accuracy model implies fewer, larger steps reduce the number of chances to fail but raise per-step difficulty; no paper cited here ablates step granularity directly.

### Gaps
- No direct ablation of step granularity (number of steps for the same solution) on MATH for 7-9B students was found.
- No equal-token-budget comparison of short vs long rationales was found.
- Mix-Long / Mix-Large results for 7-8B students are not reported.
- Distilling step-by-step and Fine-tune-CoT numbers were not retrieved beyond abstracts; both use small T5/GPT-3-family students on simpler tasks, so relevance to MATH is indirect.
- Student-aware or curriculum distillation with MATH numbers for 7-9B students: nothing verified (BRIDGE claim is GSM8K, 3B, and unverified).
- NuminaMath-CoT has no published ablations; whether its "math" 7,478 subset is train-only is not stated on the card.

## 4. Synthetic problem generation: does it help on MATH, and how is it verified

### Takeaway
New teacher-written questions seeded from MATH training problems are the single largest lever in the literature once the 7.5K training problems are exhausted: Xwin-Math took LLaMA-2-7B to 40.6 and Mistral-7B to 43.7 MATH with 480K synthetic MATH-style problems, and OpenMathInstruct-2 used 519K to reach 67.8 on Llama3.1-8B. GSM8K-style augmentation does not transfer to MATH. Verification is done by teacher self-consistency / majority vote or a solve-and-revise step, and ablations show verification is worth about 1 pt at most; decontamination against the test set is the step that cannot be skipped.

### Cited Findings
- MetaMathQA (GPT-3.5 teacher, 395K: answer augmentation 155K, rephrasing 130K, self-verification 55K, FOBAR 55K; MATH part 155K = 75K / 50K / 15K / 15K). LLaMA-2-7B trained on MATH augmentations only: SFT 4.7 -> AnsAug 12.9 -> Rephrasing only 12.4 -> AnsAug+Rephrasing 15.3 -> all four (with backward questions) 17.7. Full MetaMath: LLaMA-2-7B 19.8, Mistral-7B 28.2, Llemma-7B 30.0, 70B 26.6. — [MetaMath, Tables 1-3, 8](https://arxiv.org/html/2309.12284)
- MetaMath verification: rephrased questions were answered by GPT-3.5 with CoT and kept only if the answer matched the original ground truth (accuracy 76.30% on rephrased vs 80.74% on originals). Answer augmentation keeps only answer-matching solutions. Backward (SV/FOBAR) questions mask a number in the question and ask for it given the answer. — [MetaMath, Sec 3](https://arxiv.org/html/2309.12284)
- MetaMath, wrong-answer paths (GSM8K, LLaMA-2-7B): SFT 41.6; training on 7,473 incorrect-answer paths 43.6; correct-answer paths 52.2. — [MetaMath, Table 4](https://arxiv.org/html/2309.12284)
- Xwin-Math (GPT-4 Turbo teacher): generate a new question from a training reference question -> a single "solve, verify and modify" prompt -> generate CoT answer; up to 960K GSM8K-style and 480K MATH-style. Results: Xwin-Math-7B (LLaMA-2) 82.6 / 40.6; Mistral-7B 89.2 / 43.7; Llemma-7B 84.2 / 47.2; 13B 88.1 / 44.9; 70B 90.6 / 52.8. MATH evaluated on a 500-problem subset; sampling at T=0.7 reported as minimally different from greedy. — [Xwin-Math, Table 2](https://arxiv.org/html/2403.04706)
- Synthetic vs real at equal size (LLaMA-2-7B, MATH): 0.94K 4.2 real / 3.9 syn; 1.88K 5.6 / 4.9; 3.75K 6.6 / 6.0; 7.5K 8.4 / 7.9. Synthetic questions are nearly as good as real ones per example. Accuracy grows "almost linear or even super-linear" in log data size up to 480K. — [Xwin-Math, Table 1](https://arxiv.org/html/2403.04706)
- Question-verification ablation (MATH pass@1): with / without the verify step 28.9 / 28.1 at 7.5K; 37.6 / 36.6 at 30K; no significant effect on GSM8K. No solution-verification ablation. — [Xwin-Math, Table 5](https://arxiv.org/html/2403.04706)
- Augmentation-style comparison (all re-run with GPT-4 Turbo): add-constraint, change numbers, change background, MetaMath-style: differences negligible at 7.5K and 30K; at larger sizes "new question from reference" and "add constraint" are best. Exact values only in a figure. — [Xwin-Math, Fig 6](https://arxiv.org/html/2403.04706)
- Latent capability: LLaMA-2-7B trained on only 7.5K real MATH has Pass@256 of 72.0% but a per-sample pass ratio of 7.9%; more data mainly raises consistency, not coverage. — [Xwin-Math, Sec 2](https://arxiv.org/html/2403.04706)
- **[CONTAM]** Xwin-Math leakage check compares losses on regenerated vs reference test items (MATH: 0.59 vs 0.58; train 0.23) and concludes no leakage. — [Xwin-Math, Table 6](https://arxiv.org/html/2403.04706)
- OpenMathInstruct-2: 519.1K new MATH-style and 73.6K GSM8K-style questions from training seeds; labels = majority answer of 32 teacher samples at T=0.7; requiring more agreement hurt (Section 1). — [OpenMathInstruct-2](https://arxiv.org/html/2410.01560)
- MuggleMath: "increase complexity" was the best query-augmentation type on GSM8K. GSM8K augmentation gives "little to no help" on MATH: LLaMA-2-7B MATH 5.8 (SFT on MATH) vs 4.8-8.4 across multi-task / transfer variants. MATH-seeded augmentation (AugMATH, 300K = 7,500 x 5 x 8) does work: LLaMA-2-7B 5.8 -> 25.8, 13B 30.7, 70B 36.3; log-linear with a steeper slope after about 80K (y = -13.56 + 6.33 log x for 82.5K-307.5K). — [MuggleMath, Tables 5-6 / App A](https://arxiv.org/html/2310.05506)
- MathScale (GPT-3.5 teacher): concept-graph composition (2,018 topics, 8,892 knowledge points extracted from about 20K seed questions) -> 2M QA pairs. LLaMA-2-7B MATH 31.1 vs MetaMath-7B 20.6 on the same harness; Mistral-7B 35.2. Halving knowledge points cost 8.6% relative (macro 14.5 -> 13.2 at 25K); halving seed questions -2.9%. No validation of generated QA in the final pipeline (validation ablation showed no gain at 5K). — [MathScale, Tables 5-7](https://arxiv.org/html/2403.02884)
- **[CONTAM]** MathScale seeds from the MWPBench *training* set and decontaminates against the MWPBench test set. — [MathScale](https://arxiv.org/html/2403.02884)
- **[CONTAM]** OpenMathInstruct-2 removed about 9% of synthesized questions as paraphrases of test problems despite train-only seeds (Section 1). — [OpenMathInstruct-2](https://arxiv.org/html/2410.01560)
- DART-Math reaches 45-47 MATH on Mistral-7B / Llama3-8B with *no* new questions, matching Xwin-Math-Mistral (43.7) and KPMath-Plus-Mistral (46.8, 1.6M), using only about 590K responses to the original training questions. — [DART-Math, Table 2](https://arxiv.org/html/2407.13690)

### Inferences
- Ranking of data levers by evidence for MATH: (1) MATH-style new questions from MATH-train seeds (large, near log-linear, synthetic about as good as real per example); (2) distinct solutions to hard training problems (DART); (3) rephrasing / backward variants (small, a few points on weak bases); (4) GSM8K-style or word-problem augmentation (about zero transfer to MATH).
- Verification without ground truth: majority vote / self-consistency of a strong teacher as the label is the standard, and the measured value of extra checking is small (Xwin +0.8 to +1.0; OpenMathInstruct-2 filters +0; MathScale none; MuggleMath and OpenMathInstruct-2 majority-vote *thresholding* negative). The cheap, defensible policy is: take the teacher's (majority) answer, do not threshold hard on agreement, spend effort instead on diversity and decontamination.
- Because a teacher seeded with training problems still regenerates test-like problems (about 9% flagged in OpenMathInstruct-2) and because MATH comes from the same sources as its test split, a train-only pipeline cannot assume synthetic questions are test-free. The published fix is a decontamination pass against the test set (which itself reads test problems); if the test set may not be touched even for that, the remaining mitigation is instructing the teacher to write novel problems rather than recall known contest problems; this is a judgment, not a measured result.
- Pass@256 72% vs single-sample 7.9% suggests the student's bottleneck is reliability, which is why more consistent, uniform-format supervision (one house style) plausibly matters; this link is an inference.

### Gaps
- No ablation found comparing teacher-agreement thresholds for synthetic questions with a *frontier* teacher on MATH beyond OpenMathInstruct-2's Table 9.
- Exact Xwin-Math MATH-vs-size curve values and augmentation-style comparison values are in figures only.
- No per-token (rather than per-example) comparison of "new question + 1 solution" vs "extra solution to an existing question" was found.
- KPMath-Plus, MMIQC and Skywork-Math were not read directly; only their scores as re-reported in the DART-Math table are included.

## 5. Final-answer formatting and typical MATH accuracy of fine-tuned 7-9B models

### Takeaway
Scorer mismatch is a real, measurable loss: re-grading 3,751 models with a robust parser raised MATH scores by 4.66 points on average, mostly from extraction failures, so training targets should end with exactly one unambiguous, scorer-compatible final answer. Published greedy no-tool MATH scores for fine-tuned 7-9B models range from about 20% (MetaMath LLaMA-2-7B, 2023) through 45-47% (DART / Xwin on Mistral-7B, Llama3-8B), 53-56% (DeepSeekMath-7B-based), to 67.8% (OpenMath2-Llama3.1-8B) and 83.6% (Qwen2.5-Math-7B, reported by others).

### Cited Findings

**Formatting / scoring**
- Switching the Open LLM Leaderboard to Math-Verify: 3,751 models re-evaluated; +61 problems solved on average, +4.66 pts; Algebra +8.27, Prealgebra +6.93; some models gained nearly 90 pts on subsets; Qwen models more than doubled, DeepSeek almost tripled (the old evaluator could not extract `\boxed{}` answers). — [HF blog: Fixing Open LLM Leaderboard with Math-Verify](https://huggingface.co/blog/math_verify_leaderboard)
- Failure categories listed there: (1) extraction: answer buried in a sentence, boxed not parsed; (2) parsing: partial equations, intervals such as `(-inf,-14) U (-3,inf)`, matrices (`pmatrix`), "100%" parsed as 1, radicals such as `14 + 7\sqrt{2}`; (3) comparison: 1/3 vs 0.333333, `sqrt(1/2)*7` vs `sqrt(0.5)*7`, `k = 1` vs `1`, sets, matrices. — [HF blog](https://huggingface.co/blog/math_verify_leaderboard)
- Even Math-Verify returns False for an interval vs an equivalent inequality, and for a solution chain vs just the final number. — [Math-Verify GitHub](https://github.com/huggingface/Math-Verify)
- Hendrycks-style normalisation converts `a/b` to `\frac{a}{b}`, strips right-side units and percent signs; unit lists are hand-curated. (From a mirrored copy of the lm-eval utility, not the original repo.) — [lm-evaluation-harness utils mirror](https://huggingface.co/BayesTensor/out/blame/main/lm-evaluation-harness/lm_eval/tasks/agieval/utils.py)
- Grader differences shift reported scores by points: NVIDIA's grading script scores public ToRA outputs about 2-3% lower than ToRA's own reported numbers. — [OpenMathInstruct-1](https://arxiv.org/html/2402.10176)
- Conventions used by the successful pipelines: OpenMathInstruct-2 requires the answer in `\boxed{}`, truncates text after the boxed sentence and drops training solutions with multiple boxed entries; MetaMath ends every target with "The answer is: {gold}" and extracts only the text after that marker; Xwin-Math uses "SOLUTION: ... FINAL ANSWER:"; DART-Math extracts with regex and compares with SymPy. — [OpenMathInstruct-2](https://arxiv.org/html/2410.01560); [MetaMath App B](https://arxiv.org/html/2309.12284); [Xwin-Math App A](https://arxiv.org/html/2403.04706); [DART-Math](https://arxiv.org/html/2407.13690)
- A base (no chat role tokens) prompt template kept the student/teacher from reverting to its native verbose format (57% -> 0.1% reversion). — [OpenMathInstruct-2](https://arxiv.org/html/2410.01560)

**MATH accuracy of fine-tuned 7-9B models (greedy, zero-shot, text-only unless noted)**

| Model (base) | Data / teacher | Train size | MATH | Source |
|---|---|---|---|---|
| LLaMA-2-7B SFT on MATH train only | human solutions | 7.5K | 4.7-8.4 | [MetaMath](https://arxiv.org/html/2309.12284), [Xwin-Math](https://arxiv.org/html/2403.04706) |
| WizardMath-7B (LLaMA-2) | as re-reported | n/a | 10.7 | [Xwin-Math Table 2](https://arxiv.org/html/2403.04706) |
| MetaMath-7B (LLaMA-2) | MetaMathQA / GPT-3.5 | 395K | 19.8 | [MetaMath](https://arxiv.org/html/2309.12284) |
| MuggleMath-7B (LLaMA-2) | AugMATH / GPT-4 | about 300K | 25.8 | [MuggleMath](https://arxiv.org/html/2310.05506) |
| MetaMath-Mistral-7B | MetaMathQA | 395K | 28.2 | [MetaMath](https://arxiv.org/html/2309.12284) |
| MetaMath-Llemma-7B | MetaMathQA | 395K | 30.0 | [MetaMath](https://arxiv.org/html/2309.12284) |
| MathScale-7B (LLaMA-2) | MathScaleQA / GPT-3.5 | 2M | 31.1 | [MathScale](https://arxiv.org/html/2403.02884) |
| MathScale-Mistral-7B | MathScaleQA | 2M | 35.2 | [MathScale](https://arxiv.org/html/2403.02884) |
| Llama3-8B + MetaMathQA | re-trained by DART authors | 0.40M | 32.5 | [DART-Math](https://arxiv.org/html/2407.13690) |
| Llama3-8B + MMIQC | re-trained by DART authors | 2.3M | 39.5 | [DART-Math](https://arxiv.org/html/2407.13690) |
| Llama3-8B + vanilla rejection tuning | DeepSeekMath-7B-RL | 0.59M | 39.7 | [DART-Math](https://arxiv.org/html/2407.13690) |
| Xwin-Math-7B (LLaMA-2) | GPT-4 Turbo synthetic (MATH-500, T=0.7) | 480K MATH + 960K GSM8K | 40.6 | [Xwin-Math](https://arxiv.org/html/2403.04706) |
| Xwin-Math-Mistral-7B | same | same | 43.7 | [Xwin-Math](https://arxiv.org/html/2403.04706) |
| Xwin-Math-Llemma-7B | same | same | 47.2 | [Xwin-Math](https://arxiv.org/html/2403.04706) |
| OpenMath-Mistral-7B (**code execution**) | OpenMathInstruct-1 / Mixtral | about 1.02M | 44.5 | [OpenMathInstruct-1](https://arxiv.org/html/2402.10176) |
| DART-Math Mistral-7B (Uniform / Hard) | DeepSeekMath-7B-RL | 0.59M | 43.5 / 45.5 | [DART-Math](https://arxiv.org/html/2407.13690) |
| DART-Math Llama3-8B (Uniform / Hard) | same | 0.59M | 45.3 / 46.6 | [DART-Math](https://arxiv.org/html/2407.13690) |
| Mistral-7B + KPMath-Plus | as re-reported | 1.6M | 46.8 | [DART-Math](https://arxiv.org/html/2407.13690) |
| DeepSeekMath-7B-Instruct | as re-reported | 0.78M | 46.9 | [DART-Math](https://arxiv.org/html/2407.13690) |
| DART-Math DeepSeekMath-7B (Hard) | DeepSeekMath-7B-RL | 0.59M | 53.6 | [DART-Math](https://arxiv.org/html/2407.13690) |
| NuminaMath-7B-CoT (DeepSeekMath-7B) | NuminaMath-CoT | about 860K | 55.2-55.8 | [OpenMathInstruct-2](https://arxiv.org/html/2410.01560), [model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR) |
| NuminaMath-7B-TIR (**code execution**) | + GPT-4 TIR data | n/a | 68.1 | [model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR) |
| Llama3.1-8B-Instruct (no math fine-tune) | n/a | n/a | 51.8 | [OpenMathInstruct-2](https://arxiv.org/html/2410.01560) |
| Qwen2.5-7B-Instruct + 7.5K MATH train, long / short CoT | QwQ-32B / Qwen2.5-32B | 7.5K | 68.2 / 67.8 | [Small Models Struggle...](https://arxiv.org/html/2502.12143) |
| Qwen2.5-7B-Instruct + 7.5K MATH train, 72B teacher | Qwen2.5-72B | 7.5K | 71.3 | [Small Models Struggle...](https://arxiv.org/html/2502.12143) |
| OpenMath2-Llama3.1-8B | OpenMathInstruct-2 / Llama3.1-405B | 14M | 67.8 (maj@256 76.1) | [OpenMathInstruct-2](https://arxiv.org/html/2410.01560) |
| Qwen2.5-Math-7B-Instruct | as re-reported by NVIDIA | n/a | 83.6 | [OpenMathInstruct-2](https://arxiv.org/html/2410.01560) |

- Reference points: GPT-3.5-Turbo 34.1-37.8; GPT-4 (early) 42.5-52.6; GPT-4 Turbo (1106) 64.5 on the harnesses of those papers. — [Xwin-Math Table 2](https://arxiv.org/html/2403.04706); [MathScale Table 5](https://arxiv.org/html/2403.02884)
- Per-level difficulty: an OpenMathInstruct-1 7B model scored 72.4% on Level 1 vs 16.3% on Level 5; geometry weakest. — [OpenMathInstruct-1](https://arxiv.org/html/2402.10176)

### Inferences
- The base model dominates the absolute number: the same about-0.5M-sample recipe yields about 20% on LLaMA-2-7B, about 45% on Mistral-7B / Llama3-8B and about 54% on DeepSeekMath-7B, and a modern instruct 7B starts near 50-70% before any math SFT. A realistic target for a short, token-limited distillation is "a few points over the same model's zero-shot baseline", and the first risk is regressing below it (2026 revisit paper; Qwen2.5-7B short-CoT SFT 67.8 with unknown pre-SFT baseline in that paper).
- Most recoverable formatting loss is on easy problems (Algebra/Prealgebra gained most under a better parser), so it shows up as lost "free" points. Practical consequences for training targets (inferred from the failure lists, not separately ablated): end with exactly one final-answer marker in the form the scorer extracts; put only the bare answer there (no "x =", no sentence, no units unless the reference includes them); use `\frac{a}{b}` and simplified radicals instead of decimals; match the reference's form for intervals/sets/ordered pairs; never emit two boxed/answer expressions; normalise teacher outputs to the gold answer string of the training problem where one exists.
- Whether `\boxed{}` or an "ANSWER:" line is better depends entirely on the scorer; no paper found ablates the two. The measured fact is only that extraction failures cost several points on average and that each successful pipeline enforces a single rigid marker and filters non-conforming training samples.
- Numbers across rows are not strictly comparable: graders differ by 2-3 pts, Xwin uses a 500-problem subset with T=0.7, and some rows use code execution.

### Gaps
- No controlled ablation of final-answer format (boxed vs "The answer is" vs "ANSWER:") on MATH accuracy for a fixed model and grader was found.
- No per-category breakdown of which MATH answer types (fractions, intervals, expressions, text answers) most often fail under the original Hendrycks `is_equiv` script was retrieved from a primary source; the Hendrycks script itself was not read directly.
- Qwen2.5-Math's own paper/card was not read; its 83.6 and the ">1 trillion synthetic math tokens" remark are as relayed by the OpenMathInstruct-2 authors, who also note uncertainty about its contamination protocol **[CONTAM, unconfirmed]**.
- MATH-500-specific numbers for most of these models are not reported (most use the full 5,000-problem test set); Xwin-Math is the exception.
- Pre-fine-tuning zero-shot baselines for the Qwen2.5 students in the learnability-gap paper were not retrieved, so the net effect of 7.5K-example distillation on a strong 7B instruct model is unknown from these notes.

# Verifiable Instruction Following (IFEval-style) and RLVR for Small Open Models

Reading notes: numbers were extracted from arXiv HTML versions through an automated page reader. Tables in arXiv HTML sometimes extract with shifted columns, so any figure the report leans on heavily should be spot-checked against the PDF. Items I could not confirm are marked UNCONFIRMED or listed under Gaps.

## Tulu 3 (arXiv 2411.15124): persona IF SFT data, DPO IF data, RLVR settings, gains and pitfalls

### Takeaway
Tulu 3's 8B model went from 72.8 (SFT) to 81.1 (DPO) to 82.4 (RLVR) on IFEval; the big jump came from preference tuning with IF-targeted preference data, not from RLVR, which added only +1.3 IFEval, +3.3 GSM8K and +1.7 MATH at 8B. The IF training data was ~30k GPT-4o persona-driven prompts with 1-3 IFEval-taxonomy constraints each, and it barely transfers to unseen constraints (IFEval-OOD 17.6 to 24.3).

### Cited Findings

Persona IF SFT data
- Size: 29,980 verifiable instruction-response pairs ("If-Persona-Sft" / "Tulu 3 Persona IF"). — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Construction: authors wrote 1-2 example instructions per IFEval constraint, giving 33 seed instructions; GPT-4o generated new verifiable instructions from a data-synthesis prompt + a persona from Persona Hub (~250K personas) + one seed example; GPT-4o also generated the responses. Covers the 25 constraint types in the IFEval taxonomy. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Each prompt carries 1 to 3 verifiable constraints; fields are `id`, `prompt`, `messages`, `constraints` (e.g. `keywords:exclude words`, `format:title`); license ODC-BY; 122 downstream models listed as trained on it. — [HF dataset card](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)
- Sibling persona sets in the SFT mix: Persona MATH 149,960; Persona GSM 49,980; Persona Algebra 20,000; Persona Python 34,999. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)

SFT ablations
- Tulu 3 8B SFT full mix IFEval (prompt-level loose) = 72.8. Removing WildChat gives 70.1; removing safety data gives 71.0. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- The paper text states that removing Persona data lowers HumanEval(+), GSM8K, MATH and IFEval (exact IFEval value for the "w/o Persona" row not recovered, see Gaps). — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)

DPO stage
- Hyperparameters (8B): length-normalised DPO, LR 5e-7 (70B: 2e-7), beta = 5, 1 epoch, effective batch 128, max length 2,048, linear schedule, warmup 0.1. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Algorithm ablation: length-normalised DPO (LR 5e-7, beta 5, 1 epoch) averaged 57.3 vs 55.7 for the SFT base; SimPO and standard DPO did not beat the base overall. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- IF preference data sizes: Persona IF 19,890 preference prompts; "Tulu 3 IF-augmented" 65,530 preference prompts. Total pool 425,145 preference prompts (not all used in both mixes); the text also cites 354,192 instances for the selected mix. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Of the persona preference sets, only Persona IF improved both the average and its targeted eval (IFEval). Persona Math and Persona Code preferences did not improve their target evals and slightly hurt the average, so were excluded. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- IF-augmented with verified completions: filtering left ~26k examples and improved IFEval by only ~1 point while slightly hurting the average. Persona IF + verified IF-augmented gave the best IFEval but a slightly lower average. Final 8B mix uses unverified IF-augmented + Persona IF. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Stage results, 8B: SFT to DPO moved IFEval 72.8 to 81.1, GSM8K 76.2 to 84.3, MATH 31.5 to 42.0. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)

RLVR settings
- Algorithm: PPO. Reward v(x,y) = alpha if the verifier passes, else 0, with alpha = 10. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Shared: gamma 1.0, GAE lambda 0.95, 1 mini-batch, clip 0.2, value coefficient 0.1, grad-norm 1.0, linear LR schedule, sampling temperature 1.0, max token length 2,048, max prompt length 2,048, penalty of -10 for a missing EOS. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- 8B specifics: LR 3e-7, batch size 224, PPO update iterations K = 1, response length 1,024, 300,000 total episodes, KL beta swept over [0.05, 0.03, 0.02, 0.01], final beta = 0.05. 70B: LR 1e-7, batch 640, K = 4, response length 2,048, 100,000 episodes in the table. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Source conflict inside the paper as extracted: the hyperparameter table caption gives 70B beta = 0.07 while final-run text gave beta = 0.7 and 400,000 episodes for 70B. Treat 70B values as UNCONFIRMED. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- RLVR training data: GSM8K train 7,473; MATH train 7,500; IF verifiable 14,973; total 29,946. IF prompts are Tulu 2 SFT instructions combined with constraints from IFEval (Zhou et al. 2023). — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)

RLVR gains (DPO to RLVR)
- 8B: GSM8K 84.3 to 87.6 (+3.3); MATH 42.0 to 43.7 (+1.7); IFEval 81.1 to 82.4 (+1.3); overall average 64.4 to 64.8. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- 70B: GSM8K 93.5 to 93.5 (0); MATH 62.3 to 63.0 (+0.7); IFEval 82.6 to 83.2 (+0.6); average 75.9 to 76.0. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- 405B: GSM8K 94.2 to 95.5; MATH 59.9 to 67.3; IFEval 85.0 to 86.0. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Reference: Llama 3.1 8B Instruct scores GSM8K 83.4, MATH 42.5, IFEval 80.6. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)

Pitfalls and regressions
- Some 8B RLVR runs reached GSM8K 89.4 and IFEval 84.8 but scored worse overall, so were not selected. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Lowering the KL coefficient beta increased KL from the starting policy, which usually lowered the average score (the paper's "overoptimization"). — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- 8B non-target evals after RLVR: MMLU 68.7 to 68.2; PopQA 29.3 to 29.1; TruthfulQA 56.1 to 55.0; Safety 87.2 to 85.5; BBH 65.8 to 66.0; DROP 62.5 to 62.6; HumanEval 83.9 to 83.9; HumanEval+ 78.6 to 79.2; AlpacaEval 2 33.5 to 34.5. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Value model initialised from a general reward model gave the highest GSM8K and higher averages than other initialisations. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Verifiable rewards alone beat verifiable + reward-model scores; the reward-model term added noise. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Starting RLVR from the weaker SFT checkpoint reached similar training reward but with larger KL; stronger starting points usually gave better test scores. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Generalisation to unseen constraints is poor at every stage. IFEval vs IFEval-OOD: 8B SFT 72.8 / 17.6; 8B DPO 81.1 / 23.9; 8B final 82.4 / 24.3; 70B SFT 82.1 / 26.8; 70B DPO 82.6 / 26.4; 70B final 83.2 / 27.8. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)

### Inferences
- At 8B, the cheap lever is IF-targeted SFT + preference data: +8.3 IFEval from DPO versus +1.3 from 300k episodes of PPO. RLVR in the Tulu 3 form is a small polish step, not the main driver.
- Persona-driven prompt diversity over a fixed 25-constraint taxonomy teaches the taxonomy, not constraint-following in general (OOD stays at 18-24 while in-distribution is 73-82).
- Verifying completions in preference data mattered little (about +1 IFEval) relative to simply including IF-style preference pairs, which suggests the chosen/rejected contrast carries most of the signal.
- RLVR's side effects at 8B are small but consistently negative on knowledge/safety evals (MMLU -0.5, TruthfulQA -1.1, Safety -1.7), so a KL penalty and checkpoint selection on a broad average are needed.

### Gaps
- The exact IFEval number for the "w/o Persona data" SFT ablation row was not recoverable from the HTML extraction; only the qualitative statement (IFEval drops) is confirmed.
- Appendix B.4 "RLVR IFEval overoptimization" exists in the table of contents but its content (example degenerate outputs, scores vs beta) could not be retrieved.
- UNCONFIRMED from memory, not verified this session: the 405B RLVR run used MATH data only.
- Effect of WildChat-IF preference data on IFEval was not recovered.

## IFBench / "Generalizing Verifiable Instruction Following" (Ai2, arXiv 2507.02833)

### Takeaway
Models tuned on IFEval's 25 constraint templates score 73-82 on IFEval but only 17-31 on IFBench's 58 unseen constraints. GRPO-based IF-RLVR with multi-constraint prompts and 29 extra training constraints roughly doubles IFBench (Tulu-3-DPO 25.5 to 44.6; Qwen2.5-7B base reaches 53.7) and pushes IFEval to 88-92, at a measurable cost in general response quality.

### Cited Findings

Benchmark
- IFBench: 58 new out-of-distribution verifiable constraints with Python verifiers, in 7 categories (count, ratio, words, sentence, format, custom, copy); 300 prompts built by attaching 1-2 constraints to held-out WildChat prompts; single-turn and multi-turn (task, assistant answer, then a constrained rewrite request) settings. Main metric is prompt-level loose accuracy. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Released as `allenai/IFBench_test`, `allenai/IFBench_multi-turn`, and training data `allenai/IF_multi_constraints_upto5`; the `ifbench` package holds 83 verifiers (25 IFEval + 58 IFBench); training constraints live in open-instruct's `IFEvalG` folder. Eval uses temperature 0; reasoning chains are stripped before scoring. NeurIPS 2025 Datasets & Benchmarks. — [IFBench GitHub](https://github.com/allenai/IFBench)
- IFTrain: 29 new hand-annotated training constraints with verification functions, used alongside IFEval's 25. — [arXiv abstract](https://arxiv.org/abs/2507.02833)

Overfitting to IFEval (IFEval / IFBench)
- Tulu-3-8B-SFT 72.8 / 20.7; Tulu-3-8B-DPO 81.1 / 25.5 (another table lists 25.2); Tulu 3 8B final 82.4 / 28.9; Qwen2.5-7B-Instruct 74.7 / 31.3; OLMo2-Instruct 61.7 / 16.7. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- GPT-4.1, Claude 3.7 Sonnet, Claude 4 Sonnet and Qwen3-32B all score below 50% on IFBench. — [IFBench paper](https://arxiv.org/html/2507.02833v3)

IF-RLVR recipe
- GRPO with outcome supervision; binary reward per constraint; instance reward = sum of (verifiable reward x multiplier x weight), both defaulting to 1. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- LR 5e-7, sampling temperature 1, 16 samples per prompt, local mini-batch 32, KL beta = 0 for base models, max tokens 2,048 (10,240 for base models with a reasoning template), 8 H100s, about 1 day for 2,000 steps. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Training prompts: about 60k-100k, built from Tulu 3 SFT prompts with 1 to n constraints appended, n up to 6, conflicting combinations excluded. — [IFBench paper](https://arxiv.org/html/2507.02833v3)

Results after IF-RLVR (IFEval / IFBench)
- From base models: Llama-3.1-8B 88.2 / 54.1; Qwen2.5-7B 87.8 / 53.7; OLMo2 70.4 / 46.6. From instruct models: Tulu-3-8B-DPO 92.2 / 44.6; Qwen2.5-7B-Instruct 89.1 / 45.9; OLMo2-Instruct 74.5 / 44.6. Base models with a reasoning template get similar IFEval and better IFBench than instruct starts. — [IFBench paper](https://arxiv.org/html/2507.02833v3)

Ablations (Qwen2.5 policy; IFBench / IFEval)
- Constraints per training prompt: 1: 48.9 / 71.2; 2: 53.1 / 79.9; 3: 59.5 / 77.8; 4: 49.4 / 79.5; 5: 55.8 / 79.9; 6: 54.1 / 85.8. Multi-constraint beats single-constraint on both; IFBench is non-monotonic (peak at 3), IFEval peaks at 6. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Instances per constraint: 10: 48.6 / 73.6; 50: 52.7 / 72.8; 100: 51.7 / 74.3; 500: 51.0 / 70.1; 1000: 48.6 / 72.8. Essentially flat. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Training-constraint set: IFTrain + IFEval constraints gives the best IFEval; IFBench is less sensitive to how many IFEval constraints are included (figure only, no exact values). — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Variable ranges (e.g. the N in "at least N words"): training on ranges disjoint from test is worst; wider training ranges match or beat same-range training (figure only). — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Removing a constraint category from training: removing length or keyword constraints hurts IFEval most; removing change-case or detectable-format barely matters. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- GRPO vs DPO on the same prompts (strict accuracy; IFEval / IFBench): DPO after SFT 76.89 / 25.2; DPO after DPO 79.67 / 29.3; GRPO after SFT 85.77 / 28.6; GRPO after DPO 89.65 / 30.6. DPO used 31,751 strictly verified preference prompts with 5 completion models. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Multi-turn RLVR mainly helps multi-turn IFBench; mixed single+multi-turn training helps single-turn in some cases. — [IFBench paper](https://arxiv.org/html/2507.02833v3)

Hardest categories
- Tulu-3-DPO on IFBench: words 7.0, sentence 13.3 (overall 25.5). After IF-RLVR: words 48.1, sentence 36.7 (overall 44.6). IF-RLVR Qwen2.5: words 67.0, sentence 71.1 (overall 53.7). — [IFBench paper](https://arxiv.org/html/2507.02833v3)

Costs
- Tulu 3 vs IF-RLVR (Tulu-DPO start): AlpacaEval 2 34.5 to 21.3; GSM8K 87.6 to 83.2; MMLU 68.2 to 66.4; BBH 69.0 to 68.9. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- IF-RLVR from Qwen2.5 base: AlpacaEval 2 1.1, GSM8K 15.3, MMLU 59.4, BBH 26.0; the paper attributes this to the base model never having been post-trained. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- LLM-judge quality: base-policy responses average 7.0/10 vs 6.4 for IF-RLVR responses. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Mitigation: adding a Tulu-3-8B reward-model term with threshold alpha = 7 gives IFEval 86.1, IFBench 30, AlpacaEval 2 31.6 after 1,100 steps; it recovers quality but gives back most of the IFBench gain. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Stated limitations: only verifiable constraints are covered, some test constraints are unnatural, and over-optimising for constraints reduces general quality. — [IFBench paper](https://arxiv.org/html/2507.02833v3)

### Inferences
- The recipe that generalised best: GRPO, multiple (2-6) constraints per prompt, the union of IFTrain 29 + IFEval 25 constraint types, wide variable ranges, and 60k-100k prompts; number of instances per constraint is not the bottleneck (10 to 1000 is flat), diversity of types and compositions is.
- For an IFEval-only target, 5-6 constraints per training prompt plus keeping length and keyword constraints in training matter most (IFEval 71.2 at 1 constraint vs 85.8 at 6).
- RLVR beats DPO on in-distribution IFEval by about 9-10 points on the same prompt pool (85.77 vs 76.89 after SFT), but in that table the unseen-constraint gain is small (28.6 vs 25.2). This is the closest available evidence on "RLVR vs more rejection-sampled/verified preference data".
- The quality cost is real when IF is the only reward: AlpacaEval 2 -13 points and GSM8K -4.4 for the Tulu start. Mixing other tasks or a quality reward is needed if anything beyond IFEval is being scored.
- There is an apparent inconsistency between tables: "GRPO after DPO" is 89.65 / 30.6 (strict) while "IF-RLVR (Tulu-DPO)" is 92.2 / 44.6 (loose). Strict vs loose alone is unlikely to explain a 14-point IFBench gap; the two rows probably differ in training-constraint set or steps. UNCONFIRMED which.

### Gaps
- The individual list of the 58 IFBench constraints and the 29 IFTrain constraints was not retrieved (they are in the `ifbench` package and open-instruct `IFEvalG`).
- Per-model IFEval vs IFBench numbers for frontier models are in a figure and were not extracted.
- Exact values behind the seen/unseen-constraint and variable-range figures were not available as text.
- No GRPO vs PPO comparison is reported in the paper.

## Other work on IFEval-style improvement for small models (AutoIF, Conifer, IFEval metric, FollowBench, InFoBench)

### Takeaway
Synthetic data with executable verification (AutoIF) and progressive multi-constraint data (Conifer) each buy single-digit to low-double-digit IFEval gains at 7-8B, with the quality of the supervising model mattering more than the training algorithm. Length/counting and keyword constraints, and stacking several constraints, are the recurring hard cases.

### Cited Findings

IFEval metric mechanics
- 541 prompts, 25 verifiable instruction types in 9 groups (keywords, language, length constraints, detectable content, detectable format, combination, change cases, start with/end with, punctuation); prompts were generated with one to three instructions each. — [IFEval paper](https://arxiv.org/html/2311.07911)
- Strict = raw response must pass. Loose = pass if any of 8 variants passes: identity, strip markdown `*`/`**`, drop first line, drop last line, and all combinations of those three. Authors note loose reduces false negatives but can add false positives. — [IFEval paper](https://arxiv.org/html/2311.07911)
- Prompt-level accuracy requires every instruction in the prompt to pass; instruction-level counts each separately. — [IFEval paper](https://arxiv.org/html/2311.07911)

AutoIF (Self-play with Execution Feedback, ICLR 2025)
- Pipeline: hand-written seed instructions with one atomic constraint each; self-instruct rewrites (K = 100 in the appendix); LLM writes K verification functions and K test cases per instruction; keep only functions that compile, test cases passing > 0.5 of functions, functions passing > 0.5 of test cases, and instructions with at least one of each. Back-translate function to instruction and drop NLI contradictions (mDeBERTa; about 15% removed). Attach to 16 ShareGPT queries per instruction, rejection-sample K = 8 responses, keep responses passing > 0.5 of functions, then LLM-score query fit 1-10 and drop < 8. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Yield by supervisor: LLaMA3-70B 85K generated, 15K SFT, 6K DPO, 26% pass; Qwen2-72B 123K, 10K SFT, 4K DPO, 28% pass; GPT-4 210K, 25K SFT, 15K DPO, 34% pass. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Training: SFT LR 7e-6, 3 epochs, batch 128; DPO LR 5e-7, 2 epochs, batch 64, beta 0.3; online DPO samples at temperature 0.8. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Qwen2-7B (Qwen2-72B supervisor), IFEval Pr(S) / Pr(L) / Ins(S) / Ins(L), FollowBench SSR avg: baseline 37.7 / 43.6 / 49.4 / 53.4, 52.3; SFT 40.7 / 44.5 / 51.3 / 55.4, 53.3; SFT + offline DPO 41.2 / 44.7 / 51.4 / 56.2, 54.0; SFT + online DPO 44.0 / 46.6 / 55.0 / 57.9, 56.6. General evals roughly unchanged (MMLU 64.4 to 64.8; GSM8K 71.1 to 72.3; HumanEval 58.1 to 58.2). Note the paper's Qwen2-7B is an in-house model. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- LLaMA3-8B (base, LLaMA3-70B supervisor): baseline Pr(S) 24.6 / Pr(L) 26.1; SFT 28.7 / 40.3; SFT + offline DPO 27.9 / 41.6; SFT + online DPO 28.8 / 43.1; FollowBench avg 11.6 to 49.9. Strict barely moves (+4.2) while loose gains +17. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Supervisor ablation (Qwen2-7B, Pr(L) / Ins(L)): GPT-4 SFT 52.9 / 62.6; GPT-4 SFT + offline DPO 59.3 / 68.9; GPT-4 SFT + online DPO 59.5 / 69.4, versus 46.6 / 57.9 for the best Qwen2-72B-supervised run. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Filter ablation (Qwen2-7B online DPO; delta Pr(L) / Ins(L) / FollowBench): w/o back-translation -0.8 / -1.7 / -0.7; w/o query quality verification -1.4 / -2.4 / -1.3; w/o cross verification -1.6 / -3.0 / -1.5; w/o all -2.2 / -3.8 / -2.6. — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Data efficiency: with GPT-4 supervision, 1/64 of the DPO data already reaches about 55% loose prompt accuracy (+11.4). — [AutoIF paper](https://arxiv.org/html/2406.13542v3)
- Headline at large scale: 88.0 (Qwen2-72B) and 90.4 (LLaMA3-70B) loose instruction accuracy. — [ICLR 2025 poster](https://www.iclr.cc/virtual/2025/poster/29049)

Conifer (arXiv 2404.02823)
- Data: 6,000 ShareGPT seed queries, GPT-4 Turbo reframing and constraint recombination, difficulty levels 1-5 delivered as easy-to-hard multi-turn conversations; 13,606 conversations (avg 3.02 turns): 10,302 easy-to-hard and 3,304 process-feedback (977 internal, 2,327 external); 32 format/numerical constraints after manual review. Trained as 53k ShareGPT + 13k Conifer. — [Conifer paper](https://arxiv.org/html/2404.02823v1)
- Mistral-7B SFT: 4 epochs, LR 2e-5; DPO on UltraFeedback: beta 0.01, 1 epoch, LR 5e-7. — [Conifer paper](https://arxiv.org/html/2404.02823v1)
- Results (IFEval loose / FollowBench HSR avg / InFoBench overall): Conifer-7B SFT 50.8 / 44.9 / 79.5; Conifer-7B-DPO 52.3 / 50.0 / 82.3; Mistral-7B-ShareGPT-DPO 48.2 / 47.7 / 82.0; Deita-7B-v1.0 51.9 / 45.7 / 80.9; Zephyr-7B-beta 44.9 / 44.8 / 78.0. — [Conifer paper](https://arxiv.org/html/2404.02823v1)
- Ablations (IFEval): full 50.8; without Conifer data 43.4; single-turn instead of easy-to-hard 46.2; without process feedback 44.2; easy only 47.5; hard only 48.2; random order 48.6; hard-to-easy 49.9; without format and numerical data 48.6. — [Conifer paper](https://arxiv.org/html/2404.02823v1)
- FollowBench level 5 (five stacked constraints) is the hardest level for every model: Conifer-7B SFT falls 54.3 (L1) to 30.5 (L5); DPO version 60.3 to 41.0. The paper notes GPT-4 struggled to add format/numerical constraints and tuned models struggled with them on the hardest samples. — [Conifer paper](https://arxiv.org/html/2404.02823v1)
- General ability held: Open LLM Leaderboard average 59.60 (Conifer) vs 59.68 (ShareGPT only). — [Conifer paper](https://arxiv.org/html/2404.02823v1)
- A later reproduction on Mistral-7B-v0.3 reports different Conifer-7B-DPO figures (IFEval 55.7), so absolute numbers depend on backbone version and metric. — [Constraint Back-translation paper](https://arxiv.org/pdf/2410.24175)

FollowBench and InFoBench
- FollowBench: five constraint categories (content, situation, style, format, example), 1 to 5 incrementally added constraints; all tested models decline substantially as constraints are added. — [FollowBench paper](https://arxiv.org/abs/2310.20410v3)
- InFoBench: 500 instructions decomposed into 2,250 yes/no requirement questions; metric is Decomposed Requirements Following Ratio; authors point to numerical and linguistic constraints as areas needing improvement. — [InFoBench paper](https://arxiv.org/abs/2401.03601v1)

Hard constraint types
- In IF-RLVR training, dropping length or keyword constraints hurts IFEval most; case and detectable-format are nearly free. On unseen constraints, word-level and sentence-level constraints are the weakest (7.0 and 13.3 for Tulu-3-DPO). — [IFBench paper](https://arxiv.org/html/2507.02833v3)

### Inferences
- A large strict-vs-loose gap (LLaMA3-8B after AutoIF SFT: 28.7 strict vs 40.3 loose) indicates preamble/closing lines or markdown emphasis are breaking otherwise compliant answers, since those are exactly the three things loose scoring strips. Training responses that start directly with the answer and avoid decorative markdown should close this gap cheaply.
- Supervisor/data quality dominates algorithm choice: switching the AutoIF supervisor to GPT-4 gave about +13 Pr(L) while SFT to online DPO with a fixed supervisor gave about +2.
- Small amounts of verified data go a long way (1/64 of the DPO set gave +11.4), consistent with IFBench's finding that instances-per-constraint is flat from 10 to 1000.
- Multi-constraint composition is the common thread across Conifer (easy-to-hard, +4.6 over single-turn), FollowBench (monotone drop with constraint count) and IFBench (1 vs 6 constraints: 71.2 vs 85.8 IFEval).

### Gaps
- IFEval per-category accuracy for GPT-4/PaLM 2 is only in figures; no text numbers recovered.
- No source found with a per-category IFEval error breakdown specifically for 7-8B open models (e.g. share of failures due to word-count vs sentence-count vs preamble).
- FollowBench and InFoBench per-category results were not retrieved; only headline design and qualitative findings are confirmed.
- AutoIF's seed-instruction count was not stated in the retrieved text.

## Known pitfalls of RLVR on small models

### Takeaway
The documented pitfalls at ~8B are over-optimisation (KL drift lowering broad averages), loss of general response quality when the reward is constraint-only, and overfitting to the training constraint set; clear evidence on checker-specific reward hacking and on RLVR vs rejection-sampled SFT at this scale is thin.

### Cited Findings
- Over-optimisation: in Tulu 3, lower KL penalty meant higher KL and usually lower average scores; runs with the best target metrics (GSM8K 89.4, IFEval 84.8) were worse overall. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Quality degradation from IF-only reward: AlpacaEval 2 34.5 to 21.3, GSM8K 87.6 to 83.2, judge score 7.0 to 6.4. — [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Adding a learned reward model is not a free fix: in Tulu 3 it added noise and underperformed verifiable-only; in IF-RLVR it restored AlpacaEval (31.6) but cut IFBench to 30. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5); [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Mixed-task RLVR (GSM8K + MATH + IF, about 30k prompts) at 8B kept non-target evals within about 2 points and improved all three targets modestly. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5)
- Overfitting to the constraint set: in-distribution IFEval 82.4 vs unseen 24.3-28.9 for Tulu 3 8B. — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5); [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Starting point matters: RL from a weaker checkpoint reaches the same training reward with larger KL and worse test scores; RL from a base model with IF-only reward leaves general ability undeveloped (Qwen2.5 base: GSM8K 15.3). — [Tulu 3 paper](https://arxiv.org/html/2411.15124v5); [IFBench paper](https://arxiv.org/html/2507.02833v3)
- Verifier false positives invite shortcuts: a 2026 paper reports RLVR-trained reasoning models abandoning rule induction for enumerating instance labels when imperfect verifiers accept it. — [LLMs Gaming Verifiers, arXiv 2604.15149](https://huggingface.co/papers/2604.15149)
- Informal small-model evidence (blog post, Qwen2.5-0.5B with GRPO, not peer reviewed): a planted checker bug was exploited only when the model already sampled the exploit sometimes; in another run GRPO removed reasoning and kept a bare answer. — [LessWrong post](https://www.lesswrong.com/posts/NzzmNREX4qR54q33j/linear-probes-add-little-for-verifiable-reward-hacking)
- IFEval's own loose metric admits false positives (dropping a first/last line can make a count pass), which is the same kind of checker slack a policy could exploit. — [IFEval paper](https://arxiv.org/html/2311.07911)
- RLVR vs verified preference tuning on the same prompts: GRPO 85.77 vs DPO 76.89 IFEval after SFT; 89.65 vs 79.67 after DPO. — [IFBench paper](https://arxiv.org/html/2507.02833v3)

### Inferences
- If the score is a mix of IFEval-style and other skills, IF-only RLVR is risky: the measured trade is roughly +10 IFEval for -13 AlpacaEval 2 and -4 GSM8K. Tulu 3's mixed-task, KL-regularised PPO is the safe end (+1.3 IFEval, near-zero side effects); IF-RLVR with beta = 0 is the aggressive end.
- Math RLVR gains at 8B are small (+3.3 GSM8K, +1.7 MATH over 300k episodes), consistent with sparse binary rewards on problems the model rarely solves; most of the math gain in Tulu 3 came earlier (SFT to DPO: +8.1 GSM8K, +10.5 MATH).
- No direct published comparison of RLVR against "more rejection-sampled SFT" was found. The nearest evidence: AutoIF's rejection-sampled SFT/DPO gives +3 to +16 Pr(L) depending on supervisor, and IFBench shows GRPO beating verified DPO by about 9-10 IFEval. Rejection-sampled SFT from a strong teacher is the lower-risk first step; RLVR is the higher-ceiling, higher-risk second step.

### Gaps
- No peer-reviewed study found of an 8B model exploiting IFEval-style checkers specifically (e.g. keyword stuffing, degenerate repetition); Tulu 3's Appendix B.4 likely contains examples but could not be read.
- No head-to-head of RLVR vs rejection-sampled SFT at equal compute was found.
- No quantitative ablation of "how much task mixing helps" (IF-only vs IF+math RLVR on the same model) was found; the comparison above is across two papers with different algorithms (PPO vs GRPO) and KL settings.

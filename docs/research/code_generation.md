# Improving MBPP-style Python function generation in 7-9B models via SFT on synthetic / teacher-generated data

Reader constraint kept in view throughout: no MBPP example from any split may be used for training. Where a method or an evaluation harness touches MBPP data this is flagged with **[MBPP-DATA]**.

Verification legend: "(full text read)" = numbers taken from the paper HTML or source file fetched in this session; "(search snippet only)" = number seen only in a search-result summary and not checked against the paper table; "UNCONFIRMED" = could not be verified in this session.

---

## 1. Data-generation pipelines (OSS-Instruct/Magicoder, WaveCoder, SelfCodeAlign, Evol-Instruct/WizardCoder, NVIDIA OpenCodeInstruct): how generated, size, filtering, gains for ~7B models, ablations

### Takeaway
For ~7B models, every pipeline gives large HumanEval(+) gains over a base model (+15 to +21 points) but much smaller MBPP(+) gains (typically +2 to +7 points), and the gain shrinks further when the starting point is already a strong instruct model (Qwen2.5-Coder-7B-Instruct: MBPP+ 71.7 -> 74.9 with 5M samples). The best-supported design choices are: NL-to-code instruction format, instruction diversity seeded from real code, a strong teacher for the responses, and some correctness filter (execution in SelfCodeAlign, LLM judge in OpenCodeInstruct); dataset size beyond roughly 50-500K gives diminishing returns for MBPP.

### Cited Findings

**OSS-Instruct / Magicoder (gpt-3.5 teacher, 75K, no execution filtering)** (full text read, except Table 1 cells)
- Seeds: 80K snippets of 1-15 consecutive lines from 80K starcoderdata documents (40K Python, 5K each for C++, Java, TypeScript, Shell, C#, Rust, PHP, Swift); teacher gpt-3.5-turbo-1106 with greedy decoding; final set about 75K (about 43K Python, 32K non-Python). — [Magicoder paper](https://arxiv.org/html/2312.02120)
- No execution filtering. Only decontamination against benchmarks: 9 samples removed from OSS-Instruct, 89 from evol-codealpaca-v1. — [Magicoder paper](https://arxiv.org/html/2312.02120)
- Training: 2 epochs, lr 5e-5, 15 warmup steps, linear schedule, Adafactor, batch 512, max length 1216; MagicoderS continues on about 110K evol-codealpaca-v1 samples (max length 1024). — [Magicoder paper](https://arxiv.org/html/2312.02120)
- Results (greedy pass@1, EvalPlus leaderboard numbers): Magicoder-CL-7B HumanEval 60.4 (HumanEval+ 55.5), MBPP 64.2 (MBPP+ 52.6); MagicoderS-DS-6.7B HumanEval 76.8 (70.7), MBPP 75.7 (64.4). (search snippet of the README table; the paper's Table 1 cells did not render in the fetched HTML) — [Magicoder GitHub README](https://www.github.com/ise-uiuc/magicoder)
- Base CodeLlama-Python-7B: HumanEval+ 34.1, MBPP+ 45.4. So 75K OSS-Instruct gives +21.4 on HumanEval+ but only +7.2 on MBPP+ (52.6 vs 45.4). — [Magicoder paper, Table 7](https://arxiv.org/html/2312.02120) and [README](https://www.github.com/ise-uiuc/magicoder)
- Ablation, language mix (CodeLlama-Python-7B, HumanEval+): base 34.1; Python-only 43K -> 47.6; non-Python 32K -> 44.5; all 75K -> 55.5. Non-Python data alone still adds +10.4 on Python. — [Magicoder paper, Table 5](https://arxiv.org/html/2312.02120)
- Ablation, raw code vs synthesized instructions: fine-tuning directly on 75K comment-function pairs from the same corpus gives HumanEval+ 34.1 (no gain over base), vs 55.5 for OSS-Instruct. — [Magicoder paper, Table 6](https://arxiv.org/html/2312.02120)
- Ablation, weaker open teacher: 20K OSS-Instruct samples from Mixtral-8x7B-Instruct (itself HumanEval+ 39.6 / MBPP+ 47.4) lift CodeLlama-Python-7B from 34.1 / 45.4 to 55.5 / 50.4. The student ends above its teacher. — [Magicoder paper, Table 7](https://arxiv.org/html/2312.02120)
- Ablation, noise removal: dropping "noisy" samples (75K -> 68K) changes HumanEval+ from 55.5 to 54.9, i.e. no benefit. — [Magicoder paper, Table 10](https://arxiv.org/html/2312.02120)
- Stated limitation: only models up to 7B were tested, and the OSS-Instruct prompt is complex so "weaker models may fail to produce plausible instruction data". — [Magicoder paper, App. F](https://arxiv.org/html/2312.02120)

**Evol-Instruct / WizardCoder** (full text read)
- Seed: Code Alpaca (about 20K); evolved to about 78K samples; base models StarCoder-15B and CodeLlama-34B-Python. — [WizardCoder paper](https://arxiv.org/html/2306.08568)
- WizardCoder-15B: HumanEval 57.3, MBPP 51.8; WizardCoder-34B: HumanEval 71.5, MBPP 61.2. Note these are NOT greedy single-sample: n=200 samples, temperature 0.2, top_p 0.95; MBPP is the 500-problem test split. — [WizardCoder paper, Table 1](https://arxiv.org/html/2306.08568)
- Ablation on evolution rounds (HumanEval pass@1): sample-matched Round 0 45.7, Round 1 56.1, Round 2 53.0, Round 3 54.3, Round 4 51.2; token-matched 44.5 / 51.8 / 52.4 / 50.0 / 49.4. Most of the gain arrives after one round; more complexity does not keep helping. — [WizardCoder paper, Table 5](https://arxiv.org/html/2306.08568)
- **[MBPP-DATA]** WizardCoder selected the number of evolution rounds on an "MBPP-400 dev set" built by merging MBPP train and dev, and its MBPP prompt is three-shot. This model-selection step is not available under the reader's constraint. — [WizardCoder paper, Sec. 5 and App. A](https://arxiv.org/html/2306.08568)

**WaveCoder / CodeSeaXDataset (about 20K, generator-discriminator)** (full text read)
- 19,915 instances: generation 11,370 (57.1%), summarization 3,165, repair 3,144, translation 2,236. Raw code from CodeSearchNet (about 2M comment-code pairs), selected with KCenterGreedy over roberta-large-v1 embeddings; GPT-3.5 generates instructions from raw code with few-shot examples, GPT-4 acts as discriminator; no execution. — [WaveCoder paper v4](https://arxiv.org/html/2312.14187v4)
- Results, pass@1 on HumanEval and MBPP (500-problem test split): CodeLlama-7B 33.5 / 41.4 -> WaveCoder-CL-7B 48.1 / 47.2 (+14.6 / +5.8); DeepSeekCoder-6.7B base 49.4 / 60.6 -> WaveCoder-DS 64.0 / 62.8 (+14.6 / +2.2); WaveCoder-Pro-6.7B 72.0 / 63.6; WaveCoder-Ultra-6.7B (130K) 78.6 / 64.4. — [WaveCoder paper v4, Table 3](https://arxiv.org/html/2312.14187v4)
- Same table, comparison rows: Magicoder-DS (75K) 66.5 / 60.4, i.e. slightly BELOW the DeepSeekCoder base on MBPP (60.6); DeepSeekCoder-6.7B-instruct 73.8 / 62.8; MagicoderS-DS (185K) 76.8 / 64.6. — [WaveCoder paper v4, Table 3](https://arxiv.org/html/2312.14187v4)
- Task ablation (DeepSeekCoder-6.7B, HumanEval): all four tasks 64.0; without generation 53.6; without repair 60.9; without translation 60.9; without summarization 61.5. Generation data carries most of the gain, but the auxiliary tasks each add 2.5-3 points. — [WaveCoder paper v4, Table 6](https://arxiv.org/html/2312.14187v4)
- Size ablation (StarCoder base, HumanEval, 1K / 5K / 20K samples): the paper's Appendix C lists 44.9 vs 41.7, 45.7 vs 48.1, 47.0 vs 50.5 for the two datasets; the text as extracted is internally inconsistent about which column is which, so treat only the qualitative point (gains keep growing from 1K to 20K, with most of the gain already at 1K) as usable. — [WaveCoder paper v4, App. C](https://arxiv.org/html/2312.14187v4)
- **[MBPP-DATA]** The paper says the 500-problem MBPP test set is used for "few-shot inference of fine-tuned models", i.e. MBPP few-shot exemplars are in the evaluation prompt. — [WaveCoder paper v4](https://arxiv.org/html/2312.14187v4)

**SelfCodeAlign / Self-OSS-Instruct (self-generated, execution-validated, 74K)** (full text read, first 100K chars)
- Pipeline: extract concepts from seed functions -> generate instruction -> sample multiple responses, each paired with its own tests -> run in sandbox -> keep passing pairs. The base model generates its own data (no stronger teacher). — [SelfCodeAlign paper](https://arxiv.org/html/2410.24198)
- Seeds: 5,359,051 Python functions with docstrings from The Stack V1, filtered to 248,934 high-quality functions. Generation temperature 0.7; 21 hand-written few-shot examples (8-shot for concept and instruction generation, 1-shot for response generation). Final set 74K pairs from CodeQwen1.5-7B. — [SelfCodeAlign paper, Sec. 2 and App. B/C](https://arxiv.org/html/2410.24198)
- Training: lr 1e-5 for self-generated data (2e-5 for other-model data), warmup ratio 0.05, linear schedule, Adafactor, batch 64, truncation 1280. — [SelfCodeAlign paper](https://arxiv.org/html/2410.24198)
- Response format in the training data: prose explanation, then code, then a test block. — [SelfCodeAlign paper, App. D.1](https://arxiv.org/html/2410.24198)
- Main result (greedy pass@1): CodeQwen1.5-7B base HumanEval+ 45.7 / MBPP+ 60.2 -> SelfCodeAlign-CQ-7B 67.1 / 65.2 (+21.4 / +5.0). Same base tuned on OctoPack: 50.6 / 63.2. For reference Magicoder-S-DS-6.7B 70.7 / 65.4, CodeQwen1.5-7B-Chat 77.7 / 67.2, CodeLlama-70B-Instruct 65.2 / 61.7. — [SelfCodeAlign paper, Table 1](https://arxiv.org/html/2410.24198)
- Ablation, execution filtering (HumanEval+): random selection 27.7K samples (24.1% pass rate) -> 61.6; random 15.6K -> 61.6; failures only 15.6K -> 57.9; passes only 15.6K -> 65.2. So filtering is worth +3.6 over unfiltered at equal size, and passing vs failing data differ by 7.3. — [SelfCodeAlign paper, Table 9](https://arxiv.org/html/2410.24198)
- Ablation, seed and concept steps (HumanEval+): seed -> instruction directly 59.8; seed -> concepts -> instruction with random snippets 64.0; with filtered functions 65.2. The concept-extraction step is worth +5.4, seed quality filtering +1.2. — [SelfCodeAlign paper, Table 10](https://arxiv.org/html/2410.24198)
- Ablation, same base (CodeQwen1.5-7B), same size (74K), HumanEval+: Evol-Instruct with GPT-3.5 teacher 59.1; OSS-Instruct with GPT-3.5 teacher 61.6; direct distillation from GPT-4o (no filtering) 65.9; SelfCodeAlign 67.1. — [SelfCodeAlign paper, Table 11](https://arxiv.org/html/2410.24198)
- Ablation, who generates the data (Table 8, HumanEval+, 37K seeds): Llama-3-8B base 29.3 -> 42.7 with its own data, 43.3 with CodeQwen1.5-7B data, 41.5 with DeepSeek-Coder-33B data; StarCoder2-3B 27.4 -> 35.4 own data, 42.1 with DeepSeek-Coder-33B data; CodeQwen1.5-7B 45.7 -> 65.2 own data, 59.1 with DeepSeek-Coder-33B data, 54.9 with Llama-3-8B data. — [SelfCodeAlign paper, Table 8](https://arxiv.org/html/2410.24198)
- Self-generation yield is low: only 24.1% of randomly selected CodeQwen1.5-7B responses pass their own tests. — [SelfCodeAlign paper, Table 9](https://arxiv.org/html/2410.24198)

**NVIDIA OpenCodeInstruct (5M, open teacher, tests + LLM judge attached)** (full text read)
- Seeds: 1.43M "generic" instructions made with OSS-Instruct from GitHub Python functions with docstrings (written by Qwen2.5-32B-Instruct, solutions discarded) plus 25,443 "algorithmic" TACO questions. Genetic-Instruct (mutation = Evol-Instruct style, crossover = Self-Instruct style) for one generation gives nearly 9M instructions; removing instructions containing Python code and n-gram decontamination against benchmarks leaves about 5M. — [OpenCodeInstruct paper](https://arxiv.org/html/2504.04030v2)
- Responses, 10 assert-style unit tests per sample, and a 1-5 LLM-judge score (requirement conformance, logical correctness, edge cases; averaged) are all produced by Qwen2.5-Coder-32B-Instruct. — [OpenCodeInstruct paper](https://arxiv.org/html/2504.04030v2)
- Dataset card fields: `input`, `output`, `domain` (generic/algorithmic), `generation_algorithm` (self-instruct/evol-instruct), `llm_judgement`, `unit_tests`, `tests_execution_status`, `average_test_score`; about 4.97M rows; CC BY 4.0. Nothing is pre-filtered: the user filters. — [nvidia/OpenCodeInstruct dataset card](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
- Training: 3 epochs, lr 5e-6, 100 warmup steps, cosine, AdamW, batch 2048, max length 2048, final model = average of end-of-epoch checkpoints. — [OpenCodeInstruct paper](https://arxiv.org/html/2504.04030v2)
- Main results (HumanEval / HumanEval+ / MBPP / MBPP+): Llama-3.1-8B-Instruct 69.5 / 62.8 / 68.3 / 60.6 vs OCI-Llama-3.1-8B 78.7 / 73.2 / 77.5 / 66.4; Qwen2.5-Coder-7B-Instruct 88.4 / 84.1 / 83.5 / 71.7 vs OCI-Qwen2.5-Coder-7B 87.8 / 84.1 / 86.8 / 74.9; OpenCoder-8B-Instruct 83.5 / 78.7 / 79.1 / 69.0. — [OpenCodeInstruct paper, Table 3](https://arxiv.org/html/2504.04030v2)
- Smaller models gain more: Qwen2.5-Coder-1.5B-Instruct 69.2 / 59.4 (MBPP / MBPP+) vs OCI version 80.2 / 68.3; Llama-3.2-3B-Instruct 57.1 / 48.1 vs 69.8 / 61.1. — [OpenCodeInstruct paper, Table 3](https://arxiv.org/html/2504.04030v2)
- Ablation, filtering (Qwen2.5-Coder-7B, 500K samples each; HE / HE+ / MBPP / MBPP+): random (72.4% test pass rate) 82.9 / 77.8 / 81.0 / 70.1; unit-test failures only 80.0 / 75.3 / 80.1 / 69.8; unit-test passes only 83.1 / 78.4 / 81.4 / 70.4; LLM-judge score = 5.0 84.8 / 80.5 / 82.3 / 71.4. — [OpenCodeInstruct paper, Table 4](https://arxiv.org/html/2504.04030v2)
- Ablation, instruction format (2M each, Qwen2.5-Coder-7B): NL-to-Code 85.4 / 79.9 / 83.3 / 73.0 vs Code-to-Code 80.5 / 74.4 / 81.8 / 70.1. Llama-3.1-8B: 70.7 / 67.1 / 74.6 / 65.6 vs 68.3 / 62.2 / 71.2 / 61.1. — [OpenCodeInstruct paper, Table 5](https://arxiv.org/html/2504.04030v2)
- Ablation, generation algorithm (Qwen2.5-Coder-7B): Self-Instruct 3M 84.1 / 78.0 / 82.3 / 71.4 vs Evol-Instruct 2M 87.2 / 80.5 / 82.8 / 72.8. — [OpenCodeInstruct paper, Table 5](https://arxiv.org/html/2504.04030v2)
- Ablation, seed population (2.5M each, Qwen2.5-Coder-7B): algorithmic-small 81.7 / 76.8 / 83.3 / 71.4; algorithmic-large 84.8 / 79.3 / 83.1 / 72.2; generic-large 85.4 / 79.3 / 83.3 / 71.7. MBPP is essentially flat across seed types. — [OpenCodeInstruct paper, Table 5](https://arxiv.org/html/2504.04030v2)
- Versus plain OSS-Instruct at equal size (4M each): OSS-I-Qwen2.5-Coder-7B 82.9 / 73.8 / 84.4 / 73.3 vs OCI 86.8 / 83.2 / 87.2 / 75.3; OSS-I-Llama-3.1-8B 69.5 / 63.4 / 70.4 / 60.8 vs OCI 78.7 / 73.2 / 77.5 / 66.4. — [OpenCodeInstruct paper, Table 6](https://arxiv.org/html/2504.04030v2)
- Size scaling: the paper states 500K samples already surpass the instruct baselines and performance peaks at the full 5M; exact per-size values are only in a figure. — [OpenCodeInstruct paper, Fig. 3](https://arxiv.org/html/2504.04030v2)

### Inferences
- MBPP is far less responsive to instruction tuning than HumanEval. Across four independent papers the MBPP(+) gain for a ~7B model is +2 to +7 points (Magicoder-CL +7.2 MBPP+, WaveCoder-DS +2.2 MBPP, SelfCodeAlign +5.0 MBPP+, OCI-Qwen-7B +3.2 MBPP+ over the instruct model) while HumanEval(+) gains are +15 to +21. A realistic expectation for SFT on a 7-9B model is single-digit MBPP gains.
- From the OpenCodeInstruct 500K ablation (MBPP 81.0 random, 82.3 judge-filtered) versus the 5M result (86.8), most of the benefit of the data is obtained at 500K; a 10x increase buys roughly 4-5 MBPP points. At 500K, the tuned model is already at or slightly below the stock instruct model on MBPP (81.0-82.3 vs 83.5), so for a strong modern instruct model, small synthetic sets may not beat the off-the-shelf checkpoint at all.
- Data generated by a model close in capability/"distribution" to the student can beat data from a larger teacher (CodeQwen row of SelfCodeAlign Table 8), but for weaker students (StarCoder2-3B, and roughly a wash for Llama-3-8B) a stronger generator is as good or better. For a general-purpose 7-9B model that is not a code specialist, teacher-generated data is the safer choice.
- None of these pipelines requires MBPP data to generate the training set. MBPP dependence enters through (a) model selection on MBPP dev (WizardCoder) and (b) few-shot exemplars in the evaluation prompt (WizardCoder, WaveCoder, inspect_evals).
- OpenCodeInstruct ships with per-sample test results and judge scores, so it can be filtered directly (e.g. `average_test_score == 1` and judge = 5) without running a generation pipeline.

### Gaps
- Magicoder Table 1 and Table 4 cells did not render in the fetched HTML; the Magicoder-CL/S-DS numbers above come from the README table surfaced via search, not from the paper table itself. MagicoderS-CL-7B MBPP/MBPP+ values were not retrieved.
- SelfCodeAlign: number of responses sampled per instruction, tests per response, and the overall pass rate of the main 74K run were not found in the portion read (last ~40K characters of the paper were not read). StarCoder2-15B-Instruct final scores were not extracted.
- OpenCodeInstruct does not report base-model (pre-instruct) scores, nor the prompt template / MBPP variant used (likely EvalPlus, UNCONFIRMED), nor numeric values for the size-scaling curve.
- The `bigcode/self-oss-instruct-sc2-exec-filter-50k` and `ise-uiuc/Magicoder-OSS-Instruct-75K` dataset cards were not fetched; no numbers from them are included.
- No ablation was found that isolates dataset size in the 1K-50K range for MBPP specifically (WaveCoder's size ablation is HumanEval only and its extracted text is inconsistent).

---

## 2. Execution-verified data: how much does filtering by running solutions against tests help, and test-quality problems

### Takeaway
Evidence is mixed and depends on who wrote the tests and how hard the tasks are: self-validation filtering gave +3.6 HumanEval+ in SelfCodeAlign (where only 24% of samples pass), but only about +0.2 to +0.4 in OpenCodeInstruct (where 72% already pass with a 32B teacher), and it actively hurt on competitive-programming data (OpenCodeReasoning). An LLM-judge filter outperformed execution filtering in the one head-to-head comparison, and training exclusively on failing samples is consistently worst for function-level benchmarks.

### Cited Findings
- SelfCodeAlign (CodeQwen1.5-7B, HumanEval+, 15.6K samples each): passes only 65.2; random (24.2% pass) 61.6; failures only 57.9. — [SelfCodeAlign paper, Table 9](https://arxiv.org/html/2410.24198)
- In SelfCodeAlign the tests are written by the same model that writes the solution (the response is generated together with its test block), so "pass" means self-consistency, not independent verification. — [SelfCodeAlign paper](https://arxiv.org/html/2410.24198)
- SelfCodeAlign with execution filtering (67.1) beats unfiltered direct distillation from GPT-4o (65.9) at equal size on the same base; unfiltered GPT-3.5 OSS-Instruct gives 61.6. — [SelfCodeAlign paper, Table 11](https://arxiv.org/html/2410.24198)
- OpenCodeInstruct (Qwen2.5-Coder-7B, 500K each, MBPP / MBPP+): random 81.0 / 70.1; unit-test passes only 81.4 / 70.4; unit-test failures only 80.1 / 69.8; LLM-judge = 5.0 82.3 / 71.4. Judge-filtered data has a 77.4% test pass rate, i.e. the judge and the tests disagree on a substantial share of samples. — [OpenCodeInstruct paper, Table 4](https://arxiv.org/html/2504.04030v2)
- OpenCodeInstruct authors' explanation (per search summary of the paper): execution filtering discards unique questions whose correct solutions failed the generated tests, trading diversity for correctness. (search snippet only) — [OpenCodeInstruct paper](https://arxiv.org/abs/2504.04030)
- OpenCodeReasoning (Qwen2.5-14B-Instruct, DeepSeek-R1 teacher, competitive programming; LiveCodeBench / CodeContests): no filtering (445,618 samples) 54.1 / 16.59; correct solutions only (151,251) 47.0 / 15.34; incorrect solutions only (151,251) 52.3 / 15.53. The authors attribute this partly to incorrect solutions covering harder problems. — [OpenCodeReasoning paper, Table 3](https://arxiv.org/html/2504.01943v2)
- Magicoder used no execution filter at all and removing heuristically "noisy" samples did not help (55.5 -> 54.9 HumanEval+). — [Magicoder paper, Table 10](https://arxiv.org/html/2312.02120)
- CodeT (inference-time, not training): generating tests with the same model and selecting solutions by dual execution agreement raised code-davinci-002 HumanEval pass@1 to 65.8%, an absolute +18.8%. This shows model-written tests carry real signal for ranking candidate solutions even without ground-truth tests. — [CodeT abstract](https://arxiv.org/abs/2207.10397)
- Test quality in benchmarks themselves is poor enough to change rankings: HumanEval has 9.6 tests per task vs 764.1 in HumanEval+; adding tests lowers pass@1 by up to 19.3% (greedy up to 23.1%); 18 defects were found in HumanEval ground-truth solutions (11% of problems: 5 unhandled edge cases, 10 bad logic, 3 performance). — [EvalPlus paper](https://arxiv.org/html/2305.01210)

### Inferences
- The value of execution filtering scales with the error rate of the unfiltered pool. With a weak self-generator (24% pass) it is worth several points; with a strong 32B teacher (72% pass) it is close to noise on MBPP (+0.3-0.4). With a strong teacher, filtering is cheap insurance rather than a major lever.
- The "failures only" arm being only about 1 MBPP point below "passes only" in OpenCodeInstruct implies many "failures" are correct solutions paired with wrong generated tests. Hard filtering on LLM-written tests therefore removes good data; a softer rule (e.g. majority of tests pass, or agreement between multiple sampled solutions as in CodeT) or an LLM judge is supported as an alternative.
- For MBPP-style targets (easy, short functions) the OpenCodeReasoning result (filtering hurts) probably does not transfer: its mechanism is that correct-only data is biased to easy problems, which is the target distribution here. This is an inference, not a tested claim.
- Combining filters (tests pass AND judge = 5) was not tested in any source read.

### Gaps
- No source read reports an ablation where tests are written by a second, different model from the solution writer; SelfCodeAlign uses the same model and OpenCodeInstruct uses the same model (Qwen2.5-Coder-32B) for solutions, tests and judging. The "tests by a second model" question is unanswered by evidence found here.
- No direct measurement of the false-positive / false-negative rate of LLM-generated unit tests was found (CodeT's abstract gives none; the full paper was not read).
- CodeT's MBPP-specific numbers were not retrieved (abstract only).

---

## 3. MBPP-specific failure modes, MBPP vs sanitized vs MBPP+, prompt format effects, and the Inspect AI / inspect_evals MBPP task

### Takeaway
A meaningful share of MBPP failures are benchmark artefacts: underspecified descriptions, function names and argument conventions that exist only in the asserts, and wrong tests (EvalPlus removed about 12% of sanitized tasks across two cleanups). The inspect_evals task evaluates the sanitized test split with a 3-shot prompt that includes the test cases, asks for code only, extracts a fenced python block (falling back to the whole raw completion), and by default samples at temperature 0.5 over 5 epochs, so a model that emits prose without a fence, or a fence with trailing demo code, fails on format alone.

### Cited Findings

**inspect_evals MBPP task (source file read verbatim in this session)**
- Dataset: `google-research-datasets/mbpp`, config `sanitized`, split `test`, pinned revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`. — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- Instruction text (verbatim): "You are an expert Python programmer. You will be given a task, and the tests that your code must pass. Write the Python function to solve the task. Do not give additional explanations, just output the Python function. Only use imports that are included in Python's standard library." — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- **[MBPP-DATA]** Few-shot: after "For example:", three examples from config `full`, split `prompt`, task IDs 2, 3, 4, each rendered as `## Prompt N` (task text inside a python fence), `## Test Case N` (asserts inside a python fence), `## Completion N` (reference code inside a python fence). These are MBPP prompt-split items inside the evaluation prompt; they are not in the evaluated test split, but training on them would violate the reader's constraint. — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- Target section: "# Now, do it for the following task." then `## Prompt:` (task text in a python fence), `## Test Case:` (the test list in a python fence), then `## Completion:` left open. The model sees the asserts, hence the required function name and call signature. The whole thing is one user message via `prompt_template` followed by `generate()`. — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- Generation defaults: temperature 0.5 (task argument; docstring shows `-T temperature=0.0` as an option), 5 epochs with reducers `mean`, `pass_at_1`, `pass_at_2`, `pass_at_5`; metrics `accuracy()` and `stderr()`. So the default is NOT one greedy sample; a greedy pass@1 run needs temperature 0 (and epochs 1). — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- Extraction: `find_code` calls `extract_code_block(completion, "python")`; if it returns `None`, the entire raw completion is executed as code. The source comment states: "It is assumed that generated output is of the form: ```python [code output] ```". — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- Scoring: each target assert is appended to the extracted code as `assert ..., '<assert text>'`, and the result is run with `python -c <code>` in a docker sandbox with a 30-second timeout; CORRECT only if the process exits successfully. Any exception anywhere in the extracted block (a syntax error from stray prose, a failing self-written assert, an import that is not installed, `input()` calls) fails the sample; extra `print` output alone does not, since only the exit status is checked. — [inspect_evals mbpp.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py)
- `extract_code_block` is re-exported from a separate package `inspect_evals_utils.code` (alongside `extract_code_blocks`, `defines`, `strip_json_fence`). — [inspect_evals utils/code.py](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/utils/code.py)

**Test and task quality: MBPP vs sanitized vs MBPP+**
- Original MBPP: 974 problems with three test cases each. — [EvalPlus paper, related work](https://arxiv.org/html/2305.01210)
- MBPP-sanitized is the roughly 427-problem subset hand-verified by the original authors; MBPP+ v0.1.0 trimmed ill-formed problems to 399 and fixed incorrect reference implementations; v0.2.0 removed further tasks whose `test_list` was wrong (inherited from MBPP) leaving 378, which the maintainers say is worth about +4 points pass@1 on both base and plus tests. (search snippet of the EvalPlus release notes) — [EvalPlus repository](https://github.com/evalplus/evalplus)
- MBPP+ has about 35x more tests than MBPP. — [Magicoder paper](https://arxiv.org/html/2312.02120)
- The base-to-plus gap is large for ~7B models: Magicoder-CL-7B MBPP 64.2 vs MBPP+ 52.6; MagicoderS-DS 75.7 vs 64.4; OCI-Qwen2.5-Coder-7B 86.8 vs 74.9; Llama-3.1-8B-Instruct 68.3 vs 60.6. Roughly 8-12 points of "passing" solutions on the three original asserts are wrong on wider inputs. — [Magicoder README](https://www.github.com/ise-uiuc/magicoder), [OpenCodeInstruct paper, Table 3](https://arxiv.org/html/2504.04030v2)
- A third-party study using MBPP+ excluded a further 21 of the 378 tasks for oracle problems (350 remaining after their own removals). (search snippet only) — [The Limits of Inference Scaling Through Resampling](https://arxiv.org/pdf/2411.17501)
- MHPP error analysis of strong models (e.g. GPT-4) on MBPP: 18.82% of errors attributed to test-instance quality; 10.59% to unclear problem descriptions and 8.23% to missing constraints or incorrect test cases. (search snippet only) — [MHPP paper](https://arxiv.org/pdf/2405.11430)
- An ISSTA 2024 study notes incomplete problems are especially common in MBPP, e.g. a task asking for a dog's age in "dog years" with no definition of the conversion. (search snippet only) — [Oracle-guided Program Selection, ISSTA 2024](https://abhikrc.com/pdf/ISSTA24_Oracle_Guided.pdf)
- SpecFix found 5 of 542 HumanEval+/MBPP+ instances (<1%) where examples conflict with the reference solution, i.e. after EvalPlus cleaning, outright contradictions are rare. (search snippet only) — [SpecFix paper](https://arxiv.org/html/2505.07270)
- Perturbation study: making task descriptions ambiguous reduces pass@1 by 25-30% on average, incomplete descriptions by 20-25%. (search snippet only) — [When Prompts Go Wrong](https://arxiv.org/html/2507.20439)

**Prompt-format dependence across papers**
- WizardCoder evaluates MBPP three-shot with instruction "Create a Python script for this problem:" followed by the question and three test examples, on the 500-problem test split, n=200 samples at temperature 0.2. — [WizardCoder paper, App. A](https://arxiv.org/html/2306.08568)
- WaveCoder evaluates on MBPP (500) with few-shot inference; Magicoder / SelfCodeAlign / OpenCodeInstruct report EvalPlus-style MBPP and MBPP+ with greedy decoding. The same model gets different "MBPP" numbers under the two protocols: Magicoder-DS is 60.4 on MBPP(500) in WaveCoder's table, MagicoderS-DS 64.6 there vs 75.7 on EvalPlus MBPP. — [WaveCoder paper v4, Table 3](https://arxiv.org/html/2312.14187v4), [Magicoder README](https://www.github.com/ise-uiuc/magicoder)

### Inferences
- Under the inspect_evals harness, three format behaviours are fatal and cheap to train away: (1) prose before/after code with no fence (the raw completion is executed and raises SyntaxError); (2) code in a fence but with a different function name or argument structure than the asserts use; (3) appended example usage that raises (e.g. reading stdin, or self-written asserts that are wrong). Training targets consisting of exactly one ```python fence containing only the function (plus needed imports) match the harness's stated assumption.
- Because the prompt contains the asserts, the dominant MBPP-specific skill is reading the name, arity, argument types and expected return type/format from the test cases, not from the description. Synthetic training items should therefore have the same shape: short description + 3 asserts in the prompt, with the description sometimes deliberately terse so the asserts are the specification. This follows from the prompt format and the ambiguity findings; no paper read tests this directly.
- Whether the multi-block behaviour of `extract_code_block` returns the first or last block is unknown (see Gaps), so the safest training target emits exactly one fenced block.
- SelfCodeAlign-style training responses (explanation, then code, then a test block) are mismatched to a harness that says "Do not give additional explanations" and executes whatever is in the extracted block; if such data is used, responses should be reformatted to code-only.
- Numbers from different papers under the name "MBPP" are not comparable (974-full/500-test few-shot, sanitized 427, MBPP+ base 399/378, greedy vs sampled). Only within-paper deltas should be used for decisions.
- With temperature 0.5 and 5 epochs as the inspect_evals default, published inspect_evals MBPP numbers are means over sampled completions, not greedy pass@1; the reader's "one greedy sample" setting requires overriding the temperature.

### Gaps
- The implementation of `extract_code_block` (regex, behaviour with multiple fenced blocks, unlabeled fences, unterminated fences) lives in the separate `inspect_evals_utils` package, whose source could not be located in this session. UNCONFIRMED: whether an unlabeled ``` fence is accepted and which block is taken when several are present.
- The size of the sanitized test split used by inspect_evals (commonly cited as 257 test problems out of 427 sanitized) was not verified from the dataset card in this session. UNCONFIRMED.
- No controlled study was found that quantifies, for a fixed model, the MBPP pass@1 difference between prompts with vs without the test cases, or zero-shot vs 3-shot. The claim that function names are only inferable from tests is structurally evident from the dataset format but no source giving the percentage of tasks affected was found.
- No source was found that measures the rate of extraction/format failures (prose, missing fences) for small models on MBPP.
- MHPP, ISSTA, SpecFix and perturbation-study figures are from search snippets only and were not checked against the papers.
- The EvalPlus paper itself (v1) predates MBPP+ and contains no MBPP+ statistics; the 399/378 details come from release notes via search.

---

## 4. Short simple function tasks vs long complex instruction data; reasoning/comments before code vs code only

### Takeaway
Direct evidence is thin, but what exists points to simple NL-to-code function tasks transferring best to MBPP: MBPP scores were insensitive to whether seeds were algorithmic (competitive-programming) or generic, more Evol-Instruct complexity rounds stopped helping after round one, and NL-to-Code beat Code-to-Code instructions by 1.5-3.4 MBPP points. No controlled ablation of "explanation before code" versus "code only" for 7-9B models on MBPP was found.

### Cited Findings
- Seed type barely moves MBPP (Qwen2.5-Coder-7B, 2.5M each): algorithmic-small 83.3 / 71.4, algorithmic-large 83.1 / 72.2, generic-large 83.3 / 71.7 (MBPP / MBPP+); HumanEval moves more (81.7 / 84.8 / 85.4). For Llama-3.1-8B: 73.5 / 63.5, 74.1 / 64.9, 74.0 / 62.2. — [OpenCodeInstruct paper, Table 5](https://arxiv.org/html/2504.04030v2)
- NL-to-Code vs Code-to-Code instructions (2M each): Qwen2.5-Coder-7B MBPP 83.3 vs 81.8, MBPP+ 73.0 vs 70.1; Llama-3.1-8B MBPP 74.6 vs 71.2, MBPP+ 65.6 vs 61.1. — [OpenCodeInstruct paper, Table 5](https://arxiv.org/html/2504.04030v2)
- OpenCodeInstruct removed all instructions that contained Python code snippets during cleaning, i.e. the final set is NL-described tasks. — [OpenCodeInstruct paper](https://arxiv.org/html/2504.04030v2)
- Increasing instruction complexity via extra Evol-Instruct rounds: HumanEval 45.7 (round 0) -> 56.1 (round 1) -> 53.0 -> 54.3 -> 51.2. — [WizardCoder paper, Table 5](https://arxiv.org/html/2306.08568)
- At equal size (74K) on CodeQwen1.5-7B, Evol-Instruct data (complexity-evolved) gives HumanEval+ 59.1 vs 61.6 for OSS-Instruct and 67.1 for SelfCodeAlign, whose seeds are single documented Python functions. — [SelfCodeAlign paper, Table 11](https://arxiv.org/html/2410.24198)
- Fine-tuning on raw comment-function pairs (short functions, no instruction framing) gives no gain (HumanEval+ 34.1 = base), so "short functions" alone is insufficient; the instruction-response framing matters. — [Magicoder paper, Table 6](https://arxiv.org/html/2312.02120)
- Multi-task data helps generation a little: removing repair, translation or summarization tasks from WaveCoder's 20K each costs 2.5-3.1 HumanEval points (64.0 -> 60.9 / 60.9 / 61.5). — [WaveCoder paper v4, Table 6](https://arxiv.org/html/2312.14187v4)
- Response format: SelfCodeAlign responses contain a prose explanation, then code, then tests. OpenCodeInstruct capped responses at 1024 tokens and, because the reasoning model QwQ-32B-Preview exceeded that limit, used a code-only prefix for it; Qwen2.5-Coder-32B-Instruct was chosen as the response generator and is described as giving "a several point improvement" over alternatives (no table). — [SelfCodeAlign paper, App. D.1](https://arxiv.org/html/2410.24198), [OpenCodeInstruct paper](https://arxiv.org/html/2504.04030v2)
- Long reasoning-trace distillation (OpenCodeReasoning: 736,712 Python samples over 28,904 competitive-programming questions, DeepSeek-R1 teacher, up to 16K output tokens) is evaluated only on LiveCodeBench and CodeContests (OCR-Qwen-7B-Instruct 51.3 / 18.1); the paper reports no HumanEval or MBPP results and no reasoning-vs-code-only ablation. — [OpenCodeReasoning paper](https://arxiv.org/html/2504.01943v2)

### Inferences
- For an MBPP target, hard algorithmic data is not harmful but buys nothing over generic function tasks, so the cheaper, shorter, generic NL-to-function data is the efficient choice.
- Complexity-increasing evolution beyond one round is not supported for function-level benchmarks; breadth (many distinct simple tasks) appears to matter more than depth.
- On reasoning-before-code: the harness instruction ("just output the Python function") plus fence-or-raw extraction makes code-only targets the lower-risk format. A brief in-code comment or docstring is harmless to extraction. Whether a short pre-code explanation improves correctness for 7-9B models on MBPP is untested in the sources read; the only relevant observation is that the two strongest SFT pipelines use different formats (SelfCodeAlign: explanation + code + tests; OpenCodeInstruct: standard instruct-model answers), so no conclusion can be drawn.

### Gaps
- No paper found ablates response format (code only vs explanation-then-code vs chain-of-thought) for a 7-9B model on MBPP/HumanEval. This key question is unanswered by direct evidence.
- No paper found ablates instruction length or solution length of training samples against MBPP score.
- No paper found tests training on MBPP-shaped prompts (description + asserts) versus generic instruction prompts.

---

## 5. Typical MBPP pass@1 for fine-tuned 7-9B models, and gains from RL / self-distillation with unit-test rewards

### Takeaway
Reported numbers for fine-tuned 7-9B models span about 47-65 on the few-shot MBPP-500 protocol (2023-era code models), 64-87 on EvalPlus MBPP-base and 53-75 on MBPP+ (2024-2025 models); no paper read reports MBPP-sanitized (427/257) pass@1 under the inspect_evals prompt. RL with unit-test rewards shows large gains for sub-2B models (+13 to +19 MBPP points) but only about +3 for a 7B model, comparable to SFT.

### Cited Findings

**Per-paper numbers for ~7-9B models (protocols differ; do not compare across blocks)**
- MBPP-500 test split, few-shot prompt: CodeLlama-7B 41.4; CodeLlama-7B-instruct 44.4; WaveCoder-CL-7B 47.2; DeepSeekCoder-6.7B base 60.6; Magicoder-DS 60.4; WaveCoder-DS 62.8; DeepSeekCoder-6.7B-instruct 62.8; WaveCoder-Pro 63.6; WaveCoder-Ultra 64.4; MagicoderS-DS 64.6. For scale: StarCoder-15B 43.3, WizardCoder-15B 51.8, ChatGPT 52.2. — [WaveCoder paper v4, Table 3](https://arxiv.org/html/2312.14187v4)
- EvalPlus MBPP / MBPP+, greedy: Magicoder-CL-7B 64.2 / 52.6; MagicoderS-DS-6.7B 75.7 / 64.4. — [Magicoder README](https://www.github.com/ise-uiuc/magicoder)
- EvalPlus MBPP+, greedy: CodeLlama-Python-7B 45.4; Magicoder-CL-Mixtral-7B (20K Mixtral-generated) 50.4. — [Magicoder paper, Table 7](https://arxiv.org/html/2312.02120)
- EvalPlus MBPP+, greedy: CodeQwen1.5-7B-Base 60.2; CodeQwen1.5-7B-OctoPack 63.2; SelfCodeAlign-CQ-7B 65.2; Magicoder-S-DS-6.7B 65.4; CodeQwen1.5-7B-Chat 67.2. — [SelfCodeAlign paper, Table 1](https://arxiv.org/html/2410.24198)
- MBPP / MBPP+ (2025): Llama-3.1-8B-Instruct 68.3 / 60.6 -> OCI-Llama-3.1-8B 77.5 / 66.4; Qwen2.5-Coder-7B-Instruct 83.5 / 71.7 -> OCI-Qwen2.5-Coder-7B 86.8 / 74.9; OpenCoder-8B-Instruct 79.1 / 69.0. — [OpenCodeInstruct paper, Table 3](https://arxiv.org/html/2504.04030v2)
- Same paper, 500K-sample tuned Qwen2.5-Coder-7B: MBPP 80.1-82.3, MBPP+ 69.8-71.4 depending on filter. — [OpenCodeInstruct paper, Table 4](https://arxiv.org/html/2504.04030v2)

**RL / unit-test-reward evidence** (all search snippets only; papers not read)
- Process-supervised RL: Qwen2.5-7B-RL MBPP pass@1 62.4% -> 65.4% using a process reward model alongside unit-test outcomes. — [Process Supervision-Guided Policy Optimization for Code Generation](https://arxiv.org/pdf/2410.17621)
- Dense multi-component reward (unit-test pass signal primary, syntactic and structural rewards auxiliary) on Qwen2.5-Coder-1.5B-Instruct: MBPP 0.460 -> 0.653, MBPP+ 0.413 -> 0.556. — [Domain-Adaptable RL for Code Generation with Dense Rewards](https://arxiv.org/html/2605.21180)
- RL from verification feedback with LoRA on Qwen3-0.6B and Llama3.2-1B: combined unit-test + static-analysis reward improves MBPP pass@1 by up to 13 points for Qwen3-0.6B; a static-analysis-only reward shortens completions without improving functional correctness. — [Improving Small Language Models for Code Generation with RL from Verification Feedback](https://arxiv.org/html/2605.30478v1)
- Self-generated, self-validated data as a form of self-distillation: CodeQwen1.5-7B +5.0 MBPP+ (60.2 -> 65.2) and +21.4 HumanEval+ from 74K own-generated, execution-filtered samples; Llama-3-8B HumanEval+ 29.3 -> 42.7 from its own data. — [SelfCodeAlign paper, Tables 1 and 8](https://arxiv.org/html/2410.24198)

### Inferences
- A general-purpose 8B instruct model (Llama-3.1-8B-Instruct class) starts around 68 MBPP / 61 MBPP+ and has demonstrated headroom of about +9 / +6 from SFT on a large synthetic set; a current code-specialised 7B instruct model starts around 83.5 / 71.7 with only about +3 demonstrated headroom. The achievable gain depends far more on the starting checkpoint than on the pipeline.
- On the 7B scale, RL with unit-test rewards (+3.0 MBPP) has not been shown to beat SFT on synthetic data (+2 to +7), and the large RL gains are confined to models under 2B where baseline scores are low. RL also needs an executable task set with tests; **[MBPP-DATA]** the RL papers' training task sources were not checked, and some RL-for-code work trains on the MBPP train split, which the reader cannot do.
- Since the 1.5B RL result baseline (0.460 MBPP) is far below the same model's EvalPlus score in OpenCodeInstruct (69.2), that paper uses a different MBPP protocol; its absolute numbers should not be compared with the others.

### Gaps
- No paper read reports pass@1 on MBPP-sanitized test (the inspect_evals split) for fine-tuned 7-9B models; all numbers above are MBPP-500 few-shot or EvalPlus MBPP/MBPP+. A direct answer to "typical MBPP sanitized pass@1" is not available from these sources.
- The EvalPlus leaderboard page was not fetched; leaderboard values appear only as quoted in the papers above.
- RL papers were seen only through search summaries: training data (possible MBPP train usage), decoding settings, MBPP variant, and whether gains are over a base or instruct model are UNCONFIRMED. Two of the three have 2026 arXiv identifiers and are recent preprints.
- No source found on classic self-distillation loops (generate, keep test-passing outputs, retrain, repeat for multiple rounds) with MBPP results for 7-9B models beyond SelfCodeAlign's single round.

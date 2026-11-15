# Winning strategies from competitions where teams fine-tuned small (7-14B) open models under compute limits

Reader context: designing a LoRA fine-tune of a ~9B base (non-chat) model for joint instruction following + competition math + Python code generation, partly graded on hidden benchmarks.

Confidence legend used below: **[primary]** = read directly from the paper / slides / config / model card; **[secondary]** = from a third-party summary; **[unconfirmed]** = could not be verified against a primary source.

---

## 1. NeurIPS 2023 LLM Efficiency Challenge ("1 LLM + 1 GPU + 1 Day"): what the winners of both tracks did, and what the post-competition analysis (arXiv 2503.13507) concludes

### Takeaway
Both winners used off-the-shelf LoRA tooling on a strong base model (Mistral-7B on the 4090, Qwen-14B on the A100) and spent their effort on data curation; neither wrote custom training code. The 4090 winner ranked only 20th of 57 on the public (open) tasks but 1st on the hidden tasks, because it deliberately avoided training on the public benchmarks' data, and the organisers' analysis found open-vs-hidden score correlation was near zero (-0.08 on 4090, 0.18 on A100).

### Cited Findings

**Competition rules and scoring**
- Rules: fine-tune an open, non-instruction-tuned base model on one GPU (RTX 4090 24 GB or A100 40 GB) for at most 24 h using only open-source data; LLM-generated data (ChatGPT/GPT-4) was banned. **[primary]** — [Analysis paper, arXiv 2503.13507](https://arxiv.org/html/2503.13507); [Birbal paper, arXiv 2403.02247](https://arxiv.org/pdf/2403.02247)
- 182 registered teams, 225 submissions; top 10 per track advanced to the reproduction round. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- Evaluation used an adapted Stanford HELM. Open stage: 600 questions over six scenarios (300 min limit). Closed stage: 5,000 questions over five held-out scenarios (600 min limit). Stage score = geometric mean of mean win rates across scenarios. Final = 1/3 x open + 2/3 x closed. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- Open tasks were drawn from MMLU, TruthfulQA, BBQ, GSM8K and BIG-bench; hidden tasks from SAMSum, Corr2cause, MATH (chain-of-thought) and ETHICS. **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247)

**4090 track winner: Team Upaya, "Birbal" (Mistral-7B)**
- Base model Mistral-7B-v0.1, 4-bit QLoRA, fine-tuned ~16 h on one RTX 4090. They state that Mistral-7B and Qwen-14B were the two best options within 24 GB and chose the 7B "to fit more high-quality instructions covering multiple tasks". **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- Explicit design choice: "Evaluation included both HELM and hidden tasks; to excel in the latter, minimize reliance on HELM-sourced data." They chose to curate existing datasets rather than generate data with a 7B model, and to invest in data rather than kernel/hardware optimisation. **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- Dataset mixture (three sizes built; time budget allowed 200K x 3 epochs, 400K x 2 epochs, 700K x 1 epoch) **[primary]** — [Birbal paper, Table 1](https://arxiv.org/pdf/2403.02247):

  | Source | 200K | 400K | 700K |
  |---|---|---|---|
  | LIMA | 1K | 1K | 1K |
  | Open-Platypus (GPT-generated ~10% removed) | 25K | 25K | 25K |
  | Natural Instructions, Exact Match tasks | 50K | 110K | 220K |
  | Natural Instructions, Generation tasks | 50K | 110K | 220K |
  | OpenbookQA | 5K | 5K | 5K |
  | QuAC | 10K | 10K | 10K |
  | CNN/DailyMail | 15K | 28K | 28K |
  | MathInstruct (LLM-generated examples removed) | 50K | 100K | 200K |

- Natural Instructions curation, four stages: (1) hand-select 463 of 1,600+ tasks (dropped 576 non-English tasks, MMLU-derived tasks, question-generation, wrong-candidate-generation, math, and linguistic tagging tasks such as PoS/NER/coreference "considering the inherent strength of most LLMs" there); (2) manually label each task Exact Match vs Generation; (3) run few-shot inference with the *base* Mistral-7B on every task to measure accuracy (EM tasks) or ROUGE (generation tasks); (4) difficulty-weighted sampling: for EM tasks drop the lowest-accuracy tasks as too hard to learn in budget, then sample more from lower-accuracy tasks; for generation tasks sample 40% of examples in the ROUGE [0, 0.2) bucket and 10% from each higher bucket; then randomly cut each pool to 50K. **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- LoRA / training settings: rank 128, alpha 256, dropout 0.05; targets q, k, v, o plus MLP gate/up/down projections; lr 2e-5 cosine, 100 warmup steps, weight decay 0.01, max grad norm 0.3; optimizer paged_adamw_32bit; micro-batch 2; NEFTune noisy embeddings (alpha 5); sample packing on; max sequence length 8192; bf16 compute over NF4 4-bit base; flash attention; eval/save every 64 steps and pick the checkpoint with lowest eval loss; 2,000 examples held out as validation; trained with Axolotl. **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247); [nips_02.yml config](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge/blob/main/training/axolotl/examples/mistral/nips/nips_02.yml)
  - Minor conflict: the paper says gradient accumulation 3; the released YAML and the Xebia write-up say 4. — [Birbal paper](https://arxiv.org/pdf/2403.02247) vs [Xebia blog](https://xebia.com/blog/winning-recipe-for-finetuning-llm/)
- Training format in the config is the Alpaca-style template (`alpaca_chat.load_qa`); few-shot inference used an Alpaca-style `### Instruction: / ### Input: / ### Response:` template with demonstrations followed by the query. **[primary]** — [Upaya07 GitHub repo](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge)
- In hindsight the 2-epoch checkpoint was best on validation (of 3 epochs run). **[secondary]** — [Xebia blog](https://xebia.com/blog/winning-recipe-for-finetuning-llm/)
- Results: open stage Birbal-200K placed **20th of 57** with 0.64 (30 teams qualified); hidden stage **1st** with 0.660. Final table: Birbal open 0.52 / closed 0.61 / final 0.58; Rank-2 (Qwen-14B based) open 0.63 / closed 0.32 / final 0.42; Rank-3 (Mistral-7B based) open 0.21 / closed 0.47 / final 0.38. The paper describes this as a ">35%" margin over second place. **[primary]** — [Birbal paper, Table 2](https://arxiv.org/pdf/2403.02247)
- The 400K and 700K submissions failed to pass the open stage at all. **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- Dataset-size ablation (base Mistral-7B vs 200K x 3 ep vs 400K x 2 ep vs 700K x 1 ep) **[primary]** — [Birbal paper, Table 3](https://arxiv.org/pdf/2403.02247):

  | Task (stage) | Base | 200K | 400K | 700K |
  |---|---|---|---|---|
  | MMLU EM (open) | 0.64 | 0.63 | 0.62 | 0.62 |
  | TruthfulQA EM (open) | 0.56 | 0.59 | 0.41 | 0.46 |
  | BIG-bench EM (open) | 0.29 | 0.33 | 0.38 | 0.37 |
  | GSM8K EM (open) | 0.33 | 0.44 | 0.61 | 0.56 |
  | BBQ EM (open) | 0.80 | 0.74 | 0.67 | 0.64 |
  | corr2cause EM (closed) | 0.46 | 0.61 | 0.55 | 0.56 |
  | MATH CoT (closed) | 0.02 | 0.12 | 0.12 | 0.12 |
  | ethics-virtue EM (closed) | 0.68 | 0.89 | 0.79 | 0.81 |
  | ethics-utilitarianism EM (closed) | 0.61 | 0.72 | 0.66 | 0.64 |

- Authors' reading of the ablation: "adding more data points led to a drop in performance in a number of open tasks. However, performance on closed tasks improves as we scale a number of data points." Of 31 evaluations, the base model was best on 6, 200K on 12, 400K on 10, 700K on 10. **[primary]** — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- The team also credits a member for "post-processing rules for model output" in the submission. **[primary]** — [Upaya07 GitHub repo](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge)

**A100 track winner: Team Percent-BFD (Qwen-14B)**
- Base model Qwen-14B, LoRA via LLaMA-Factory. Selected after LoRA-fine-tuning Qwen-14B, Llama2-7B, Llama2-13B and Llama-33B on the same data: Qwen-14B had the best scores and the second-fastest training speed. **[primary]** — [Percent-BFD NeurIPS slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf)
- Hyperparameters: `lora_rank 8`, `lora_alpha 32`, `lora_dropout 0.1`, `lora_target c_attn,c_proj`, batch 2 x grad-accum 4, cosine schedule, `learning_rate 5e-5`, `num_train_epochs 1.0`, bf16, no quantisation. **[primary]** — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf)
- Prompt template: a Vicuna-style system line ("A chat between a curious user and an artificial intelligence assistant...") then `Human: {query}` / `Assistant: {answer}`. They report: "w/wo template, the score difference is less than 0.01." **[primary]** — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf)
- Dataset iteration log **[primary]** — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf):
  - V1 baseline: MMLU (high-school subjects, moral), BIG-bench (logical, safety), LIMA, databricks-dolly-15k, oasst1-en, alpaca-gpt4, CodeAlpaca-20k, competition-math.
  - V2 (on Qwen-14B): added MetaMathQA and summarize_from_feedback_tldr_3_filtered. Result: "No significant effect in math & sum"; "Too much data requires 30h training". Both removed again.
  - V3: added GPT4-LLM-Cleaned; "Overall score improved".
  - V4: "Remove all llm generated data" (dropped GPT4-LLM-Cleaned, alpaca-gpt4, CodeAlpaca-20k) and competition-math; added cnn_dailymail, truthful_qa, gsm8k; "CNN, truthful_qa improved".
  - V5: re-added summarize_from_feedback_tldr_3_filtered; "MMLU, BBQ, CNN improved".
- Final composition: MMLU (all high-school subjects, philosophy, moral_disputes, moral_scenarios; random 10% subsample); BIG-bench (analytic_entailment, causal_judgment, empirical_judgments, known_unknowns, logical_deduction, emoji_movie, strange_stories, snarks, dark_humor_detection); CNN/DailyMail (random 10%); TruthfulQA; GSM8K; LIMA duplicated 10x; databricks-dolly-15k; oasst1-en; summarize_from_feedback_tldr_3_filtered; then a 67% random subset of the whole. **[primary]** — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf)
- Conflict on final size: the slides say "final training set size 24K"; the Xebia write-up says about 247,000 rows. I could not resolve which is right (the dataset `Percent-BFD/nips_data_v7` on Hugging Face would settle it). — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf) vs [Xebia blog](https://xebia.com/blog/winning-recipe-for-finetuning-llm/)
- Note the contrast with the 4090 winner: Percent-BFD trained directly on the public benchmarks' own data (MMLU, BIG-bench, TruthfulQA, GSM8K), whereas Upaya avoided it. **[primary]** — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf); [Birbal paper](https://arxiv.org/pdf/2403.02247)

**Post-competition analysis paper (arXiv 2503.13507, Saroufim et al., March 2025)**
- Headline conclusions: top-performing models "overfit benchmark datasets"; "data curation is essential"; "the winning submissions utilized standard open-source libraries and focused primarily on data curation"; "None of them created custom code for fine-tuning." **[primary]** — [arXiv 2503.13507](https://arxiv.org/abs/2503.13507)
- "Interestingly, the winning entries did not achieve the highest scores on the open evaluation tasks", which the authors read as the winners being less overfitted. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- Open-vs-closed score correlation: about -0.08 (4090) and 0.18 (A100). Full-vs-hidden correlation: about 0.85 (4090) and 0.20 (A100). The top-ranked final model was best in only 1 of 8 scenarios. Some submissions scored "at about chance level" on some closed tasks. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- Caveat stated by the authors: low agreement means either overfitting or that open and closed tasks measure different skills; the body does not establish which. **[primary]** — [ar5iv mirror of 2503.13507](https://ar5iv.labs.arxiv.org/html/2503.13507)
- Their recommendation against overfitting: use diverse fine-tuning samples rather than data drawn only from the open evaluation tasks. **[primary]** — [ar5iv mirror](https://ar5iv.labs.arxiv.org/html/2503.13507)
- Operational failure modes: ~30% of submissions did not finish all questions in the time limit; ~10% ran out of memory at inference; "More than half of the submitted entries failed to build due to unpinned dependencies." **[primary]** — [ar5iv mirror](https://ar5iv.labs.arxiv.org/html/2503.13507)
- Cost: most fine-tunes took 15-20 h, two took ~2 h; roughly USD 7 on a 4090 and USD 20 on an A100 at Vast.ai rates. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- 4090 second place used an ensemble: regular expressions classified the task type and routed each query to a specialised model. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- Most popular base models across all submissions: Qwen-14B, Mistral-7B, Llama2-7B. **[primary]** — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- Inconsistency inside the analysis paper: it attributes a LIMA + Open-Platypus + Dolly-15k + OASST1 mixture and LLaMA-Factory to the A100 winner and cites the Birbal paper (Jindal et al.) for both winners. Open-Platypus was in Upaya's (4090) mix, not in Percent-BFD's slide list; LLaMA-Factory was Percent-BFD's tool. Treat the analysis paper's per-team details as blended and prefer the teams' own artefacts. — [arXiv 2503.13507](https://arxiv.org/html/2503.13507) vs [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf) and [Birbal paper](https://arxiv.org/pdf/2403.02247)

**Other entrant**
- nextAI fine-tuned LLaMA-2 70B with QLoRA + Flash Attention 2 on one A100 40 GB in 24 h using the Alpaca prompt format; placement not stated on the abstract page. **[primary, partial]** — [arXiv 2604.09034](https://arxiv.org/abs/2604.09034)

### Inferences
- The two winners' LoRA settings are at opposite ends (rank 128/alpha 256 on all attention + MLP modules at lr 2e-5 for 3 epochs, vs rank 8/alpha 32 on two attention modules at lr 5e-5 for 1 epoch). Since both won, adapter hyperparameters were not the deciding factor; data mixture and base model were.
- The Birbal ablation is the cleanest evidence available on the data-size/epoch trade-off for a ~7B model: math skills (GSM8K 0.33 -> 0.44 -> 0.61) scaled with more MathInstruct, while behaviours the base model already had (TruthfulQA, BBQ, MMLU) degraded as the mixture grew or drifted. For a three-skill target, expect a per-skill tension rather than a single optimum, and track each skill separately.
- MATH CoT went 0.02 -> 0.12 at every dataset size (50K, 100K, 200K MathInstruct). The jump likely reflects learning the answer format/CoT behaviour rather than new math ability, since quadrupling math data added nothing. That suggests format alignment is a large share of gain from a base (non-chat) model on a strictly parsed benchmark.
- Birbal's sampling rule (measure base-model few-shot performance per task; drop tasks that are hopeless; oversample tasks that are weak but learnable) is a directly reusable recipe for allocating a fixed example budget across instruction-following, math and code sub-tasks.
- Percent-BFD's finding that a system-prompt template changed scores by <0.01 applies to few-shot HELM evaluation, where demonstrations set the format. It should not be assumed to hold for zero-shot evaluation of a base model, where the template is the only format signal.
- Percent-BFD's V2 experience (adding MetaMathQA and a summarisation set gave "no significant effect" but pushed training to 30 h) is a warning that adding a large single-domain corpus can cost budget without moving the metric.
- Rank-3 in the 4090 track scored 0.0 on both BIG-bench and GSM8K yet finished third on the strength of hidden tasks, showing how heavily the 2/3 hidden weighting rewarded general robustness over public-task tuning.

### Gaps
- I could not confirm whether either winner deliberately formatted training examples to mirror HELM's few-shot evaluation prompts; the analysis paper is silent and neither team's material states it. Birbal's use of few-shot inference was for data selection.
- Percent-BFD's final dataset size is unresolved (24K per slides vs ~247K per Xebia).
- No per-version numeric scores for Percent-BFD's V1-V5 iterations are in the slide text (they are in chart images I could not read).
- A100 2nd/3rd place and 4090 2nd place base models and datasets are not described in the sources I could read.
- No team reported a controlled LoRA-rank or learning-rate ablation.

---

## 2. Kaggle AI Mathematical Olympiad progress prizes (1, 2, 3) and NuminaMath: recipes, and what transfers to a single greedy sample without tools or voting

### Takeaway
AIMO-1 (Numina) and AIMO-2 (NVIDIA NemoSkills) both won with full fine-tuning of a base (non-instruct) 7B/14B model on large chain-of-thought (CoT) math corpora followed by a tool-use stage, then heavy sampling and voting at inference. The CoT stage alone is what transfers to a greedy, tool-free setting: it took DeepSeekMath-7B from 46.8-51.7% to 56.3% on MATH zero-shot greedy, but it was nearly useless on olympiad-level problems (8/50 even with maj@32).

### Cited Findings

**AIMO Progress Prize 1 (2024): Numina, NuminaMath-7B-TIR**
- Winning score 29/50 on the private test set. **[primary]** — [HF blog: How NuminaMath Won the 1st AIMO Progress Prize](https://huggingface.co/blog/winning-aimo-progress-prize). The model card states 29/50 on "public and private" sets. — [NuminaMath-7B-TIR model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
- Base: `deepseek-ai/deepseek-math-7b-base` (a base, not instruct, model). **Full fine-tuning, no LoRA/DoRA.** Two-stage recipe following MuMath-Code. **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)
- Stage 1 (CoT): several hundred thousand natural-language problem/solution pairs with solutions templated as chain of thought, drawn from Chinese high-school exercises and US/international olympiad problems. The released dataset NuminaMath-CoT is described as "860k+ math competition problem-solution pairs". **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize); [NuminaMath-7B-CoT model card](https://huggingface.co/AI-MO/NuminaMath-7B-CoT)
- Stage 2 (TIR): ~60,000 problems with numerical (mostly integer) answers; GPT-4 generated ToRA-style reasoning with code execution; solutions whose final answer mismatched the reference were filtered out; process repeated three times. **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)
- Hyperparameters **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize):

  | | Stage 1 (CoT) | Stage 2 (TIR) |
  |---|---|---|
  | Learning rate | 2.0e-5 | 2.0e-5 |
  | Total batch size | 32 | 32 |
  | Block size | 2048 | 1024 |
  | Epochs | 3 | 4 |
  | Scheduler | cosine | cosine |
  | Warmup ratio | 0.0 | 0.1 |

  Other: TRL SFTTrainer with packing, gradient checkpointing, DeepSpeed ZeRO-3, Adam (0.9, 0.999), one node of 8x H100, ~10 h. — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize); [NuminaMath-7B-TIR model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
- MATH benchmark, zero-shot greedy **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize):

  | Model | CoT | TIR |
  |---|---|---|
  | DeepSeekMath-7B-Instruct | 46.8 | 57.4 |
  | DeepSeekMath-7B-RL | 51.7 | 58.8 |
  | NuminaMath-7B-CoT (stage 1 only) | 56.3 | - |
  | NuminaMath-7B-TIR (stage 2) | - | 68.2 |

- Other greedy vs voting numbers for the TIR model: GSM8K 84.6% and MATH 68.1% (0-shot); AMC 2023 20/40 greedy vs 31/40 maj@64; AIME 2024 5/30 greedy vs 10/30 maj@64. **[primary]** — [NuminaMath-7B-TIR model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
- Public leaderboard progression: stage-1-only CoT model 8/50 with maj@32; MMOS-style single-step Python model 16/50 with maj@32; on-policy KTO model 27/50. **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)
- Model-card limitation: with greedy decoding the model handles AMC-12-level problems but "often struggles generate a valid solution on harder problems at the AIME and Math Olympiad level"; weak on geometry; "should not be used for general chat applications". **[primary]** — [NuminaMath-7B-TIR model card](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
- Inference (SC-TIR): N = 48 sampled candidates, depth M = 4 rounds of code execution, majority vote; raising N or M did not help. 8-bit AutoGPTQ quantisation calibrated on training data, with a "small drop in accuracy". **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)
- Validation discipline: with only 50 public problems and 2 submissions/day they relied on four internal sets: AMC (83 problems, AMC12 2022-23), AIME (90 problems, 2022-24), MATH level 4 (754) and MATH level 5 (721), integer-answer subsets; seed-to-seed variance with SC-TIR was ~1-3% across 5-10 seeds. **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)
- What did not work: larger models (InternLM-20B, CodeLlama-33B, Mixtral-8x7B) could not beat DeepSeekMath-7B; PPO/RLOO with code-execution rewards gave no significant gains; model merging (DARE, TIES, WARP) of SFT with KTO or with public DeepSeekMath models caused "significant regressions". KTO on on-policy samples (4 completions per prompt, labelled correct/incorrect) gave "a few percentage points" over SFT on internal evals. **[primary]** — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)

**AIMO Progress Prize 2 (2025): NVIDIA NemoSkills, OpenMath-Nemotron**
- Winning Kaggle submission: an intermediate OpenMath-Nemotron-14B built on **Qwen2.5-14B-Base**; 34/50 on the private leaderboard. **[primary]** — [AIMO-2 Winning Solution paper, arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- Data (OpenMathReasoning): ~540K unique problems; 3.2M CoT solutions (kept from 5.2M generated); 1.7M TIR solutions; 566K GenSelect examples; total SFT mix 5.5M samples. Kaggle CoT model was trained on 2.2M DeepSeek-R1-generated solutions for 8 epochs. **[primary]** — [arXiv 2504.16891](https://arxiv.org/abs/2504.16891); [HTML](https://arxiv.org/html/2504.16891v1)
- Final-model hyperparameters: AdamW, weight decay 0.01, cosine decay with 10% linear warmup; peak lr 3e-4 (1.5B), 2e-4 (7B), 1e-4 (14B, 32B), decaying to 1000x lower; batch 1024; 6 epochs; average of 4 checkpoints. Then a second SFT round on 2.2M harder problems for 4 epochs (problems with pass rate above 0.3 removed; solutions under 5,000 tokens removed). **[primary]** — [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- Kaggle TIR stage: only 400 steps at constant lr 1e-5 on a 15K filtered TIR set; final submission was a linear weight merge CoT x 0.3 + TIR x 0.7, which beat mergekit methods. Merge ablation on Comp-Math-24-25, maj@16: CoT 62.9, TIR 66.8, merged 69.1. **[primary]** — [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- Single-sample vs voting, Comp-Math-24-25, pass@1 (maj@64) **[primary]** — [arXiv 2504.16891, Table 7](https://arxiv.org/html/2504.16891v1). The fetched table's column mapping for AIME/HMMT was unreliable, so only the Comp-Math column is reported here and should be re-checked against the PDF before quoting:

  | Model | CoT | TIR |
  |---|---|---|
  | 1.5B | 61.6 (80.0) | 52.0 (83.3) |
  | 7B | 74.8 (80.0) | 72.9 (83.3) |
  | 14B | 76.3 (83.3) | 76.3 (86.7) |
  | 32B | 76.5 (86.7) | 78.4 (93.3) |

- Second-round "harder problems" SFT: 14B CoT 62.9 -> 65.2 (helped); 14B TIR 74.6 -> 73.4 (flat/slightly worse); 1.5B CoT 55.1 -> 58.2. **[primary]** — [arXiv 2504.16891, Table 6](https://arxiv.org/html/2504.16891v1)
- Mixing CoT and TIR data in one run vs training sequentially gave similar accuracy. **[primary]** — [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- Leaderboard noise: 50 public problems in random order, one submission/day, large variance between identical submissions; local evals sometimes disagreed with the leaderboard; 7B models matched or beat the 14B locally but not on the leaderboard. **[primary]** — [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- Inference config (not transferable to greedy single-sample, listed for completeness): 4x L4, 5 h limit, TensorRT-LLM FP8 + ReDrafter speculative decoding, batches of 16 samples per question with early stop once ~4-5 agree, up to 6 code calls per generation. **[primary]** — [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)

**AIMO Progress Prize 3 (2025-26)**
- **[unconfirmed]** I could not find an official winner write-up. An informal Lean Zulip post says top entries used tool calls with a model around the size of gpt-oss-120b (quantised), i.e. far outside the 7-14B range. — [Lean Zulip archive thread](https://leanprover-community.github.io/archive/stream/219941-Machine-Learning-for-Theorem-Proving/topic/Kaggle.2C.2038.2F50.20already.20via.20open.20models.2C.20.242M.20prize.20fund.html)
- A participant blog describes an AIMO3 solver built on GPT-OSS-120B using prompt diversity and voting, with no fine-tuning; it is a personal project, not the winning solution. — [AIMO3 solver blog](https://cyberzzhhss.github.io/machine%20learning/computer%20science/mathematics/aimo3_solver_blog/)

### Inferences
- **What transfers to one greedy sample with no tools:** the stage-1 CoT fine-tune. Its measured value on a 7B base is roughly +5 to +10 points on MATH greedy relative to the vendor's own instruct/RL tunes (56.3 vs 46.8/51.7). Everything that produced the headline AIMO scores (TIR, SC-TIR with 48 samples, maj@64, GenSelect, early-stop voting) is unavailable in that setting.
- **Size of the voting gap to budget for:** greedy is far below voting on hard problems (AIME 2024: 5/30 greedy vs 10/30 maj@64; AMC 2023: 20/40 vs 31/40; NemoSkills 7B CoT 74.8 pass@1 vs 80.0 maj@64). On MATH-level difficulty the model is mostly consistent; the gap opens on AIME/olympiad difficulty. For a greedy-graded MATH-style benchmark, CoT data at MATH/AMC difficulty is the relevant part of NuminaMath-CoT; olympiad-proof data is unlikely to pay off at 9B greedy.
- **Hyperparameter caution:** both AIMO winners did *full* fine-tuning at lr 2e-5 (Numina) or 1e-4 with batch 1024 (NemoSkills) on hundreds of thousands to millions of examples for 3-8 epochs. These learning rates are not LoRA learning rates and the data volumes exceed a small budget; the transferable parts are the data source, CoT templating, answer-format consistency (final boxed/integer answer) and filtering by answer correctness, not the numeric settings.
- **Answer-verified filtering is a common thread:** Numina discarded generated solutions whose final answer mismatched the reference; NemoSkills kept 3.2M of 5.2M generated CoT solutions. Training only on solutions that reach the right answer in the target format appears to be a standard ingredient.
- **Base (non-chat) model precedent:** both winners started from base models (DeepSeekMath-7B-base, Qwen2.5-14B-Base) and imposed their own prompt/solution template through SFT, which matches the reader's ~9B base-model setting.
- **Merging is fragile:** Numina saw regressions from DARE/TIES merging; NemoSkills found only a plain linear merge of two checkpoints *from the same lineage* worked. Merging separately trained math/code/instruction adapters should be treated as risky versus one mixed-data run (NemoSkills found mixed vs sequential training gave similar accuracy).
- Specialisation cost: the NuminaMath card warns the model is unfit for general chat, i.e. a math-only fine-tune of this kind is expected to lose instruction-following ability, which matters for a joint objective.

### Gaps
- No source gives NuminaMath-7B-CoT's AMC/AIME greedy numbers, or a greedy-vs-voting comparison for the final AIMO-1 system.
- NuminaMath-CoT prompt template text is not shown on the model cards (only "use the tokenizer chat template").
- AIMO-3 official results, winner and method are unconfirmed; unsourced figures seen in search results (e.g. 36/50, 46/50) conflict and come from non-authoritative pages, so they are not reported as findings.
- AIMO-2 2nd/3rd place write-ups were not retrieved.
- No AIMO team reported a LoRA variant of their recipe, so there is no direct evidence on how much of the CoT gain survives under LoRA at small data budgets.

---

## 3. Other relevant contests and public course projects (Kaggle LLM fine-tuning competitions, university multi-task projects on IFEval/MATH/MBPP, Tinker-based projects)

### Takeaway
I found no public write-up of a course or contest that fine-tunes a small model jointly on IFEval + MATH + MBPP, and no Tinker-based competition results. The only directly relevant primary material is the Tinker cookbook's own warning that scores on exactly these benchmarks move substantially with evaluation configuration.

### Cited Findings
- The Tinker cookbook (Thinking Machines) ships an evaluation framework covering 12 benchmarks including GSM8K, MATH-500, IFEval, MBPP, IFBench, AIME 2025 and AIME 2026, and states: "Benchmark scores are sensitive to evaluation configuration", adding that system prompts, `max_tokens`, temperature and timeout settings "can shift results significantly". **[primary]** — [tinker-cookbook GitHub](https://github.com/thinking-machines-lab/tinker-cookbook)
- The cookbook's recipes include Chat SFT (e.g. on Tulu3), Math RL with verifiable rewards, Code RL with sandboxed execution, distillation and preference learning; a `hyperparam_utils` module provides learning-rate scaling for LoRA; the README example uses rank 32 illustratively. The top-level README gives no benchmark scores. **[primary]** — [tinker-cookbook GitHub](https://github.com/thinking-machines-lab/tinker-cookbook)
- Tinker is a managed LoRA fine-tuning API (the service trains a LoRA adapter rather than full weights). **[secondary]** — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/thinking-machines-new-tinker-api-makes-it-easier-to-fine-tune-models-on-many-gpus)
- Kaggle LLM Science Exam (Aug-Oct 2023, MAP@3): searches returned only the competition page and lower-placed repos; public approaches centred on retrieval ("open book") with DeBERTa-v3. I did not retrieve a 1st-place write-up or any 7B/13B LoRA details. **[unconfirmed for winner details]** — [Kaggle competition page](https://kaggle.com/competitions/kaggle-llm-science-exam); [10th-place repo](https://www.github.com/Lizhecheng02/Kaggle-LLM_Science_Exam)

### Inferences
- Because Tinker's cookbook already bundles IFEval, MATH-500 and MBPP evaluators, a course built on it most likely grades with those harnesses; reading the cookbook's eval README for the exact prompt, `max_tokens` and answer-extraction rules is probably higher-value than any further contest reading. (This is an inference about the grading setup, not a confirmed fact.)
- The Tinker statement on config sensitivity independently corroborates the competition lesson that format/extraction details can move scores as much as training choices.

### Gaps
- No university course project reports or leaderboards for joint IFEval/MATH/MBPP fine-tuning were found in the searches I ran; this does not prove none exist.
- No Tinker-based competition or project write-ups with numbers were found.
- The Tinker cookbook eval README (with the verified reference scores and exact settings) and `hyperparam_utils` LoRA learning-rate formula were not read; they are the obvious next primary sources.
- Kaggle LLM Science Exam winner details, and any other Kaggle LLM fine-tuning competition (e.g. LMSYS, Prompt Recovery), were not covered.

---

## 4. Recurring lessons: matching the evaluation format, balancing data across tasks, generalising to hidden test sets

### Takeaway
Across both competitions the durable pattern is: pick the strongest base model that fits, spend effort on a curated and balanced mixture rather than on adapter hyperparameters, validate on several held-out proxies instead of the public score, and expect public-task tuning to anti-correlate with hidden-task performance.

### Cited Findings
- **Public score is a poor guide to hidden score.** Open-vs-closed correlation was -0.08 (4090) and 0.18 (A100); the winners were not the top scorers on open tasks. — [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- **The clearest single case:** Birbal placed 20th/57 on open tasks and 1st on hidden tasks; the Qwen-14B runner-up scored higher on open (0.63 vs 0.52) but about half on closed (0.32 vs 0.61). — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- **Stated cause:** Birbal's team deliberately minimised reliance on data from the public evaluation suite in order to do well on hidden tasks. — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- **Organisers' advice:** use diverse fine-tuning samples rather than data drawn only from the open evaluation tasks. — [ar5iv mirror of 2503.13507](https://ar5iv.labs.arxiv.org/html/2503.13507)
- **More data is not monotonic and trades off across tasks:** 200K x 3 epochs beat 400K x 2 and 700K x 1 on the open stage (the larger two failed to qualify); GSM8K rose 0.44 -> 0.61 with more math data while TruthfulQA fell 0.59 -> 0.41 and BBQ fell 0.74 -> 0.67. Fine-tuning also lowered BBQ below the untouched base model (0.80). — [Birbal paper, Table 3](https://arxiv.org/pdf/2403.02247)
- **Balance explicitly:** Birbal capped each task family at a fixed quota (50K EM, 50K generation, 50K math, 25K Platypus, small QA/summarisation slices) and sampled inversely to base-model performance. — [Birbal paper](https://arxiv.org/pdf/2403.02247)
- **Small high-quality sets get upweighted:** Percent-BFD duplicated the 1K-example LIMA set 10x, while subsampling MMLU and CNN/DailyMail to 10%. — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf)
- **Iterate on the mixture with a fixed eval, adding and removing one source at a time:** Percent-BFD's V1-V5 log shows sources being added, measured, and removed when they gave "no significant effect". — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf)
- **Adapter settings varied widely among winners:** rank 8/alpha 32/lr 5e-5/1 epoch (A100 winner) vs rank 128/alpha 256/lr 2e-5/3 epochs (4090 winner). — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf); [Birbal paper](https://arxiv.org/pdf/2403.02247)
- **Checkpoint selection:** Birbal selected by minimum validation loss on 2,000 held-out examples, and the 2-epoch checkpoint turned out best of 3 epochs. NemoSkills averaged 4 checkpoints. — [Birbal paper](https://arxiv.org/pdf/2403.02247); [Xebia blog](https://xebia.com/blog/winning-recipe-for-finetuning-llm/); [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- **Use several internal validation sets and multiple seeds:** Numina used four sets (83 + 90 + 754 + 721 problems) and measured 1-3% seed variance; NemoSkills reported leaderboard decisions were hard because 50-problem scores varied a lot between submissions. — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize); [arXiv 2504.16891](https://arxiv.org/html/2504.16891v1)
- **Prompt template sensitivity is context-dependent:** Percent-BFD measured <0.01 difference with vs without their SFT template under few-shot HELM evaluation; the Tinker cookbook says system prompt, max_tokens and temperature shift IFEval/MATH/MBPP-type scores significantly. — [Percent-BFD slides](https://neurips.cc/media/neurips-2023/Slides/83962.pdf); [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- **Filter training targets for correctness and for banned/contaminated data:** Numina kept only generated solutions matching the reference answer; NeurIPS winners removed non-English data, possible eval contamination and LLM-generated data. — [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize); [arXiv 2503.13507](https://arxiv.org/html/2503.13507)
- **Operational failures eliminate entries:** ~30% of NeurIPS submissions timed out, ~10% hit OOM at inference, and over half failed to build. Birbal's team used output post-processing rules. — [ar5iv mirror of 2503.13507](https://ar5iv.labs.arxiv.org/html/2503.13507); [Upaya07 GitHub repo](https://github.com/Upaya07/NeurIPS-llm-efficiency-challenge)
- **Larger is not automatically better under fixed budgets:** Upaya chose 7B over 14B to fit more data/epochs in the time limit and won by a wide margin over a 14B entry; Numina could not beat a 7B math-specialised base with 20-33B general models. — [Birbal paper](https://arxiv.org/pdf/2403.02247); [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize)

### Inferences
- For a grade split between known and hidden benchmarks, the NeurIPS evidence argues for treating the known benchmarks as *format and difficulty references* rather than as training sources, and for building the mixture from broader neighbours (general instruction data with explicit constraints, CoT math at MATH/AMC level, diverse Python tasks). Percent-BFD did train on public-benchmark data and still won its track, so this is a risk-weighted recommendation, not an iron rule; the A100 full-vs-hidden correlation (0.20) suggests that track's ranking was noisier.
- A reasonable budget-allocation procedure, synthesised from Birbal and Percent-BFD: (1) score the untouched base model per sub-task with few-shot prompts; (2) set per-skill quotas; (3) oversample weak-but-learnable sub-tasks and drop hopeless ones; (4) upweight small high-quality sets by duplication; (5) change one source at a time against a fixed multi-benchmark dev suite; (6) select the checkpoint on held-out loss or dev score, expecting the optimum before the final epoch.
- Since fine-tuning demonstrably reduced some abilities the base model already had (BBQ, MMLU, TruthfulQA in Birbal's table), every candidate should be compared against the untouched base on each of the three target skills, not only against the previous candidate.
- Given seed variance of 1-3% on sets of ~80-750 problems (Numina), differences smaller than that between two mixtures on a small dev set should not drive decisions.
- For a base (non-chat) model evaluated zero-shot, the training template effectively *is* the evaluation interface; the weak template effect Percent-BFD saw should not be relied on, and the Tinker cookbook warning points the other way. Matching the grader's exact prompt wrapper, stop conditions and answer-extraction format in the training targets is the safest reading of the combined evidence, though no competition source ablates this directly.

### Gaps
- No source provides a controlled ablation of "training prompt format matched to eval format" vs "mismatched" for a base model under zero-shot grading.
- No source provides a controlled ablation of task-mixture ratios for a joint instruction/math/code objective; the Birbal table confounds dataset size, math fraction and epoch count.
- No competition evidence was found on code generation (MBPP/HumanEval-style) specifically: Percent-BFD dropped CodeAlpaca because LLM-generated data was banned, not on performance grounds, and neither NeurIPS evaluation nor AIMO graded code generation.
- No competition evidence was found on IFEval-style verifiable-constraint following.

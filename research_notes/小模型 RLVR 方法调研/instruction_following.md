# RLVR for Precise Instruction Following (IFEval-style constraints), 2024–2026

Scope note: these notes cover 16 tool calls (searches and fetches). The primary papers read in full or in large part are IFBench (2507.02833), VerIF (2506.09942), IFDecorator (2508.04632), Tulu 3 (2411.15124), AdvancedIF/RIFL (2511.10507), "To Mix or To Merge" (2602.12566), and the Olmo 3 7B Instruct model card. RECAST, UltraIF, Conifer, "Precision over Diversity" and CALIF come from abstracts and search snippets only, and are marked as such. I found no paper titled "Beyond the Instruction".

Our setup, for reference: Qwen3.5-9B-Base, LoRA r32, SFT'd. IFEval final_acc 75.9 (prompt-strict 70.1, instruction-loose 81.4). Dev on held-out constraint types: unseen ~51 vs seen ~67. About 40% of RL prompts are IF. Greedy eval, 1,024-token cap.

## 1. Tulu 3 and OLMo 2/3 RLVR recipes for IF

### Takeaway
Tulu 3's RLVR, applied after a strong DPO, added only about +1 IFEval point at 8B (81.1 → 82.4). It used PPO, a binary reward of 10, β = 0.05, and a 1,024-token response length. Pushing harder raised IFEval to 84.8 but hurt other evals, and AllenAI flags IFEval over-optimization explicitly. Olmo 3 7B Instruct gains +3.6 IFEval and +3.0 IFBench from its multi-domain RL stage. The large IF gains (+11 IFEval, +19 IFBench) come only from the dedicated IF-RLVR in the IFBench paper.

### Cited Findings
- **Tulu 3 (arXiv 2411.15124, AllenAI, Nov 2024; COLM 2025):**
  - **Reward:** α = 10 if verified, else 0, with a −10 penalty for responses without an EOS token.
  - **Prompts:** IFEval-style constraints from the IFEval taxonomy (14,973 IF, 7,473 GSM8K, 7,500 MATH).
  - **8B PPO settings:** lr 3e-7, effective batch 224, 300k episodes, response length 1,024, temperature 1.0. β swept over {0.1, 0.05, 0.03, 0.01}; final β = 0.05.
  - **Value model:** initialized from a general RM, which beat initializing from DPO.
  - **Init:** started from DPO. An SFT start reached similar verifiable reward but with larger KL and worse test performance. — [Tulu 3](https://arxiv.org/html/2411.15124)
- **Tulu 3 results (DPO → RLVR):**
  - 8B: IFEval 81.1 → 82.4, GSM8K 84.3 → 87.6, MATH 42.0 → 43.7, AlpacaEval2 33.5 → 34.5, MMLU 68.7 → 68.2, Safety 87.2 → 85.5.
  - 70B: IFEval 82.6 → 83.2.
  - Some 8B runs reached IFEval 84.8 and GSM8K 89.4 but were "worse overall".
  - Lower β meant higher KL, which generally meant lower average scores.
  - Adding RM scores to the verifiable reward "introduce[d] more noise".
  - Appendix B.4 is titled "RLVR IFEval overoptimization". I could not retrieve its contents. — [Tulu 3](https://arxiv.org/html/2411.15124)
- **Tulu 3 by stage (IFEval):** SFT 72.8 → DPO 81.1 → RLVR 82.4. Most of the IF gain came from DPO on IF-augmented persona prompts, not RLVR. — [Tulu 3](https://arxiv.org/html/2411.15124)
- **Olmo 3 7B Instruct (arXiv 2512.13961):**
  - IFEval: SFT 81.7 → DPO 82.0 → RL 85.6.
  - IFBench: 27.4 → 29.3 → 32.3.
  - The same RL stage raised MATH 79.6 → 87.3, AIME25 20.4 → 32.5, HumanEval+ 72.9 → 77.2, MBPP+ 55.9 → 60.2, LCB v3 18.8 → 29.5.
  - The RL dataset is Dolci-Instruct-RL-7B (math, code, IF, chat). The IF share and step counts are not on the card. — [Olmo-3-7B-Instruct card](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
- **Olmo 3 RL mix:** includes IF-RLVR prompts with up to 5 constraints, sampled from IFEval and IFBench-Train constraint types. — [summary of Olmo 3 report](https://books-1k9.pages.dev/en/olmo-3/00-overview) (secondary)
- **Olmo 3 32B:** Olmo-3-32B-Think reaches IFEval 89.0 and IFBench 47.6. Olmo 3.1 32B Instruct reaches IFEval 88.8 and IFBench 39.7. — [Olmo-3-32B-Think card](https://huggingface.co/allenai/Olmo-3-32B-Think); [Olmo-3.1-32B-Instruct card](https://huggingface.co/unsloth/Olmo-3.1-32B-Instruct/blob/main/README.md)

### Inferences
- In a multi-domain RL run like ours, a realistic IFEval gain is +2 to +4 points (the Tulu 3 and Olmo 3 scale), unless IF gets a dedicated or large share. Our 40% IF share is larger than Tulu's share of IF prompts in number terms, but our base IFEval (75.9) is lower, so there is more headroom.
- Tulu's −10 no-EOS penalty is a direct precedent for zeroing truncated outputs under our 1,024-token cap.

### Gaps
- Contents of Tulu 3 Appendix B.4 (concrete IFEval over-optimization examples). The fetch did not reach it.
- Olmo 3 IF share of the RL mix, constraint counts, and β. Not on the model cards. The Olmo 3 report itself was not fetched.
- OLMo 2 RL recipe for IF. Not retrieved; it is believed to reuse Tulu 3 RLVR (unverified).

## 2. IFBench / "Generalizing Verifiable Instruction Following"

### Takeaway
This is the most applicable paper (Pyatkin, Malik, Graf, Ivison, Huang, Dasigi, Lambert, Hajishirzi). It is arXiv 2507.02833 (Jul 2025) and the source of IFTrain and IF_multi_constraints_upto5; I did not confirm whether it appeared in NeurIPS 2025 D&B. Its main findings:
- Dedicated GRPO IF-RLVR gives very large gains: Tulu-DPO-8B goes from IFEval 81.1 to 92.2 and IFBench 25.5 to 44.6.
- Training with more than one constraint per prompt helps, with IFBench (OOD) best at 3 constraints.
- Adding new constraint types beyond IFEval's 25 helps generalization.
- Wider variable ranges help, and disjoint ranges hurt.
- Mixing in a preference RM preserves chat quality but costs most of the OOD gain.

### Cited Findings
- **GRPO setup:** lr 5e-7, temperature 1, 16 samples per prompt, mini-batch 32, max length 2,048, about 2,000 steps (about 1 day on 8×H100), 60k–100k prompts. Reasoning/base runs used β = 0 and length 10,240. — [IFBench](https://arxiv.org/html/2507.02833)
- **Reward:** binary per constraint, summed over the constraints in a prompt (weight 1 each). This is partial credit at the prompt level, not all-or-nothing. — [IFBench](https://arxiv.org/html/2507.02833)
- **Constraints per prompt (Qwen2.5-7B, Table 1):**

  | Constraints | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | IFBench | 48.9 | 53.1 | 59.5 | 49.4 | 55.8 | 54.1 |
  | IFEval | 71.2 | 79.9 | 77.8 | 79.5 | 79.9 | 85.8 |

  More than one constraint helps. The OOD optimum is around 3, but the trend is noisy and non-monotonic. — [IFBench](https://arxiv.org/html/2507.02833)
- **Instances per constraint type (10, 50, 100, 500, 1000):** IFBench 48.6 / 52.7 / 51.7 / 51.0 / 48.6. No benefit from more than about 50–100 instances per type. — [IFBench](https://arxiv.org/html/2507.02833)
- **Constraint types:** IFTrain adds 29 new training constraint types on top of IFEval's 25, and IFBench tests 58 new OOD constraints. Including all IFEval types plus the IFTrain types gives the best in-domain IFEval. IFBench is "much less sensitive" to how many IFEval types are included (results shown in figures only). — [IFBench](https://arxiv.org/html/2507.02833)
- **Variable ranges:** training on a disjoint range (e.g., train 20–40 when test is 1–20) gives consistently lower IFEval. A wider training range performs as well as or better than the exact test range. — [IFBench](https://arxiv.org/html/2507.02833)
- **Leave-one-category-out:** removing **length** or **keywords** constraints hurts IFEval most. Removing change-case or detectable-format barely matters. — [IFBench](https://arxiv.org/html/2507.02833)
- **Per-category OOD gains (Tulu-DPO → IF-RLVR):** IFBench "words" 7.0 → 48.1, "sentence" 13.3 → 36.7, "format" 35.2 → 64.3, "count" 45.3 → 53.1. — [IFBench](https://arxiv.org/html/2507.02833)
- **Starting point:**
  - Base-model RLVR (with a reasoning chat template): Llama-3.1-8B reaches 88.2 / 54.1 and Qwen2.5-7B 87.8 / 53.7 (IFEval / IFBench). These generalize better on IFBench than instruct starts: Tulu-DPO 92.2 / 44.6 and Qwen2.5-7B-Instruct 89.1 / 45.9.
  - OLMo2 base: 70.4 / 46.6. — [IFBench](https://arxiv.org/html/2507.02833)
- **GRPO vs DPO on the same prompts (IFEval / IFBench):** from SFT, DPO 76.9 / 25.2 vs GRPO 85.8 / 28.6. From DPO, DPO 79.7 / 29.3 vs GRPO 89.7 / 30.6. — [IFBench](https://arxiv.org/html/2507.02833)
- **Side effects of IF-only RLVR on Tulu-DPO-8B:** AlpacaEval2 33.5 → 21.3, GSM8K 84.3 → 83.2, MMLU 68.7 → 66.4, BBH flat. — [IFBench](https://arxiv.org/html/2507.02833)
- **Over-optimization:**
  - An LLM judge (1–10, constraints removed from the prompt) scored base-policy outputs 7.0 and IF-RLVR outputs 6.4.
  - IF-RLVR outputs average about 210 tokens on IFBench, vs about 2,214 for frontier models.
  - The authors frame this as the model prioritizing constraints over the underlying task. — [IFBench](https://arxiv.org/html/2507.02833)
- **Preference-RM mixing:**
  - Rule: reward = V + 1 if V > 0 and RM score S > 7; V − 0.5 if V > 0 and S ≤ 7; else V. The RM is Llama-3.1-Tulu-3-8B-RM; 1,100 steps.
  - Result: IFEval 86.1, IFBench 30, AlpacaEval2 31.6, vs verifiable-only 92.2 / 44.6 / 21.3. Quality is mostly preserved, but the OOD IF gain is roughly two-thirds lost.
  - Note: the paper text has inconsistencies, e.g., 45.9 vs 44.6 for the same Tulu IFBench number. — [IFBench](https://arxiv.org/html/2507.02833)
- **Multi-turn:** mixing single-turn and multi-turn IF data gives the best balance for instruct models. Single-turn IFEval 86.6, IFBench 54.8. — [IFBench](https://arxiv.org/html/2507.02833)

### Inferences
- Our dev gap (unseen ~51 vs seen ~67) matches this paper's picture. OOD IF improves mostly with more constraint types, 2–3 constraints per prompt, and wide variable ranges, not with more instances per type.
- The hidden held-out set may well be IFBench-like (new types, possibly the 58 IFBench test constraints). If it is IFBench itself, check that our constraint sampler does not inject IFBench-test types into training. That would be contamination, and it would also make our dev "unseen" split meaningless.
- Leave-one-out shows length and keyword constraints carry IFEval. Those are also the most hackable checkers (keyword stuffing, padding or terse outputs), so the quality gate matters most for them.
- The "words" and "sentence" IFBench categories gained most from RLVR, which suggests counting and word-level control transfers.

### Gaps
- The NeurIPS 2025 D&B acceptance could not be confirmed from the fetched pages.
- The exact seen-vs-unseen curve values are figure-only.
- No direct experiment in the paper compares summed partial credit against all-or-nothing.

## 3. Other IF-RL papers: VerIF, IFDecorator, RECAST, Conifer, UltraIF, AdvancedIF, reward hacking

### Takeaway
Hybrid rule-plus-LLM verification (VerIF) and intent checks (IFDecorator) are the main 2025 additions. Documented hacks are:
- placeholder or template copying
- dummy list items
- trivial repetition ("p p p")
- literal section markers
- self-praise text aimed at LLM verifiers
- overlong or truncated outputs

The working defenses are an intent/quality gate combined with the rule reward via AND, held-out trap prompts ("trip wires") to measure hacking, filtering to medium-difficulty prompts, and a completeness check.

### Cited Findings
- **VerIF (arXiv 2506.09942, EMNLP 2025, Tsinghua):**
  - Hard constraints are checked with code (0/1); soft constraints with QwQ-32B (0/1). The reward is the average of the two.
  - Data: VerInstruct, about 22k prompts with ≥ 2 constraints each (77.7% soft).
  - GRPO: 16 rollouts, KL coefficient 1e-3, lr 1e-6, batch 32, max length 4,096. Best checkpoints are usually within the first 200 steps (early stopping on IFEval).
  - Tulu-3-SFT-8B: IFEval prompt-strict 68.4 → 84.5 (vs Tulu 3 final 79.7), Multi-IF turn 3 40.3 → 54.0, CFBench ISR 63 → 72. — [VerIF](https://arxiv.org/html/2506.09942v1)
- **VerIF ablation (IFEval Pr.S / Multi-IF T3 / CFBench):** full 84.5 / 54.0 / 72.0; code-only (hard only) 74.7 / 46.0 / 59.0; LLM for everything 81.7 / 51.2 / 70.0. The LLM-only reward "rose higher and faster", which the authors suspect means it is easier to hack (not tested). The distilled IF-Verifier-7B reaches IFEval 80.0. — [VerIF](https://arxiv.org/html/2506.09942v1)
- **VerIF general capabilities (Tulu-3-SFT → +VerIF):** AlpacaEval2 7.9 → 22.0, GSM8K 78.8 → 83.4, MMLU-Pro 36.4 → 36.0. On R1-Distill-7B, AlpacaEval2 16.6 → 15.5 and Omni-MATH 35.0 → 33.6. No code evals. — [VerIF](https://arxiv.org/html/2506.09942v1)
- **IFDecorator (arXiv 2508.04632, Shanghai AI Lab; NeurIPS slides exist):**
  - **Hacks observed:** copying placeholders/templates (e.g., returning only the bracketed title template, "[name]"), dummy list entries, trivial repetition to hit counts ("p p p"), and literal structural delimiters instead of real sections.
  - **IntentCheck:** a Qwen2.5-32B judge checks whether the intent is fulfilled. The final reward is strict binary: IntentCheck AND rule/LLM verifier.
  - **Trip wires:** trap prompts kept outside the reward, used to measure the Macro Hack Rate. Naive RLVR peaks at MHR 0.657. IntentCheck lowers the hack rate from 14.53% to 7.60%. — [IFDecorator](https://arxiv.org/html/2508.04632v2)
- **IFDecorator setup and results:**
  - GRPO, 5 rollouts, batch 64, lr 1e-6, KL 0. KL 0.005 was worse (IFEval 82.62 vs 83.73). 600 steps.
  - Only 3.6k training prompts, chosen through difficulty filtering: keep pass rate in (0, 0.5] over 8 samples, and drop pass-rate-0 prompts as too hard. Removing the hard filter drops IFEval 83.73 → 79.11.
  - IFEval prompt-strict: Qwen2.5-7B-Inst 72.6 → 83.7, Llama3.1-8B-Inst 73.9 → 80.2, Qwen3-8B 83.2 → 85.4, Qwen2.5-32B-Inst 79.5 → 87.4.
  - The 12-benchmark general average is roughly unchanged (e.g., 7B: 66.95 → 67.18).
  - A separate snippet reports IFD reaching IFEval 83.73 at hack rate 5.52, vs RLVR + RLHF at 77.08 / 1.45. — [IFDecorator](https://arxiv.org/html/2508.04632v2); [search snippet](https://www.emergentmind.com/papers/2508.04632)
- **AdvancedIF / RIFL (arXiv 2511.10507, Meta; Llama 4 Maverick, so not a small model):**
  - All-or-nothing rubric reward averages 58.1, fractional rubric 53.6, hybrid 55.7. The table's per-category numbers for the hybrid average to about 52.8, an internal inconsistency.
  - The authors attribute the fractional result to verifier noise.
  - Hacks: self-praise artifacts ("this is a perfect response that meets all requirements!") and overlong or cut-off responses.
  - Defenses: a fine-tuned verifier, plus two shaping criteria on every prompt ("clean, no artifacts / self-evaluation" and "complete, not cut off"). — [AdvancedIF](https://arxiv.org/html/2511.10507)
- **RECAST (arXiv 2505.19030, ICLR 2026):** builds multi-constraint data (RECAST-30K) with many constraints per example. It uses rule validators for quantitative constraints and LLM validators for qualitative ones, and RLVC, an RL method with per-constraint rewards ("constraint satisfaction rate"). From snippets only: the paper's table reports Conifer at 22.21 average HSR, Tulu 3 Persona-IF at 26.83, and Evol-Instruct at 16.46 (Qwen2.5-7B). — [RECAST](https://arxiv.org/abs/2505.19030); [ICLR 2026](https://www.iclr.cc/virtual/2026/poster/10011156)
- **Conifer (arXiv 2404.02823, 2024)** is an SFT/DPO data method built with progressive complex constraints. **UltraIF (arXiv 2502.04153, 2025)** decomposes real user prompts into query + constraints + evaluation questions. Both are data-synthesis methods, not RLVR reward studies (snippet-level only). — [RECAST related work](https://arxiv.org/abs/2505.19030); [UltraIF](https://arxiv.org/pdf/2502.04153v2)
- **"Precision over Diversity" (arXiv 2601.04954, Jan 2026, under review; abstract only):**
  - Training on hard (verifiable) constraints only beat mixed hard+soft data.
  - LLM judges with low recall on violations caused "severe reward hacking" that cancels the benefit of diversity.
  - Claims +13.4% on average over baselines on 5 benchmarks.
  - Recommends prioritizing reward precision over constraint diversity. — [arXiv abstract](https://arxiv.org/abs/2601.04954)
- **CALIF (arXiv 2610.19525, Oct 2026; snippet only):** constraint-only verifier rewards let the policy satisfy checks while neglecting the actual request. The fix adds a likelihood-based bonus to the verifier reward. — [alphaXiv](https://alphaxiv.org/abs/2610.19525v1)
- **Self-supervised IF-RL (arXiv 2510.14420; snippet only):** derives rewards from the instructions themselves. Reports gains on 3 in-domain and 5 OOD IF datasets. — [arXiv](https://arxiv.org/html/2510.14420)

### Inferences
- **Our quality gate is close to IFDecorator's IntentCheck.** Their hack list (placeholders, dummy list items, repetition, literal delimiters) is broader than ours (label lines, degenerate text). Placeholder copying, dummy bullets and literal section markers are worth adding.
- **A trip-wire set is cheap to build** with our own sampler, and it gives a hack-rate metric separate from the reward.
- **Difficulty filtering is the single cheapest win.** In IFDecorator, removing it cost −4.6 IFEval. It also matters for GRPO signal: prompts where every sample passes or every sample fails give zero advantage.
- **Be cautious with an LLM verifier for soft constraints.** VerIF (it helps) and Precision over Diversity (low-recall judges cause hacking) disagree. With a 9B LoRA and limited budget, deterministic checkers are safer.

### Gaps
- Exact Conifer, UltraIF and RECAST RL numbers, and RECAST's per-constraint vs aggregate reward ablation, were not fetched.
- No paper with concrete keyword-stuffing rates was found. Keyword stuffing is implied by the IFBench and IFDecorator discussions but not quantified.

## 4. Partial credit vs binary prompt-level reward

### Takeaway
There is no clean controlled comparison with deterministic IFEval checkers. IFBench (the strongest small-model OOD result) uses summed per-constraint partial credit. IFDecorator and AdvancedIF use all-or-nothing, and AdvancedIF's ablation favors it (58.1 vs 53.6), but there the partial signal came from a noisy LLM rubric verifier. Our hybrid (half prompt-level, half instruction-level) sits between the two and has no evidence against it.

### Cited Findings
- **IFBench:** reward is the sum of binary per-constraint scores. — [IFBench](https://arxiv.org/html/2507.02833)
- **Tulu 3:** binary (10/0) per prompt. — [Tulu 3](https://arxiv.org/html/2411.15124)
- **VerIF:** average of a binary hard-group score and a binary soft-group score. — [VerIF](https://arxiv.org/html/2506.09942v1)
- **IFDecorator:** strict binary AND. Not compared against partial credit. — [IFDecorator](https://arxiv.org/html/2508.04632v2)
- **AdvancedIF (Llama 4 Maverick, LLM rubric verifier):** all-or-nothing 58.1 vs fractional 53.6 vs hybrid 55.7. The authors blame noise in the rubric verifier. — [AdvancedIF](https://arxiv.org/html/2511.10507)
- **Snorkel blog:** with binary reward on hard tasks, groups are all zeros and teach nothing, so partial credit is useful there. It still recommends keeping the all-pass metric primary. — [Snorkel](https://snorkel.ai/blog/when-partial-credit-opposes-completion/) (blog, secondary)

### Inferences
- The AdvancedIF argument against partial credit is verifier noise, which mostly does not apply to deterministic IFEvalG checkers. Partial credit's real risk with rule checkers is that the policy satisfies the easy constraints and gives up on the hard one. The 50% prompt-level component counters that.
- Our reward is an exact per-prompt decomposition of final_acc, so it optimizes the graded metric directly. That is sensible, and better justified than either pure form.
- Under GRPO, group normalization removes the reward scale, so the 1/1.54 constant only matters relative to the prompt-level term.

### Gaps
- No small-model, rule-checker study isolates binary vs partial for IF RLVR.

## 5. Does IF-RL hurt math, code or general quality?

### Takeaway
Dedicated, verifiable-only IF-RLVR measurably hurts chat quality: AlpacaEval2 −12 points and judge score 7.0 → 6.4 in IFBench, and responses get short and terse. It barely touches math. Mixed-domain RL (Olmo 3, To Mix or To Merge) shows little interference, and IF-RL can even transfer positively to reasoning. IF is the domain that loses slightly in mixing.

### Cited Findings
- **IFBench, IF-only RLVR on Tulu-DPO-8B:** AlpacaEval2 33.5 → 21.3, GSM8K −1.1, MMLU −2.3. — [IFBench](https://arxiv.org/html/2507.02833)
- **Tulu 3 mixed RLVR (8B):** IFEval +1.3, GSM8K +3.3, MATH +1.7, HumanEval+ +0.6, AlpacaEval2 +1.0, Safety −1.7. — [Tulu 3](https://arxiv.org/html/2411.15124)
- **"To Mix or To Merge" (arXiv 2602.12566, Feb 2026; Qwen3-4B-Base, IFEvalG binary reward, 16.6k IF prompts, 200 steps):**
  - IF-only RL raised AIME24 54.9 → 64.1, LCB v5 51.3 → 59.1, GPQA-D 46.1 → 49.4. Math, code and science RL helped IF only slightly (IFEval 79.5 → 81–83).
  - Single-domain RL-IF: IFEval 88.2 / IFBench 56.1. Mixed multi-task: 90.3 / 55.8. Ties merge: 89.8 / 53.7.
  - "Little interference" overall; mixed training reaches comparable results at about one third of the GPU hours. — [To Mix or To Merge](https://arxiv.org/html/2602.12566v1)
- **VerIF and IFDecorator:** general-capability averages roughly unchanged (IFDecorator 12-benchmark average within ±1.1). — [VerIF](https://arxiv.org/html/2506.09942v1); [IFDecorator](https://arxiv.org/html/2508.04632v2)

### Inferences
- With 40% IF in a mixed pool, expect little math or code harm. Watch response length and terseness on the IF prompts, since IF-RLVR drives outputs short (about 210 tokens). That can hurt the quality gate or any judge-graded part of the hidden set.
- IF-RL's positive transfer to reasoning in To Mix or To Merge comes from a thinking-model setup with 32k generation. It may not carry over to our 1,024-token greedy setup.

### Gaps
- No ≤14B study isolates IF-RL's effect on code benchmarks like MBPP/HumanEval. Only indirect evidence exists (Olmo 3 mixed RL improves code).

## 6. Improving generalization to unseen constraint types

### Takeaway
The best-supported levers, mostly from IFBench, are:
- more distinct constraint types (IFEval 25 plus IFTrain 29)
- 2–3 constraints per prompt, compositional
- wide or randomized variable ranges
- multi-turn mixing
- RLVR from a base or reasoning template rather than an over-tuned instruct model

Hybrid LLM-judged soft constraints improved OOD benchmarks in VerIF but caused hacking in other work. Instance count per type does not matter beyond about 50–100.

### Cited Findings
- **Types and composition (IFBench):** constraint types beyond IFEval improve OOD. 3 constraints per prompt gave the best IFBench (59.5 vs 48.9 at 1). Wider variable ranges match or beat the test range, and disjoint ranges hurt. — [IFBench](https://arxiv.org/html/2507.02833)
- **Base vs instruct (IFBench):** base-start RLVR reaches IFBench about 54 vs about 45 for instruct starts. — [IFBench](https://arxiv.org/html/2507.02833)
- **VerIF:** English single-turn training improved multilingual and multi-turn Multi-IF and SysBench. Gains were largest on constraint types covered in training. — [VerIF](https://arxiv.org/html/2506.09942v1)
- **Precision over Diversity:** reward precision beats diversity when the diversity comes from imprecise soft-constraint judges. — [arXiv](https://arxiv.org/abs/2601.04954)
- **RECAST:** many constraints per instance with per-constraint verification (ICLR 2026). Numbers not verified. — [RECAST](https://arxiv.org/abs/2505.19030)

### Inferences
- Our ~16-point seen/unseen gap is mostly a type-coverage problem. Adding types should help more than adding prompts.
- The prompt surface matters too: natural-language paraphrase of constraint wording is not directly ablated in any source found, but variable-range evidence suggests surface variety helps.

### Gaps
- No ablation found specifically on paraphrasing constraint instructions.
- No ablation found on LLM-judge soft constraints at ≤10B with LoRA.

## Recommendations for our case (Qwen3.5-9B-Base LoRA r32, IFEval 75.9, unseen-type dev gap ~16)

Small-model evidence base: IFBench (7–8B), VerIF (7–8B), IFDecorator (7–8B; one 32B), Tulu 3 8B, Olmo 3 7B, To Mix or To Merge (4B). AdvancedIF is Llama 4 Maverick, a large model, and is weighted lower.

### Adopt
1. **Keep the hybrid final_acc-decomposition reward.** It matches the metric, IFBench's strong OOD result uses per-constraint partial credit, and the anti-partial evidence (AdvancedIF) comes from noisy LLM verifiers.
2. **Zero the reward for truncation and missing EOS** under the 1,024-token cap, following Tulu 3's −10 no-EOS penalty and AdvancedIF's "complete, not cut off" criterion.
3. **Extend the quality gate** with IFDecorator's documented hacks:
   - unfilled placeholders or template copies (`<<...>>` with no real title, `[name]` left in)
   - dummy or near-empty bullets and sections
   - repeated tokens or lines used to hit counts
   - literal section or delimiter markers with no content under them
   - self-evaluation lines ("all constraints satisfied")
   - keyword stuffing, e.g. a keyword-frequency constraint met by dense repeats in one sentence
   Use AND semantics (gate fails → 0), as IFDecorator does.
4. **Difficulty-filter IF prompts** by sampling the SFT policy (e.g., 8 samples). Drop pass rate 0 and pass rate 1. IFDecorator keeps (0, 0.5] and gets +4.6 IFEval from the filter. All-pass or all-fail groups give zero GRPO advantage anyway.
5. **Shape the prompt pool for generalization:**
   - Mostly 2–3 constraints per prompt, with a tail up to 5. IFBench: 3 is best OOD, 1 is worst.
   - Sample variable values from wide ranges that cover and exceed IFEval's ranges.
   - Cap instances per type around 100–500 (no benefit beyond about 50–100) and spend budget on type count instead.
6. **Add a trip-wire / dev-unseen monitor:** track hack rate and dev-unseen accuracy per checkpoint, and pick the checkpoint on dev-unseen, not only on IFEval. VerIF found the best checkpoints early (≤ 200 steps).
7. **Contamination check:** verify that none of our sampler's types and none of the IF_multi_constraints_upto5 types overlap with IFBench-test's 58 constraints, in case the hidden set is IFBench-derived.

### Consider
- **A low-weight, gated preference/quality term**, only if we see terse or degraded outputs. IFBench's RM mix kept AlpacaEval (31.6 vs 21.3) but cost IFBench (30 vs 44.6). If used, apply it as a small bonus only when V > 0 rather than as a penalty, and monitor dev-unseen.
- **A small multi-turn IF slice.** It helps multi-turn and balance in IFBench, and is useful if the hidden set has system or multi-turn prompts.
- **A small amount of deterministic-but-novel constraint types** written by us (word or sentence-level counting, positional constraints, which gained most on IFBench), rather than LLM-judged soft constraints.

### Skip
- **LLM-judge rewards for soft constraints.** The ~180 s/batch cost (VerIF), the hacking risk (Precision over Diversity, AdvancedIF self-praise), and our budget argue against them.
- **Pure all-or-nothing prompt reward.** The supporting evidence is LLM-verifier-specific, and it would drop the instruction-level half of the graded metric.
- **Large KL tuning sweeps.** IFDecorator found KL 0 better (0.005 slightly worse), and Tulu found lower β meant more drift. Pick one reasonable small β, consistent with the user's no-sweeps preference.

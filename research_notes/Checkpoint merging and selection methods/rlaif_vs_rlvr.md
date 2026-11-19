# RLAIF (AI-feedback rewards) vs RLVR vs RLHF-style reward models for a small LoRA model graded on verifiable benchmarks (2023-2026)

Scenario: Qwen3.5-9B-Base, LoRA r=32 on Tinker. v4 SFT gets IFEval 75.9, MATH-500 71.8, MBPP 73.2. The plan is about 200 GRPO-style RLVR steps with 512 rollouts per step, on IF (programmatic checkers), math (answer match) and code (unit tests). All grading is programmatic. The budget is about $135 in total.

## 1. RLAIF vs RLHF, and vs RLVR on verifiable tasks: is there any head-to-head?

### Takeaway
On open-ended tasks, AI feedback roughly matches human-preference RMs (Lee et al.). On *verifiable* tasks, no paper shows a pure LLM-judge or RM reward beating a correct programmatic verifier on the programmatically graded metric. The head-to-heads that exist go the other way: an RM-only or judge-only reward regresses IFEval, while verifier-based RL gives the largest gains. AI feedback helps in two places: as a *secondary* signal for response quality and coherence, and as a fallback verifier where rules give false negatives.

### Cited Findings
- **Lee et al. 2023/ICML 2024 (Google, PaLM 2)**, summarization: RLAIF vs SFT 71% win rate and RLHF vs SFT 73% (not significant), RLAIF vs RLHF 50%. Helpful dialogue: 63% / 64% / 52%. Harmless rate: RLAIF 88%, RLHF 76%, SFT 64%. d-RLAIF (the LLM scores directly inside the RL loop, with no distilled RM) beats SFT 74% on summarization and 66% on helpfulness. All of these are human-preference metrics, not verifiable ones. — [Lee et al., arXiv 2309.00267](https://arxiv.org/abs/2309.00267)
- **Constitutional AI** (Anthropic, 2022) uses AI preference labels guided by a written constitution for harmlessness RL (RLAIF). Its evaluations are harmlessness/helpfulness preferences, not verifiable benchmarks. — [Bai et al., arXiv 2212.08073](https://arxiv.org/abs/2212.08073)
- **RM-only and judge-only rewards hurt IFEval at 7B** (RLCF paper, Qwen2.5-7B-Instruct, offline DPO). IFEval avg of the 4 metrics: base 77.3; +DPO(Skywork RM) 76.0; +DPO(ArmoRM) 76.0; +DPO(UltraFeedback) 74.6; +DPO(direct AI judge, Qwen2.5-72B) 75.2; +DPO(checklist, RLCF) 78.6. FollowBench HSR: base 71.4; Skywork 69.5; ArmoRM 70.4; AI judge 70.3; RLCF 75.3. The RMs raise AlpacaEval and Arena-Hard (Skywork AE 33.5→44.8) while lowering IFEval. — [Viswanathan et al., arXiv 2507.18624](https://arxiv.org/html/2507.18624v2)
- **An LLM judge disagrees with constraint satisfaction.** On prompts with the constraints removed, GPT-4.1 as judge scored base-policy completions 7/10 on average and IF-RLVR completions 6.4/10. Qwen2.5-72B-Instruct's IF accuracy was the most negatively correlated with judge scores among the models tested. The authors call this over-optimization or reward hacking of the verifiable reward. — [Pyatkin et al., IFBench, arXiv 2507.02833](https://arxiv.org/html/2507.02833v3)
- **Math, offline vs online with a verifiable reward** (Llama-3.1-8B-Instruct, NuminaMath, verifiable reward): MATH500 seed 47.4, offline DPO 53.7, semi-online DPO s=100 58.9, online DPO 58.7, GRPO 58.1. Online DPO and GRPO are statistically tied. With an Athene-RM-8B reward on WildChat, the policy "tends to hack its length bias", and checkpoints were chosen by length-normalized reward. — [Lanchantin et al., arXiv 2506.21495](https://arxiv.org/html/2506.21495v1)
- **RM-only training wrecks math.** Training the same 8B with online DPO on WildChat (RM reward) only: MATH500 47.4→35.0, AMC23 23.7→15.0. Mixing NuminaMath (verifiable) with WildChat (RM): MATH500 57.3 (vs 58.7 verifiable-only), AlpacaEval LC 65.6, ArenaHard 57.1. The mix costs about 1.4 MATH points and buys large chat gains. — [Lanchantin et al.](https://arxiv.org/html/2506.21495v1)

### Inferences
- The question "RLAIF vs RLHF" is mostly moot for us: both are preference proxies, and neither is the metric we are graded on. When the grader is a program, the program is the highest-fidelity reward. Any AI-feedback term is either a regularizer or a source of new hacking surface.
- Our graded metrics are IFEval, MATH and MBPP plus hidden sets of the same kinds. AI-feedback-only rewards would very likely *lower* scores, based on the RLCF RM baselines and the Lanchantin WC-only row.

### Gaps
- I found no head-to-head of d-RLAIF (LLM judge as the scalar reward) vs RLVR on MATH or MBPP at 7-9B with identical compute.
- The CAI paper's numbers were not extracted; it is cited only for the method.

## 2. IF-specific evidence: Tulu 3, IFBench/IF-RLVR, VerIF, RLCF, Rubrics as Rewards, OLMo 3

### Takeaway
On IFEval, verifier-based GRPO gives the largest gains (+11 to +16 points from SFT/DPO starts at 7-8B). Hybrid rule+LLM verification (VerIF) adds a few more points on IFEval *when the dataset includes soft constraints*. Gating the verifiable reward on an RM score (as in IFBench) *cost* 6 IFEval and 15 IFBench points while gaining 10 AlpacaEval points. The safest "hybrid" for IFEval-graded work is OLMo 3's approach: programmatic IF reward on IF prompts, with LM-judge reward only on separate chat prompts in the mix.

### Cited Findings
**Tulu 3 / IF-RLVR (Pyatkin et al., AI2, NeurIPS 2025 D&B)** — [arXiv 2507.02833](https://arxiv.org/html/2507.02833v3)
- Tulu-3-8B: SFT IFEval 72.8 / IFBench 20.7 → DPO 81.1 / 25.5 → final Tulu 3 (RLVR) 82.4 / 28.9. The original Tulu 3 RLVR gave only +1.3 IFEval over DPO.
- The paper's own IF-RLVR (GRPO, 29 new constraints plus IFEval-type ones, multi-constraint prompts): Tulu-3-8B-DPO 81.1/25.5 → 92.2/44.6; Qwen2.5-7B-Instruct 74.7/31.3 → 89.1/45.9; from base, Qwen2.5-7B 87.8/53.7 and Llama-3.1-8B 88.2/54.1.
- **DPO vs GRPO on the same IF data** (Table 5, prompt-level strict): from SFT, DPO 76.89/25.2 vs GRPO 85.77/28.6; from DPO, DPO 79.67/29.3 vs GRPO 89.65/30.6. GRPO beats DPO by about 9-10 IFEval points.
- **Verifiable × RM hybrid (Appendix E):** reward = V+1 if V>0 and RM score S>7; V−0.5 if V>0 and S≤7; V otherwise. The RM was Tulu-3-8B-RM, run for 1100 steps, batch 512, 8 samples/prompt. Result: IFEval 86.1 (vs 92.2 verifiable-only), IFBench 30 (vs 44.6), AlpacaEval 2 31.6 (vs 21.3).
- **Overfitting and generalization levers.** Leading models score under 50% on IFBench (GPT-4.1, Claude 3.7/4 Sonnet, Qwen3-32B). More constraints per training instance helps: 1 constraint gives IFEval 71.2 / IFBench 48.9, 6 constraints give 85.8 / 54.1. Training on a wider variable range than test performs comparably or better, while a different range hurts IFEval. Removing the length or keyword constraint categories hurts IFEval most. IF-RLVR responses average about 210 tokens vs about 2214 for frontier models on IFBench.
- GRPO settings: LR 5e-7, temperature 1, 16 samples/prompt, max 2048 tokens, about 2000 steps/day on 8×H100, 60k-100k prompts.

**VerIF (Tsinghua, EMNLP 2025), hybrid code + reasoning-LLM verification** — [arXiv 2506.09942](https://arxiv.org/html/2506.09942v1)
- Code checks hard constraints. QwQ-32B checks soft constraints (style, content). Training uses GRPO, 16 rollouts, KL 1e-3, LR 1e-6, max 4096 tokens. **The best checkpoints usually appear within the first 200 steps.**
- IFEval prompt strict/loose: Tulu-3-SFT 68.4/71.7 → +VerIF 84.5/87.1 (vs Tulu 3 final 79.7/82.8). R1-Distill-Qwen-7B 59.9/65.1 → 75.6/79.5. Multi-IF turn 3: 40.3→54.0. CFBench ISR: 63→72.
- Ablations (IFEval strict / Multi-IF T3 / CFBench ISR): full 84.5/54.0/72.0; "w/o code" (LLM-only) 76.2/52.0/73.0; "w/o LLM" (code-only on hard constraints) 74.7/46.0/59.0; code-only variant ("w/o code*") 81.7/51.2/70.0. Using Qwen2.5-72B-Instruct (non-reasoning) as the verifier gives 76.9. A distilled IF-Verifier-7B gives 80.0. *Caveat:* code-only underperforms partly because VerInstruct contains soft constraints that code cannot check, which is not our situation.
- Cost: QwQ-32B reward took about 180 s/batch on 8×H800, about 80% of step time. Total about 1,900 H800-hours. General capabilities: GSM8K 78.8→83.4, AlpacaEval 2 7.9→22.0, MMLU-Pro flat.

**RLCF, "Checklists are better than reward models" (CMU/Apple, 2025)** — [arXiv 2507.18624](https://arxiv.org/html/2507.18624v2)
- Instruction-specific checklists are scored by a Qwen2.5-72B judge (25 samples per item, averaged) and averaged with a generated verification program when one is confident. Pairs train **offline DPO** on Qwen2.5-7B-Instruct, using about 130k WildChat instructions with the top 40% of pairs kept.
- IFEval prompt loose/strict 75.0/72.5 → 77.3/72.6 (strict essentially flat, avg +1.3). FollowBench HSR 71.4→75.3. InFoBench 78.1→84.1. Arena-Hard 51.3→54.6. This is the only method that improved every benchmark, but the IFEval gain is small compared with GRPO+verifier gains.
- Removing program verification hurts badly in the base-model setting (Arena-Hard 23.1, AlpacaEval 11.0 in that table).
- Cost: grading took about 4 days on 8×H100 for 130k instructions. Using 3 vs 25 judge samples took 32 vs 92 h.

**Rubrics as Rewards (Gunjal et al., ICLR 2026)** — [arXiv 2507.17746](https://arxiv.org/abs/2507.17746v2)
- Rubric-based judge rewards beat direct Likert LLM-judge rewards by up to 31% relative on HealthBench and 7% on GPQA-Diamond. Rubrics give better alignment for smaller judges and less variance across judge scale. It targets *non-verifiable* domains and reports no IFEval/MATH/MBPP. One secondary summary says the setup is GRPO on Qwen2.5-7B with gpt-4o-mini as judge ([alphaXiv overview](https://alphaxiv.org/overview/2507.17746v2)), which I did not verify in the paper.

**OLMo 3 (AI2, Dec 2025), the production-style mix** — [arXiv 2512.13961](https://arxiv.org/pdf/2512.13961)
- RL rewards: IF uses constraint functions with reward 1 only if *all* constraints pass. Chat uses an LM judge (Qwen3-32B, thinking off) scoring [0,1] against a reference or open-ended. Math and code are verifiable. IF prompts come from IF-RLVR with ≤5 constraints sampled from IFEval and IFBench-Train.
- "Training on IFEval alone, higher IFEval scores correlate with lower AlpacaEval". The mixed run keeps AlpacaEval "without compromising IFEval", "as the LM-judge reward ensures that the model continues to produce well-formed chat responses". Mixed runs get lower train reward but equal or better downstream scores, which the authors read as less reward hacking.
- IFEval/IFBench by stage, 7B: Think SFT 77.9/30.0 → Think RL 88.2/41.6; Instruct SFT 81.7/27.4 → DPO 82.0/29.3 → RL 85.8/32.3. At 32B, Instruct RL reaches 88.8/39.7.
- DPO preference data: delta-learning pairs (big vs small model) plus "delta-aware" GPT-4.1-judged pairs are complementary. On a 7B dev checkpoint the average goes 51.9 → 60.4, but IFEval moves 83.2 → 83.0, so AI-preference DPO did not raise IFEval. LLM judges "tend to prefer longer responses".

### Inferences
- **For an IFEval-graded target, every reported way of multiplying or gating the verifiable IF reward by an RM or judge score lowered IFEval and IFBench** (Pyatkin App. E). The judge gains show up on AlpacaEval, which we are not graded on.
- Hybrid verification (VerIF) wins where constraints are *soft*. IFEval, IFBench and our hidden IF sets are fully programmatic, so an LLM verifier adds no accuracy on our graded constraints. Its value would be catching degenerate outputs (empty, gibberish, constraint-gaming text) that satisfy the checker.
- Rough expected IF effect sizes for us (v4 SFT 75.9 IFEval): GRPO with good multi-constraint verifiers is the dominant lever, with +10 to +14 points seen from comparable SFT starts at 7-8B (72.8→85.8 for Tulu SFT; 68.4→84.5 VerIF) over ~200-2000 steps. AI-preference DPO gives roughly 0 to +1.3. A LoRA r=32 run of about 200 steps should land well below the full-FT, 2000-step numbers.
- Hidden held-out IF sets "of the same kinds" reward the IFBench lessons: multi-constraint prompts, wide variable ranges, and constraint types beyond the IFEval 25. These matter more than adding a judge.

### Gaps
- FollowBench numbers for VerIF could not be cleanly extracted, because of misaligned table columns.
- No paper tested a *soft* coherence gate (for example, judge can only veto gibberish, with a low threshold) rather than Pyatkin's quality threshold S>7. Its effect on IFEval is untested.
- RECAST was not researched in this pass (search budget).

## 3. Reward-hacking risks of LLM judges for small models; judge cost per rollout at our scale

### Takeaway
LLM judges used as rewards are exploitable. Trivial "master key" strings get up to ~90% false-positive rates on 70B judges, and RL policies discover them and collapse. RM and judge rewards also reliably induce length inflation. At our scale (200 steps × 512 rollouts ≈ 102k judgments), a cheap OpenAI judge costs about $9-60 at standard pricing, which is 7-45% of the $135 budget. Gating the judge to IF rollouts that already pass the verifier cuts this to about $2-10.

### Cited Findings
- **"One Token to Fool LLM-as-a-Judge"** (Tencent, NeurIPS 2025): bare symbols or reasoning openers ("Solution:", ":") get FPRs up to ~90% on LLaMA3-70B-Instruct and Qwen2.5-72B-Instruct judges. Proprietary judges are also affected. In an RLVR run with a Qwen2.5-72B judge, the policy collapsed to very short responses. The fix is Master-RM, trained with truncated outputs as negatives. — [arXiv 2507.08794](https://arxiv.org/html/2507.08794v1); [NeurIPS page](https://neurips.cc/virtual/2025/131135)
- RM length hacking in online DPO/GRPO (Athene-RM-8B): lengths grew, and checkpoints had to be selected by length-normalized reward. — [Lanchantin et al.](https://arxiv.org/html/2506.21495v1)
- LLM judges prefer longer responses (OLMo 3 DPO pipeline). — [OLMo 3 report](https://arxiv.org/pdf/2512.13961)
- A judge's quality scores anti-correlate with strict constraint following (GPT-4.1 judge, IFBench section 5), so a judge term pulls *against* the IF verifier. — [Pyatkin et al.](https://arxiv.org/html/2507.02833v3)
- **Cases where a model verifier helps:** rule-based math verifiers produce false negatives. 38.5% of responses that Prime Verifier marked wrong on Big-Math were actually correct. TinyV (a fine-tuned Qwen2.5-1.5B) is queried *only when the rule verifier says wrong*. It raised pass rates by up to 10% (on its own HardVerify-Math set; gains elsewhere are modest) with about 6% GRPO overhead. — [TinyV, arXiv 2505.14625](https://arxiv.org/html/2505.14625v2)
- OpenAI standard pricing per 1M tokens (input / cached / output): gpt-5-nano $0.05/$0.005/$0.40; gpt-4o-mini $0.15/$0.075/$0.60; gpt-5-mini $0.25/$0.025/$2.00; gpt-4.1-mini $0.40/$0.10/$1.60. Batch tier is 50% off. — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)

### Cost estimate (my arithmetic, from the prices above)
Assumptions: 200 steps × 512 rollouts = 102,400 judge calls. Each call has about 1,300 input tokens (≈400-token judge rubric prefix, ≈300 prompt, ≈600 response) and either about 30 output tokens (non-reasoning model, score only) or about 300 (reasoning model with minimal effort). Input totals about 133M tokens.

| Judge | Input $ | Output $ | Total (all rollouts) | IF-only (~1/3) | IF-only and verifier-passed (~1/6) |
|---|---|---|---|---|---|
| gpt-5-nano (~50 out tok, minimal reasoning) | 6.7 | 2.0 | ≈ $9 | ≈ $3 | ≈ $1.5 |
| gpt-5-nano (~300 out tok) | 6.7 | 12.3 | ≈ $19 | ≈ $6 | ≈ $3 |
| gpt-4o-mini (~30 out tok) | 20.0 | 1.8 | ≈ $22 (≈ $19 with prefix caching) | ≈ $7 | ≈ $4 |
| gpt-4.1-mini (~30 out tok) | 53.2 | 4.9 | ≈ $58 | ≈ $19 | ≈ $10 |
| gpt-5-mini (~300 out tok) | 33.3 | 61.4 | ≈ $95 | ≈ $32 | ≈ $16 |

- The Batch API halves these costs, but its asynchronous turnaround is incompatible with a synchronous per-step RL loop.
- 512 judge calls per step adds latency and rate-limit exposure per step.

### Inferences
- A 9B policy with LoRA trained for about 200 steps has less capacity and time to find exotic exploits than frontier-scale runs. Length inflation and format gaming (the cheapest exploits) still appear quickly in the reports above. A small judge such as gpt-5-nano or gpt-4o-mini is the *most* exploitable class, according to the master-key results.
- Cost is affordable only with a nano-class judge and with judging restricted to a subset of rollouts. Even then it competes with Tinker compute for a budget where extra RL steps or rollouts likely buy more graded points.

### Gaps
- The Tinker per-token price for sampling the 397B Qwen as a judge was not found, so its cost could not be estimated. Staff approval is also pending.
- The output-token counts for gpt-5-nano at minimal reasoning are assumptions, not measurements. A 100-call pilot would pin them down.

## 4. DPO / online DPO with AI preferences vs GRPO-style RLVR at 7-9B on IFEval / MATH / code

### Takeaway
With the *same verifiable reward*, online DPO is about equal to GRPO on math (8B). Offline DPO is clearly worse on math (−5 MATH500) and on IF (−9 to −10 IFEval). DPO on *AI-judged preferences* (no verifier) gives roughly 0 to +1.3 IFEval at 7B and can hurt. No study shows AI-preference DPO beating verifier-based GRPO on IFEval, MATH or code.

### Cited Findings
- Math (Llama-3.1-8B-Instruct, verifiable reward): offline DPO 53.7, online DPO 58.7, GRPO 58.1 MATH500; AMC23 28.8 / 32.9 / 33.6. — [Lanchantin et al.](https://arxiv.org/html/2506.21495v1)
- IF (Tulu-3 8B, verifiable IF data): from SFT, DPO 76.89 vs GRPO 85.77 IFEval; from DPO, 79.67 vs 89.65. — [Pyatkin et al. Table 5](https://arxiv.org/html/2507.02833v3)
- AI-preference DPO on IFEval (Qwen2.5-7B-Instruct): AI-judge DPO 75.2 avg vs base 77.3; checklist DPO 78.6. — [RLCF](https://arxiv.org/html/2507.18624v2)
- OLMo 3 Instruct 7B: DPO (delta + GPT-judged) moves IFEval 81.7→82.0, and the RL stage moves it to 85.8. On the dev checkpoint, AI-preference DPO left IFEval at about 83 while raising AlpacaEval 23.8→49.8 and MATH 65.5→80.0 (the MATH gain came mostly from delta-learning pairs). — [OLMo 3](https://arxiv.org/pdf/2512.13961)
- Tulu 3 8B: DPO stage gave +8.3 IFEval (72.8→81.1); the RLVR stage gave +1.3 more. — [Pyatkin et al. Table 3](https://arxiv.org/html/2507.02833v3)

### Inferences
- For our graded metrics, AI-preference DPO is not a substitute for RLVR. It could be a cheap pre-RL step, but our SFT is already at 75.9 IFEval and the evidence for AI-pref DPO gains on IFEval/MATH/MBPP at this level is weak and mixed.
- Code (MBPP): I found no 7-9B head-to-head of AI-judge rewards vs unit-test rewards. Unit tests are the standard reward in OLMo 3 and Tulu 3-style pipelines.

### Gaps
- No direct comparison of AI-pref DPO vs GRPO on MBPP or HumanEval at 7-9B was found.
- No LoRA-specific comparison was found. All the numbers above are full fine-tunes.

## 5. Recommendation: hybrid (verifiable × judge gate) vs pure RLVR for programmatically graded benchmarks; cost

### Takeaway
**Stay with RLVR as the reward on every IF, math and code rollout. Do not multiply or gate it by a judge quality score.** The only head-to-head (Pyatkin App. E) shows a multiplicative RM gate costs about 6 IFEval and 15 IFBench points. The evidence-backed complements are cheap:
- (a) OLMo 3-style *domain mixing* to prevent single-domain over-optimization;
- (b) deterministic degeneracy penalties (length caps, repetition, language checks) instead of an LLM coherence gate;
- (c) optionally, a TinyV-style *fallback* model verifier for math false negatives;
- (d) optionally, a small separate chat-prompt slice with a judge reward, if hidden evals include free-form quality, which the brief says they do not.

### Cited Findings
- Verifiable+RM gate: IFEval 92.2→86.1, IFBench 44.6→30, AlpacaEval 21.3→31.6. — [Pyatkin et al.](https://arxiv.org/html/2507.02833v3)
- A separate-prompt judge reward in the mix keeps chat quality "without compromising IFEval". — [OLMo 3](https://arxiv.org/pdf/2512.13961)
- Code-only vs hybrid verification gap (VerIF 81.7 vs 84.5 IFEval strict) depends on soft constraints in the training data. — [VerIF](https://arxiv.org/html/2506.09942v1)
- Rule-verifier false negatives in math: 38.5% of "incorrect" labels were correct on Big-Math. A fallback LLM verifier adds ~6% overhead. — [TinyV](https://arxiv.org/html/2505.14625v2)
- Judges are hackable (FPR up to ~90%), and RM rewards inflate length. — [Zhao et al.](https://arxiv.org/html/2507.08794v1); [Lanchantin et al.](https://arxiv.org/html/2506.21495v1)

### Inferences
- **Decision matrix:**
  - *Switch to RLAIF:* no. All evidence predicts lower IFEval/MATH/MBPP.
  - *Mix (multiplicative gate):* no. Measured −6 IFEval and −15 IFBench.
  - *Mix (veto-only gate)*, where the judge zeroes the reward only for clearly broken outputs and only when V=1: untested. Plausibly neutral to slightly positive on hidden sets. Cost about $2-4 with gpt-5-nano or gpt-4o-mini on verifier-passed IF rollouts. Hacking risk is limited because the judge can only subtract. Worth it only if pilot rollouts show checker-gaming (for example, keyword stuffing or empty sections satisfying constraints).
  - *Stay with RLVR plus programmatic hygiene:* recommended. Judge cost is $0, and the budget goes to steps and rollouts.
- If a judge is used anyway:
  - use a checklist/rubric-style prompt (RLCF, RaR), not Likert;
  - make the judge only able to veto;
  - strip "Solution:"-style openers and test the master-key strings before training;
  - log length per step;
  - cap judge spend at about $10 (gpt-5-nano or gpt-4o-mini, IF verifier-passed rollouts only, about 17k calls).
- Use the 397B Tinker teacher (if approved) for something with better-evidenced returns than a judge reward, for example generating multi-constraint IF prompts with wider variable ranges or new constraint types (the IFBench generalization levers).

### Gaps
- No published experiment uses a veto-only coherence judge on top of IF-RLVR. Its effect is an inference, not a measurement.
- LoRA r=32 with about 200 steps may underfit relative to the full-FT, 1000-2000-step runs cited. The magnitudes of all effects (including judge-induced regressions) may be smaller in our setting.

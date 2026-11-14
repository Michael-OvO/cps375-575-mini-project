# Run log

One entry per run: name, date, cost, scores. Costs are estimates from token counts unless marked as read from the Tinker console. The token-count estimate for the base eval matched the console ($2.09 vs ~$2.1).

| Run | Date | What | IFEval | MATH-500 | MBPP | Tinker cost (est.) | Notes |
|---|---|---|---|---|---|---|---|
| base | 2026-11-14 | Untrained `Qwen/Qwen3.5-9B-Base`, full public eval (541 / 500 / 257 samples) | 47.9 | 15.8 | 65.4 | $2.09 (user's Tinker console; token-count estimate was ~$2.1) | Handout lists 47.5 / 17.0 / 63.8. Sampled tokens: IFEval 0.38M, MATH 1.79M, MBPP 0.33M. MATH averaged ~3,580 output tokens per problem against a 4,096 cap, i.e. the base model rarely stops on its own. Results file: `evaluation/baseline_results.json` (git-ignored). |

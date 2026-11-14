# Run log

One entry per run: name, date, cost, scores. Costs are estimates from token counts unless marked as read from the Tinker console. Token-count estimates have run about 40% below the console figure, so scale them up by 1.4 when budgeting.

| Run | Date | What | IFEval | MATH-500 | MBPP | Tinker cost (est.) | Notes |
|---|---|---|---|---|---|---|---|
| base | 2026-11-14 | Untrained `Qwen/Qwen3.5-9B-Base`, full public eval (541 / 500 / 257 samples) | 47.9 | 15.8 | 65.4 | $3 (user's Tinker console; my token-count estimate was ~$2.1, includes a 5-sample smoke test) | Handout lists 47.5 / 17.0 / 63.8. Sampled tokens: IFEval 0.38M, MATH 1.79M, MBPP 0.33M. MATH averaged ~3,580 output tokens per problem against a 4,096 cap, i.e. the base model rarely stops on its own. Results file: `evaluation/baseline_results.json` (git-ignored). |

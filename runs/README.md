# Run log

One entry per run: name, date, cost, scores. Costs are estimates from token counts unless marked as read from the Tinker console. The token-count estimate for the base eval matched the console ($2.09 vs ~$2.1).

| Run | Date | What | IFEval | MATH-500 | MBPP | Tinker cost (est.) | Notes |
|---|---|---|---|---|---|---|---|
| base | 2026-11-14 | Untrained `Qwen/Qwen3.5-9B-Base`, full public eval (541 / 500 / 257 samples) | 47.9 | 15.8 | 65.4 | $2.09 (user's Tinker console; token-count estimate was ~$2.1) | Handout lists 47.5 / 17.0 / 63.8. Sampled tokens: IFEval 0.38M, MATH 1.79M, MBPP 0.33M. MATH averaged ~3,580 output tokens per problem against a 4,096 cap, i.e. the base model rarely stops on its own. Results file: `evaluation/baseline_results.json` (git-ignored). |
| smoke | 2026-11-14 | 1,000 examples (500 MATH-train, 200 code, 200 IF, 100 general), 32 steps, rank 32, lr 4.73e-4; test-overlap filter on | — | — | 70.0 | ~$0.63 train + ~$0.2 MBPP | Loss 0.73 → ~0.5. Format OK (math ends on `ANSWER:`, code single block, IF no preamble, stops cleanly). Dev (same 100 items, base → smoke): math 17 → 80, code 65 → 59, IF seen 35 → 55, IF unseen 21 → 44. Code dev drop: base reasons in `<think>` before code; direct-code targets lose that on harder dev tasks (lost 19, won 13), but public MBPP rose 65.4 → 70.0. Base dev (full, 325/337/165/166): math 17.2, code 65.0, IF seen 34.7, unseen 25.1. |

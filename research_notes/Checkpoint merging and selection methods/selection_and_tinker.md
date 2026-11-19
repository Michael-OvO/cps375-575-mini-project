# Checkpoint selection with small dev sets, and building merged LoRA checkpoints on Tinker

Scope: (A) a statistically sound protocol for picking among about 20 RLVR checkpoints plus merged candidates using private dev sets of a few hundred items per task, and (B) what tinker==0.39.0 / tinker-cookbook==0.6.1 allow for producing a sampler checkpoint from weights computed locally.

Source note: Tinker findings come from reading the installed package source at `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker` and `.../tinker_cookbook`. These are cited as file paths with line numbers. No Tinker endpoint was called. The simulation numbers are from a local Monte-Carlo script (`/private/tmp/claude-501/-Users-michael-Documents-Github-cps375-575-mini-project/fd3751cd-c27f-4703-86d9-73297678389c/scratchpad/wc.py`; numpy only, R = 4,000 to 20,000 replicates). They are computed, not taken from the literature.

## (B) Can we produce a Tinker sampler checkpoint from locally computed (merged) LoRA weights?

### Takeaway
Yes, but only through an experimental API that tinker 0.39.0 already ships. The path is: `RestClient.external_weights_upload_start(base_model, lora_config)`, then HTTP PUT each file, then `external_weights_upload_finish(upload_id)`. That returns `tinker://<run>/external_weights/<name>`. Load it with `ServiceClient.create_training_client_from_state(path)`, then call `save_weights_for_sampler(...)` to get a `sampler_weights` path. Downloading works the other way: `save_weights_external` gives PEFT-style `adapter_model.safetensors` plus `adapter_config.json`, via `get_external_weights_urls` or `tinker checkpoint download`. The upload is validated strictly. Files must be exactly what `save_weights_external` would export for the same base_model and LoraConfig. So a merged adapter has to keep the export's key names, shapes and config, and its rank must match the declared `LoraConfig.rank`.

### Cited Findings
- **Upload exists (experimental).** `RestClient.external_weights_upload_start(base_model, lora_config, destination_project_id=None)` is marked `@experimental` ("may change or be removed in any release without a deprecation period"). Its docstring: "Start uploading weights to become an external weights checkpoint of a new training run, which cannot be trained on... Upload each file in the result's `urls` with an HTTP PUT of its bytes, then call `external_weights_upload_finish`. The files must be exactly what `save_weights_external` exports for `base_model` and `lora_config`; finishing rejects anything else." Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/lib/public_interfaces/rest_client.py` (lines 505-561).
- **Finish step.** `external_weights_upload_finish(upload_id)` will "Validate an upload whose files were all put to their signed URLs, and register it as an external weights checkpoint. Retrying returns the same checkpoint... Load the checkpoint into a training client of the same base model and LoRA config with `load_state` (weights only)." It returns `ExternalWeightsUploadFinishResult(tinker_path=...)`. Source: same file (lines 563-597). The REST endpoints are `POST /api/v1/external_weights_upload_start` and `POST /api/v1/external_weights_upload_finish`. Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/resources/weights.py` (lines 150-171).
- **Request schema.** `ExternalWeightsUploadStartRequest{base_model: str, lora_config: LoraConfig, destination_project_id}`. The result holds `upload_id`, `urls: Mapping[str,str]` ("Signed URL per file to upload, keyed by file name; upload each file with an HTTP PUT") and `expires`. Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/types/external_weights_upload.py` (lines 14-57). `LoraConfig` fields: `rank`, `seed`, `train_unembed=True`, `train_mlp=True`, `train_attn=True`. Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/types/lora_config.py` (lines 8-25).
- **Turning an uploaded checkpoint into a training client.** `ServiceClient.create_training_client_from_state(path, base_model=None, ...)` "loads only the model weights, not optimizer state". It sends a single `LoadWeightsRequest(path=...)`. `TrainingClient.load_state` is deprecated in favour of this call. Sources: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/lib/public_interfaces/service_client.py` (lines 384-530) and `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/lib/public_interfaces/training_client.py` (lines 997-1025). The upload docstring explicitly names "load_state (weights only)" as the way to consume an uploaded checkpoint. Loading external_weights through `create_training_client_from_state` is not a separate guess: `tinker checkpoint download` itself loads checkpoints with that same call.
- **Sampler checkpoints.** `save_weights_for_sampler()` creates a SAMPLER_WEIGHTS checkpoint that "contains weights only and can be sampled from" but "cannot initialize or resume a training run". It is used with `create_sampling_client(model_path=...)`. Source: [Tinker docs llms-full](https://tinker-docs.thinkingmachines.ai/llms-full.txt); [Tinker weights tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/).
- **Download path.** `TrainingClient.save_weights_external(name, ttl_seconds, age_encryption_recipients)` saves "model weights in an external (e.g. HuggingFace) format". `RestClient.get_external_weights_urls(tinker_path)` returns one signed URL per file and only accepts `external_weights/` paths. Sources: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/lib/public_interfaces/training_client.py` (lines 1198-1258) and `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/lib/public_interfaces/rest_client.py` (lines 450-488). The web docs give the same code pattern. Source: [Tinker weights tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/).
- **CLI download.** `tinker checkpoint download` accepts `weights/` (training) or `external_weights/` paths. Sampler checkpoints are refused: "Sampler checkpoints cannot be downloaded". For a training checkpoint, the CLI starts a training session on the base model (`create_training_client_from_state`), exports a temporary external_weights checkpoint (TTL `_TEMPORARY_EXPORT_TTL_SECONDS`), downloads it, then deletes it. Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/cli/commands/checkpoint.py` (lines 534-608 and 1187-1260). The cookbook's `tinker_cookbook.weights.download(tinker_path=..., output_dir=...)` does the same thing, with age encryption to a throwaway key. Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker_cookbook/weights/_download.py` (lines 34-76).
- **Format of the exported adapter.** The cookbook's adapter tools take a Tinker adapter directory that "must contain `adapter_model.safetensors` and `adapter_config.json`". The config has `lora_alpha` and `r`, and merge scaling is `lora_alpha / r`. Sources: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker_cookbook/weights/_adapter.py` (lines 88-120) and `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker_cookbook/weights/_merge_utils.py` (lines 33-57). Tinker key names use a `base_model.model.` prefix and `unembed_tokens`. For Qwen3.5, Tinker trains separate `in_proj_q/k/v` LoRA adapters (which may differ in size) for the fused `in_proj_qkv`. Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker_cookbook/weights/README.md` ("Qwen3.5 / Qwen3.6 Split QKV", "Key Remapping"). The README defaults list `lora_alpha` 32 at rank 32, "Common to set alpha = rank". Same README.
- **Copying is not merging.** `copy_weights(path, destination_project_id)` only duplicates a checkpoint reference into a project: "Storage is shared with the source, so no bytes are duplicated." Source: `/Users/michael/Documents/Github/cps375-575-mini-project/.venv/lib/python3.11/site-packages/tinker/lib/public_interfaces/rest_client.py` (lines 720-772).
- **The public docs do not describe the upload path.** A web search and the weights tutorial cover only export and download. Sources: [Tinker weights tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/); [Downloading Weights](https://tinker-docs.thinkingmachines.ai/download-weights). The first 100k characters of llms-full.txt do not mention it either. Source: [llms-full](https://tinker-docs.thinkingmachines.ai/llms-full.txt).

### Inferences
- **End-to-end recipe**, all from source above; nothing was run.
  1. For each checkpoint to merge, call `create_training_client_from_state(tinker://.../weights/<step>)` then `save_weights_external(name, ttl_seconds=...)`. Alternatively run `tinker checkpoint download tinker://.../weights/<step>`. The RL loop's saved `weights/` checkpoints work for this; `sampler_weights/` cannot be downloaded. Each export opens a short training session, which probably costs a little.
  2. Load each `adapter_model.safetensors` locally and compute the merge.
  3. Write the result with byte-compatible structure: same tensor keys, dtypes and shapes as the export, and an unchanged `adapter_config.json` unless rank changes.
  4. Call `external_weights_upload_start(base_model="Qwen/Qwen3.5-9B-Base", lora_config=LoraConfig(rank=32, train_*=same as training))`. PUT every file to its URL, then call `external_weights_upload_finish`.
  5. Call `create_training_client_from_state(<external path>)`, then `save_weights_for_sampler("merged_x")` to get the submission path. Sample a few dev prompts from this sampler path and compare them with local expectations to confirm the round trip, for example by uploading the unmodified v4 export and checking that its outputs match v4.
- **Math caveat: averaging LoRA factors is not averaging deltas.** Each checkpoint's update is dW_i = s·B_i·A_i with s = alpha/r. The mean of the B_i times the mean of the A_i is not equal to the mean of B_i·A_i. Three ways to handle it:
  - **(a) Exact merge at higher rank by concatenation.** Use B' = [w_1 B_1, ..., w_N B_N] and A' = [A_1; ...; A_N]. Then B'A' = sum of w_i B_i A_i exactly, at rank N·r. Example: a v4/RL interpolation (or extrapolation) of 2 checkpoints is exactly rank 64.
  - **(b) Rank-32 truncated SVD of the merged delta.** Compute the delta per module and take the best rank-32 approximation (Eckart-Young). Keep the A and B convention: B = U·sqrt(S), A = sqrt(S)·V^T, divided by s.
  - **(c) Naive per-factor averaging.** This is roughly right only when all checkpoints share nearly the same A. That may hold within one RL run that started from v4 at low LR, but it is unverified.
  Option (a) changes `LoraConfig.rank`. Whether the server accepts rank 64 for Qwen3.5-9B-Base, and whether alpha stays fixed (which would change s = alpha/r, so B must be rescaled), cannot be told from the client code. Read the `adapter_config.json` of a real export first. Option (b) at rank 32 is the safest bet for passing "finishing rejects anything else". For interpolation and extrapolation between v4 and the RL endpoint, SVD truncation of a rank-64 delta to 32 is probably lossy but small; this is not measured.
- **If the upload fails or is withdrawn** (it is experimental), no pure-API weight arithmetic exists. There is no `set_weights` or `add` operation on TrainingClient in 0.39.0; a grep found only load, save and copy. The fallbacks:
  - Distill the merged target into a fresh rank-32 LoRA, using the target's samples or logprobs as SFT data. This needs the target to be sampleable, which is circular unless it runs locally.
  - Logit or ensemble averaging across several sampling clients at decode time. This is not a single checkpoint, so it cannot be submitted as one `sampler_weights` path.
  - Restrict candidates to the native checkpoints.

### Gaps
- The exact server-side validation rules for uploads (allowed ranks, dtype, whether `adapter_config.json` may differ, whether `alpha` is fixed) are not visible in client code or public docs. Test with a round trip of an unmodified export before relying on the upload.
- Whether uploading or exporting is billed, and how much, was not found.
- Whether the course or staff allows submitting a sampler path that derives from uploaded (externally modified) weights is a policy question, not a technical one.

## (A1) Winner's curse: optimism from picking the best of N checkpoints on n dev items

### Takeaway
At 70-80% accuracy, one task's dev SE is about 1.8-2.6 points for n = 300-500. Taking the argmax of N = 10-40 checkpoints inflates the winner's dev score. If all candidates were truly equal and errors independent, the inflation would be about 2.7-5.6 points. With realistic shared-item correlation it is about 1.6-4.4 points. Real differences between neighbouring RL checkpoints are often smaller than this, so the argmax mostly picks noise. Treat the winner's dev score as biased and re-measure it on the vault exactly once.

### Cited Findings
- **Cawley and Talbot.** Over-fitting the model-selection criterion produces optimistically biased performance estimates. An unbiased estimate needs data not used in selection, i.e. a nested or held-out evaluation. Source: [Cawley & Talbot 2010, JMLR 11](https://jmlr.org/papers/v11/cawley10a.html). This is the standard citation; it is cited from prior knowledge and the page was not fetched in this session.
- **Reporting error bars (Anthropic).** Report SEM with every eval score, 95% CI = mean ± 1.96·SEM. Cluster SEs when items share a source (clustered SEs can be more than 3× the naive SE). Use paired differences between models on the same items: the variance of the mean difference falls as the per-item score correlation rises, and frontier models correlate 0.3-0.7. Use power analysis to size the eval. Source: [Anthropic, "A statistical approach to model evaluations" (Miller 2024)](https://www.anthropic.com/research/statistical-approach-to-model-evals); paper [arXiv:2411.00640](https://arxiv.org/abs/2411.00640).
- **Signal and Noise (Ai2).** Defines signal as a benchmark's ability to separate models and noise as score fluctuation between consecutive training checkpoints. Benchmarks with a higher signal-to-noise ratio make small-scale decisions more reliable. Averaging scores over several neighbouring checkpoints reduces noise and consistently improves decisions such as early-stopping predictions. Covers 30 benchmarks and 465 models from 60M to 32B. Sources: [arXiv:2508.13144](https://arxiv.org/abs/2508.13144); [Ai2 blog](https://allenai.org/blog/signal-noise).
- **Quantifying Variance in Evaluation Benchmarks.** Measures seed variance and checkpoint-to-checkpoint monotonicity and urges practitioners to "carefully factor in variance when comparing models". Item-analysis and IRT methods did not meaningfully reduce variance. Source: [Madaan et al. 2024, arXiv:2406.10229](https://arxiv.org/abs/2406.10229). Only the abstract was read; specific numbers were not extracted.
- **Local simulation, scenario 1** (N truly equal checkpoints, independent binomial noise). The expected optimism, E[max dev acc] − true acc, in points:

  | p | n | SE | N=10 | N=20 | N=40 |
  |---|---|---|---|---|---|
  | 0.70 | 300 | 2.6 | 4.0 | 4.9 | 5.6 |
  | 0.70 | 500 | 2.0 | 3.1 | 3.8 | 4.4 |
  | 0.80 | 300 | 2.3 | 3.5 | 4.2 | 4.8 |
  | 0.80 | 500 | 1.8 | 2.7 | 3.3 | 3.8 |

  These match the analytic approximation optimism ≈ SE·E[max of N standard normals], with that expected max ≈ 1.54, 1.87 and 2.16 for N = 10, 20 and 40.
- **Local simulation, scenario 2** (shared items; latent per-item correlation r between checkpoints; p = 0.75; all truly equal). Optimism of the winner:

  | n | r | N=10 | N=20 | N=40 |
  |---|---|---|---|---|
  | 300 | 0.5 | 3.2 | 3.8 | 4.4 |
  | 300 | 0.8 | 2.5 | 3.0 | 3.5 |
  | 300 | 0.9 | 2.1 | 2.5 | 3.0 |
  | 500 | 0.5 | 2.5 | 2.9 | 3.5 |
  | 500 | 0.8 | 2.0 | 2.3 | 2.7 |
  | 500 | 0.9 | 1.6 | 2.1 | 2.3 |

  Here the "true" accuracy is defined over the item universe. The winner's dev score minus the candidates' mean is the same quantity.
- **Local simulation, scenario 3** (N = 20; true accuracy has a smooth bump with a 3-point range, peaking mid-run; n = 400; r = 0.8). Argmax selection: regret 0.35 points versus the true best, but the winner's reported dev score overstates its true accuracy by 2.25 points. Choosing the argmax of a 3-checkpoint moving average cut regret to 0.24 points.

### Inferences
- **Regret versus bias.** With smooth true curves, argmax selection rarely costs much true accuracy (regret about 0.3 points), but its reported dev score is inflated by about 2 points. The real danger is believing a merge or checkpoint "beats v4 by 2 points" on dev when it does not.
- **Rule of thumb.** For one task with n ≈ 400, a dev gain below about 2·SE_diff is not evidence. Paired SE_diff ≈ sqrt(discordant fraction / n). With 10% discordant items at n = 400, SE_diff ≈ 1.6 points.
- **Macro-averaging tasks helps.** A macro-average over three tasks has about 1/sqrt(3) the per-task noise if the tasks are independent: about 1.2 points at n ≈ 400 each. Select on the macro-average of the three dev tasks, not per-task argmaxes. Per-task argmaxes triple the number of comparisons.

### Gaps
- Real per-item agreement rates between neighbouring RL checkpoints for our model are unknown. Measure them from the dev runs, since they set the discordant fraction and therefore paired power.
- The 2025-26 literature was not searched for a closed-form "best-of-N checkpoint" correction specific to LLMs. The extreme-value approximation above is standard statistics.

## (A2) Paired tests, multiple comparisons, smoothing, one-shot vault

### Takeaway
- Compare candidates against one fixed reference (v4 or the final checkpoint) on the same items with paired tests: McNemar's test or a paired bootstrap.
- Reduce the effective number of candidates before testing. Smooth across adjacent checkpoints, pre-register a short list, and select on the macro-average.
- Control multiplicity with Holm, or simply by requiring the winner's margin to exceed the expected-max optimism.
- Open the vault once, for at most 1-3 pre-chosen finalists, and accept its verdict even if it disagrees with dev.

### Cited Findings
- Paired-difference analysis removes question-difficulty variance and should be reported with mean differences, SEs, CIs and correlations. Source: [Anthropic error-bars research](https://www.anthropic.com/research/statistical-approach-to-model-evals).
- Neighbouring-checkpoint averaging reduces noise in development decisions. Source: [Signal and Noise, arXiv:2508.13144](https://arxiv.org/abs/2508.13144).
- An unbiased performance estimate for a selected model needs data unseen by the selection. Source: [Cawley & Talbot 2010](https://jmlr.org/papers/v11/cawley10a.html).
- Local simulation (scenario 3): smoothing with a 3-window moving average lowered selection regret from 0.35 to 0.24 points.

### Inferences
Recommended concrete protocol, assembled from the findings above:
1. **Freeze dev-set usage.** Use greedy decoding (the grading setting) on all three private dev sets. Use one score per candidate: the macro-average of IFEval-like, MATH-like and MBPP-like dev accuracy, after deciding on any weighting in advance.
2. **Stage 1, trajectory screen.** Score all ~20 RL checkpoints. Compute a centred 3-checkpoint moving average of the macro score. Pick the argmax window, and take its centre checkpoint plus its neighbours as candidates. If the curve is flat (all within about 2·SE), prefer the latest stable checkpoint, or the average of the last K checkpoints (see the merging notes from other researchers).
3. **Stage 2, a small pre-registered merge set.** At most about 3-5 merges, e.g. the uniform average of the top window and one or two v4↔RL interpolation coefficients. Do not sweep. This also matches the user's "first best guess" preference. Each extra candidate adds roughly 0.3-0.5 points of expected optimism at N ≈ 10-20 (scenario 1/2 tables).
4. **Paired comparison against v4 and against the best native checkpoint.**
   - Per task: McNemar's test on discordant items (b, c), z = (b − c)/sqrt(b + c).
   - Macro score: paired bootstrap over items, resampling items within each task, 10k resamples.
   - Apply Holm correction across the candidates compared.
   - Pick a merge over the native checkpoint only if its lower 95% bound is above 0, or at least its margin exceeds the expected-max optimism for that N. Otherwise prefer the simpler candidate.
5. **Vault, once.** Evaluate at most 2 finalists, the stage-4 winner and the safest fallback (e.g. the latest stable or smoothed checkpoint), on the vault in a single pass. Report the vault score as the unbiased estimate. Use it to choose between those two only. If it reverses dev by less than the paired SE, keep the pre-declared default rather than re-opening selection.
6. **Never** touch the public IFEval, MATH-500 or MBPP test sets in any of the above.
- **Regressions per task matter.** Grading includes hidden held-out sets per task, so also require no task to regress more than about 2 SE versus v4. A macro gain bought by an IFEval loss is risky; Tulu 3 notes that models "generally overfit to IFEval" (see A3).

### Gaps
- Bootstrap code and the exact item-agreement statistics depend on our own dev outputs, which are not yet produced.
- No source tells us how correlated private-dev gains are with the hidden held-out sets. The vault is the only check.

## (A3) How labs choose final checkpoints

### Takeaway
Public reports describe dev/unseen splits and merging or souping more than an explicit "argmax checkpoint" rule. Tulu 3 tuned only on a development suite, never looked at its unseen suite during development, and later used the unseen suite to measure overfitting. One finding listed there is that models generally overfit to IFEval. Detailed RLVR checkpoint-selection rules were not retrieved in this session.

### Cited Findings
- Tulu 3 uses a development evaluation suite for decisions and an unseen suite it "did not examine... when developing our models". The unseen suite exists to measure "how much we may have overfit to particular evaluations" across data, algorithm and hyperparameter decisions. Source: [Tulu 3, arXiv:2411.15124](https://arxiv.org/html/2411.15124v4) (Section 2.2).
- Tulu 3 Section 7.4, "Evaluating the Development Process Using the Unseen Suite", includes the finding "Models generally overfit to IFEval". Only the table of contents entry was retrieved, not its body. Source: [Tulu 3](https://arxiv.org/html/2411.15124v4).
- Tulu 3 lists "Random Seeds and Model Soups" (Section 4.3.1) and says it "explored model merging techniques" for SFT. The body was not retrieved. Source: [Tulu 3](https://arxiv.org/html/2411.15124v4).
- Ai2's Signal and Noise recommends checkpoint-averaging of scores for development decisions. Source: [Signal and Noise](https://arxiv.org/abs/2508.13144).

### Inferences
- Our dev/vault split mirrors Tulu 3's development/unseen design. IFEval-like dev gains in particular deserve extra scepticism, given Tulu 3's overfitting finding.

### Gaps
- The exact Tulu 3 RLVR final-checkpoint rule (Section 6.2) was not retrieved; the fetch truncated at 100k characters.
- The Llama 3 checkpoint/model-averaging practice and the OLMo 2 final-souping practice were not verified in this session. I recall that Llama 3 averages models from different data/hyperparameter runs at each post-training stage and that OLMo 2 souped final checkpoints, but both are unverified, so they are left out of the cited findings.

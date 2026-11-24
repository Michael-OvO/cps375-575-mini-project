# Data versioning

One private Hugging Face dataset repo holds all our data. Code stays on GitHub; GitHub records which data
version each run used.

## Layout

| Folder | What | Safe to train on directly? |
|---|---|---|
| `mixtures/<run>/` | Finished SFT mixtures (`train.jsonl` + `manifest.json`), dev-filtered and leakage-checked | Yes |
| `sources/` | Inputs to `select.py` (teacher outputs, tasks, self-distilled data) | **No**: some rows are dev or vault items; build mixtures with `select.py`, which drops them |
| `rl/pools/` | rl1 prompt pools (dev-filtered) | Yes, for RL |
| `eval/` | Dev files (vault items flagged `"vault": true`) and RL trap prompts | **Never** |

`MANIFEST.json` lists every file with its sha256, size and row count. `CHANGELOG.md` says what changed and why.

## Rules

1. **Every change is one push** with a message: `python -m pipeline.data_release push -m "..."`.
   Run `plan` first to see what will upload. Add a line to `docs/data/CHANGELOG.md` in the same GitHub commit.
2. **Tag only what a training run used.** The tag is the run name (`v4`, `rl1`, `v5`). Tags are never moved or
   reused. Between runs, `main` is the working state.
3. **Record the tag** in the run's EXPERIMENTS.md entry: "data: `cps375-data@v5`". With the code commit, that
   pins exactly what produced a model.
4. **Never edit a released mixture.** A changed mixture is a new run name (`mixtures/v5/`, then `mixtures/v6/`).
   Source files may be updated; history keeps the old copies.
5. **Never train on `eval/`.** Measure with `dev_eval --split dev`. The vault is evaluated only on a frozen final
   candidate, and each use is logged in EXPERIMENTS.md.
6. **No secrets or logs**: the script uploads only the folders above (no `.done`, usage files, logs or keys).

## Common commands

```bash
export HF_DATA_REPO=<org>/cps375-data
.venv/bin/hf download $HF_DATA_REPO --repo-type dataset --local-dir data_hub/             # latest
.venv/bin/hf download $HF_DATA_REPO --repo-type dataset --revision v4 --local-dir data_v4/ # a version
.venv/bin/python -m pipeline.data_release plan
.venv/bin/python -m pipeline.data_release push -m "add rl1 shortest-correct distilled data"
.venv/bin/python -m pipeline.data_release tag v5 -m "v5 SFT trained on this"
```

To use downloaded sources with the pipeline, copy `sources/*` and `eval/*` into `data/`.

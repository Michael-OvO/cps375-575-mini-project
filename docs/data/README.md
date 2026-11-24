---
license: other
pretty_name: CPS 375 project data (private)
---

# CPS 375 project data (private)

Training and evaluation data for our Qwen3.5-9B-Base LoRA SFT and RL runs. Private to the team.

**Read before using:**
- `mixtures/<run>/train.jsonl`: finished SFT mixtures, safe to train on.
- `sources/`: inputs to `pipeline/select.py`. Some rows are dev or vault items; never train on these files
  directly, build a mixture with `select.py`.
- `rl/pools/`: RL prompt pools, safe for RL.
- `eval/`: dev and vault items. **Never train on them.**

Versioning rules: `docs/data/VERSIONING.md` in the code repo. File hashes: `MANIFEST.json`. History:
`CHANGELOG.md`.

Contains outputs from teacher models (OpenAI, Anthropic, Codex) and data derived from MATH train,
allenai/tulu-3-sft-mixture, allenai/IF_multi_constraints_upto5 and nvidia/OpenCodeInstruct; their licenses and
terms apply. Not for redistribution.

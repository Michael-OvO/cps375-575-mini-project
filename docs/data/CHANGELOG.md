# Data changelog

Newest first. One entry per push; note the tag when a training run used the data.

## 2026-11-24: first release (tag `rl1`)
- `mixtures/v1`–`v4`: the SFT mixtures behind v1–v4 (v4 is the base of rl1 and the current submission).
- `sources/`: every select.py input used through v4 (MATH-train solutions, teacher math, verified code tasks,
  OCI code, IF-RLVR answers, self-distilled data).
- `rl/pools/`: the rl1 prompt pools.
- `eval/`: dev files with vault flags, and RL trap prompts.

"""Paths, model ids and constants shared by the whole pipeline."""

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"

STUDENT = "Qwen/Qwen3.5-9B-Base"
# Fable 5.2 wrote the first part of the pool (until its usage limit on 2026-11-14);
# Opus 5.5 is the default from then on. The served model is recorded per call in usage.json.
CLAUDE_MODEL = os.environ.get("CLAUDE_TEACHER_MODEL", "claude-opus-5-5")
# GPT-6.1 Sol (2026-11-15): near GPT-6 Astra on OpenAI's reasoning and coding evals at a fifth of the price.
GPT_MODEL = os.environ.get("OPENAI_TEACHER_MODEL", "").strip() or "gpt-6.1-sol"
TINKER_TEACHER = "Qwen/Qwen3.5-397B-A17B"

SEED = 20261113
MAX_TRAIN_TOKENS = 2048
ALLOW_TEST_OVERLAP_CHECK = os.environ.get("ALLOW_TEST_OVERLAP_CHECK") == "1"

# Datasets that must never be a training source, in any split.
EXCLUDED_DATASETS = (
    "google/IFEval",
    "HuggingFaceH4/MATH-500",
    "google-research-datasets/mbpp",
)

# Exact dataset ids that may be used as training sources (checked case-sensitively).
ALLOWED_TRAIN_DATASETS = (
    "DigitalLearningGmbH/MATH-lighteval",
    "allenai/tulu-3-sft-mixture",
    "teacher-generated",
)

# Lower-case fragments that disqualify a training source by dataset id or source name.
EXCLUDED_KEYWORDS = ("mbpp", "ifeval", "math-500", "math500", "humaneval", "ifbench")


def _bucket(item_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 1000


def is_dev(item_id: str, per_mille: int) -> bool:
    """True for a fixed pseudo-random `per_mille`/1000 share of ids."""
    return _bucket(item_id, "dev") < per_mille


def is_vault(item_id: str) -> bool:
    """A third of dev items are held back for final candidates only."""
    return _bucket(item_id, "vault") < 333

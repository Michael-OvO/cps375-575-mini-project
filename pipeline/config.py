"""Paths, model ids and constants shared by the whole pipeline."""

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"

STUDENT = "Qwen/Qwen3.5-9B-Base"
CLAUDE_MODEL = "claude-fable-5-2"
GPT_MODEL = os.environ.get("OPENAI_TEACHER_MODEL", "")
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


def _bucket(item_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 1000


def is_dev(item_id: str, per_mille: int) -> bool:
    """True for a fixed pseudo-random `per_mille`/1000 share of ids."""
    return _bucket(item_id, "dev") < per_mille


def is_vault(item_id: str) -> bool:
    """A third of dev items are held back for final candidates only."""
    return _bucket(item_id, "vault") < 333

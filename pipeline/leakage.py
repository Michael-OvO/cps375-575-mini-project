"""Guards against evaluation data reaching the training set."""

import re

from pipeline.config import ALLOW_TEST_OVERLAP_CHECK, EXCLUDED_DATASETS

ALLOWED_SPLITS = {"train", "generated"}
_TOKEN = re.compile(r"[a-z0-9]+")


def check_sources(sources: list[dict]) -> None:
    for src in sources:
        if any(ex.lower() in src["dataset"].lower() for ex in EXCLUDED_DATASETS):
            raise ValueError(f"excluded dataset used as a training source: {src}")
        if src["split"] not in ALLOWED_SPLITS:
            raise ValueError(f"non-train split used as a training source: {src}")


def ngrams(text: str, n: int = 8) -> set[tuple]:
    tokens = _TOKEN.findall(text.lower())
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class OverlapFilter:
    def __init__(self, test_texts: list[str]):
        self.grams: set[tuple] = set()
        for text in test_texts:
            self.grams |= ngrams(text)

    def is_leak(self, text: str, threshold: float = 0.3) -> bool:
        grams = ngrams(text)
        if not grams:
            return False
        return len(grams & self.grams) / len(grams) >= threshold


def load_overlap_filter() -> OverlapFilter | None:
    """Load test-set texts for filtering only. Disabled unless staff have allowed it."""
    if not ALLOW_TEST_OVERLAP_CHECK:
        return None
    from datasets import load_dataset

    texts: list[str] = []
    texts += list(load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")["problem"])
    texts += list(load_dataset("google/IFEval", split="train")["prompt"])
    for split in ("train", "test", "validation", "prompt"):
        texts += list(load_dataset("google-research-datasets/mbpp", "sanitized", split=split)["prompt"])
    return OverlapFilter(texts)

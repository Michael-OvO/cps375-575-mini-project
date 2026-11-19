"""Guards against evaluation data reaching the training set."""

import re
from collections import Counter

from pipeline.config import (
    ALLOW_TEST_OVERLAP_CHECK,
    ALLOWED_TRAIN_DATASETS,
    EXCLUDED_DATASETS,
    EXCLUDED_KEYWORDS,
)

ALLOWED_SPLITS = {"train", "generated"}
_TOKEN = re.compile(r"[a-z0-9]+")

# Test texts with at least this many tokens are also matched as contiguous phrases.
_MIN_PHRASE_TOKENS = 4
_PHRASE_KEY_LEN = 4
# A candidate is flagged if this share of any single test item's n-grams appears in it.
_ITEM_COVERAGE = 0.5


def check_sources(sources: list[dict]) -> None:
    """Fail closed: every source must be an allowlisted dataset in a training split."""
    for src in sources:
        try:
            dataset = src["dataset"]
            split = src["split"]
        except KeyError as exc:
            raise ValueError(f"training source missing key {exc}: {src}") from exc
        name = str(src.get("name", "")).lower()
        if dataset not in ALLOWED_TRAIN_DATASETS:
            raise ValueError(f"training source not on the allowlist: {src}")
        for text in (dataset.lower(), name):
            if any(kw in text for kw in EXCLUDED_KEYWORDS):
                raise ValueError(f"excluded keyword in training source: {src}")
            if any(ex.lower() in text for ex in EXCLUDED_DATASETS):
                raise ValueError(f"excluded dataset used as a training source: {src}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"non-train split used as a training source: {src}")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def ngrams(text: str, n: int = 8) -> set[tuple]:
    tokens = _tokens(text)
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class OverlapFilter:
    def __init__(self, test_texts: list[str], n: int = 8):
        self.n = n
        self.grams: set[tuple] = set()  # all n-grams of all test texts
        self.whole: set[str] = set()  # normalised whole test texts
        self.phrases: dict[tuple, list[tuple]] = {}  # first tokens -> phrases to match
        self.postings: dict[tuple, list[int]] = {}  # n-gram -> indices of test items
        self.item_sizes: list[int] = []  # n-gram count of each test item
        for text in test_texts:
            tokens = _tokens(text)
            if not tokens:
                continue
            self.whole.add(" ".join(tokens))
            if len(tokens) >= _MIN_PHRASE_TOKENS:
                phrase = tuple(tokens)
                self.phrases.setdefault(phrase[:_PHRASE_KEY_LEN], []).append(phrase)
            if len(tokens) >= n:
                grams = ngrams(text, n)
                self.grams |= grams
            else:
                grams = {tuple(tokens)}  # short item: index it as one n-gram of its own length
            item = len(self.item_sizes)
            self.item_sizes.append(len(grams))
            for gram in grams:
                self.postings.setdefault(gram, []).append(item)

    def _contains_phrase(self, tokens: list[str]) -> bool:
        for i in range(len(tokens) - _PHRASE_KEY_LEN + 1):
            for phrase in self.phrases.get(tuple(tokens[i:i + _PHRASE_KEY_LEN]), ()):
                if tuple(tokens[i:i + len(phrase)]) == phrase:
                    return True
        return False

    def is_leak(self, text: str, threshold: float = 0.3) -> bool:
        tokens = _tokens(text)
        if not tokens:
            return False
        if " ".join(tokens) in self.whole or self._contains_phrase(tokens):
            return True
        grams = ngrams(text, self.n)
        if not grams:
            return False
        if len(grams & self.grams) / len(grams) >= threshold:
            return True
        hits: Counter = Counter()
        for gram in grams:
            for item in self.postings.get(gram, ()):
                hits[item] += 1
        return any(count / self.item_sizes[item] >= _ITEM_COVERAGE for item, count in hits.items())


def eval_texts() -> list[str]:
    """Every graded or held-out evaluation text that must never reach training."""
    from datasets import load_dataset

    texts: list[str] = []
    texts += list(load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")["problem"])
    texts += list(load_dataset("google/IFEval", split="train")["prompt"])
    for config, column in (("sanitized", "prompt"), ("full", "text")):
        for split in ("train", "test", "validation", "prompt"):
            texts += list(load_dataset("google-research-datasets/mbpp", config, split=split)[column])
    texts += list(load_dataset("openai/openai_humaneval", split="test")["prompt"])
    texts += list(load_dataset("allenai/IFBench_test", split="train")["prompt"])
    return texts


def load_overlap_filter() -> OverlapFilter | None:
    """Load test-set texts for filtering only. Disabled unless staff have allowed it."""
    if not ALLOW_TEST_OVERLAP_CHECK:
        return None
    return OverlapFilter(eval_texts())

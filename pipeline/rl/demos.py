"""Gated-SFT demonstrations (spec §6.4): verified answers already in v4's training set. A v4 row matches a pool
item when the item id is a dash-boundary prefix of the row id and the item's raw text appears in the row's user
message, so a shared prefix of two different problems can never match."""

from pipeline import jsonl
from pipeline.config import RUNS

TASKS = ("math", "code")


def _norm(s: str) -> str:
    return " ".join(s.split())


def _content(messages: list[dict], role: str) -> str:
    return next(m["content"] for m in messages if m["role"] == role)


def id_prefixes(example_id: str) -> list[str]:
    parts = example_id.split("-")
    return ["-".join(parts[:k]) for k in range(2, len(parts) + 1)]


def load(path=RUNS / "v4" / "train.jsonl") -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for r in jsonl.read(path):
        if r["task"] in TASKS:
            for p in id_prefixes(r["id"]):
                index.setdefault(p, []).append(r)
    return index


def pick(item: dict, index: dict[str, list[dict]]) -> list[dict] | None:
    text = _norm(item["text"])
    cands = [r for r in index.get(item["id"], []) if text in _norm(_content(r["messages"], "user"))]
    if not cands:
        return None
    return min(cands, key=lambda r: len(_content(r["messages"], "assistant")))["messages"]

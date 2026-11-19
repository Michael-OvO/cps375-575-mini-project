"""Paired comparison of per-item dev results against v4 (McNemar exact test, agreement, paired standard error)."""

import math
import sys

from pipeline import jsonl
from pipeline.config import RUNS


def load_v4_items(task: str) -> dict[str, bool]:
    name = {"math": "math", "code": "code", "if": "if"}[task]
    return {r["id"]: bool(r["correct"]) for r in jsonl.read(RUNS / "v4" / "errors" / f"{name}.jsonl")}


def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(base: dict[str, bool], new: dict[str, bool]) -> dict:
    ids = sorted(set(base) & set(new))
    n = len(ids)
    b = sum(base[i] and not new[i] for i in ids)
    c = sum(new[i] and not base[i] for i in ids)
    d = (c - b) / n
    se = math.sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
    return {"n": n, "b": b, "c": c, "p_mcnemar": _binom_two_sided(min(b, c), b + c),
            "agreement": 1 - (b + c) / n, "diff": d, "se": se}


def per_task(items: list[dict]) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {"math": {}, "code": {}, "if": {}}
    for r in items:
        out[r["task"]][r["id"]] = bool(r["correct"])
    return out


def gate(items_path) -> dict:
    new = per_task(jsonl.read(items_path))
    res = {t: compare(load_v4_items(t), new[t]) for t in ("math", "code", "if")}
    res["passed"] = all(r["p_mcnemar"] > 0.05 and r["agreement"] >= 0.85 for r in res.values())
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(gate(sys.argv[1]), indent=1))

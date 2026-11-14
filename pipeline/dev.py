"""Build the private dev sets for code and instruction following.

Usage:
    python -m pipeline.dev            # code_dev.jsonl and if_dev.jsonl
    python -m pipeline.dev --if-new   # only if_dev_new.jsonl (new dev-only constraint types)
"""

import argparse
import random
import re

from pipeline import jsonl
from pipeline.config import DATA, SEED, is_dev, is_vault
from pipeline.constraints.sample import render_prompt, sample_constraints

CODE_DEV_PER_MILLE = 80
IF_DEV_PER_MILLE = 80  # must match generate/ifeval.py DEV_PER_MILLE
IF_DEV_SIZE = 500
IF_DEV_NEW_SIZE = 200


def normalize_code_text(text: str) -> str:
    """Lower-case [a-z0-9]+ tokens joined by spaces: identical descriptions normalise identically."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def is_code_dev(text: str) -> bool:
    """Dev membership of a code task, decided by its normalised description text (never its id),
    so the same description can never be on both the dev and the training side."""
    return is_dev(normalize_code_text(text), CODE_DEV_PER_MILLE)


def dedupe_code_tasks(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """(kept, duplicates): keep the first task for each normalised description."""
    seen: set[str] = set()
    kept, dupes = [], []
    for task in tasks:
        key = normalize_code_text(task["text"])
        (dupes if key in seen else kept).append(task)
        seen.add(key)
    return kept, dupes


def build_if_dev(tasks: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 1)
    items = []
    for i, task in enumerate(tasks):
        kind = "seen" if i % 2 == 0 else "unseen"
        constraints = sample_constraints(rng, task["task"], mode=f"dev_{kind}")
        items.append({
            "id": f"ifdev-{task['id']}",
            "task": task["task"],  # base task, for the training-side dev-overlap filter in select
            "prompt": render_prompt(task["task"], constraints),
            "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in constraints],
            "kind": kind,
            "vault": is_vault(task["id"]),
        })
    return items


def build_if_dev_new(tasks: list[dict], n: int = IF_DEV_NEW_SIZE) -> list[dict]:
    """Rows whose first constraint is one of the new (post-v1) dev-only types; same format as if_dev."""
    rng = random.Random(SEED + 2)
    items = []
    for task in tasks[:n]:
        constraints = sample_constraints(rng, task["task"], mode="dev_unseen_new")
        items.append({
            "id": f"ifdevnew-{task['id']}",
            "task": task["task"],
            "prompt": render_prompt(task["task"], constraints),
            "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in constraints],
            "kind": "unseen_new",
            "vault": is_vault(task["id"]),
        })
    return items


def _if_dev_base() -> list[dict]:
    """The dev-side IF base tasks used by the IF dev sets, in file order."""
    base = [t for t in jsonl.read(DATA / "if_base_tasks.jsonl") if is_dev(t["id"], IF_DEV_PER_MILLE)]
    return base[:IF_DEV_SIZE]


def main_if_new() -> None:
    rows = build_if_dev_new(_if_dev_base())
    jsonl.write(DATA / "if_dev_new.jsonl", rows)
    print(f"instruction-following dev (new unseen types): {len(rows)}")


def main() -> None:
    tasks, _ = dedupe_code_tasks(jsonl.read(DATA / "code_tasks.jsonl"))
    code = [dict(t, vault=is_vault(t["id"])) for t in tasks if is_code_dev(t["text"])][:500]
    jsonl.write(DATA / "code_dev.jsonl", code)
    if_dev = build_if_dev(_if_dev_base())
    jsonl.write(DATA / "if_dev.jsonl", if_dev)
    print(f"code dev: {len(code)}; instruction-following dev: {len(if_dev)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-new", action="store_true", help="only write data/if_dev_new.jsonl")
    if parser.parse_args().if_new:
        main_if_new()
    else:
        main()

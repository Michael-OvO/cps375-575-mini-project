"""Build the private dev sets for code and instruction following.

Usage: python -m pipeline.dev
"""

import random

from pipeline import jsonl
from pipeline.config import DATA, SEED, is_dev, is_vault
from pipeline.constraints.sample import render_prompt, sample_constraints

CODE_DEV_PER_MILLE = 80
IF_DEV_PER_MILLE = 80  # must match generate/ifeval.py DEV_PER_MILLE
IF_DEV_SIZE = 500


def is_code_dev(task_id: str) -> bool:
    return is_dev(task_id, CODE_DEV_PER_MILLE)


def build_if_dev(tasks: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 1)
    items = []
    for i, task in enumerate(tasks):
        kind = "seen" if i % 2 == 0 else "unseen"
        constraints = sample_constraints(rng, task["task"], mode=f"dev_{kind}")
        items.append({
            "id": f"ifdev-{task['id']}",
            "prompt": render_prompt(task["task"], constraints),
            "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in constraints],
            "kind": kind,
            "vault": is_vault(task["id"]),
        })
    return items


def main() -> None:
    code = [dict(t, vault=is_vault(t["id"])) for t in jsonl.read(DATA / "code_tasks.jsonl")
            if is_code_dev(t["id"])][:500]
    jsonl.write(DATA / "code_dev.jsonl", code)
    base = [t for t in jsonl.read(DATA / "if_base_tasks.jsonl") if is_dev(t["id"], IF_DEV_PER_MILLE)]
    if_dev = build_if_dev(base[:IF_DEV_SIZE])
    jsonl.write(DATA / "if_dev.jsonl", if_dev)
    print(f"code dev: {len(code)}; instruction-following dev: {len(if_dev)}")


if __name__ == "__main__":
    main()

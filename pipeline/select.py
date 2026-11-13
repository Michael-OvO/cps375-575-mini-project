"""Turn verified pools into one training file plus a manifest.

Usage:
    python -m pipeline.select --name v1 --math 30000 --ifeval 6000 --code 6000 --general 2000
    python -m pipeline.select --name smoke --math 500 --ifeval 200 --code 200 --general 100
"""

import argparse
import json
import random

from pipeline import formats, jsonl, leakage
from pipeline.config import DATA, MAX_TRAIN_TOKENS, RUNS, SEED, STUDENT
from pipeline.dev import is_code_dev

MIN_SOLUTION_WORDS = 40
LEVEL_WEIGHT = {1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}
TRAIN_PRICE_PER_M = 1.463


def _unique_by_id(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            out.append(row)
    return out


def _read_pool(name: str) -> list[dict]:
    return _unique_by_id(jsonl.read(DATA / name))


def _shortest_per_problem(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        n = len(row["solution"].split())
        if n < MIN_SOLUTION_WORDS:
            continue
        if row["pid"] not in best or n < len(best[row["pid"]]["solution"].split()):
            best[row["pid"]] = row
    return list(best.values())


def pick_math(train_rows: list[dict], new_rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    train = _shortest_per_problem(train_rows)
    rng.shuffle(train)
    picked = train[:n]
    remaining = n - len(picked)
    new = _shortest_per_problem(new_rows)
    if remaining > 0 and new:
        # Weighted sampling without replacement: sort by weighted random key.
        keyed = sorted(new, key=lambda r: rng.random() ** (1.0 / LEVEL_WEIGHT.get(r["level"], 1.0)),
                       reverse=True)
        picked += keyed[:remaining]
    return picked


def math_example(row: dict, rng: random.Random) -> dict:
    prompt = formats.math_prompt(row["problem"], rng)
    return {"id": row["id"], "task": "math",
            "messages": formats.to_messages(prompt, formats.math_target(row["solution"], row["target"]))}


def code_example(row: dict, rng: random.Random) -> dict:
    prompt = formats.code_prompt(row["text"], row["tests"], rng)
    return {"id": row["id"], "task": "code",
            "messages": formats.to_messages(prompt, formats.code_target(row["code"]))}


def if_example(row: dict) -> dict:
    return {"id": row["id"], "task": "ifeval", "messages": formats.to_messages(row["prompt"], row["response"])}


def general_example(row: dict) -> dict:
    return {"id": row["id"], "task": "general", "messages": row["messages"]}


def _token_counter():
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(STUDENT)
    renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(STUDENT), tokenizer)

    def count(messages: list[dict]) -> int:
        datum = conversation_to_datum(messages, renderer, max_length=None)
        return len(datum.model_input.to_ints())

    return count


def build(name: str, sizes: dict) -> dict:
    rng = random.Random(SEED)
    math_train = _read_pool("math_train_solutions.jsonl")
    math_new = _read_pool("math_new_solutions.jsonl")
    code_pool = [t for t in _read_pool("code_tasks.jsonl") if not is_code_dev(t["id"])]
    if_pool = _read_pool("if_responses.jsonl")
    general_pool = _read_pool("general.jsonl")
    for pool in (code_pool, if_pool, general_pool):
        rng.shuffle(pool)

    math_rows = pick_math(math_train, math_new, sizes["math"], rng)
    code_rows = code_pool[: sizes["code"]]
    if_rows = if_pool[: sizes["ifeval"]]
    general_rows = general_pool[: sizes["general"]]

    # Each candidate: (example, raw text for the overlap filter, manifest source, teacher).
    # The raw text is never written to train.jsonl.
    candidates = (
        [(math_example(r, rng), r["problem"], r.get("source", "math_train"), r.get("teacher")) for r in math_rows]
        + [(code_example(r, rng), r["text"], "code", None) for r in code_rows]
        + [(if_example(r), r["prompt"], "instruction_following", r.get("teacher")) for r in if_rows]
        + [(general_example(r), None, "general", None) for r in general_rows]
    )

    overlap = leakage.load_overlap_filter()
    count = _token_counter()
    kept, per_source, teachers = [], {}, set()
    dropped_long = dropped_leak = tokens = 0
    for ex, raw, source, teacher in candidates:
        if overlap is not None and raw is not None and overlap.is_leak(raw):
            dropped_leak += 1
            continue
        n = count(ex["messages"])
        if n > MAX_TRAIN_TOKENS:
            dropped_long += 1
            continue
        tokens += n
        kept.append(ex)
        per_source[source] = per_source.get(source, 0) + 1
        if teacher:
            teachers.add(teacher)
    rng.shuffle(kept)

    sources = [
        {"name": "math_train", "dataset": "DigitalLearningGmbH/MATH-lighteval", "split": "train",
         "count": per_source.get("math_train", 0)},
        {"name": "math_new", "dataset": "teacher-generated", "split": "generated",
         "count": per_source.get("math_new", 0)},
        {"name": "code", "dataset": "teacher-generated", "split": "generated",
         "count": per_source.get("code", 0)},
        {"name": "instruction_following", "dataset": "teacher-generated", "split": "generated",
         "count": per_source.get("instruction_following", 0)},
        {"name": "general", "dataset": "allenai/tulu-3-sft-mixture", "split": "train",
         "count": per_source.get("general", 0)},
    ]
    leakage.check_sources(sources)
    manifest = {
        "name": name, "seed": SEED, "sources": sources,
        "total_examples": len(kept), "total_tokens": tokens,
        "estimated_cost_usd": round(tokens / 1e6 * TRAIN_PRICE_PER_M, 2),
        "dropped_too_long": dropped_long, "dropped_overlap": dropped_leak,
        "overlap_filter_enabled": overlap is not None,
        "teachers": sorted(teachers),
    }
    out = RUNS / name
    jsonl.write(out / "train.jsonl", kept)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--math", type=int, default=30000)
    p.add_argument("--ifeval", type=int, default=6000)
    p.add_argument("--code", type=int, default=6000)
    p.add_argument("--general", type=int, default=2000)
    args = p.parse_args()
    manifest = build(args.name, {"math": args.math, "ifeval": args.ifeval,
                                 "code": args.code, "general": args.general})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

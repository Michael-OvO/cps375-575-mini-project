"""Turn verified pools into one training file plus a manifest.

Usage:
    python -m pipeline.select --name v1 --math 30000 --ifeval 6000 --code 6000 --general 2000
    python -m pipeline.select --name smoke --math 500 --ifeval 200 --code 200 --general 100
"""

import argparse
import json
import random

from pipeline import formats, jsonl, leakage
from pipeline.config import DATA, MAX_TRAIN_TOKENS, RUNS, SEED, STUDENT, is_dev
from pipeline.dev import IF_DEV_PER_MILLE, dedupe_code_tasks, is_code_dev
from pipeline.generate.math import DEV_PER_MILLE as MATH_DEV_PER_MILLE
from pipeline.teachers import has_control_chars

MIN_SOLUTION_WORDS = 40
LEVEL_WEIGHT = {1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}
TRAIN_PRICE_PER_M = 1.463


def _unique_by_id(rows: list[dict], dupes: list[dict] | None = None) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            out.append(row)
        elif dupes is not None:
            dupes.append(row)
    return out


def _read_pool(name: str, dupes: list[dict] | None = None, missing: list[str] | None = None) -> list[dict]:
    if missing is not None and not (DATA / name).exists():
        missing.append(name)
    return _unique_by_id(jsonl.read(DATA / name), dupes)


def _shortest_per_problem(rows: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for row in rows:
        n = len(row["solution"].split())
        if n < MIN_SOLUTION_WORDS:
            continue
        if row["pid"] not in best or n < len(best[row["pid"]]["solution"].split()):
            best[row["pid"]] = row
    return list(best.values())


def math_candidates(train_rows: list[dict], new_rows: list[dict], rng: random.Random) -> list[dict]:
    """Full ordered candidate list: solved train problems first, then new problems weighted to hard levels."""
    train = _shortest_per_problem(train_rows)
    rng.shuffle(train)
    new = _shortest_per_problem(new_rows)
    # Weighted sampling without replacement: sort by weighted random key.
    keyed = sorted(new, key=lambda r: rng.random() ** (1.0 / LEVEL_WEIGHT.get(r["level"], 1.0)),
                   reverse=True)
    return train + keyed


def pick_math(train_rows: list[dict], new_rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    return math_candidates(train_rows, new_rows, rng)[:n]


def math_example(row: dict, rng: random.Random) -> dict:
    prompt = formats.math_prompt(row["problem"], rng)
    return {"id": row["id"], "task": "math",
            "messages": formats.to_messages(prompt, formats.math_target(row["solution"], row["target"]))}


def code_example(row: dict, rng: random.Random) -> dict:
    prompt = formats.code_prompt(row["text"], row["tests"], rng)
    return {"id": row["id"], "task": "code",
            "messages": formats.to_messages(prompt, formats.code_target(row["code"]))}


def if_example(row: dict) -> dict:
    return {"id": row["id"], "task": "instruction_following",
            "messages": formats.to_messages(row["prompt"], row["response"])}


def _strip_content(msg):
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return {**msg, "content": msg["content"].strip()}
    return msg


def general_example(row: dict) -> dict:
    messages = row["messages"]
    if isinstance(messages, list):
        messages = [_strip_content(m) for m in messages]
    return {"id": row["id"], "task": "general", "messages": messages}


def _user_text(row: dict) -> str | None:
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and isinstance(msgs[0].get("content"), str):
        return msgs[0]["content"]
    return None


# ---- dev membership and dev overlap (private dev sets only: train-side data, no flag needed) ----

def is_dev_member(task: str, row: dict) -> bool:
    if task == "math":
        pid = str(row.get("pid", ""))
        return pid.startswith("mt-") and is_dev(pid, MATH_DEV_PER_MILLE)
    if task == "code":
        return is_code_dev(row["text"])
    if task == "ifeval":
        base_id = row["id"][len("if-"):] if row["id"].startswith("if-") else row["id"]
        return is_dev(base_id, IF_DEV_PER_MILLE)
    return False


DEV_FILES = ("math_dev.jsonl", "code_dev.jsonl", "if_dev.jsonl")


def load_dev_filter(missing: list[str] | None = None) -> leakage.OverlapFilter | None:
    """Overlap filter over the private dev texts on disk (math problems, code texts, IF base tasks)."""
    texts: list[str] = []
    for name, key in (("math_dev.jsonl", "problem"), ("code_dev.jsonl", "text"), ("if_dev.jsonl", None)):
        if not (DATA / name).exists():
            if missing is not None:
                missing.append(name)
            continue
        for row in jsonl.read(DATA / name):
            text = row.get(key) if key else (row.get("task") or row.get("prompt"))
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return leakage.OverlapFilter(texts) if texts else None


class OverlapCheckNotDecided(RuntimeError):
    pass


OVERLAP_DECISION_MESSAGE = (
    "The selection includes teacher-written math (math_new) and/or code, but the test-set overlap "
    "filter is off. Choose one:\n"
    "  1. If staff have allowed the overlap check, set ALLOW_TEST_OVERLAP_CHECK=1 and rerun, so these "
    "examples are filtered against the evaluation sets.\n"
    "  2. If a human has decided to train on them without that filter, rerun with "
    "--no-overlap-check-ok (recorded in the manifest).")


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


MATH_SOURCES = ("math_train", "math_new")
TASKS = ("math", "code", "ifeval", "general")
# Per-example task label written to train.jsonl and dropped.jsonl, by TASKS key.
TASK_LABEL = {"math": "math", "code": "code", "ifeval": "instruction_following", "general": "general"}
DROP_REASONS = ("invalid", "too_long", "overlap", "dev_overlap", "dev_member", "duplicate")


def _valid(example: dict) -> bool:
    msgs = example.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False
    for msg, role in zip(msgs, ("user", "assistant")):
        if not isinstance(msg, dict) or msg.get("role") != role:
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip() or has_control_chars(content, allow_tab=True):
            return False
    return True


def _check_counts(sources: list[dict], kept: list[dict]) -> None:
    total = sum(src["count"] for src in sources)
    if total != len(kept):
        raise ValueError(f"manifest source counts ({total}) do not match kept examples ({len(kept)})")


def build(name: str, sizes: dict, no_overlap_check_ok: bool = False) -> dict:
    rng = random.Random(SEED)
    dupes = {t: [] for t in TASKS}
    missing_pools: list[str] = []
    math_train = _read_pool("math_train_solutions.jsonl", dupes["math"], missing_pools)
    math_new = _read_pool("math_new_solutions.jsonl", dupes["math"], missing_pools)
    # Code: one task per normalised description (first kept), before shuffling.
    code_pool, code_text_dupes = dedupe_code_tasks(_read_pool("code_tasks.jsonl", dupes["code"], missing_pools))
    dupes["code"] += code_text_dupes
    if_pool = _read_pool("if_responses.jsonl", dupes["ifeval"], missing_pools)
    general_pool = _read_pool("general.jsonl", dupes["general"], missing_pools)
    for pool in (code_pool, if_pool, general_pool):
        rng.shuffle(pool)

    math_pool = math_candidates(math_train, math_new, rng)
    for row in math_pool:
        if row.get("source") not in MATH_SOURCES:
            raise ValueError(f"math row {row['id']!r} has unexpected source {row.get('source')!r}")

    # Per task: (pool in selection order, builder -> example, raw text for the staff-allowed test-set
    # overlap filter, raw text for the private-dev overlap filter, manifest source).
    # Raw texts are never written to train.jsonl.
    plans = {
        "math": (math_pool, lambda r: math_example(r, rng), lambda r: r["problem"], lambda r: r["problem"],
                 lambda r: r["source"]),
        "code": (code_pool, lambda r: code_example(r, rng), lambda r: r["text"], lambda r: r["text"],
                 lambda r: "code"),
        # Dev overlap uses the base task (no constraint boilerplate); rows without one skip that check.
        "ifeval": (if_pool, if_example, lambda r: r["prompt"], lambda r: r.get("task"),
                   lambda r: "instruction_following"),
        "general": (general_pool, general_example, _user_text, lambda r: None, lambda r: "general"),
    }

    overlap = leakage.load_overlap_filter()
    missing_dev: list[str] = []
    dev_filter = load_dev_filter(missing_dev)
    count = _token_counter()
    kept, per_source, teachers = [], {}, set()
    kept_per_task = dict.fromkeys(TASKS, 0)
    dropped_rows: list[dict] = []
    dropped_by_task = {t: dict.fromkeys(DROP_REASONS, 0) for t in TASKS}
    tokens = 0

    def drop(task: str, row: dict, reason: str) -> None:
        dropped_by_task[task][reason] += 1
        dropped_rows.append({"id": row.get("id"), "task": TASK_LABEL[task], "reason": reason})

    for task in TASKS:
        for row in dupes[task]:
            drop(task, row, "duplicate")
    for task in TASKS:
        pool, make, raw_of, dev_raw_of, source_of = plans[task]
        for row in pool:
            if kept_per_task[task] >= sizes[task]:
                break
            if is_dev_member(task, row):
                drop(task, row, "dev_member")
                continue
            ex = make(row)
            if not _valid(ex):
                drop(task, row, "invalid")
                continue
            raw = raw_of(row)
            if overlap is not None and raw is not None and overlap.is_leak(raw):
                drop(task, row, "overlap")
                continue
            dev_raw = dev_raw_of(row)
            if dev_filter is not None and dev_raw and dev_filter.is_leak(dev_raw):
                drop(task, row, "dev_overlap")
                continue
            n = count(ex["messages"])
            if n > MAX_TRAIN_TOKENS:
                drop(task, row, "too_long")
                continue
            tokens += n
            kept.append(ex)
            kept_per_task[task] += 1
            source = source_of(row)
            per_source[source] = per_source.get(source, 0) + 1
            if row.get("teacher"):
                teachers.add(row["teacher"])
    rng.shuffle(kept)

    if overlap is None and not no_overlap_check_ok and (per_source.get("math_new", 0) or per_source.get("code", 0)):
        raise OverlapCheckNotDecided(OVERLAP_DECISION_MESSAGE)

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
    _check_counts(sources, kept)

    def total(reason: str) -> int:
        return sum(d[reason] for d in dropped_by_task.values())

    shortfall = {t: max(0, sizes[t] - kept_per_task[t]) for t in TASKS}
    manifest = {
        "name": name, "seed": SEED, "sources": sources,
        "requested": {t: sizes[t] for t in TASKS},
        "shortfall": shortfall,
        "total_examples": len(kept), "total_tokens": tokens,
        "estimated_cost_usd": round(tokens / 1e6 * TRAIN_PRICE_PER_M, 2),
        "dropped_too_long": total("too_long"), "dropped_overlap": total("overlap"),
        "dropped_invalid": total("invalid"), "dropped_dev_overlap": total("dev_overlap"),
        "dropped_dev_member": total("dev_member"), "dropped_duplicate": total("duplicate"),
        "dropped_by_task": dropped_by_task,
        "overlap_filter_enabled": overlap is not None,
        "no_overlap_check_acknowledged": bool(no_overlap_check_ok),
        "dev_overlap_filter_enabled": dev_filter is not None,
        "missing_pools": missing_pools, "missing_dev_files": missing_dev,
        "teachers": sorted(teachers),
    }
    for task, short in shortfall.items():
        if short:
            print(f"WARNING: {task}: shortfall of {short} (requested {sizes[task]}, kept {kept_per_task[task]})")
    for pool_name in missing_pools:
        print(f"WARNING: pool file missing: {DATA / pool_name}")
    for dev_name in missing_dev:
        print(f"WARNING: dev file missing, not used by the dev-overlap filter: {DATA / dev_name}")
    out = RUNS / name
    jsonl.write(out / "train.jsonl", kept)
    jsonl.write(out / "dropped.jsonl", dropped_rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--math", type=int, default=30000)
    p.add_argument("--ifeval", type=int, default=6000)
    p.add_argument("--code", type=int, default=6000)
    p.add_argument("--general", type=int, default=2000)
    p.add_argument("--no-overlap-check-ok", action="store_true",
                   help="a human has decided to use teacher-written math/code without the test-set overlap filter")
    args = p.parse_args()
    try:
        manifest = build(args.name, {"math": args.math, "ifeval": args.ifeval,
                                     "code": args.code, "general": args.general},
                         no_overlap_check_ok=args.no_overlap_check_ok)
    except OverlapCheckNotDecided as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

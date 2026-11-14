"""Verified self-distillation: sample a Tinker model on training-side prompts, keep only verified outputs.

math: base model, hard problems with trusted answers; keep finished, correct, short-enough, loop-free traces.
code: base model, our verified code tasks; keep outputs that pass given + hidden tests with exactly one
      ```python block (the MBPP scorer grades the first one).
if:   a checkpoint (v1), fresh base tasks with sampled train-mode constraints; keep strict passes.

Usage:
    python -m pipeline.self_distill math --limit 200 --n 4 --out data/sd_math_pilot.jsonl
    python -m pipeline.self_distill code --limit 200 --n 4 --out data/sd_code_pilot.jsonl
    python -m pipeline.self_distill if --checkpoint_path tinker://... --limit 200 --n 8 --out data/sd_if_pilot.jsonl
"""

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

from pipeline import formats, jsonl
from pipeline.config import DATA, SEED, STUDENT, is_dev
from pipeline.constraints.sample import render_prompt, sample_constraints
from pipeline.dev import IF_DEV_PER_MILLE, is_code_dev
from pipeline.generate.math import DEV_PER_MILLE as MATH_DEV_PER_MILLE
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

PRICE_SAMPLE, PRICE_PREFILL = 0.807, 0.30  # $/M tokens, Qwen3.5-9B on Tinker
MAX_TOKENS = {"math": 4096, "code": 4096, "if": 1024}
KEEP_TOKENS = {"math": 3000, "code": 1500, "if": 900}
MAX_IF_WORDS = 600
CONCURRENCY = 32
_FENCE = re.compile(r"```python\n")


def has_loop(text: str, tail: int = 200) -> bool:
    return len(text) > 2 * tail and text[-tail:] in text[:-tail]


# ---------------------------------------------------------------- item sources

def math_items_v2(min_level: int, max_new: int) -> list[dict]:
    """Every hard MATH-train problem first, then up to max_new hard teacher-new problems."""
    items = math_items(min_level)
    train = [i for i in items if i["source"] == "math_train"]
    new = [i for i in items if i["source"] == "math_new"]
    return train + new[:max_new]


def math_items(min_level: int) -> list[dict]:
    rows = {}
    for r in jsonl.read(DATA / "math_train_solutions.jsonl"):
        pid = r.get("pid", r["id"])
        if pid.startswith("mt-") and is_dev(pid, MATH_DEV_PER_MILLE):
            continue
        rows.setdefault(pid, {"id": pid, "problem": r["problem"], "target": r.get("answer") or r["target"],
                              "level": r.get("level"), "source": "math_train"})
    for r in jsonl.read(DATA / "math_new_solutions.jsonl"):
        if r.get("verified"):
            pid = r.get("pid", r["id"])
            rows.setdefault(pid, {"id": pid, "problem": r["problem"], "target": r.get("answer") or r["target"],
                                  "level": r.get("level"), "source": "math_new"})
    items = [r for r in rows.values() if (r["level"] or 0) >= min_level
             and not re.search(r"(?i)no such|does not exist", str(r["target"]))]
    random.Random(SEED).shuffle(items)
    return items


def code_items() -> list[dict]:
    items = [r for r in jsonl.read(DATA / "code_tasks.jsonl") if not is_code_dev(r["text"])]
    random.Random(SEED).shuffle(items)
    return items


def if_items() -> list[dict]:
    used = {r["id"][len("if-"):] for r in jsonl.read(DATA / "if_responses.jsonl")}
    used |= {i[len("if-"):] if i.startswith("if-") else i for i in jsonl.done_ids(DATA / "if_responses.jsonl")}
    tasks = [t for t in jsonl.read(DATA / "if_base_tasks.jsonl")
             if t["id"] not in used and not is_dev(t["id"], IF_DEV_PER_MILLE)]
    rng = random.Random(SEED + 1)
    rng.shuffle(tasks)
    items = []
    for t in tasks:
        cons = sample_constraints(random.Random(f"sd-{t['id']}"), t["task"], mode="train")
        items.append({"id": f"sd-{t['id']}", "task": t["task"], "prompt": render_prompt(t["task"], cons),
                      "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in cons]})
    return items


# ---------------------------------------------------------------- per-sample checks

async def check_math(item: dict, text: str, ntok: int) -> tuple[bool, str]:
    if ntok >= MAX_TOKENS["math"] - 1:
        return False, "truncated"
    if has_loop(text):
        return False, "loop"
    if not await vm.equivalent(vm.answer_of(text), item["target"]):
        return False, "wrong"
    if ntok > KEEP_TOKENS["math"]:
        return False, "too_long"
    return True, "kept"


async def check_code(item: dict, text: str, ntok: int) -> tuple[bool, str]:
    if ntok >= MAX_TOKENS["code"] - 1:
        return False, "truncated"
    if len(_FENCE.findall(text)) != 1:
        return False, "not_one_block"
    tests = item["tests"] + item.get("hidden_tests", [])
    if not await asyncio.to_thread(vc.run_tests, vc.extract_code(text), tests, timeout=10.0):
        return False, "fails_tests"
    if ntok > KEEP_TOKENS["code"]:
        return False, "too_long"
    return True, "kept"


async def check_if(item: dict, text: str, ntok: int) -> tuple[bool, str]:
    if ntok >= MAX_TOKENS["if"] - 1:
        return False, "truncated"
    if has_loop(text):
        return False, "loop"
    if not vi.passes(item["prompt"], item["constraints"], text):
        return False, "fails_constraints"
    if len(text.split()) > MAX_IF_WORDS or ntok > KEEP_TOKENS["if"]:
        return False, "too_long"
    return True, "kept"


PROMPT = {"math": lambda it: formats.math_prompt(it["problem"]),
          "code": lambda it: formats.code_prompt(it["text"], it["tests"]),
          "if": lambda it: it["prompt"]}
CHECK = {"math": check_math, "code": check_code, "if": check_if}


# ---------------------------------------------------------------- run

async def run(task: str, items: list[dict], checkpoint_path: str | None, n: int, temperature: float,
              out: Path) -> dict:
    import tinker
    from pipeline.student import Sampler, _text

    s = Sampler(STUDENT, checkpoint_path)
    stop = s.renderer.get_stop_sequences()
    sem = asyncio.Semaphore(CONCURRENCY)
    done = jsonl.done_ids(out)
    stats = Counter()
    lens_kept: list[int] = []

    async def one(item):
        if item["id"] in done:
            return
        text_prompt = PROMPT[task](item)
        prompt = s.renderer.build_generation_prompt([{"role": "user", "content": text_prompt}])
        params = tinker.SamplingParams(temperature=temperature, max_tokens=MAX_TOKENS[task], stop=stop, top_p=1.0)
        async with sem:
            res = await s.client.sample_async(prompt=prompt, sampling_params=params, num_samples=n)
        stats["prompt_tokens"] += prompt.length * n
        kept = []
        for seq in res.sequences:
            text = _text(s.renderer.parse_response(seq.tokens)[0]["content"])
            stats["sampled_tokens"] += len(seq.tokens)
            ok, why = await CHECK[task](item, text, len(seq.tokens))
            stats[why] += 1
            if ok:
                kept.append({"text": text, "tokens": len(seq.tokens)})
        stats["items"] += 1
        stats["items_with_keep"] += bool(kept)
        kept.sort(key=lambda k: k["tokens"])
        rows = [{**{k: v for k, v in item.items()}, "id": f"{item['id']}-sd{j}", "item_id": item["id"],
                 "response": k["text"], "tokens": k["tokens"], "sampler": checkpoint_path or STUDENT}
                for j, k in enumerate(kept[:2])]
        lens_kept.extend(k["tokens"] for k in kept[:2])
        jsonl.append_many(out, rows)
        jsonl.mark_done(out, item["id"])

    await asyncio.gather(*(one(i) for i in items))
    stats = dict(stats)
    stats["cost_usd"] = round(stats.get("sampled_tokens", 0) * PRICE_SAMPLE / 1e6
                              + stats.get("prompt_tokens", 0) * PRICE_PREFILL / 1e6, 2)
    if lens_kept:
        lens_kept.sort()
        stats["kept_tokens_median"] = lens_kept[len(lens_kept) // 2]
        stats["kept_tokens_p90"] = lens_kept[int(0.9 * len(lens_kept))]
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["math", "code", "if"])
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--min_level", type=int, default=4)
    p.add_argument("--max_new", type=int, default=None,
                   help="math: all hard MATH-train problems plus this many hard teacher-new problems")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.task == "if" and not args.checkpoint_path:
        p.error("if sampling uses a trained checkpoint (--checkpoint_path)")
    math_src = (lambda: math_items_v2(args.min_level, args.max_new)) if args.max_new is not None \
        else (lambda: math_items(args.min_level))
    items = {"math": math_src, "code": code_items, "if": if_items}[args.task]()
    items = items[: args.limit]
    out = Path(args.out)
    stats = asyncio.run(run(args.task, items, args.checkpoint_path, args.n, args.temperature, out))
    stats.update(task=args.task, n=args.n, temperature=args.temperature, limit=args.limit)
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

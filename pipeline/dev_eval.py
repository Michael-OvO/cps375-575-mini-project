"""Score a checkpoint on the private dev sets with the same prompts and scorers as the public eval.

Usage:
    python -m pipeline.dev_eval --out runs/base/dev.json
    python -m pipeline.dev_eval --checkpoint_path tinker://... --out runs/v1/dev.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from pipeline import formats, jsonl
from pipeline.config import DATA, STUDENT
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

CONCURRENCY = 64
CODE_TIMEOUT = 30  # the public MBPP scorer's timeout, in seconds


def if_metric(scores: list[dict]) -> float:
    if not scores:
        return 0.0
    n_inst = sum(s["n"] for s in scores)
    parts = [
        sum(s["prompt_strict"] for s in scores) / len(scores),
        sum(s["prompt_loose"] for s in scores) / len(scores),
        sum(s["inst_strict"] for s in scores) / n_inst,
        sum(s["inst_loose"] for s in scores) / n_inst,
    ]
    return sum(parts) / 4


async def score_items(sampler, math: list[dict], code: list[dict], ifd: list[dict]) -> dict:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def ask(prompt: str, max_tokens: int) -> str:
        async with sem:
            return (await sampler.answer(prompt, max_tokens=max_tokens))[0]

    async def one_math(item):
        out = await ask(formats.math_prompt(item["problem"]), 4096)
        return await vm.equivalent(vm.answer_of(out), item["target"])

    async def one_code(item):
        out = await ask(formats.code_prompt(item["text"], item["tests"]), 4096)
        tests = item["tests"] + item.get("hidden_tests", [])
        return await asyncio.to_thread(vc.run_tests, vc.extract_code(out), tests, timeout=CODE_TIMEOUT)

    async def one_if(item):
        out = await ask(item["prompt"], 1024)
        return item["kind"], vi.score(item["prompt"], item["constraints"], out)

    math_ok = await asyncio.gather(*(one_math(i) for i in math))
    code_ok = await asyncio.gather(*(one_code(i) for i in code))
    if_scores = await asyncio.gather(*(one_if(i) for i in ifd))
    seen = [s for kind, s in if_scores if kind == "seen"]
    unseen = [s for kind, s in if_scores if kind == "unseen"]
    return {
        "math": sum(math_ok) / max(len(math_ok), 1),
        "code": sum(code_ok) / max(len(code_ok), 1),
        "ifeval_seen": if_metric(seen),
        "ifeval_unseen": if_metric(unseen),
        "counts": {"math": len(math), "code": len(code),
                   "ifeval_seen": len(seen), "ifeval_unseen": len(unseen)},
    }


def _load(name: str, split: str, limit: int | None) -> list[dict]:
    """Items of one dev file for the split. A missing file or zero items is an error, never a 0.0 score."""
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"dev file missing: {path}. Build the dev sets first.")
    rows = [r for r in jsonl.read(path) if r["vault"] == (split == "vault")][:limit]
    if not rows:
        raise ValueError(f"{path} has no items for split {split!r} (limit={limit}).")
    return rows


async def evaluate(sampler, split: str = "dev", limit: int | None = None) -> dict:
    return await score_items(
        sampler,
        _load("math_dev.jsonl", split, limit),
        _load("code_dev.jsonl", split, limit),
        _load("if_dev.jsonl", split, limit),
    )


def main() -> None:
    from pipeline.student import Sampler

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--split", choices=["dev", "vault"], default="dev")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = asyncio.run(evaluate(Sampler(STUDENT, args.checkpoint_path), args.split, args.limit))
    result["checkpoint_path"] = args.checkpoint_path
    result["split"] = args.split
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

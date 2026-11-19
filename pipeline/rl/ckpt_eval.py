"""Checkpoint eval for rl1: greedy dev (split 'dev' only), pass@1/pass@8 at T=1 on 100 seeded dev items per task,
and the 50 trap prompts. Every sampled token is recorded in the spend ledger (phase 'eval')."""

import argparse
import asyncio
import json
import random
from pathlib import Path

from pipeline import dev_eval, formats, jsonl
from pipeline.config import DATA, RUNS, SEED, STUDENT
from pipeline.rl import traps
from pipeline.rl.ledger import Ledger
from pipeline.student import Sampler
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

PASSK_N, K = 100, 8
EST_USD = 1.6  # reserve per checkpoint eval; the actual cost is recorded after


def _tok(sampler: Sampler, texts: list[str]) -> int:
    return sum(len(sampler.renderer.tokenizer.encode(t)) for t in texts)


async def passk(sampler: Sampler, math, code, ifd) -> tuple[dict, list[str], list[str]]:
    rng = random.Random(SEED)
    subsets = {"math": rng.sample(math, min(PASSK_N, len(math))), "code": rng.sample(code, min(PASSK_N, len(code))),
               "if": rng.sample(ifd, min(PASSK_N, len(ifd)))}
    prompts, outs = [], []
    sem = asyncio.Semaphore(16)

    async def one(task, item):
        if task == "math":
            p, cap = formats.math_prompt(item["problem"]), 4096
        elif task == "code":
            p, cap = formats.code_prompt(item["text"], item["tests"]), 4096
        else:
            p, cap = item["prompt"], 1024
        async with sem:
            texts = await sampler.answer(p, max_tokens=cap, temperature=1.0, n=K)
        prompts.append(p)
        outs.extend(texts)
        if task == "math":
            oks = [await vm.equivalent(vm.answer_of(t), item["target"]) for t in texts]
        elif task == "code":
            tests = item["tests"] + item.get("hidden_tests", [])
            oks = [await asyncio.to_thread(vc.run_tests, vc.extract_code(t), tests, 30) for t in texts]
        else:
            oks = [vi.score(item["prompt"], item["constraints"], t)["prompt_strict"] for t in texts]
        return task, oks

    res = await asyncio.gather(*(one(t, i) for t, items in subsets.items() for i in items))
    out = {}
    for task in subsets:
        rows = [oks for t, oks in res if t == task]
        out[task] = {"pass1": sum(sum(o) / K for o in rows) / len(rows),
                     "pass8": sum(any(o) for o in rows) / len(rows)}
    return out, prompts, outs


async def run(path: str, step: int) -> dict:
    led = Ledger()
    led.reserve(EST_USD, f"ckpt_eval step {step}")
    sampler = Sampler(STUDENT, path)
    math, code, ifd = dev_eval.load_split("dev")
    items: list = []
    greedy = await dev_eval.score_items(sampler, math, code, ifd, items=items)
    pk, pk_prompts, pk_outs = await passk(sampler, math, code, ifd)
    trap_rows = jsonl.read(DATA / "rl_traps.jsonl")
    trap_outs = await asyncio.gather(*(sampler.answer(r["prompt"], max_tokens=1024) for r in trap_rows))
    pairs = [(r, o[0]) for r, o in zip(trap_rows, trap_outs)]
    trap_if = sum(vi.score(r["prompt"], r["constraints"], t)["prompt_strict"] for r, t in pairs) / len(pairs)
    gen = [r["output"] for r in items] + pk_outs + [t for _, t in pairs]
    led.record("eval", "sample", _tok(sampler, gen), note=f"ckpt_eval step {step}")
    led.record("eval", "prefill", _tok(sampler, pk_prompts) + 2000 * len(items) // 4, note=f"ckpt_eval step {step}")
    return {"step": step, "checkpoint_path": path, "greedy": greedy, "passk": pk,
            "trap_hack_rate": traps.hack_rate(pairs), "trap_if_score": trap_if, "items": items}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = asyncio.run(run(a.checkpoint_path, a.step))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))
    if a.step > 0:  # spec §8 eval rules; train_rl checks PAUSED before every step
        from pipeline.rl import monitor

        base = json.loads((RUNS / "rl1" / "evals" / "step0.json").read_text())["trap_hack_rate"]
        pauses, warns = monitor.eval_checks(res, base, a.step)
        monitor.warn(warns, a.step)
        if pauses:
            (RUNS / "rl1" / "PAUSED").write_text("\n".join(pauses) + "\n")
            print("PAUSED:", pauses)


if __name__ == "__main__":
    main()

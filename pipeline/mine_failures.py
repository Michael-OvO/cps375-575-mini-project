"""Failure mining: run a checkpoint once (greedy, evaluation prompts and limits) on training-side items
and record pass/fail, so teacher generation can target what the student currently gets wrong.

Never touches dev or test items: math uses hard training problems (MATH train and verified teacher-new),
code uses training-side code tasks (dev members excluded).

Usage:
    python -m pipeline.mine_failures math --checkpoint_path tinker://... --max_new 6000
    python -m pipeline.mine_failures code --checkpoint_path tinker://...
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from pipeline import formats, jsonl
from pipeline.config import DATA, STUDENT
from pipeline.self_distill import PRICE_PREFILL, PRICE_SAMPLE, code_items, math_items_v2
from pipeline.verify import code as vc
from pipeline.verify import math as vm

OUT = {"math": DATA / "v1_mined_math.jsonl", "code": DATA / "v1_mined_code.jsonl"}
CONCURRENCY = 64


async def run(task: str, items: list[dict], checkpoint_path: str) -> dict:
    import tinker
    from pipeline.student import Sampler, _text

    s = Sampler(STUDENT, checkpoint_path)
    params = tinker.SamplingParams(temperature=0.0, max_tokens=4096, stop=s.renderer.get_stop_sequences(), top_p=1.0)
    sem = asyncio.Semaphore(CONCURRENCY)
    out = OUT[task]
    done = jsonl.done_ids(out)
    stats = Counter()

    async def one(item):
        if item["id"] in done:
            return
        text = formats.math_prompt(item["problem"]) if task == "math" else formats.code_prompt(item["text"], item["tests"])
        prompt = s.renderer.build_generation_prompt([{"role": "user", "content": text}])
        async with sem:
            res = await s.client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        seq = res.sequences[0]
        reply = _text(s.renderer.parse_response(seq.tokens)[0]["content"])
        stats["prompt_tokens"] += prompt.length
        stats["sampled_tokens"] += len(seq.tokens)
        if task == "math":
            ok = await vm.equivalent(vm.answer_of(reply), item["target"])
            row = {"id": item["id"], "correct": ok, "tokens": len(seq.tokens), "level": item["level"],
                   "source": item["source"], "answer": vm.answer_of(reply)}
        else:
            tests = item["tests"] + item.get("hidden_tests", [])
            ok = await asyncio.to_thread(vc.run_tests, vc.extract_code(reply), tests, timeout=10.0)
            row = {"id": item["id"], "correct": ok, "tokens": len(seq.tokens), "category": item.get("category")}
        stats["correct"] += ok
        stats["items"] += 1
        jsonl.append(out, row)
        jsonl.mark_done(out, item["id"])

    await asyncio.gather(*(one(i) for i in items))
    stats = dict(stats)
    stats["cost_usd"] = round(stats.get("sampled_tokens", 0) * PRICE_SAMPLE / 1e6
                              + stats.get("prompt_tokens", 0) * PRICE_PREFILL / 1e6, 2)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["math", "code"])
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--max_new", type=int, default=6000)
    args = p.parse_args()
    items = math_items_v2(4, args.max_new) if args.task == "math" else code_items()
    stats = asyncio.run(run(args.task, items, args.checkpoint_path))
    Path(str(OUT[args.task]) + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

"""Proxy benchmarks, never trained on: MATH test outside MATH-500, and HumanEval.

This is the only pipeline module that loads evaluation datasets, and only to score.

Usage: python -m pipeline.proxy_eval --checkpoint_path tinker://... --out runs/v1/proxy.json
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from inspect_ai import Epochs, eval_async, task_with
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import GenerateConfig, Model
from inspect_evals.humaneval import humaneval
from inspect_evals.math.math import math

from evaluation.math_scorer import math_answer_match
from pipeline.config import ROOT, SEED, STUDENT


def math_rest_samples(limit: int) -> list:
    """MATH test samples that are not in MATH-500, shuffled with the fixed seed."""
    in_500 = set(load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"])
    task = math(shuffle=False)
    rest = [s for s in task.dataset if s.input not in in_500]
    random.Random(SEED).shuffle(rest)
    return rest[:limit]


def math_rest_task(config: GenerateConfig, limit: int):
    task = math(shuffle=False)
    dataset = MemoryDataset(math_rest_samples(limit), name="math_rest")
    return task_with(task, dataset=dataset, scorer=[math_answer_match()], config=config)


async def run(args) -> dict:
    import tinker
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
    from tinker_cookbook.model_info import get_recommended_renderer_name

    service = tinker.ServiceClient()
    if args.checkpoint_path:
        sampling = service.create_sampling_client(model_path=args.checkpoint_path).result()
    else:
        sampling = service.create_sampling_client(base_model=STUDENT).result()
    api = InspectAPIFromTinkerSampling(
        renderer_name=get_recommended_renderer_name(STUDENT), model_name=STUDENT, sampling_client=sampling
    )
    config = GenerateConfig(temperature=0.0, top_p=1.0, max_tokens=4096)
    model = Model(api=api, config=config)
    tasks = [
        math_rest_task(config, args.limit),
        task_with(humaneval(sandbox="local"), config=config, epochs=Epochs(1, "mean")),
    ]
    results = await eval_async(
        tasks=tasks, model=[model], sandbox="local", debug_errors=True, retry_on_error=5,
        fail_on_error=False, log_dir=str(ROOT / "runs" / "inspect-logs"), max_connections=256,
    )
    out = {"checkpoint_path": args.checkpoint_path}
    for name, result in zip(["math_rest", "humaneval"], results):
        metrics = result.results.scores[0].metrics if result.results and result.results.scores else {}
        out[name] = metrics["accuracy"].value if "accuracy" in metrics else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.environ["PATH"] = f"{ROOT / '.venv' / 'bin'}:{os.environ['PATH']}"
    result = asyncio.run(run(args))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

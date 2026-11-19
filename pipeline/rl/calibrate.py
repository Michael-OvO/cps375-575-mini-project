"""Phase 2 calibration on v4r: 300 prompts per task, G=8 at T=1 plus one greedy decode, scored with the RL rewards.
Produces per-stratum statistics for selection, per-domain baselines for pause rules and EMA init, and review files."""

import asyncio
import json
import math
import random
from collections import defaultdict

import tinker

from pipeline import jsonl
from pipeline.config import RUNS, SEED, STUDENT
from pipeline.rl import data_gates, rewards
from pipeline.rl.ledger import Ledger
from pipeline.student import Sampler, _text

OUT = RUNS / "rl1" / "calibration"
PER_TASK, G = 300, 8
CAPS = {"math": 4096, "code": 4096, "if": 1024}
EST_USD = 4.5


def _norm(task: str, s: rewards.Scored) -> float:
    return s.info.get("norm", s.reward) if task == "if" else s.reward


def stratum_stats(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r)
    out = {}
    for s, rs in by.items():
        tot = sum(sum(r["norms"]) for r in rs)
        cnt = sum(len(r["norms"]) for r in rs)
        pq = [(sum(r["norms"]) / len(r["norms"])) * (1 - sum(r["norms"]) / len(r["norms"])) for r in rs]
        out[s] = {"n": len(rs), "p_hat": (tot + 1) / (cnt + 2),
                  "e_pq": (sum(pq) + 0.25) / (len(rs) + 1),  # shrunk toward 0.25 with one pseudo-prompt
                  "informative": sum(max(r["norms"]) - min(r["norms"]) > 1e-9 for r in rs) / len(rs),
                  "greedy_fail": sum(r["greedy"] < 1.0 - 1e-9 for r in rs) / len(rs),
                  "mean_len": sum(sum(r["lens"]) for r in rs) / cnt,
                  "trunc": sum(sum(r["trunc"]) for r in rs) / cnt}
    return out


def domain_baseline(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    out = {}
    for task, rs in by.items():
        adv = []
        for r in rs:
            if max(r["rewards"]) - min(r["rewards"]) > 1e-9:
                m = sum(r["rewards"]) / len(r["rewards"])
                adv += [x - m for x in r["rewards"]]
        n = sum(len(r["lens"]) for r in rs)
        vphf = sum(v is True for r in rs for v in r["vis"])  # visible tests passed, hidden failed
        out[task] = {"S": math.sqrt(sum(a * a for a in adv) / len(adv)) if adv else 0.3,
                     "L": sum(sum(r["lens"]) for r in rs) / n,
                     "mean_len": sum(sum(r["lens"]) for r in rs) / n,
                     "trunc": sum(sum(r["trunc"]) for r in rs) / n,
                     "gate": sum(x in ("gate", "guard") for r in rs for x in r["reasons"]) / n,
                     "vphf": vphf / n if task == "code" else 0.0}
    return out


def pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin over strata, random within each, so every stratum is represented."""
    by = defaultdict(list)
    for r in pool:
        by[r["stratum"]].append(r)
    strata = sorted(by)
    out = []
    while len(out) < n and any(by.values()):
        for s in strata:
            if by[s] and len(out) < n:
                out.append(by[s].pop(rng.randrange(len(by[s]))))
    return out


async def calibrate(sampler: Sampler, items: list[dict], led: Ledger) -> list[dict]:
    sem = asyncio.Semaphore(32)
    stop = sampler.renderer.get_stop_sequences()
    rows = []

    async def one(item):
        # Same prompt stripping and completion text as student.Sampler.answer, which the official eval mirrors.
        prompt = sampler.renderer.build_generation_prompt([{"role": "user", "content": item["prompt"].strip()}])
        cap = CAPS[item["task"]]
        async with sem:
            t1 = await sampler.client.sample_async(prompt=prompt, num_samples=G, sampling_params=tinker.SamplingParams(
                temperature=1.0, top_p=1.0, max_tokens=cap, stop=stop))
            t0 = await sampler.client.sample_async(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=cap, stop=stop))
        seqs = list(t1.sequences) + list(t0.sequences)
        texts = [_text(sampler.renderer.parse_response(s.tokens)[0]["content"]) for s in seqs]
        truncs = [s.stop_reason == "length" for s in seqs]
        scored = [await rewards.score(item, t, tr, check_visible=True) for t, tr in zip(texts, truncs)]
        led.record("2", "sample", sum(len(s.tokens) for s in seqs), note=item["id"])
        led.record("2", "prefill", 2 * prompt.length, note=item["id"])
        rows.append({"id": item["id"], "task": item["task"], "stratum": item["stratum"],
                     "rewards": [s.reward for s in scored[:G]], "norms": [_norm(item["task"], s) for s in scored[:G]],
                     "greedy": _norm(item["task"], scored[G]), "lens": [len(s.tokens) for s in seqs[:G]],
                     "trunc": truncs[:G], "reasons": [s.info.get("reason") for s in scored[:G]],
                     "vis": [s.info.get("visible_pass") for s in scored[:G]], "samples": texts[:2]})

    await asyncio.gather(*(one(i) for i in items))
    return rows


def write_review(rows: list[dict]) -> None:
    rng = random.Random(SEED)
    for task in ("math", "if", "code"):
        rs = [r for r in rows if r["task"] == task]
        lines = [f"# Calibration review: {task}\n"]
        for r in rng.sample(rs, min(30, len(rs))):
            lines.append(f"## {r['id']} ({r['stratum']}) rewards={r['rewards']} reasons={r['reasons']}\n")
            lines.append("```\n" + r["samples"][0][:3000] + "\n```\n")
        (OUT / f"review_{task}.md").write_text("\n".join(lines))


def main() -> None:
    data_gates.require_passed()
    v4r = json.loads((RUNS / "v4r" / "checkpoint.json").read_text())["checkpoint_path"]
    led = Ledger()
    led.reserve(EST_USD, "calibration")
    rng = random.Random(SEED)
    items = []
    for task in ("math", "if", "code"):
        items += pick(jsonl.read(RUNS / "rl1" / "pools" / f"{task}.jsonl"), PER_TASK, rng)
    rows = asyncio.run(calibrate(Sampler(STUDENT, v4r), items, led))
    OUT.mkdir(parents=True, exist_ok=True)
    jsonl.write(OUT / "items.jsonl", rows)
    summary = {"checkpoint": v4r, "strata": stratum_stats(rows), "baseline": domain_baseline(rows),
               "spent_total": led.total()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    write_review(rows)
    print(json.dumps(summary["baseline"], indent=1), "\nledger total", round(led.total(), 2))


if __name__ == "__main__":
    main()

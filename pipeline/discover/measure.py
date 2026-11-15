"""Score a checkpoint on the probes (greedy, evaluation prompts and limits), then rank cells.

Usage:
    python -m pipeline.discover.measure score --name v1 --checkpoint_path tinker://...
    python -m pipeline.discover.measure rank --name v1         # writes reports/weakness_registry.md

priority = Wilson lower bound of the failure rate x relevance x learnability x coverage gap
- learnability: teacher pass rate on the cell's probes;
- coverage gap: 1 / (1 + training examples sharing the cell's rarest facet value / 200);
- relevance: 1, or 0.6 when the cell's key facet value is one the staff description does not suggest.
"""

import argparse
import asyncio
import json
import math as _m
from collections import defaultdict

from pipeline import formats, jsonl
from pipeline.config import ROOT, STUDENT
from pipeline.discover import coverage
from pipeline.discover.probes import probes_path
from pipeline.discover.taxonomy import DISCOVER, FACETS, describe
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

LIMITS = {"math": 4096, "code": 4096, "if": 1024}
CONCURRENCY = 64
REPORT = ROOT / "reports" / "weakness_registry.md"
# facet values outside what the staff described (they may still appear, so they are discounted, not dropped)
SPECULATIVE = {("if", "genre"): {"a riddle or joke"}}


def scores_path(name: str):
    return DISCOVER / f"scores_{name}.jsonl"


def load_probes(task: str) -> list[dict]:
    p = probes_path(task)
    return [r for r in jsonl.read(p) if not r.get("empty")] if p.exists() else []


async def score(name: str, checkpoint_path: str | None) -> None:
    import tinker
    from pipeline.student import Sampler, _text

    s = Sampler(STUDENT, checkpoint_path)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def ask(text: str, max_tokens: int) -> tuple[str, int]:
        prompt = s.renderer.build_generation_prompt([{"role": "user", "content": text}])
        params = tinker.SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                       stop=s.renderer.get_stop_sequences(), top_p=1.0)
        async with sem:
            res = await s.client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        seq = res.sequences[0]
        return _text(s.renderer.parse_response(seq.tokens)[0]["content"]), len(seq.tokens)

    async def one(p: dict) -> dict:
        t = p["task"]
        if t == "math":
            out, n = await ask(formats.math_prompt(p["problem"]), LIMITS[t])
            ok = await vm.equivalent(vm.answer_of(out), p["target"])
        elif t == "code":
            out, n = await ask(formats.code_prompt(p["text"], p["tests"]), LIMITS[t])
            ok = await asyncio.to_thread(vc.run_tests, vc.extract_code(out), p["tests"] + p["hidden_tests"])
        else:
            out, n = await ask(p["prompt"], LIMITS[t])
            per = vi.check(p["prompt"], p["constraints"], out)
            ok = all(per)
        row = {"id": p["id"], "cell_id": p["cell_id"], "task": t, "correct": bool(ok), "tokens": n,
               "truncated": n >= LIMITS[t] - 1, "teacher_pass": p["teacher_pass"], "output": out}
        if t == "if":
            row["per_constraint"] = per
        return row

    probes = [p for t in FACETS for p in load_probes(t)]
    rows = await asyncio.gather(*(one(p) for p in probes))
    jsonl.write(scores_path(name), rows)
    for t in FACETS:
        rs = [r for r in rows if r["task"] == t]
        if rs:
            print(t, f"{sum(r['correct'] for r in rs)}/{len(rs)} correct, teacher {sum(r['teacher_pass'] for r in rs)}/{len(rs)}")


# ---------------------------------------------------------------- ranking

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * _m.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell_table(name: str) -> list[dict]:
    rows = list(jsonl.read(scores_path(name)))
    probes = {p["id"]: p for t in FACETS for p in load_probes(t)}
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell_id"]].append(r)
    out = []
    for cid, rs in by_cell.items():
        cell = probes[rs[0]["id"]]["cell"]
        task = rs[0]["task"]
        n = len(rs)
        fails = sum(not r["correct"] for r in rs)
        # learnability counts only probes the teacher passes: failures on those are the fixable ones
        teach = [r for r in rs if r["teacher_pass"]]
        fixable = sum(not r["correct"] for r in teach)
        lo, hi = wilson(fixable, len(teach)) if teach else (0.0, 1.0)
        counts = {f: coverage.count(task, f, cell[f]) for f in FACETS[task] if f in cell}
        rarest = min(counts, key=counts.get) if counts else None
        gap = 1 / (1 + (counts[rarest] if rarest else 0) / 200)
        spec = any(cell.get(f) in vals for (t, f), vals in SPECULATIVE.items() if t == task)
        relevance = 0.6 if spec else 1.0
        learn = len(teach) / n
        out.append({"cell_id": cid, "task": task, "desc": describe(task, cell), "n": n, "fail": fails / n,
                    "fixable_fail": fixable / max(1, len(teach)), "fix_lo": lo, "fix_hi": hi, "teacher": learn,
                    "rarest": f"{rarest}={cell.get(rarest)} ({int(counts[rarest])})" if rarest else "",
                    "truncated": sum(r["truncated"] for r in rs) / n, "relevance": relevance,
                    "priority": lo * relevance * learn * gap})
    return sorted(out, key=lambda r: -r["priority"])


def facet_slices(name: str) -> list[dict]:
    """Failure rate per facet value across all cells: more probes per value than any single cell."""
    rows = list(jsonl.read(scores_path(name)))
    probes = {p["id"]: p for t in FACETS for p in load_probes(t)}
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        p = probes[r["id"]]
        keys = [(r["task"], f, p["cell"][f]) for f in FACETS[r["task"]] if f in p["cell"]]
        if r["task"] == "if":
            keys += [("if", "regime", g) for g in set(p.get("regimes", []))]
            keys += [("if", "type_seen", "dev-only type" if any(p.get("dev_only", [])) else "training types only")]
        for k in keys:
            a = agg[k]
            a[0] += 1
            a[1] += not r["correct"]
            a[2] += r["teacher_pass"]
    out = []
    for (t, f, v), (n, fails, teach) in agg.items():
        lo, hi = wilson(fails, n)
        out.append({"task": t, "facet": f, "value": v, "n": n, "fail": fails / n, "lo": lo, "hi": hi,
                    "teacher": teach / n})
    return sorted(out, key=lambda r: (r["task"], r["facet"], -r["fail"]))


def write_registry(name: str) -> None:
    cells = cell_table(name)
    slices = facet_slices(name)
    lines = [f"# Weakness registry ({name})", "",
             "Generated by `python -m pipeline.discover.measure rank`. Probes are fresh Opus-written items, verified, "
             "never trained on. `fixable fail` counts only probes the teacher itself passes; the interval is a 95% "
             "Wilson interval on it. Priority = interval lower bound x relevance x teacher pass rate x coverage gap. "
             "Twelve probes per cell separate a 30% cell from a 70% one, not close cells.", "",
             "## Cells by priority", "",
             "| # | cell | task | probes | v1 fail | fixable fail [95% CI] | teacher | truncated | rarest facet (train count) | priority | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cells, 1):
        lines.append(f"| {i} | {c['desc']} | {c['task']} | {c['n']} | {c['fail']:.0%} | {c['fixable_fail']:.0%} "
                     f"[{c['fix_lo']:.0%}, {c['fix_hi']:.0%}] | {c['teacher']:.0%} | {c['truncated']:.0%} | "
                     f"{c['rarest']} | {c['priority']:.3f} | open |")
    lines += ["", "## Failure rate by facet value (all cells pooled)", "",
              "| task | facet | value | probes | v1 fail [95% CI] | teacher |", "|---|---|---|---|---|---|"]
    for s in slices:
        lines.append(f"| {s['task']} | {s['facet']} | {s['value']} | {s['n']} | {s['fail']:.0%} "
                     f"[{s['lo']:.0%}, {s['hi']:.0%}] | {s['teacher']:.0%} |")
    REPORT.write_text("\n".join(lines) + "\n")
    (DISCOVER / f"ranked_{name}.json").write_text(json.dumps(cells, indent=1))
    print(f"wrote {REPORT} ({len(cells)} cells)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["score", "rank"])
    p.add_argument("--name", default="v1")
    p.add_argument("--checkpoint_path", default=None)
    a = p.parse_args()
    if a.command == "score":
        asyncio.run(score(a.name, a.checkpoint_path))
    write_registry(a.name)


if __name__ == "__main__":
    main()

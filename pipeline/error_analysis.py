"""Per-item error analysis on the private dev sets.

`collect` samples a checkpoint on every dev item (same prompts and limits as dev_eval) and writes one
row per item with the output, its token length and a failure category. `summary` compares runs.

Usage:
    python -m pipeline.error_analysis collect --name v1 --checkpoint_path tinker://...
    python -m pipeline.error_analysis collect --name base
    python -m pipeline.error_analysis summary --names base smoke v1
"""

import argparse
import asyncio
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from pipeline import dev_eval, formats, jsonl
from pipeline.config import RUNS, STUDENT
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

LIMITS = {"math": 4096, "code": 4096, "if": 1024}
TEST_TIMEOUT = 10.0


def _out_dir(name: str) -> Path:
    return RUNS / name / "errors"


# ---------------------------------------------------------------- code failure categories

def _run(program: str, timeout: float = TEST_TIMEOUT) -> tuple[str, str]:
    """('ok' | 'timeout' | 'error', last stderr line)."""
    with tempfile.TemporaryDirectory() as workdir:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True,
                               timeout=timeout, cwd=workdir)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    if r.returncode == 0:
        return "ok", ""
    lines = r.stderr.decode("utf-8", "replace").strip().splitlines()
    return "error", (lines[-1] if lines else "")


def code_category(output: str, tests: list[str], hidden: list[str]) -> dict:
    """Why a code answer fails: first matching category in a fixed order."""
    fenced = bool(re.search(r"```python\n", output))
    code = vc.extract_code(output)
    status, err = _run(code)
    if status == "timeout":
        return {"category": "timeout_on_import", "fenced": fenced, "detail": ""}
    if status == "error":
        cat = "syntax_error" if err.startswith(("SyntaxError", "IndentationError")) else "import_or_toplevel_error"
        return {"category": cat if fenced else "no_code_block", "fenced": fenced, "detail": err[:200]}
    failed_given, failed_hidden, detail = 0, 0, ""
    for kind, group in (("given", tests), ("hidden", hidden)):
        for t in group:
            status, err = _run(code + "\n" + t + "\n")
            if status == "ok":
                continue
            if kind == "given":
                failed_given += 1
            else:
                failed_hidden += 1
            if not detail:
                detail = f"{kind}: {t[:160]} -> {status} {err[:120]}"
            if status == "timeout":
                return {"category": "timeout", "fenced": fenced, "detail": detail,
                        "failed_given": failed_given, "failed_hidden": failed_hidden}
    if failed_given == failed_hidden == 0:
        cat = "pass"
    elif "NameError" in detail and failed_given == len(tests):
        cat = "wrong_function_name"
    elif failed_given:
        cat = "fails_given_tests"
    else:
        cat = "fails_hidden_only"
    return {"category": cat, "fenced": fenced, "detail": detail,
            "failed_given": failed_given, "failed_hidden": failed_hidden}


# ---------------------------------------------------------------- collection

async def collect(name: str, checkpoint_path: str | None, limit: int | None) -> None:
    from pipeline.student import Sampler
    import tinker

    math, code, ifd = dev_eval.load_split("dev", limit)
    s = Sampler(STUDENT, checkpoint_path)
    sem = asyncio.Semaphore(dev_eval.CONCURRENCY)

    async def ask(text: str, max_tokens: int) -> tuple[str, int]:
        prompt = s.renderer.build_generation_prompt([{"role": "user", "content": text}])
        params = tinker.SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                       stop=s.renderer.get_stop_sequences(), top_p=1.0)
        async with sem:
            res = await s.client.sample_async(prompt=prompt, sampling_params=params, num_samples=1)
        seq = res.sequences[0]
        from pipeline.student import _text
        return _text(s.renderer.parse_response(seq.tokens)[0]["content"]), len(seq.tokens)

    async def one_math(item):
        out, ntok = await ask(formats.math_prompt(item["problem"]), LIMITS["math"])
        ans = vm.answer_of(out)
        ok = await vm.equivalent(ans, item["target"])
        has_line = bool(re.search(r"(?im)^\s*ANSWER\s*:", out))
        cat = ("pass" if ok else "truncated" if ntok >= LIMITS["math"] - 1
               else "no_answer_line" if not has_line else "wrong_answer")
        return {"id": item["id"], "level": item["level"], "subject": item["subject"],
                "target": item["target"], "answer": ans, "correct": ok, "category": cat,
                "tokens": ntok, "output": out}

    async def one_code(item):
        out, ntok = await ask(formats.code_prompt(item["text"], item["tests"]), LIMITS["code"])
        info = await asyncio.to_thread(code_category, out, item["tests"], item.get("hidden_tests", []))
        if ntok >= LIMITS["code"] - 1 and info["category"] != "pass":
            info["category"] = "truncated"
        return {"id": item["id"], "category_topic": item.get("category"), "text": item["text"],
                "tests": item["tests"], "correct": info["category"] == "pass", "tokens": ntok,
                "output": out, **info}

    async def one_if(item):
        out, ntok = await ask(item["prompt"], LIMITS["if"])
        per = vi.check(item["prompt"], item["constraints"], out)
        return {"id": item["id"], "kind": item["kind"], "prompt": item["prompt"],
                "constraints": [c["id"] for c in item["constraints"]], "kwargs": [c.get("kwargs") for c in item["constraints"]],
                "per_constraint": per, "correct": all(per), "tokens": ntok,
                "truncated": ntok >= LIMITS["if"] - 1, "output": out}

    rows = {
        "math": await asyncio.gather(*(one_math(i) for i in math)),
        "code": await asyncio.gather(*(one_code(i) for i in code)),
        "if": await asyncio.gather(*(one_if(i) for i in ifd)),
    }
    out_dir = _out_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, rs in rows.items():
        jsonl.write(out_dir / f"{task}.jsonl", rs)
        print(task, f"{sum(r['correct'] for r in rs)}/{len(rs)}")


# ---------------------------------------------------------------- summary

def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}" if b else "-"


def summary(names: list[str]) -> str:
    data = {n: {t: {r["id"]: r for r in jsonl.read(_out_dir(n) / f"{t}.jsonl")} for t in LIMITS}
            for n in names}
    lines = []
    for task in LIMITS:
        lines.append(f"\n## {task}")
        lines.append("| run | acc | " + " | ".join(["categories"]) + " |")
        for n in names:
            rs = list(data[n][task].values())
            cats = Counter(r.get("category") or ("pass" if r["correct"] else "fail") for r in rs)
            lines.append(f"| {n} | {_pct(sum(r['correct'] for r in rs), len(rs))} | "
                         + ", ".join(f"{k} {v}" for k, v in cats.most_common()) + " |")
        if len(names) >= 2:
            a, b = names[-2], names[-1]
            ids = data[a][task].keys() & data[b][task].keys()
            won = sum(data[b][task][i]["correct"] and not data[a][task][i]["correct"] for i in ids)
            lost = sum(data[a][task][i]["correct"] and not data[b][task][i]["correct"] for i in ids)
            lines.append(f"\n{a} -> {b}: won {won}, lost {lost} of {len(ids)}")
    # breakdowns for the last run
    last = names[-1]
    groups = defaultdict(lambda: [0, 0])
    for r in data[last]["math"].values():
        for key in (f"level {r['level']}", r["subject"]):
            groups[key][0] += r["correct"]; groups[key][1] += 1
    lines.append(f"\n## {last} math by level / subject")
    lines += [f"- {k}: {_pct(c, t)} ({c}/{t})" for k, (c, t) in sorted(groups.items())]
    groups = defaultdict(lambda: [0, 0])
    for r in data[last]["code"].values():
        groups[r["category_topic"]][0] += r["correct"]; groups[r["category_topic"]][1] += 1
    lines.append(f"\n## {last} code by topic (worst first)")
    lines += [f"- {k}: {_pct(c, t)} ({c}/{t})" for k, (c, t) in sorted(groups.items(), key=lambda kv: kv[1][0] / kv[1][1])]
    groups = defaultdict(lambda: [0, 0])
    kinds = {}
    for r in data[last]["if"].values():
        for cid, ok in zip(r["constraints"], r["per_constraint"]):
            groups[cid][0] += ok; groups[cid][1] += 1; kinds[cid] = r["kind"]
    lines.append(f"\n## {last} IF pass rate per constraint type (worst first)")
    lines += [f"- {k} [{kinds[k]}]: {_pct(c, t)} ({c}/{t})" for k, (c, t) in sorted(groups.items(), key=lambda kv: kv[1][0] / kv[1][1])]
    by_n = defaultdict(lambda: [0, 0])
    for r in data[last]["if"].values():
        by_n[len(r["constraints"])][0] += r["correct"]; by_n[len(r["constraints"])][1] += 1
    lines.append(f"\n## {last} IF prompt pass rate by number of constraints")
    lines += [f"- {k}: {_pct(c, t)} ({c}/{t})" for k, (c, t) in sorted(by_n.items())]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    c.add_argument("--checkpoint_path", default=None)
    c.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("summary")
    s.add_argument("--names", nargs="+", required=True)
    args = p.parse_args()
    if args.cmd == "collect":
        asyncio.run(collect(args.name, args.checkpoint_path, args.limit))
    else:
        print(summary(args.names))


if __name__ == "__main__":
    main()

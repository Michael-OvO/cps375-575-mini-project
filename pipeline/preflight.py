"""Preflight gate: checks a training set against how the model is graded, before any training run.

Every training example is verified for correctness when generated, but nothing checked that its format and
prompt style match the evaluation harnesses. v2 slipped through that gap: plan-first code answers broke the
HumanEval harness (67 -> 31), and function-naming task texts pulled the model away from reading the asserts.

Checks (runs/<name>/preflight.json records the result; train_sft refuses to start unless it passed):
- FAIL rules: answer formats that a harness would score wrong (code must open with one ```python block and end
  with it; math must end with an `ANSWER:` line without \\boxed; no empty assistant turn).
- FAIL: a sample of code targets must pass the asserts shown in their own prompt (harness-style extraction).
- WARN: style shares that moved by more than 10 points (or length medians by more than 1.5x) against a baseline
  training set. A warning needs an explicit `--ack <feature>` with a reason recorded in preflight.json.

Usage:
    python -m pipeline.preflight --name v3 --baseline v1 [--ack code.prompt_names_function="reason"]
"""

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

from pipeline import jsonl
from pipeline.config import RUNS, SEED
from pipeline.verify import code as vc

FENCE_OPEN = re.compile(r"\A```python\n")
ASSERT_LINE = re.compile(r"^assert .+$", re.M)
SHARE_DELTA = 0.10
LENGTH_RATIO = 1.5
CODE_EXEC_SAMPLE = 300
FEWSHOT_MARK = "# Now, do it for the following task."


def _user(ex: dict) -> str:
    return ex["messages"][0]["content"]


def _answer(ex: dict) -> str:
    return ex["messages"][-1]["content"]


def code_task_text(prompt: str) -> str:
    """The task part of a code prompt: after the few-shot block, without the assert lines."""
    if FEWSHOT_MARK in prompt:
        prompt = prompt.split(FEWSHOT_MARK, 1)[1]
    return ASSERT_LINE.sub("", prompt)


def called_name(prompt: str) -> str | None:
    tests = ASSERT_LINE.findall(prompt.split(FEWSHOT_MARK, 1)[-1])
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*(?:math\.isclose\()?\s*([A-Za-z_]\w*)\s*\(", tests[0]) if tests else None
    return m.group(1) if m else None


def format_errors(ex: dict) -> list[str]:
    task, ans = ex["task"], _answer(ex)
    if not ans.strip():
        return ["empty_answer"]
    errs = []
    if task == "code":
        if not FENCE_OPEN.match(ans):
            errs.append("code_not_fence_first")  # completion-style harnesses append the answer to the prompt
        if ans.count("```") != 2 or not ans.rstrip().endswith("```"):
            errs.append("code_not_single_block")
    elif task == "math":
        last = ans.rstrip().splitlines()[-1]
        if not re.match(r"ANSWER:\s*\S", last):
            errs.append("math_no_answer_line")
        elif "\\boxed" in last:
            errs.append("math_boxed_answer")
    return errs


def features(ex: dict) -> dict[str, float]:
    """Style features whose share or median must not drift silently between versions."""
    task, prompt, ans = ex["task"], _user(ex), _answer(ex)
    f = {"answer_chars": float(len(ans)), "prompt_chars": float(len(prompt))}
    if task == "code":
        text = code_task_text(prompt)
        name = called_name(prompt)
        f["prompt_names_function"] = float(bool(name) and re.search(rf"\b{re.escape(name)}\b", text) is not None)
        f["prompt_docstring_style"] = float('"""' in text or "'''" in text)
        f["prompt_harness_template"] = float("## Prompt:" in prompt)
        f["answer_prose_first"] = float(not FENCE_OPEN.match(ans))
    elif task == "math":
        f["answer_markdown"] = float(bool(re.search(r"\*\*|^#+ |^- ", ans, re.M)))
    elif task == "instruction_following":
        f["prompt_rules_first"] = float(not prompt[:1].isupper() or prompt.split(".")[0].lower().startswith(
            ("your", "in your", "do not", "use ", "write in", "answer with", "wrap", "include", "make sure")))
    return f


def profile(rows: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    for ex in rows:
        counts[ex["task"]] += 1
        for k, v in features(ex).items():
            by[ex["task"]][k].append(v)
    total = sum(counts.values())
    out = {"counts": dict(counts), "share": {t: c / total for t, c in counts.items()}, "features": {}}
    for task, fs in by.items():
        for k, vals in fs.items():
            key = f"{task}.{k}"
            out["features"][key] = statistics.median(vals) if k.endswith("_chars") else sum(vals) / len(vals)
    return out


def drift(new: dict, base: dict) -> list[str]:
    warns = []
    for t in set(new["share"]) | set(base["share"]):
        a, b = new["share"].get(t, 0), base["share"].get(t, 0)
        if abs(a - b) > SHARE_DELTA:
            warns.append(f"mix.{t}: {b:.0%} -> {a:.0%} of examples")
    for k, a in new["features"].items():
        b = base["features"].get(k)
        if b is None:
            continue
        if k.endswith("_chars"):
            if b and not (1 / LENGTH_RATIO <= a / b <= LENGTH_RATIO):
                warns.append(f"{k}: median {b:.0f} -> {a:.0f}")
        elif abs(a - b) > SHARE_DELTA:
            warns.append(f"{k}: {b:.0%} -> {a:.0%}")
    return warns


def code_exec_failures(rows: list[dict], n: int = CODE_EXEC_SAMPLE) -> list[str]:
    """Extract code the way the MBPP scorer does and run it against the asserts shown in its own prompt."""
    code = [ex for ex in rows if ex["task"] == "code"]
    sample = random.Random(SEED).sample(code, min(n, len(code)))
    bad = []
    for ex in sample:
        tests = ASSERT_LINE.findall(_user(ex).split(FEWSHOT_MARK, 1)[-1])
        if tests and not vc.run_tests(vc.extract_code(_answer(ex)), tests):
            bad.append(ex["id"])
    return bad


def run(name: str, baseline: str | None, acks: dict[str, str]) -> dict:
    rows = list(jsonl.read(RUNS / name / "train.jsonl"))
    errs = Counter()
    examples = defaultdict(list)
    for ex in rows:
        for e in format_errors(ex):
            errs[e] += 1
            if len(examples[e]) < 3:
                examples[e].append(ex["id"])
    exec_bad = code_exec_failures(rows)
    prof = profile(rows)
    warns = drift(prof, profile(list(jsonl.read(RUNS / baseline / "train.jsonl")))) if baseline else []
    unacked = [w for w in warns if w.split(":")[0] not in acks]
    fails = [f"{e}: {c} examples (e.g. {examples[e]})" for e, c in errs.items()]
    if exec_bad:
        fails.append(f"code targets failing their own prompt's asserts: {len(exec_bad)} of sample (e.g. {exec_bad[:3]})")
    result = {"name": name, "baseline": baseline, "ok": not fails and not unacked, "fails": fails,
              "warnings": warns, "acknowledged": acks, "unacknowledged": unacked, "profile": prof}
    (RUNS / name / "preflight.json").write_text(json.dumps(result, indent=2))
    return result


def require_passed(name: str) -> None:
    """Called by train_sft: refuse to train a set that has not passed preflight."""
    p = RUNS / name / "preflight.json"
    if not p.exists() or not json.loads(p.read_text()).get("ok"):
        raise SystemExit(f"preflight has not passed for {name}: run `python -m pipeline.preflight --name {name} "
                         f"--baseline <previous>` and fix or acknowledge its findings")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument("--ack", action="append", default=[], help='feature="reason" for an intended style change')
    a = p.parse_args()
    acks = dict(x.split("=", 1) for x in a.ack)
    r = run(a.name, a.baseline, acks)
    for f in r["fails"]:
        print("FAIL", f)
    for w in r["warnings"]:
        print("ACK " if w.split(":")[0] in acks else "WARN", w)
    print("preflight", "PASSED" if r["ok"] else "BLOCKED", f"-> {RUNS / a.name / 'preflight.json'}")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()

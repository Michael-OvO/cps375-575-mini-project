"""Code pool from nvidia/OpenCodeInstruct (train split, CC BY 4.0; a staff-suggested dataset).

Why: a classmate's run with OCI as its only code source matched our MBPP score with a quarter of our data, and
OCI's 5M self-instruct/evol problems are far more varied than our teacher-written tasks. We keep a row only if
our own executor confirms its reference code passes its unit tests, so no teacher call is needed.

Filters, every drop counted in data/oci_tasks.jsonl.report.json:
  1. OCI's own test pass rate is 1.0 and the LLM judge gave 5/5 on every axis.
  2. One ```python block; the code keeps only imports, functions and classes (example-usage prints dropped);
     at least one function, no classes (MBPP and the held-out set grade plain functions).
  3. Spec of 15-250 words, at least 4 assert tests, each calling a function the code defines.
  4. Exact and near-duplicate specs (MinHash, as in if_rlvr), eval overlap (MATH test, IFEval, MBPP all
     splits, HumanEval), our code dev set and discovery probes.
  5. The code passes all kept tests in our sandbox (3 visible, the rest hidden, as in code_tasks.jsonl).

Usage:
    ALLOW_TEST_OVERLAP_CHECK=1 python -m pipeline.generate.oci --shards 0 17 34 --limit 12000
"""

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pipeline import jsonl, leakage
from pipeline.config import DATA, SEED
from pipeline.dev import is_code_dev, normalize_code_text
from pipeline.generate.if_rlvr import NearDup
from pipeline.verify import code as vc

DATASET = "nvidia/OpenCodeInstruct"
OUT = DATA / "oci_tasks.jsonl"
MIN_WORDS, MAX_WORDS, MIN_TESTS, VISIBLE = 15, 250, 4, 3
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def judged_perfect(row: dict) -> bool:
    try:
        j = json.loads(row["llm_judgement"])
    except (TypeError, ValueError):
        return False
    return bool(j) and all(isinstance(v, dict) and v.get("score") == 5 for v in j.values())


def clean_code(output: str) -> str | None:
    """The single python block, keeping only imports and functions; None if there is no usable function."""
    blocks = _BLOCK.findall(output or "")
    if len(blocks) != 1:
        return None
    try:
        tree = ast.parse(blocks[0])
    except SyntaxError:
        return None
    if any(isinstance(n, ast.ClassDef) for n in tree.body):
        return None
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            and not _is_typing_import(n)]
    if not any(isinstance(n, ast.FunctionDef) for n in keep):
        return None
    # Type hints go: the harness-safe format moves imports inside each function (HumanEval keeps only the
    # function body), which would leave `List[int]` in a signature undefined.
    for node in ast.walk(ast.Module(body=keep, type_ignores=[])):
        if isinstance(node, ast.FunctionDef):
            node.returns = None
        elif isinstance(node, ast.arg):
            node.annotation = None
    code = ast.unparse(ast.Module(body=keep, type_ignores=[]))
    return code if "typing" not in code and "List[" not in code and "Dict[" not in code else None


def _is_typing_import(node) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.module == "typing"
    return isinstance(node, ast.Import) and all(a.name == "typing" for a in node.names)


def clean_tests(unit_tests, defined: set[str]) -> list[str]:
    tests = unit_tests if isinstance(unit_tests, list) else []
    if isinstance(unit_tests, str):
        try:
            tests = json.loads(unit_tests)
        except ValueError:
            tests = []
    out = []
    for t in (t.strip() for t in tests if isinstance(t, str)):
        if t.startswith("assert ") and "\n" not in t and set(_CALL.findall(t)) & defined and t not in out:
            out.append(t)
    return out


def _defined(code: str) -> set[str]:
    return {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}


def candidates(shards: list[int], drops: Counter) -> list[dict]:
    from datasets import load_dataset
    files = [f"data/train-{s:05d}-of-00050.parquet" for s in shards]
    rows = list(load_dataset(DATASET, data_files=files, split="train", verification_mode="no_checks"))
    random.Random(SEED).shuffle(rows)
    out = []
    for r in rows:
        if str(r.get("average_test_score")) not in ("1", "1.0") or not judged_perfect(r):
            drops["not_perfect"] += 1
            continue
        text = (r.get("input") or "").strip()
        if not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
            drops["spec_length"] += 1
            continue
        code = clean_code(r.get("output"))
        if code is None:
            drops["no_plain_function"] += 1
            continue
        tests = clean_tests(r.get("unit_tests"), _defined(code))
        if len(tests) < MIN_TESTS:
            drops["few_tests"] += 1
            continue
        out.append({"id": "oci-" + hashlib.sha1(r["id"].encode()).hexdigest()[:12], "oci_id": r["id"],
                    "text": text, "code": code, "tests": tests, "domain": r.get("domain"),
                    "generation_algorithm": r.get("generation_algorithm")})
    return out


def _probe_texts() -> list[str]:
    path = DATA / "discover" / "probes_code.jsonl"
    return [r.get("text") or r.get("prompt") or "" for r in jsonl.read(path)] if path.exists() else []


def build(shards: list[int], limit: int) -> dict:
    eval_filter = leakage.load_overlap_filter()
    if eval_filter is None:
        raise SystemExit("set ALLOW_TEST_OVERLAP_CHECK=1: the pool must be checked against the eval sets")
    from pipeline.generate.if_rlvr import _humaneval_texts
    humaneval = leakage.OverlapFilter(_humaneval_texts())
    probes = leakage.OverlapFilter(_probe_texts())
    drops = Counter()
    rows = candidates(shards, drops)
    seen, near, picked = set(), NearDup(), []
    for r in rows:
        key = normalize_code_text(r["text"])
        if key in seen:
            drops["dup_exact"] += 1
            continue
        if eval_filter.is_leak(r["text"]) or humaneval.is_leak(r["text"]):
            drops["eval_overlap"] += 1
            continue
        if is_code_dev(r["text"]) or probes.is_leak(r["text"]):
            drops["dev_or_probe"] += 1
            continue
        if not near.add(r["text"]):
            drops["dup_near"] += 1
            continue
        seen.add(key)
        picked.append(r)
        if len(picked) >= int(limit * 1.15):  # headroom for sandbox failures
            break
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda r: vc.run_tests(r["code"], r["tests"]), picked))
    kept = []
    for r, passed in zip(picked, ok):
        if not passed:
            drops["sandbox_fail"] += 1
            continue
        kept.append({**r, "tests": r["tests"][:VISIBLE], "hidden_tests": r["tests"][VISIBLE:], "author": "oci"})
    kept = kept[:limit]
    jsonl.write(OUT, kept)
    report = {"dataset": DATASET, "split": "train", "shards": shards, "candidates": len(rows), "kept": len(kept),
              "drops": dict(drops), "domains": dict(Counter(r["domain"] for r in kept)),
              "algorithms": dict(Counter(r["generation_algorithm"] for r in kept))}
    (OUT.parent / (OUT.name + ".report.json")).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=int, nargs="+", default=[0, 17, 34])
    p.add_argument("--limit", type=int, default=12000)
    args = p.parse_args()
    print(json.dumps(build(args.shards, args.limit), indent=2))


if __name__ == "__main__":
    main()

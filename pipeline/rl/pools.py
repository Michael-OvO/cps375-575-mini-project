"""Candidate prompt pools for rl1, with every exclusion of spec §6.1 applied. Selection by pass rate happens
later (schedule.py) from calibration statistics; this module only builds clean, stratified candidates."""

import ast
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pipeline import formats, jsonl
from pipeline.code_format import modules_in_tests
from pipeline.config import DATA, RUNS, SEED
from pipeline.constraints.sample import DEV_ONLY_IDS, render_prompt, sample_constraints
from pipeline.dev import is_code_dev
from pipeline.generate.if_rlvr import HELD_OUT
from pipeline.rl import data_gates, sandbox
from pipeline.rl.tests_split import Check, check, split_tests
from pipeline.verify import ifeval as vi

OUT = RUNS / "rl1" / "pools"
COUNTING = {"keywords:frequency", "length_constraints:number_words", "keywords:forbidden_words",
            "length_constraints:number_sentences", "keywords:letter_frequency",
            "change_case:capital_word_frequency", "change_case:english_lowercase"}
BIN_SHARE = {"1": 0.25, "2-3": 0.60, "4-5": 0.15}
TYPE_CAP = 400  # a prompt counts toward every type it contains (spec §6.1)
IF_TARGET = 9000
STUBS = ["return None", "return 0", "return ''", "return []", "return {}", "return True", "return False"]
GUESSES = (True, False, None, 0, 1)
# References whose output depends on set iteration order: they passed in some Phase 0 runs and failed in others.
SET_ORDER_FLAKY = frozenset({"oci-53cf9feefc60", "oci-f78717d1a20c", "oci-4c408a856586", "oci-586084257469"})


def math_stratum(subject: str, level: int) -> str:
    return f"{subject}|{'1-3' if level <= 3 else level}"


def count_bin(k: int) -> str:
    return "1" if k == 1 else ("2-3" if k <= 3 else "4-5")


def if_stratum(constraints: list[dict]) -> str:
    return f"{constraints[0]['id'].split(':')[0]}|{count_bin(len(constraints))}"


def code_stratum(row: dict) -> str:
    return f"oci|{row.get('domain', 'generic')}" if row["id"].startswith("oci-") else f"ct|{row.get('category', 'other')}"


def _called_function(code: str, tests: list[str]) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and re.search(rf"\b{re.escape(n.name)}\(", tests[0]):
            return n
    return None


def stub_program(tests: list[str], name: str, params: str, first: str | None) -> str:
    bodies = STUBS + ([f"return {first}"] if first else [])
    stubs = [f"def {name}({params}):\n    {b}\n" for b in bodies]
    imports = "".join(f"import {m}\n" for m in sorted(modules_in_tests(tests)))
    return (f"_tests = {tests!r}\n_stubs = {stubs!r}\n_hit = []\n"
            "for _i, _s in enumerate(_stubs):\n    _g = {}\n    try:\n"
            f"        exec({imports!r}, _g)\n        exec(_s, _g)\n"
            "        for _t in _tests:\n            exec(_t, _g)\n        _hit.append(_i)\n"
            "    except BaseException:\n        pass\nprint('STUBHITS', len(_hit))\n")


def stub_passes(task: dict) -> bool:
    """True if some trivial stub passes every assert (visible and hidden): the tests cannot tell it apart."""
    tests = task["tests"] + task.get("hidden_tests", [])
    fn = _called_function(task["code"], tests)
    if fn is None:
        return False  # class-based or unparsable: keep, the reference gate already checked it
    first = fn.args.args[0].arg if fn.args.args else None
    res = sandbox.run_program(stub_program(tests, fn.name, ast.unparse(fn.args), first))
    return "STUBHITS" in res.stdout and not res.stdout.strip().endswith("STUBHITS 0")


def guessable_constant(checks: list[Check]) -> bool:
    """True if one constant in GUESSES passes every check (all truthy, all falsy, all == / is the same constant):
    a program that returns it without computing anything would earn full reward."""
    return bool(checks) and any(all(check(c, g) for c in checks) for g in GUESSES)


def code_drop_reason(r: dict, ref_fail: set[str], leak, dev=None) -> str | None:
    """Why a code task cannot enter the pool (cheap, sandbox-free rules), or None. The stub rule runs after.
    leak: eval-set overlap filter; dev: fuzzy overlap filter over the private dev texts (dev_filter())."""
    tests = r["tests"] + r.get("hidden_tests", [])
    if len(tests) < 5:
        return "few_asserts"
    checks = split_tests(tests)
    if checks is None:
        return "unsupported_tests"
    if is_code_dev(r["text"]):
        return "dev"
    if dev and dev.is_leak(r["text"]):
        return "dev_overlap"
    if r["id"] in ref_fail:
        return "reference_fails"
    if r["id"] in SET_ORDER_FLAKY:
        return "set_order_flaky"
    if guessable_constant(checks):
        return "guessable_constant"
    if leak and leak.is_leak(r["text"]):
        return "leak"
    return None


def _weight(c: list[dict]) -> float:
    return 1.5 if any(x["id"] in COUNTING for x in c) else 1.0


def select_if(cands: list[dict], target: int, rng: random.Random) -> list[dict]:
    keyed = sorted(cands, key=lambda c: rng.random() ** (1.0 / _weight(c["constraints"])), reverse=True)
    quota = {b: round(target * s) for b, s in BIN_SHARE.items()}
    per_type: Counter = Counter()
    out = []
    for c in keyed:
        b = count_bin(len(c["constraints"]))
        ids = [x["id"] for x in c["constraints"]]
        if quota[b] <= 0 or any(per_type[i] >= TYPE_CAP for i in ids):
            continue
        quota[b] -= 1
        per_type.update(ids)
        out.append(c)
    return out


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class _AnyFilter:
    def __init__(self, filters: list):
        self.filters = filters

    def is_leak(self, text: str) -> bool:
        return any(f.is_leak(text) for f in self.filters)


def dev_filter():
    """The SFT fuzzy dev-overlap filter (select.load_dev_filter: 8-gram over math_dev problems, code_dev texts and
    if_dev base tasks) plus the if_dev_new base tasks. Fails closed: any dev file missing is an error."""
    from pipeline.leakage import OverlapFilter
    from pipeline.select import load_dev_filter

    missing: list[str] = []
    base = load_dev_filter(missing)
    if not (DATA / "if_dev_new.jsonl").exists():
        missing.append("if_dev_new.jsonl")
    if missing or base is None:
        raise SystemExit(f"dev files missing, pools cannot be checked against dev: {missing or 'all'}")
    new = [r["task"] for r in jsonl.read(DATA / "if_dev_new.jsonl") if r.get("task", "").strip()]
    return _AnyFilter([base] + ([OverlapFilter(new)] if new else []))


def build_math(leak, dev=None) -> tuple[list[dict], dict]:
    from pipeline.generate.math import load_train

    drops: Counter = Counter()

    def why(problem: str, is_dev: bool = False) -> str | None:
        if is_dev:  # vault items are a subset of dev
            return "dev"
        if dev and dev.is_leak(problem):
            return "dev_overlap"
        if leak and leak.is_leak(problem):
            return "leak"
        return None

    out = []
    for r in load_train():
        w = why(r["problem"], r["dev"])
        if w:
            drops[w] += 1
            continue
        out.append({"id": r["id"], "task": "math", "prompt": formats.math_prompt(r["problem"]), "text": r["problem"],
                    "stratum": math_stratum(r["subject"], r["level"]), "target": r["target"]})
    # teacher-written problems: keep only those where >= 2 independent solutions agree on the target
    by_pid: dict[str, list[dict]] = {}
    for s in jsonl.read(DATA / "math_new_solutions.jsonl"):
        by_pid.setdefault(s["pid"], []).append(s)
    for pid, sols in by_pid.items():
        targets = Counter(s["target"] for s in sols)
        target, votes = targets.most_common(1)[0]
        w = "no_agreement" if votes < 2 else why(sols[0]["problem"])
        if w:
            drops[f"teacher_{w}"] += 1
            continue
        s = sols[0]
        out.append({"id": pid, "task": "math", "prompt": formats.math_prompt(s["problem"]), "text": s["problem"],
                    "stratum": math_stratum(s["subject"], int(s["level"])), "target": target})
    return out, {"n": len(out), "drops": drops}


def _if_dev_tasks() -> set[str]:
    rows = jsonl.read(DATA / "if_dev.jsonl") + jsonl.read(DATA / "if_dev_new.jsonl")
    return {_norm(r["task"]) for r in rows}


def build_if(leak, excluded_types: set[str], trap_bases: set[str], rng: random.Random,
             nondet_ids: set[str] = frozenset(), nondet_dicts: list[dict] = (), dev=None) -> tuple[list[dict], dict]:
    """trap_bases: the trap prompts' base texts; a candidate whose base task equals one (normalized), or whose
    prompt contains one, is dropped. nondet_ids / nondet_dicts: Phase 0's non-deterministic prompt ids and kwargs
    dicts (dropped as well as re-checked here). dev: fuzzy dev-overlap filter, run on the base task."""
    banned = set(HELD_OUT) | set(DEV_ONLY_IDS) | excluded_types
    bad_dicts = {data_gates.key(d) for d in nondet_dicts}
    traps = {_norm(t) for t in trap_bases if t.strip()}
    dev_tasks = _if_dev_tasks()
    drops: Counter = Counter()
    cands = []

    def ok(pid: str, base: str, prompt: str, cons: list[dict]) -> bool:
        if any(c["id"] in banned for c in cons):
            drops["banned_type"] += 1
            return False
        try:
            [vi.route(c["id"]) for c in cons]
        except KeyError:
            drops["no_checker"] += 1
            return False
        if _norm(base) in dev_tasks:
            drops["dev"] += 1
            return False
        p = _norm(prompt)
        if _norm(base) in traps or any(t in p for t in traps):
            drops["trap"] += 1
            return False
        if pid in nondet_ids or any(data_gates.key(c) in bad_dicts for c in cons):
            drops["nondeterministic_phase0"] += 1
            return False
        if dev and dev.is_leak(base):
            drops["dev_overlap"] += 1
            return False
        if leak and leak.is_leak(base):
            drops["leak"] += 1
            return False
        if not all(data_gates.kwargs_deterministic(c, prompt) for c in cons):
            drops["nondeterministic"] += 1
            return False
        return True

    for r in jsonl.read(DATA / "if_rlvr_prompts.jsonl"):
        if ok(r["id"], r["task"], r["prompt"], r["constraints"]):
            cands.append({"id": r["id"], "task": "if", "prompt": r["prompt"], "constraints": r["constraints"]})
    for t in jsonl.read(DATA / "if_base_tasks.jsonl"):
        cons = sample_constraints(rng, t["task"], mode="train")
        prompt = render_prompt(t["task"], cons)
        if ok(f"ifs-{t['id']}", t["task"], prompt, cons):
            cands.append({"id": f"ifs-{t['id']}", "task": "if", "prompt": prompt, "constraints": cons})
    chosen = select_if(cands, IF_TARGET, rng)
    for c in chosen:
        c["stratum"] = if_stratum(c["constraints"])
    return chosen, {"candidates": len(cands), "drops": drops, "target": IF_TARGET, "chosen": len(chosen)}


def build_code(leak, ref_fail: set[str], dev=None) -> tuple[list[dict], dict]:
    """ref_fail: every Phase 0 reference scoring below 1 (failing, harness-unsafe, guarded, unsupported)."""
    rows = jsonl.read(DATA / "code_tasks.jsonl") + jsonl.read(DATA / "oci_tasks.jsonl")
    drops: Counter = Counter()
    keep = []
    for r in rows:
        why = code_drop_reason(r, ref_fail, leak, dev)
        if why:
            drops[why] += 1
        else:
            keep.append(r)
    with ThreadPoolExecutor(sandbox.WORKERS) as ex:
        stubbed = list(ex.map(stub_passes, keep))
    out = []
    for r, s in zip(keep, stubbed):
        if s:
            drops["stub_passes"] += 1
            continue
        out.append({"id": r["id"], "task": "code", "prompt": formats.code_prompt(r["text"], r["tests"]),
                    "text": r["text"], "stratum": code_stratum(r), "tests": r["tests"], "hidden_tests": r.get("hidden_tests", [])})
    return out, {"kept": len(out), "drops": drops}


def main() -> None:
    from pipeline.leakage import load_overlap_filter

    p0 = data_gates.require_passed()
    leak = load_overlap_filter()
    if leak is None:
        raise SystemExit("set ALLOW_TEST_OVERLAP_CHECK=1: pools must be checked against every eval set")
    dev = dev_filter()
    try:
        from pipeline.rl.traps import BASES as trap_bases
    except ImportError:
        raise SystemExit("pipeline.rl.traps.BASES (Task 11) is required: trap prompts stay out of every pool")
    rng = random.Random(SEED)
    math, math_rep = build_math(leak, dev)
    det = p0["determinism"]
    ifp, if_rep = build_if(leak, set(p0["if_router"]["excluded_types"]), set(trap_bases), rng,
                           set(det["nondeterministic_prompt_ids"]), det["nondeterministic_dicts"], dev)
    code, code_rep = build_code(leak, set(p0["code"]["ref_fail_ids"]) | set(p0["code"]["unsupported_ids"]), dev)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("math", math), ("if", ifp), ("code", code)):
        jsonl.write(OUT / f"{name}.jsonl", rows)
    report = {"math": math_rep | {"strata": Counter(r["stratum"] for r in math)},
              "if": if_rep | {"strata": Counter(r["stratum"] for r in ifp),
                              "bins": Counter(count_bin(len(r["constraints"])) for r in ifp)},
              "code": code_rep | {"strata": Counter(r["stratum"] for r in code)}}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=dict))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "strata"} for k, v in report.items()},
                     indent=1, default=dict))


if __name__ == "__main__":
    main()

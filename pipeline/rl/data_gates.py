"""Phase 0 data gates, run before any paid phase (spec §5):
1. code: guards fire on <= 0.2% of reference solutions and <= 2% of references fail their own tests (rates over
   items whose asserts the parent can verify); every reference scoring below 1 leaves the pool;
2. IF router: the routed checkers accept the stored verified IF-RLVR answers; a type with > 1% mismatch is
   dropped, and the gate fails if dropped types cover >= 5% of IF prompts;
3. kwargs determinism: every pool kwargs dict builds and gives the same arguments over 5 seeds;
4. IF quality gate: < 1% firing per type (n >= 50) on verified teacher answers that pass the v4 SFT filter.
Writes runs/rl1/phase0.json; train_rl and calibrate refuse to run unless passed."""

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from pipeline import jsonl
from pipeline.config import DATA, RUNS
from pipeline.rl import code_guard, sandbox
from pipeline.rl.gates import gate_reasons
from pipeline.rl.rewards import code_reward
from pipeline.rl.tests_split import split_tests
from pipeline.select import if_quality_problem
from pipeline.verify import ifeval as vi

OUT = RUNS / "rl1" / "phase0.json"
IF_ANSWERS = ("if_rlvr.gpt.jsonl", "if_rlvr.codex.jsonl", "if_rlvr.codex.retry.jsonl")
HARD_FAIL = {"fail", "timeout", "oom"}  # the reference ran and did not pass its own tests
GUARD_MAX, REF_FAIL_MAX, MISMATCH_MAX, EXCLUDED_SHARE_MAX, GATE_MAX, GATE_MIN_N = 0.002, 0.02, 0.01, 0.05, 0.01, 50
SEEDS = 5


def kwargs_deterministic(constraint: dict, prompt: str) -> bool:
    args = []
    for seed in range(SEEDS):  # a random default collides across all five draws with negligible odds
        random.seed(seed)
        try:
            inst = vi._instance(constraint, prompt)
        except Exception:
            return False
        get = getattr(inst, "get_instruction_args", None)
        args.append(repr(get()) if get else repr(constraint.get("kwargs")))
    return len(set(args)) == 1


def gate_rates(rows: list[dict]) -> dict[str, dict]:
    per = defaultdict(lambda: {"n": 0, "fired": 0, "examples": []})
    for r in rows:
        cons = r["constraints"]
        fired = gate_reasons(r["response"], cons)
        for c in cons:
            d = per[c["id"]]
            d["n"] += 1
            if fired:
                d["fired"] += 1
                if len(d["examples"]) < 5:
                    d["examples"].append({"id": r.get("id"), "gate": fired})
    return {k: v | {"rate": v["fired"] / v["n"]} for k, v in sorted(per.items())}


def code_summary(results: list[tuple]) -> dict:
    """results: (id, supported, guard violations, Scored) per reference. Guard and failure rates are over
    supported items (asserts the parent can verify); unsupported items leave the pool and are counted apart."""
    sup = [r for r in results if r[1]]
    n_sup = max(len(sup), 1)
    guard = [i for i, _, g, _ in sup if g]
    hard = [i for i, _, _, s in sup if s.info["reason"] in HARD_FAIL]
    return {"n": len(results), "n_supported": len(sup),
            "unsupported_ids": [i for i, ok, _, _ in results if not ok],
            "unsupported_rate": (len(results) - len(sup)) / max(len(results), 1),
            "guard_fire_ids": guard, "guard_rate": len(guard) / n_sup,
            "hard_fail_ids": hard, "ref_fail_rate": len(hard) / n_sup,
            "harness_unsafe_ids": [i for i, _, _, s in sup if s.info["reason"] == "harness_unsafe"],
            "ref_fail_ids": [i for i, _, _, s in results if s.reward != 1.0],  # every non-1 reference leaves
            "excluded_rate": sum(s.reward != 1.0 for *_, s in results) / max(len(results), 1),
            "reasons": dict(Counter(s.info["reason"] for *_, s in results))}


def code_gate() -> dict:
    tasks = jsonl.read(DATA / "code_tasks.jsonl") + jsonl.read(DATA / "oci_tasks.jsonl")
    tasks = [t for t in tasks if len(t["tests"]) + len(t.get("hidden_tests", [])) >= 5]

    def one(t):
        tests = t["tests"] + t.get("hidden_tests", [])
        s = code_reward(f"```python\n{t['code']}\n```", t["tests"], t.get("hidden_tests", []), False)
        return t["id"], split_tests(tests) is not None, code_guard.violations(t["code"]), s

    with ThreadPoolExecutor(sandbox.WORKERS) as ex:
        results = list(ex.map(one, tasks))
    s = code_summary(results)
    guards = Counter(v for _, ok, g, _ in results if ok for v in g)
    return s | {"guard_violations": dict(guards.most_common())}


def _pool_nondeterministic() -> list[str]:
    """Types with a kwargs dict (stored answers or IF-RLVR prompts) that fails to build or changes over seeds."""
    seen, bad = set(), set()
    for name in IF_ANSWERS + ("if_rlvr_prompts.jsonl",):
        for r in jsonl.read(DATA / name):
            for c in r["constraints"]:
                key = (c["id"], json.dumps(c.get("kwargs"), sort_keys=True, default=str))
                if key not in seen:
                    seen.add(key)
                    if not kwargs_deterministic(c, r["prompt"]):
                        bad.add(c["id"])
    return sorted(bad)


def if_router_gate() -> dict:
    rows = [r for name in IF_ANSWERS for r in jsonl.read(DATA / name)]
    n, bad = Counter(), Counter()
    examples = defaultdict(list)
    for r in rows:
        cons = r["constraints"]
        try:
            res = vi.check(r["prompt"], cons, r["response"])
        except Exception:  # unknown id or kwargs that do not build
            res = [False] * len(cons)
        for c, ok in zip(cons, res):
            n[c["id"]] += 1
            bad[c["id"]] += not ok
            if not ok and len(examples[c["id"]]) < 5:
                examples[c["id"]].append(r["id"])
    mism = {k: bad[k] / n[k] for k in sorted(n)}
    nondet = _pool_nondeterministic()
    excluded = sorted({k for k, v in mism.items() if v > MISMATCH_MAX} | set(nondet))
    prompts = jsonl.read(DATA / "if_rlvr_prompts.jsonl")
    hit = sum(any(c["id"] in excluded for c in p["constraints"]) for p in prompts)
    return {"n": len(rows), "n_by_type": dict(n), "mismatch_by_type": mism, "mismatch_examples": dict(examples),
            "nondeterministic_types": nondet, "excluded_types": excluded,
            "excluded_prompt_share": hit / len(prompts)}


def if_gate_gate() -> dict:
    rows = [r for name in IF_ANSWERS + ("if_responses.jsonl",) for r in jsonl.read(DATA / name)]
    good = [r for r in rows if if_quality_problem(r["response"], 0) is None]
    per = gate_rates(good)
    big = {k: v for k, v in per.items() if v["n"] >= GATE_MIN_N}
    return {"n": len(rows), "n_good": len(good), "per_type": per,
            "failing_types": sorted(k for k, v in big.items() if v["rate"] >= GATE_MAX),
            "max_rate": max((v["rate"] for v in big.values()), default=0.0),
            "fired_by_rule": dict(Counter(g for r in good for g in gate_reasons(r["response"], r["constraints"])))}


def main() -> None:
    code, router, gates = code_gate(), if_router_gate(), if_gate_gate()
    checks = {"code_guard": code["guard_rate"] <= GUARD_MAX, "code_ref_fail": code["ref_fail_rate"] <= REF_FAIL_MAX,
              "if_router": router["excluded_prompt_share"] < EXCLUDED_SHARE_MAX, "if_gates": not gates["failing_types"]}
    passed = all(checks.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"passed": passed, "checks": checks, "code": code, "if_router": router,
                               "if_gates": gates}, indent=1))
    print(json.dumps({"passed": passed, "checks": checks, "n_code": code["n"], "n_supported": code["n_supported"],
                      "unsupported_rate": code["unsupported_rate"], "guard_rate": code["guard_rate"],
                      "ref_fail_rate": code["ref_fail_rate"], "excluded_rate": code["excluded_rate"],
                      "reasons": code["reasons"], "guard_violations": code["guard_violations"],
                      "excluded_types": router["excluded_types"],
                      "nondeterministic_types": router["nondeterministic_types"],
                      "excluded_prompt_share": router["excluded_prompt_share"],
                      "gate_failing_types": gates["failing_types"], "gate_max_rate": gates["max_rate"],
                      "gate_fired_by_rule": gates["fired_by_rule"]}, indent=1))


def require_passed() -> dict:
    if not OUT.exists() or not json.loads(OUT.read_text())["passed"]:
        raise SystemExit(f"Phase 0 gates not passed: run python -m pipeline.rl.data_gates ({OUT})")
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    main()

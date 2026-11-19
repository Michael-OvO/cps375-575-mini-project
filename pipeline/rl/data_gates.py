"""Phase 0 data gates, run before any paid phase (spec §5):
1. code, over supported, harness-safe references (the ones that can enter pools): guards fire on <= 0.2% and
   <= 2% fail their own tests; every reference scoring below 1 leaves the pool;
2. IF router, non-IFEval ids only: the routed checkers accept the stored verified IF-RLVR answers; a type with
   > 1% mismatch is dropped, and the gate fails if dropped types cover >= 5% of IF prompts. IFEval ids always use
   the official checker (it grades us); their mismatch with stored labels is reported as information;
3. kwargs determinism: every pool kwargs dict builds and gives the same arguments over 5 seeds; failing dicts
   (not types) are dropped, and the gate fails if they cover >= 5% of IF prompts;
4. IF quality gate, on verified teacher answers that pass the v4 SFT filter: fires on < 1.5% of answers and on
   < 3% of every type with n >= 50.
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
GUARD_MAX, REF_FAIL_MAX, MISMATCH_MAX, SHARE_MAX = 0.002, 0.02, 0.01, 0.05
GATE_ALL_MAX, GATE_TYPE_MAX, GATE_MIN_N = 0.015, 0.03, 50
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


def _key(c: dict) -> tuple[str, str]:
    return c["id"], json.dumps(c.get("kwargs"), sort_keys=True, default=str)


def nondeterministic(rows: list[dict]) -> tuple[list[dict], list[str], int]:
    """Kwargs dicts that fail to build or change over seeds (each listed once), the ids of rows holding one, and
    the number of distinct dicts checked."""
    verdict, dicts, ids = {}, [], []
    for r in rows:
        hit = False
        for c in r["constraints"]:
            k = _key(c)
            if k not in verdict:
                verdict[k] = kwargs_deterministic(c, r["prompt"])
                if not verdict[k]:
                    dicts.append({"id": c["id"], "kwargs": c.get("kwargs")})
            hit |= not verdict[k]
        if hit:
            ids.append(r["id"])
    return dicts, ids, len(verdict)


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


def if_gate_summary(good: list[dict]) -> dict:
    fired = [gate_reasons(r["response"], r["constraints"]) for r in good]
    per = gate_rates(good)
    big = {k: v for k, v in per.items() if v["n"] >= GATE_MIN_N}
    overall = sum(map(bool, fired)) / max(len(good), 1)
    failing = sorted(k for k, v in big.items() if v["rate"] >= GATE_TYPE_MAX)
    return {"n_good": len(good), "overall_rate": overall, "per_type": per, "failing_types": failing,
            "max_rate": max((v["rate"] for v in big.values()), default=0.0),
            "fired_by_rule": dict(Counter(g for f in fired for g in f)),
            "passed": bool(good) and bool(big) and overall < GATE_ALL_MAX and not failing}


def code_summary(results: list[tuple], recheck_pass: set[str] = frozenset()) -> dict:
    """results: (id, supported, guard violations, Scored) per reference. Guard and failure rates are over
    supported, harness-safe items (the only ones that can enter pools); the rest leave the pool, counted apart.
    A first-run hard failure in recheck_pass passed its serial recheck: it is flaky, not a hard failure (it still
    leaves the pool, as does every reference that scored below 1)."""
    sup = [r for r in results if r[1]]
    pool = [r for r in sup if r[3].info["reason"] != "harness_unsafe"]
    n_pool = max(len(pool), 1)
    guard = [i for i, _, g, _ in pool if g]
    first = [i for i, _, _, s in pool if s.info["reason"] in HARD_FAIL]
    hard = [i for i in first if i not in recheck_pass]
    return {"n": len(results), "n_supported": len(sup), "n_eligible": len(pool),
            "unsupported_ids": [i for i, ok, _, _ in results if not ok],
            "unsupported_rate": (len(results) - len(sup)) / max(len(results), 1),
            "guard_fire_ids": guard, "guard_rate": len(guard) / n_pool,
            "hard_fail_ids": hard, "flaky_ids": [i for i in first if i in recheck_pass],
            "ref_fail_rate": len(hard) / n_pool,
            "harness_unsafe_ids": [i for i, _, _, s in sup if s.info["reason"] == "harness_unsafe"],
            "ref_fail_ids": [i for i, _, _, s in results if s.reward != 1.0],  # every non-1 reference leaves
            "excluded_rate": sum(s.reward != 1.0 for *_, s in results) / max(len(results), 1),
            "reasons": dict(Counter(s.info["reason"] for *_, s in results))}


def code_gate() -> dict:
    tasks = jsonl.read(DATA / "code_tasks.jsonl") + jsonl.read(DATA / "oci_tasks.jsonl")
    tasks = [t for t in tasks if len(t["tests"]) + len(t.get("hidden_tests", [])) >= 5]

    def score(t):
        return code_reward(f"```python\n{t['code']}\n```", t["tests"], t.get("hidden_tests", []), False)

    def one(t):
        tests = t["tests"] + t.get("hidden_tests", [])
        return t["id"], split_tests(tests) is not None, code_guard.violations(t["code"]), score(t)

    with ThreadPoolExecutor(sandbox.WORKERS) as ex:
        results = list(ex.map(one, tasks))
    by_id = {t["id"]: t for t in tasks}
    # Serial recheck (one at a time, no load): a hard failure counts only if it fails again.
    recheck_pass = {i for i, _, _, s in results
                    if s.info["reason"] in HARD_FAIL and score(by_id[i]).reward == 1.0}
    s = code_summary(results, recheck_pass)
    guards = Counter(v for _, ok, g, sc in results if ok and sc.info["reason"] != "harness_unsafe" for v in g)
    return s | {"guard_violations": dict(guards.most_common())}


def if_router_gate(prompts: list[dict]) -> dict:
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

    def official(cid: str) -> bool:
        try:
            return vi.route(cid) == "official"
        except KeyError:
            return False

    mism = {k: bad[k] / n[k] for k in sorted(n) if not official(k)}
    excluded = sorted(k for k, v in mism.items() if v > MISMATCH_MAX)
    hit = sum(any(c["id"] in excluded for c in p["constraints"]) for p in prompts)
    return {"n": len(rows), "n_prompts": len(prompts), "n_by_type": dict(n), "mismatch_by_type": mism,
            "official_mismatch_by_type": {k: bad[k] / n[k] for k in sorted(n) if official(k)},  # information only
            "mismatch_examples": dict(examples), "excluded_types": excluded,
            "excluded_prompt_share": hit / max(len(prompts), 1)}


def determinism_gate(prompts: list[dict]) -> dict:
    answers = [r for name in IF_ANSWERS for r in jsonl.read(DATA / name)]
    dicts, prompt_ids, n_prompt = nondeterministic(prompts)
    more, answer_ids, n_answer = nondeterministic(answers)
    seen = {_key(d) for d in dicts}
    dicts += [d for d in more if _key(d) not in seen]
    return {"n_checked": n_prompt + n_answer, "n_prompts": len(prompts), "nondeterministic_types": dict(Counter(d["id"] for d in dicts)),
            "nondeterministic_dicts": dicts, "nondeterministic_prompt_ids": prompt_ids,
            "nondeterministic_answer_ids": answer_ids,
            "nondeterministic_prompt_share": len(prompt_ids) / max(len(prompts), 1)}


def if_gate_gate() -> dict:
    rows = [r for name in IF_ANSWERS + ("if_responses.jsonl",) for r in jsonl.read(DATA / name)]
    good = [r for r in rows if if_quality_problem(r["response"], 0) is None]
    return {"n": len(rows)} | if_gate_summary(good)


def gate_checks(code: dict, router: dict, det: dict, gates: dict) -> dict[str, bool]:
    """Each check fails closed: an empty population (no data read) never passes."""
    has_code = code["n_eligible"] > 0
    return {"code_guard": has_code and code["guard_rate"] <= GUARD_MAX,
            "code_ref_fail": has_code and code["ref_fail_rate"] <= REF_FAIL_MAX,
            "if_router": router["n"] > 0 and router["n_prompts"] > 0 and router["excluded_prompt_share"] < SHARE_MAX,
            "kwargs_determinism": det["n_checked"] > 0 and det["n_prompts"] > 0 and det["nondeterministic_prompt_share"] < SHARE_MAX,
            "if_gates": gates["passed"]}


def main() -> None:
    prompts = jsonl.read(DATA / "if_rlvr_prompts.jsonl")
    code, router, det, gates = code_gate(), if_router_gate(prompts), determinism_gate(prompts), if_gate_gate()
    checks = gate_checks(code, router, det, gates)
    passed = all(checks.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"passed": passed, "checks": checks, "code": code, "if_router": router,
                               "determinism": det, "if_gates": gates}, indent=1))
    print(json.dumps({"passed": passed, "checks": checks, "n_code": code["n"], "n_supported": code["n_supported"],
                      "n_eligible": code["n_eligible"], "unsupported_rate": code["unsupported_rate"],
                      "guard_rate": code["guard_rate"], "ref_fail_rate": code["ref_fail_rate"],
                      "excluded_rate": code["excluded_rate"], "reasons": code["reasons"],
                      "guard_violations": code["guard_violations"], "excluded_types": router["excluded_types"],
                      "excluded_prompt_share": router["excluded_prompt_share"],
                      "official_mismatch_over_1pct": {k: v for k, v in router["official_mismatch_by_type"].items()
                                                      if v > MISMATCH_MAX},
                      "nondeterministic_types": det["nondeterministic_types"],
                      "nondeterministic_prompt_share": det["nondeterministic_prompt_share"],
                      "gate_overall_rate": gates["overall_rate"], "gate_failing_types": gates["failing_types"],
                      "gate_max_rate": gates["max_rate"], "gate_fired_by_rule": gates["fired_by_rule"]}, indent=1))


def require_passed() -> dict:
    if not OUT.exists() or not json.loads(OUT.read_text())["passed"]:
        raise SystemExit(f"Phase 0 gates not passed: run python -m pipeline.rl.data_gates ({OUT})")
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    main()

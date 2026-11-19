from pipeline.rl import data_gates as G
from pipeline.rl.rewards import Scored


def test_kwargs_deterministic_true_for_full_kwargs():
    c = {"id": "keywords:frequency", "kwargs": {"keyword": "tree", "frequency": 2, "relation": "at least"}}
    assert G.kwargs_deterministic(c, "p")


def test_kwargs_deterministic_false_when_official_kwarg_missing():
    c = {"id": "keywords:frequency", "kwargs": {"keyword": "tree"}}  # frequency/relation drawn at random
    assert not G.kwargs_deterministic(c, "p")


def test_kwargs_deterministic_false_for_unknown_id():
    assert not G.kwargs_deterministic({"id": "nope:nope", "kwargs": {}}, "p")


def test_gate_rates_counts_per_type():
    rows = [{"constraints": [{"id": "keywords:frequency", "kwargs": {}}], "response": "Keyword: x\nlong enough text here"},
            {"constraints": [{"id": "keywords:frequency", "kwargs": {}}], "response": "A clean answer about trees here."}]
    r = G.gate_rates(rows)
    assert r["keywords:frequency"]["n"] == 2 and r["keywords:frequency"]["fired"] == 1


def test_code_summary_rates_over_supported_harness_safe_items():
    results = [
        ("a", True, [], Scored(1.0, {"reason": "pass"})),
        ("b", True, [], Scored(0.0, {"reason": "fail"})),
        ("c", True, ["exit"], Scored(0.0, {"reason": "guard"})),
        ("d", True, [], Scored(0.0, {"reason": "harness_unsafe"})),
        ("e", False, [], Scored(0.0, {"reason": "unsupported_tests"})),
        ("f", False, ["eval"], Scored(0.0, {"reason": "guard"})),  # unsupported: neither guard nor fail count
        ("g", True, ["exit"], Scored(0.0, {"reason": "harness_unsafe"})),  # harness-unsafe never enters pools
    ]
    s = G.code_summary(results)
    assert s["n"] == 7 and s["n_supported"] == 5 and s["n_eligible"] == 3 and s["unsupported_ids"] == ["e", "f"]
    assert s["hard_fail_ids"] == ["b"] and s["ref_fail_rate"] == 1 / 3
    assert s["guard_fire_ids"] == ["c"] and s["guard_rate"] == 1 / 3
    assert s["harness_unsafe_ids"] == ["d", "g"]
    assert s["ref_fail_ids"] == ["b", "c", "d", "e", "f", "g"] and s["excluded_rate"] == 6 / 7
    assert s["reasons"] == {"pass": 1, "fail": 1, "guard": 2, "harness_unsafe": 2, "unsupported_tests": 1}


def test_code_summary_counts_timeout_and_oom_as_hard_failures():
    results = [("a", True, [], Scored(0.0, {"reason": "timeout"})), ("b", True, [], Scored(0.0, {"reason": "oom"}))]
    assert G.code_summary(results)["hard_fail_ids"] == ["a", "b"]


def test_nondeterministic_drops_dicts_not_types():
    good = {"id": "keywords:frequency", "kwargs": {"keyword": "tree", "frequency": 2, "relation": "at least"}}
    bad = {"id": "keywords:frequency", "kwargs": {"keyword": "tree"}}
    rows = [{"id": "p1", "prompt": "p", "constraints": [good]},
            {"id": "p2", "prompt": "p", "constraints": [good, bad]},
            {"id": "p3", "prompt": "p", "constraints": [bad]}]
    dicts, ids = G.nondeterministic(rows)
    assert dicts == [bad] and ids == ["p2", "p3"]


def _row(i, cid, response):
    return {"id": i, "constraints": [{"id": cid, "kwargs": {}}], "response": response}


def test_if_gate_summary_thresholds():
    clean = "A clean answer about trees and their shade in the city."
    dirty = "Keyword: x\nlong enough text here"
    rows = [_row(i, "a:a", clean) for i in range(98)] + [_row(98, "a:a", dirty)] + [_row(99, "b:b", clean)]
    s = G.if_gate_summary(rows)
    assert s["overall_rate"] == 0.01 and s["failing_types"] == [] and s["passed"]
    rows += [_row(100 + i, "c:c", dirty) for i in range(2)]  # type c has n < 50: ignored per type
    s = G.if_gate_summary(rows)
    assert s["failing_types"] == [] and not s["passed"]  # 3/102 > 1.5% overall
    rows = [_row(i, "a:a", clean) for i in range(194)] + [_row(194 + i, "a:a", dirty) for i in range(6)]
    s = G.if_gate_summary(rows)
    assert s["failing_types"] == ["a:a"] and not s["passed"]

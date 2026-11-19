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


def test_code_summary_rates_over_supported_items_only():
    results = [
        ("a", True, [], Scored(1.0, {"reason": "pass"})),
        ("b", True, [], Scored(0.0, {"reason": "fail"})),
        ("c", True, ["exit"], Scored(0.0, {"reason": "guard"})),
        ("d", True, [], Scored(0.0, {"reason": "harness_unsafe"})),
        ("e", False, [], Scored(0.0, {"reason": "unsupported_tests"})),
        ("f", False, ["eval"], Scored(0.0, {"reason": "guard"})),  # unsupported: neither guard nor fail count
        ("g", True, ["exit"], Scored(0.0, {"reason": "harness_unsafe"})),  # guard measured even past harness
    ]
    s = G.code_summary(results)
    assert s["n"] == 7 and s["n_supported"] == 5 and s["unsupported_ids"] == ["e", "f"]
    assert s["hard_fail_ids"] == ["b"] and s["ref_fail_rate"] == 1 / 5
    assert s["guard_fire_ids"] == ["c", "g"] and s["guard_rate"] == 2 / 5
    assert s["harness_unsafe_ids"] == ["d", "g"]
    assert s["ref_fail_ids"] == ["b", "c", "d", "e", "f", "g"] and s["excluded_rate"] == 6 / 7
    assert s["reasons"] == {"pass": 1, "fail": 1, "guard": 2, "harness_unsafe": 2, "unsupported_tests": 1}


def test_code_summary_counts_timeout_and_oom_as_hard_failures():
    results = [("a", True, [], Scored(0.0, {"reason": "timeout"})), ("b", True, [], Scored(0.0, {"reason": "oom"}))]
    assert G.code_summary(results)["hard_fail_ids"] == ["a", "b"]

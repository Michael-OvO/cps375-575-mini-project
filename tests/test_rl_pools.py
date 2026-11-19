import random
from collections import Counter

from pipeline.rl import data_gates as G
from pipeline.rl import pools as P
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
    dicts, ids, n = G.nondeterministic(rows)
    assert dicts == [bad] and ids == ["p2", "p3"] and n == 2


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


def test_code_summary_flaky_hard_failure_is_not_counted():
    results = [("a", True, [], Scored(0.0, {"reason": "fail"})), ("b", True, [], Scored(0.0, {"reason": "timeout"}))]
    s = G.code_summary(results, recheck_pass={"a"})
    assert s["hard_fail_ids"] == ["b"] and s["flaky_ids"] == ["a"] and s["ref_fail_rate"] == 1 / 2
    assert s["ref_fail_ids"] == ["a", "b"]  # flaky references still leave the pool


def _ok_inputs():
    code = G.code_summary([("a", True, [], Scored(1.0, {"reason": "pass"}))])
    router = {"n": 1, "n_prompts": 1, "excluded_prompt_share": 0.0}
    det = {"n_checked": 1, "n_prompts": 1, "nondeterministic_prompt_share": 0.0}
    gates = G.if_gate_summary([_row(i, "a:a", "A clean answer about trees and their shade here.") for i in range(50)])
    return code, router, det, gates


def test_gate_checks_pass_on_good_inputs():
    assert all(G.gate_checks(*_ok_inputs()).values())


def test_gate_checks_fail_closed_on_empty_code():
    code, router, det, gates = _ok_inputs()
    for empty in ([], [("a", True, [], Scored(0.0, {"reason": "harness_unsafe"}))]):
        c = G.gate_checks(G.code_summary(empty), router, det, gates)
        assert not c["code_guard"] and not c["code_ref_fail"]


def test_gate_checks_fail_closed_on_empty_router():
    code, router, det, gates = _ok_inputs()
    assert not G.gate_checks(code, router | {"n": 0}, det, gates)["if_router"]
    assert not G.gate_checks(code, router | {"n_prompts": 0}, det, gates)["if_router"]


def test_gate_checks_fail_closed_on_empty_determinism():
    code, router, det, gates = _ok_inputs()
    assert not G.gate_checks(code, router, det | {"n_checked": 0}, gates)["kwargs_determinism"]
    assert not G.gate_checks(code, router, det | {"n_prompts": 0}, gates)["kwargs_determinism"]


def test_if_gate_fails_closed_on_empty_or_small_types():
    assert not G.if_gate_summary([])["passed"]
    small = [_row(i, "a:a", "A clean answer about trees and their shade here.") for i in range(49)]
    assert not G.if_gate_summary(small)["passed"]  # no type with n >= 50


def test_strata():
    assert P.math_stratum("Algebra", 2) == "Algebra|1-3"
    assert P.math_stratum("Geometry", 5) == "Geometry|5"
    cons = [{"id": "keywords:frequency"}, {"id": "punctuation:no_comma"}]
    assert P.if_stratum(cons) == "keywords|2-3"
    assert P.code_stratum({"id": "oci-1", "domain": "generic"}) == "oci|generic"
    assert P.code_stratum({"id": "ct-1", "category": "combinatorics"}) == "ct|combinatorics"


def test_stub_rule_catches_trivial_tasks():
    weak = {"code": "def f(xs):\n    return []", "tests": ["assert f([]) == []"] * 5}
    strong = {"code": "def f(xs):\n    return sorted(xs)", "tests": ["assert f([2, 1]) == [1, 2]"] * 5}
    assert P.stub_passes(weak) and not P.stub_passes(strong)


def test_select_if_respects_type_cap_and_bins():
    rng = random.Random(0)
    cands = []
    for i in range(3000):
        k = rng.choice([1, 2, 3, 4, 5])
        ids = rng.sample([f"fam{j}:t{j}" for j in range(40)], k)
        cands.append({"id": f"c{i}", "constraints": [{"id": x} for x in ids]})
    out = P.select_if(cands, target=600, rng=random.Random(1))
    per_type = Counter(c["id"] for o in out for c in o["constraints"])
    assert max(per_type.values()) <= 120
    bins = Counter(P.count_bin(len(o["constraints"])) for o in out)
    assert abs(bins["2-3"] / len(out) - 0.60) < 0.05


GOOD_TESTS = [f"assert f({i}) == {i * i + 2}" for i in range(5)]


def _task(**kw):
    # text chosen so that is_code_dev is False (asserted below)
    return {"id": "ct-x", "text": "Return n squared plus two.", "code": "def f(n):\n    return n * n + 2",
            "tests": GOOD_TESTS[:3], "hidden_tests": GOOD_TESTS[3:]} | kw


class _Leak:
    def __init__(self, bad):
        self.bad = bad

    def is_leak(self, text):
        return text == self.bad


def test_code_drop_reason_keeps_a_clean_task():
    assert not P.is_code_dev(_task()["text"])
    assert P.code_drop_reason(_task(), set(), _Leak("other")) is None


def test_code_drop_reason_few_asserts():
    assert P.code_drop_reason(_task(hidden_tests=[]), set(), None) == "few_asserts"


def test_code_drop_reason_unsupported_split():
    t = _task(hidden_tests=GOOD_TESTS[3:4] + ["assert f(9) > 2"])  # a comparison form the parent cannot verify
    assert P.code_drop_reason(t, set(), None) == "unsupported_tests"


def test_code_drop_reason_phase0_reference_fails_and_unsupported_ids():
    assert P.code_drop_reason(_task(), {"ct-x"}, None) == "reference_fails"


def test_code_drop_reason_set_order_flaky():
    for i in P.SET_ORDER_FLAKY:
        assert P.code_drop_reason(_task(id=i), set(), None) == "set_order_flaky"
    assert P.SET_ORDER_FLAKY == {"oci-53cf9feefc60", "oci-f78717d1a20c", "oci-4c408a856586", "oci-586084257469"}


def test_code_drop_reason_leak():
    assert P.code_drop_reason(_task(), set(), _Leak("Return n squared plus two.")) == "leak"


def test_code_drop_reason_guessable_constant():
    cases = [[f"assert f({i})" for i in range(5)],  # all truthy
             [f"assert not f({i})" for i in range(5)],  # all falsy
             [f"assert f({i}) is None" for i in range(5)],
             [f"assert f({i}) == 0" for i in range(5)],
             [f"assert f({i}) == 1" for i in range(4)] + ["assert f(9) is True"]]  # True == 1 and True is True
    for tests in cases:
        assert P.code_drop_reason(_task(tests=tests[:3], hidden_tests=tests[3:]), set(), None) == "guessable_constant"
    mixed = [f"assert f({i}) is True" for i in range(4)] + ["assert f(9) is False"]
    assert P.code_drop_reason(_task(tests=mixed[:3], hidden_tests=mixed[3:]), set(), None) is None


def _prompt_row(pid, kwargs):
    return {"id": pid, "task": f"Write a note about trees {pid}.", "prompt": f"Write a note about trees {pid}.",
            "constraints": [{"id": "keywords:frequency", "kwargs": kwargs}]}


def test_build_if_drops_phase0_nondeterministic_ids_and_dicts(monkeypatch):
    full = {"keyword": "tree", "frequency": 2, "relation": "at least"}
    flagged = {"keyword": "oak", "frequency": 3, "relation": "at least"}  # deterministic, but listed by Phase 0
    files = {"if_rlvr_prompts.jsonl": [_prompt_row("p1", full), _prompt_row("p2", full), _prompt_row("p3", flagged)],
             "if_base_tasks.jsonl": [], "if_dev.jsonl": [], "if_dev_new.jsonl": []}
    monkeypatch.setattr(P.jsonl, "read", lambda path: files[path.name])
    chosen, rep = P.build_if(None, set(), set(), random.Random(0), nondet_ids={"p2"},
                             nondet_dicts=[{"id": "keywords:frequency", "kwargs": flagged}])
    assert [c["id"] for c in chosen] == ["p1"]
    assert rep["drops"]["nondeterministic_phase0"] == 2

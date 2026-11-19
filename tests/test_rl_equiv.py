import pytest

from pipeline.rl import equiv as E


def test_compare_counts_and_mcnemar():
    base = {str(i): i < 50 for i in range(100)}
    new = dict(base)
    for i in range(0, 5):   # 5 regressions
        new[str(i)] = False
    for i in range(50, 52):  # 2 fixes
        new[str(i)] = True
    r = E.compare(base, new)
    assert (r["n"], r["b"], r["c"]) == (100, 5, 2)
    assert r["agreement"] == pytest.approx(0.93)
    assert r["diff"] == pytest.approx(-0.03)
    assert 0.2 < r["p_mcnemar"] < 0.5  # exact two-sided binomial, k=2 of 7


def test_identical_is_p_one():
    base = {"a": True, "b": False}
    assert E.compare(base, dict(base))["p_mcnemar"] == 1.0


def _gate_setup(tmp_path, monkeypatch, rows):
    from pipeline import jsonl

    base = {"a": True, "b": False, "c": True}
    monkeypatch.setattr(E, "load_v4_items", lambda t: base)
    p = tmp_path / "items.jsonl"
    jsonl.write(p, rows)
    return p


def test_gate_requires_full_coverage(tmp_path, monkeypatch):
    rows = [{"task": t, "id": i, "correct": True} for t in ("math", "code", "if") for i in ("a", "b")]
    with pytest.raises(AssertionError, match="covers 2 of 3"):
        E.gate(_gate_setup(tmp_path, monkeypatch, rows))
    with pytest.raises(AssertionError, match="code: items file covers 0 of 3"):
        E.gate(_gate_setup(tmp_path, monkeypatch, [{"task": "math", "id": i, "correct": True} for i in "abc"]))


def test_gate_min_agree(tmp_path, monkeypatch):
    # one flip of three per task: agreement 2/3, p = 1.0
    rows = [{"task": t, "id": i, "correct": c} for t in ("math", "code", "if")
            for i, c in (("a", True), ("b", True), ("c", True))]
    p = _gate_setup(tmp_path, monkeypatch, rows)
    assert E.gate(p, min_agree=0.6)["passed"] is True
    assert E.gate(p)["passed"] is False

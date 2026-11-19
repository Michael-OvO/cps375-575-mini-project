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

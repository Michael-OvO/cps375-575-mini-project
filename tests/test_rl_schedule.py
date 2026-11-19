import pytest

from pipeline.rl import calibrate as C


def _row(stratum, norms, greedy=1.0, lens=None, task="math"):
    return {"task": task, "stratum": stratum, "norms": norms, "rewards": norms, "greedy": greedy,
            "lens": lens or [100] * len(norms), "trunc": [False] * len(norms),
            "reasons": ["pass"] * len(norms), "vis": [None] * len(norms)}


def test_stratum_stats_shrinkage_and_informative():
    rows = [_row("A|4", [1, 0, 1, 0, 1, 0, 1, 0], greedy=0.0), _row("A|4", [1] * 8)]
    st = C.stratum_stats(rows)["A|4"]
    assert st["n"] == 2
    assert st["p_hat"] == pytest.approx((12 + 1) / (16 + 2))
    assert st["informative"] == pytest.approx(0.5)
    assert st["greedy_fail"] == pytest.approx(0.5)


def test_domain_baseline_s_and_l():
    rows = [_row("A|4", [1, 0, 1, 0, 1, 0, 1, 0], lens=[200] * 8), _row("A|4", [1] * 8, lens=[100] * 8)]
    b = C.domain_baseline(rows)["math"]
    assert b["S"] == pytest.approx(0.5)  # only the non-constant group counts
    assert b["L"] == pytest.approx(150.0)

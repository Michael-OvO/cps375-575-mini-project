import random

import pytest

from pipeline.rl import calibrate as C
from pipeline.rl import schedule as S


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


def test_stratum_weights_exclude_uninformative_and_prefer_greedy_fails():
    strata = {"a|4": {"n": 100, "e_pq": 0.2, "greedy_fail": 0.5, "informative": 0.6, "p_hat": 0.5},
              "b|4": {"n": 100, "e_pq": 0.2, "greedy_fail": 0.0, "informative": 0.6, "p_hat": 0.5},
              "c|4": {"n": 100, "e_pq": 0.2, "greedy_fail": 0.5, "informative": 0.1, "p_hat": 0.5}}
    w = S.stratum_weights(strata, "math")
    assert w["c|4"] == 0 and w["a|4"] == pytest.approx(1.5 * w["b|4"])


def test_task_weights_move_at_most_ten_points():
    summary = {"strata": {}, "informative": {"if": 0.9, "math": 0.1, "code": 0.1}}
    w = S.task_weights(summary, {"if": 0.5, "math": 0.3, "code": 0.3}, {"if": 1.0, "math": 1.0, "code": 1.0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["if"] <= 0.50 + 1e-9 and w["math"] >= 0.25 - 1e-9 and w["code"] >= 0.15 - 1e-9


def test_groups_per_step_sum_to_64():
    g = S.groups_per_step({"if": 0.4, "math": 0.35, "code": 0.25})
    assert sum(g.values()) == 64 and g == {"if": 26, "math": 22, "code": 16}


def test_step_usd_counts_prefill_sample_and_train():
    one = S.step_usd({"math": 1}, {"math": {"L": 1000}}, G=8)
    raw = (0.30 * 300 + 0.81 * 8 * 1000 + 1.46 * 8 * (300 + 1000)) / 1e6
    assert one == pytest.approx(S.MARGIN * raw)


def test_build_schedule_no_repeats():
    pools = {"math": [{"id": f"m{i}", "stratum": "a|4"} for i in range(200)],
             "if": [{"id": f"i{i}", "stratum": "x|1"} for i in range(200)],
             "code": [{"id": f"c{i}", "stratum": "ct|y"} for i in range(200)]}
    st = {"a|4": {"n": 1, "e_pq": 0.2, "greedy_fail": 0.1, "informative": 0.5, "p_hat": 0.5},
          "x|1": {"n": 1, "e_pq": 0.2, "greedy_fail": 0.1, "informative": 0.5, "p_hat": 0.5},
          "ct|y": {"n": 1, "e_pq": 0.2, "greedy_fail": 0.1, "informative": 0.5, "p_hat": 0.5}}
    sched = S.build(pools, {"strata": st}, {"if": 10, "math": 5, "code": 3}, steps=5, rng=random.Random(0))
    flat = [i for step in sched for i in step]
    assert len(sched) == 5 and len(flat) == len(set(flat)) == 90

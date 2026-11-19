from pipeline.rl import monitor as M

BASE = {"math": {"mean_len": 500, "trunc": 0.02, "gate": 0.0, "vphf": 0.0},
        "if": {"mean_len": 300, "trunc": 0.01, "gate": 0.02, "vphf": 0.0},
        "code": {"mean_len": 200, "trunc": 0.0, "gate": 0.01, "vphf": 0.05}}


def _rows(n, **over):
    rows = []
    for s in range(n):
        r = {"step": s, "kl_sample_train": 0.001}
        for d, b in BASE.items():
            r |= {f"{d}/len": b["mean_len"], f"{d}/trunc": b["trunc"], f"{d}/gate": b["gate"],
                  f"{d}/vphf": b["vphf"], f"{d}/reward": 0.5}
        r |= over
        rows.append(r)
    return rows


def test_quiet_run_has_no_pauses_or_warnings():
    assert M.train_checks(_rows(40), BASE) == ([], [])


def test_length_blowup_warns_in_warmup_then_pauses():
    p, w = M.train_checks(_rows(12, **{"math/len": 1200}), BASE)
    assert not p and any("math/len" in x for x in w)
    p, w = M.train_checks(_rows(25, **{"math/len": 1200}), BASE)
    assert any("math/len" in x for x in p)


def test_ratio_kl_pauses_even_in_warmup():
    p, _ = M.train_checks(_rows(12, kl_sample_train=0.02), BASE)
    assert any("kl_sample_train" in x for x in p)


def test_gate_and_vphf_only_warn():
    p, w = M.train_checks(_rows(40, **{"if/gate": 0.08, "code/vphf": 0.10}), BASE)
    assert not p and any("if/gate" in x for x in w) and any("code/vphf" in x for x in w)


def test_falling_reward_only_warns():
    rows = _rows(40)
    for i, r in enumerate(rows):
        r["if/reward"] = 0.8 - 0.01 * i
    p, w = M.train_checks(rows, BASE)
    assert not p and any("if/reward" in x for x in w)


def test_kl_to_v4_warns():
    assert M.kl_warning(0.06) and not M.kl_warning(0.01)


def _items(task, wrong):
    return [{"task": task, "id": f"{task}{i}", "correct": i not in wrong} for i in range(100)]


def _ev(wrong_math=(), trap=0.0):
    return {"trap_hack_rate": trap, "items": _items("math", set(wrong_math)) + _items("code", set()) + _items("if", set())}


def _patch_v4(monkeypatch):
    monkeypatch.setattr(M.equiv, "load_v4_items", lambda t: {r["id"]: r["correct"] for r in _items(t, set())})


def test_dev_drop_warns_before_step_40_then_pauses(monkeypatch):
    _patch_v4(monkeypatch)
    ev = _ev(wrong_math=range(30))
    p, w = M.eval_checks(ev, 0.0, 30)
    assert not p and any("dev math" in x for x in w)
    p, _ = M.eval_checks(ev, 0.0, 40)
    assert any("dev math" in x for x in p)


def test_trap_hack_pauses_at_any_step(monkeypatch):
    _patch_v4(monkeypatch)
    p, _ = M.eval_checks(_ev(trap=0.25), 0.10, 10)
    assert any("trap" in x for x in p)
    assert M.eval_checks(_ev(trap=0.15), 0.10, 10) == ([], [])


def test_warn_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "RUNS", tmp_path)
    M.warn(["a", "b"], 7)
    M.warn([], 8)
    lines = (tmp_path / "rl1" / "warnings.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"step": 7' in lines[0]


def test_kl_pauses_at_any_step_without_full_window():
    p, _ = M.train_checks(_rows(3, kl_sample_train=0.02), BASE)
    assert any("kl_sample_train" in x for x in p)


def test_kl_latest_row_spike_pauses():
    rows = _rows(12)
    rows[-1]["kl_sample_train"] = 0.05
    p, _ = M.train_checks(rows, BASE)
    assert any("kl_sample_train" in x for x in p)


def test_warmup_boundary_is_zero_based():
    p, w = M.train_checks(_rows(20, **{"math/len": 1200}), BASE)  # last step 19
    assert not p and any("math/len" in x for x in w)
    p, _ = M.train_checks(_rows(21, **{"math/len": 1200}), BASE)  # last step 20
    assert any("math/len" in x for x in p)


def test_missing_key_raises():
    rows = _rows(10)
    for r in rows:
        del r["math/len"]
    import pytest
    with pytest.raises(KeyError):
        M.train_checks(rows, BASE)


def test_eval_requires_full_coverage(monkeypatch):
    import pytest
    _patch_v4(monkeypatch)
    ev = _ev()
    ev["items"] = ev["items"][1:]
    with pytest.raises(AssertionError):
        M.eval_checks(ev, 0.0, 50)

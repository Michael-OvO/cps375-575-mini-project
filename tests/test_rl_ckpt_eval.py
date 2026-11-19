import asyncio

import pytest

from pipeline.rl import ckpt_eval as C


def test_missing_baseline_fails_before_spend(tmp_path, monkeypatch):
    def no_ledger(*a, **k):
        raise AssertionError("ledger touched before the baseline check")

    monkeypatch.setattr(C, "Ledger", no_ledger)
    with pytest.raises(FileNotFoundError, match="baseline eval missing"):
        asyncio.run(C.run("tinker://x", 5, tmp_path / "step0.json"))


def test_step_zero_needs_no_baseline(tmp_path):
    C.check_baseline(0, tmp_path / "missing.json")
    (tmp_path / "b.json").write_text("{}")
    C.check_baseline(3, tmp_path / "b.json")

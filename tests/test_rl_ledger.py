import pytest

from pipeline.rl import ledger as L


def test_cost_uses_list_prices():
    assert L.cost("train", 1_000_000) == pytest.approx(1.46)
    assert L.cost("sample", 2_000_000) == pytest.approx(1.62)
    assert L.cost("prefill", 1_000_000) == pytest.approx(0.30)
    assert L.cost("teacher_prefill", 1_000_000) == pytest.approx(1.65)


def test_record_persists_and_reserve_refuses(tmp_path):
    path = tmp_path / "ledger.jsonl"
    a = L.Ledger(path, cap_usd=2.0)
    a.record("1", "train", 1_000_000, note="replay")
    b = L.Ledger(path, cap_usd=2.0)  # a restart reads the same file
    assert b.total() == pytest.approx(1.46)
    b.reserve(0.5, "ok")
    with pytest.raises(L.CapExceeded):
        b.reserve(0.6, "too much")


def test_billing_sync_dominates_when_higher(tmp_path):
    led = L.Ledger(tmp_path / "l.jsonl", cap_usd=10.0)
    led.record("1", "sample", 1_000_000)  # 0.81 at list price
    led.record_billing(3.0)
    assert led.total() == pytest.approx(3.0)
    led.record_billing(0.5)  # the latest sync counts, but never below the local sum
    assert led.total() == pytest.approx(0.81)


def test_missing_ledger_is_zero(tmp_path):
    assert L.Ledger(tmp_path / "none.jsonl", cap_usd=1.0).total() == 0.0


def test_billing_windows_are_hour_aligned(monkeypatch):
    import sys
    import types
    from datetime import datetime, timedelta, timezone

    calls = []

    class Resp:
        data = [types.SimpleNamespace(estimated_cost_usd=1.0), types.SimpleNamespace(estimated_cost_usd=None)]

    class Fut:
        def result(self):
            return Resp()

    class FakeAdmin:
        def get_billing_usage(self, a, b):
            calls.append((a, b))
            return Fut()

    monkeypatch.setitem(sys.modules, "tinker", types.SimpleNamespace(AdminClient=FakeAdmin))
    start = datetime(2026, 11, 1, 3, 25, 41, 123, tzinfo=timezone.utc)
    end = datetime(2026, 11, 20, 9, 5, tzinfo=timezone.utc)
    total = L.billing_spend(start, end)
    assert len(calls) == 2 and total == pytest.approx(2.0)
    assert calls[0][0] == datetime(2026, 11, 1, 3, tzinfo=timezone.utc)
    assert calls[-1][1] == datetime(2026, 11, 20, 10, tzinfo=timezone.utc)
    for a, b in calls:
        assert a.minute == a.second == a.microsecond == 0
        assert b.minute == b.second == b.microsecond == 0
        assert timedelta(0) < b - a <= timedelta(days=14)

"""Persistent spend ledger for rl1. Every Tinker call in every phase records its cost at list prices, the
Tinker billing API is synced as a second source, and new work that would pass the approved cap is refused."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline import jsonl
from pipeline.config import RUNS

PRICES = {"train": 1.46, "sample": 0.81, "prefill": 0.30, "teacher_prefill": 1.65}  # USD per 1M tokens
LEDGER = RUNS / "rl1" / "ledger.jsonl"
BUDGET = RUNS / "rl1" / "budget.json"


class CapExceeded(RuntimeError):
    pass


def cost(kind: str, tokens: int) -> float:
    return PRICES[kind] * tokens / 1e6


class Ledger:
    def __init__(self, path: Path = LEDGER, cap_usd: float | None = None):
        self.path = Path(path)
        self.cap = float(json.loads(BUDGET.read_text())["cap_usd"] if cap_usd is None else cap_usd)

    def rows(self) -> list[dict]:
        return jsonl.read(self.path) if self.path.exists() else []

    def total(self) -> float:
        rows = self.rows()
        local = sum(r["usd"] for r in rows if r["kind"] != "billing_sync")
        syncs = [r["usd"] for r in rows if r["kind"] == "billing_sync"]
        return max(local, syncs[-1] if syncs else 0.0)

    def reserve(self, usd: float, what: str) -> None:
        spent = self.total()
        if spent + usd > self.cap:
            raise CapExceeded(f"{what}: ${spent:.2f} spent + ${usd:.2f} planned > cap ${self.cap:.2f}")

    def record(self, phase: str, kind: str, tokens: int, note: str = "") -> float:
        usd = cost(kind, tokens)
        jsonl.append(self.path, {"ts": time.time(), "phase": phase, "kind": kind,
                                 "tokens": int(tokens), "usd": usd, "note": note})
        return usd

    def record_billing(self, usd_since_start: float) -> None:
        jsonl.append(self.path, {"ts": time.time(), "phase": "-", "kind": "billing_sync", "tokens": 0,
                                 "usd": float(usd_since_start), "note": "Tinker billing, cumulative since rl1 start"})


def billing_spend(start: datetime, end: datetime | None = None) -> float:
    """Actual Tinker spend in [start, end), summed over the API's 14-day maximum windows."""
    import tinker

    admin = tinker.AdminClient()
    end = end or datetime.now(timezone.utc)
    total, t = 0.0, start
    while t < end:
        u = min(t + timedelta(days=14), end)
        resp = admin.get_billing_usage(t, u).result()
        total += sum((e.estimated_cost_usd or 0.0) for e in resp.data)
        t = u
    return total


def sync_billing(led: Ledger) -> float:
    start = datetime.fromisoformat(json.loads(BUDGET.read_text())["start_utc"])
    usd = billing_spend(start)
    led.record_billing(usd)
    return usd


def main() -> None:
    """CLI for paid commands outside pipeline/rl (dev_eval, proxy_eval, public evals):
    `python -m pipeline.rl.ledger reserve <usd> <what>` before (exits 4 past the cap), `... sync` after."""
    import sys

    led = Ledger()
    if sys.argv[1] == "reserve":
        try:
            led.reserve(float(sys.argv[2]), " ".join(sys.argv[3:]))
        except CapExceeded as e:
            print("CAP:", e)
            sys.exit(4)
        print(f"ok: ${led.total():.2f} spent of ${led.cap:.2f}")
    elif sys.argv[1] == "sync":
        print(f"billing since start: ${sync_billing(led):.2f}; ledger total ${led.total():.2f}")
    else:
        raise SystemExit("usage: python -m pipeline.rl.ledger reserve <usd> <what> | sync")


if __name__ == "__main__":
    main()

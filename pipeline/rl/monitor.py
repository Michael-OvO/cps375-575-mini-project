"""Pause rules (spec §8). Hard pauses mean a bug or real harm: they save state and wait for review. Everything
else is a warning, logged to runs/rl1/warnings.jsonl and read at each review, so early noise never stops the
run. Only the spend cap and three consecutive crashed steps stop the run outright (handled in train_rl)."""

import json
import time

from pipeline.config import RUNS
from pipeline.rl import equiv

WINDOW = 10
WARMUP = 20  # before this step, length/truncation and dev drops warn instead of pausing
DOMAINS = ("math", "if", "code")


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        raise KeyError(key)
    return sum(vals) / len(vals)


TRUNC_MIN_ROWS = 5  # after a reward change: steps needed before the windowed truncation check resumes
TRUNC_BACKSTOP = 0.06  # until then, any single step above this pauses


def _trunc_check(rows: list[dict], d: str, b: dict, late: bool, since: int | None) -> tuple[list[str], list[str]]:
    """Windowed truncation rule. `since` (0-based step of a reward change for this domain) restarts the window:
    steps before it are no evidence about the current reward. Until TRUNC_MIN_ROWS new steps exist, a single step
    above TRUNC_BACKSTOP pauses, so the domain is never unwatched."""
    msg_late = [], []
    if since is None:
        w = rows[-WINDOW:]
    else:
        w = [r for r in rows if r["step"] >= since][-WINDOW:]
        if len(w) < TRUNC_MIN_ROWS:
            last = w[-1][f"{d}/trunc"] if w else None
            if last is not None and last > TRUNC_BACKSTOP:
                return [f"{d}/trunc {last:.3f} > backstop {TRUNC_BACKSTOP} at step {w[-1]['step']} "
                        f"(window restarted at step {since})"], []
            return msg_late
    tr = _mean(w, f"{d}/trunc")
    if tr > 2 * b["trunc"] + 0.02:
        msg = f"{d}/trunc {tr:.3f} > 2x baseline {b['trunc']:.3f} + 0.02"
        return ([msg], []) if late else ([], [msg])
    return msg_late


def train_checks(rows: list[dict], base: dict, trunc_since: dict[str, int] | None = None
                 ) -> tuple[list[str], list[str]]:
    trunc_since = trunc_since or {}
    pauses, warns = [], []
    # sampler and trainer disagree on logprobs: a datum bug, never noise; any step, no window, no warmup.
    # None: the step had no RL datums (every group constant), so there is nothing to compare.
    if rows and rows[-1]["kl_sample_train"] is not None and rows[-1]["kl_sample_train"] > 0.01:
        pauses.append(f"kl_sample_train {rows[-1]['kl_sample_train']:.4f} > 0.01 at step {rows[-1]['step']}")
    w = rows[-WINDOW:]
    if len(w) < WINDOW:
        return pauses, warns
    late = w[-1]["step"] >= WARMUP  # train_log steps are 0-based (Task 18 logs the step index from 0)
    for d in DOMAINS:
        b = base[d]
        ln = _mean(w, f"{d}/len")
        if ln > 2 * b["mean_len"] and ln - b["mean_len"] > 50:
            (pauses if late else warns).append(f"{d}/len {ln:.0f} > 2x baseline {b['mean_len']:.0f}")
        p, wn = _trunc_check(rows, d, b, late, trunc_since.get(d))
        pauses += p
        warns += wn
        g = _mean(w, f"{d}/gate")
        if g > b["gate"] + 0.03:
            warns.append(f"{d}/gate {g:.3f} > baseline {b['gate']:.3f} + 0.03")
        if d == "code":
            v = _mean(w, "code/vphf")
            if v > b["vphf"] + 0.03:
                warns.append(f"code/vphf {v:.3f} > baseline {b['vphf']:.3f} + 0.03")
        if len(rows) >= 30:
            a, m, z = (_mean(rows[-30:-20], f"{d}/reward"), _mean(rows[-20:-10], f"{d}/reward"),
                       _mean(rows[-10:], f"{d}/reward"))
            if z < m < a and a - z > 0.02:
                warns.append(f"{d}/reward falling over 30 steps ({a:.3f} -> {z:.3f})")
    return pauses, warns


def kl_warning(kl_v4: float) -> list[str]:
    return [f"KL to v4 {kl_v4:.4f} > 0.05 nats/token"] if kl_v4 > 0.05 else []


def eval_checks(ev: dict, base_trap: float, step: int) -> tuple[list[str], list[str]]:
    pauses, warns = [], []
    new = equiv.per_task(ev["items"])
    for t in ("math", "code", "if"):
        base = equiv.load_v4_items(t)
        shared = len(set(base) & set(new[t]))
        assert shared == len(base), f"{t}: eval items cover {shared} of {len(base)} v4 items"
        r = equiv.compare(base, new[t])
        if r["diff"] < -2 * r["se"]:
            (pauses if step >= WARMUP * 2 else warns).append(
                f"dev {t} {r['diff']:+.3f} below v4 by more than 2 SE ({r['se']:.3f})")
    if ev["trap_hack_rate"] > base_trap + 0.10:
        pauses.append(f"trap hack rate {ev['trap_hack_rate']:.2f} > baseline {base_trap:.2f} + 0.10")
    return pauses, warns


def warn(msgs: list[str], step: int) -> None:
    if not msgs:
        return
    path = RUNS / "rl1" / "warnings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for m in msgs:
            f.write(json.dumps({"step": step, "time": time.time(), "warning": m}) + "\n")
    print("WARN:", msgs)

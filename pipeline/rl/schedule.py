"""Selection (spec §6.2): stratum weights from calibration, task weights from the prior moved at most
+-10 points toward informative x dev-fail / cost, then the whole prompt schedule drawn up front without repeats."""

import argparse
import json
import random

from pipeline import jsonl
from pipeline.config import RUNS, SEED
from pipeline.rl.ledger import cost

PRIOR = {"if": 0.40, "math": 0.35, "code": 0.25}
MAX_MOVE = 0.10
MIN_INFORMATIVE = 0.25
CAPS = {"math": 4096, "code": 4096, "if": 1024}
SMOKE_STEPS = 3
PROMPT_TOKENS = 300  # planning estimate per prompt; the ledger records the real tokens
MARGIN = 1.25 * 1.10  # 25% general margin, 10% for gated SFT


def step_usd(per_step: dict[str, float], base: dict, G: int = 8) -> float:
    """Estimated cost of one step: prompt prefill once per group, G sampled completions, and training on
    G x (prompt + completion) tokens, at ledger list prices. The one estimator used for budgets and reservations."""
    usd = 0.0
    for t, n in per_step.items():
        L = base[t]["L"]
        usd += n * (cost("prefill", PROMPT_TOKENS) + cost("sample", G * L) + cost("train", G * (PROMPT_TOKENS + L)))
    return MARGIN * usd


def stratum_weights(strata: dict, task: str) -> dict[str, float]:
    out = {}
    for s, st in strata.items():
        if st["informative"] < MIN_INFORMATIVE:
            out[s] = 0.0
            continue
        w = st["n"] * st["e_pq"] * (1 + st["greedy_fail"])
        if task == "if" and st["p_hat"] > 0.6:
            w *= 0.5
        out[s] = w
    return out


def task_weights(summary: dict, dev_fail: dict[str, float], cost_per_group: dict[str, float]) -> dict[str, float]:
    raw = {t: summary["informative"][t] * dev_fail[t] / cost_per_group[t] for t in PRIOR}
    tot = sum(raw.values()) or 1.0
    target = {t: v / tot for t, v in raw.items()}
    lo = {t: PRIOR[t] - MAX_MOVE for t in PRIOR}
    hi = {t: PRIOR[t] + MAX_MOVE for t in PRIOR}
    w = {t: min(hi[t], max(lo[t], target[t])) for t in PRIOR}
    gap = 1.0 - sum(w.values())  # restore the sum inside the +-10 bounds (plain renormalizing would break them)
    room = {t: (hi[t] - w[t]) if gap > 0 else (w[t] - lo[t]) for t in PRIOR}
    total_room = sum(room.values())
    if total_room > 0:
        w = {t: w[t] + gap * room[t] / total_room for t in PRIOR}
    return w


def groups_per_step(weights: dict[str, float], groups: int = 64) -> dict[str, int]:
    exact = {t: w * groups for t, w in weights.items()}
    out = {t: int(v) for t, v in exact.items()}
    for t in sorted(exact, key=lambda t: exact[t] - out[t], reverse=True)[: groups - sum(out.values())]:
        out[t] += 1
    return out


def build(pools: dict[str, list[dict]], summary: dict, per_step: dict[str, int], steps: int,
          rng: random.Random, exclude: frozenset = frozenset()) -> list[list[str]]:
    sched = [[] for _ in range(steps)]
    for task, k in per_step.items():
        by: dict[str, list[str]] = {}
        for r in pools[task]:
            if r["id"] not in exclude:
                by.setdefault(r["stratum"], []).append(r["id"])
        for ids in by.values():
            rng.shuffle(ids)
        # n_s is the pool size of the stratum; strata absent from calibration are not scheduled (spec §6.2)
        w = stratum_weights({s: summary["strata"][s] | {"n": len(by[s])} for s in by if s in summary["strata"]}, task)
        for step in range(steps):
            for _ in range(k):
                live = [s for s in by if by[s] and w.get(s, 0) > 0]
                if not live:
                    raise RuntimeError(f"{task}: pool exhausted at step {step}")
                s = rng.choices(live, weights=[w[x] for x in live])[0]
                sched[step].append(by[s].pop())
    return sched


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--budget", type=float, help="USD for main-run steps; steps = budget / step_usd")
    g.add_argument("--steps", type=int, help="explicit number of main-run steps")
    p.add_argument("--summary", default=str(RUNS / "rl1" / "calibration" / "summary.json"))
    a = p.parse_args()
    summary = json.loads(open(a.summary).read())
    pools = {t: jsonl.read(RUNS / "rl1" / "pools" / f"{t}.jsonl") for t in PRIOR}
    cal = jsonl.read(RUNS / "rl1" / "calibration" / "items.jsonl")
    summary["informative"] = {t: sum(max(r["norms"]) - min(r["norms"]) > 1e-9 for r in cal if r["task"] == t)
                              / max(1, sum(r["task"] == t for r in cal)) for t in PRIOR}
    v4dev = json.loads((RUNS / "v4" / "dev.json").read_text())
    if_keys = [k for k in ("ifeval_seen", "ifeval_unseen", "ifeval_unseen_new") if k in v4dev]
    dev_fail = {"math": 1 - v4dev["math"], "code": 1 - v4dev["code"],
                "if": 1 - sum(v4dev[k] for k in if_keys) / len(if_keys)}
    base = summary["baseline"]
    per_group = {t: step_usd({t: 1}, base) for t in PRIOR}
    weights = task_weights(summary, dev_fail, per_group)
    per_step = groups_per_step(weights)
    usd = step_usd(per_step, base)
    steps = a.steps if a.steps is not None else int(a.budget / usd)
    cfg_path = RUNS / "rl1" / "config.json"
    old = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    sched = build(pools, summary, per_step, steps + SMOKE_STEPS, random.Random(SEED))  # smoke uses the tail
    (RUNS / "rl1" / "schedule.json").write_text(json.dumps(sched))
    v4r_state = [r for r in jsonl.read(RUNS / "v4r" / "train_log.jsonl") if "final_state_path" in r][-1]
    cfg = old | {"lr": 2e-5, "G": 8, "groups": 64, "rank": 32, "caps": CAPS,
                 "start_state": v4r_state["final_state_path"],
                 "v4_sampler": json.loads((RUNS / "v4" / "checkpoint.json").read_text())["checkpoint_path"],
                 "teacher": old.get("teacher", {"on": False, "lam": 0.0}),
                 "gated_sft": old.get("gated_sft", {"on": True, "mu": 0.5, "max_per_step": 8}), "save_every": 10,
                 "ttl_seconds": 21 * 24 * 3600, "task_weights": weights, "groups_per_step": per_step,
                 "ema_init": old.get("ema_init", {t: [base[t]["S"], base[t]["L"]] for t in PRIOR}),
                 "baseline": old.get("baseline", base), "steps": steps}
    cfg_path.write_text(json.dumps(cfg, indent=1))
    print(json.dumps({"task_weights": weights, "groups_per_step": per_step, "usd_per_step": round(usd, 3),
                      "steps": steps}, indent=1))


if __name__ == "__main__":
    main()

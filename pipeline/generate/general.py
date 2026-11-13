"""General chat slice: 2,000 single-turn conversations from the Tulu-3 SFT mixture (train split).

Keeps everyday English chat: skips subsets aimed at the graded skills and the least chat-like
subsets, caps any one source at 40%, and samples uniformly within each source.

Usage: python -m pipeline.generate.general
"""

import os
import random
import sys
from collections import Counter
from typing import Iterable

from pipeline import jsonl
from pipeline.config import DATA, SEED

OUT = DATA / "general.jsonl"
N = 2000
MAX_SOURCE_SHARE = 0.4
SCAN = 1_000_000  # covers the whole train split (about 939k rows); shards are source-ordered
# Skip Tulu subsets that target the three graded skills (those come from our own verified data)
# and the least chat-like ones (flan, cot, table_gpt, sciriff; wildjailbreak and wildguard would otherwise
# make refusal data over half the slice; coconot is mostly refusals too).
SKIP = ("math", "code", "persona", "ifdata", "numina", "gsm", "flan", "cot",
        "table_gpt", "sciriff", "wildjailbreak", "wildguard", "coconot")
MIN_ASCII = 0.9


def _qualifies(row: dict) -> dict | None:
    source = str(row.get("source", "")).lower()
    msgs = row.get("messages") or []
    if any(s in source for s in SKIP) or len(msgs) != 2:
        return None
    user, assistant = msgs
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    u, a = user.get("content") or "", assistant.get("content") or ""
    if not (20 <= len(a.split()) <= 500) or len(u.split()) > 300:
        return None
    text = u + a
    if not text or sum(c.isascii() for c in text) / len(text) < MIN_ASCII:
        return None
    return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}],
            "source": f"tulu-3:{row.get('source', 'unknown')}"}


def _allocate(counts: dict[str, int], caps: dict[str, int], n: int) -> dict[str, int]:
    """Split n across sources in proportion to counts, never exceeding each source's cap."""
    alloc: dict[str, int] = {}
    active = set(counts)
    remaining = n
    while active:
        total = sum(counts[s] for s in active)
        over = [s for s in active if remaining * counts[s] / total >= caps[s]]
        if not over:
            break
        for s in over:
            alloc[s] = caps[s]
            remaining -= caps[s]
            active.remove(s)
    if active:
        total = sum(counts[s] for s in active)
        exact = {s: remaining * counts[s] / total for s in active}
        for s in active:
            alloc[s] = min(int(exact[s]), caps[s])
        leftover = n - sum(alloc.values())
        for s in sorted(active, key=lambda s: exact[s] - int(exact[s]), reverse=True):
            if leftover <= 0:
                break
            if alloc[s] < caps[s]:
                alloc[s] += 1
                leftover -= 1
    return alloc


def select(rows: Iterable[dict], n: int, rng: random.Random, max_share: float = MAX_SOURCE_SHARE,
           limit: int | None = None) -> tuple[list[dict], dict]:
    """Pick n conversations: uniform within each source, no source above max_share of n.

    Per-source reservoir sampling over qualifying rows only. Returns (rows with ids, stats).
    """
    cap = int(n * max_share)
    reservoirs: dict[str, list[dict]] = {}
    counts: Counter = Counter()
    scanned = 0
    for row in rows:
        scanned += 1
        if limit is not None and scanned > limit:
            scanned -= 1
            break
        item = _qualifies(row)
        if item is None:
            continue
        src = item["source"]
        counts[src] += 1
        res = reservoirs.setdefault(src, [])
        if len(res) < cap:
            res.append(item)
        else:
            j = rng.randrange(counts[src])
            if j < cap:
                res[j] = item
    caps = {s: len(r) for s, r in reservoirs.items()}
    alloc = _allocate(dict(counts), caps, n)
    kept: list[dict] = []
    for src in sorted(alloc):
        res = reservoirs[src][:]
        rng.shuffle(res)  # reservoir order is not random in the first `cap` slots
        kept.extend(res[: alloc[src]])
    rng.shuffle(kept)
    for i, item in enumerate(kept):
        item["id"] = f"gen-{i}"
    stats = {"scanned": scanned, "qualified": sum(counts.values()),
             "kept": len(kept), "per_source": dict(Counter(k["source"] for k in kept))}
    return kept, stats


def main() -> None:
    from datasets import load_dataset

    rng = random.Random(SEED)
    stream = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    stream = stream.shuffle(seed=SEED, buffer_size=10_000)
    kept, stats = select(stream, N, rng, limit=SCAN)
    jsonl.write(OUT, kept)
    print(f"wrote {len(kept)} conversations to {OUT}")
    print(f"scanned={stats['scanned']} qualified={stats['qualified']} kept={stats['kept']}")
    for src, c in sorted(stats["per_source"].items(), key=lambda x: -x[1]):
        print(f"  {c:5d} {src}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # Hugging Face streaming threads keep the interpreter alive after main() returns

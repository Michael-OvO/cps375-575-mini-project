"""IF pool from the IF-RLVR training prompts (allenai/IF_multi_constraints_upto5, train split, ODC-BY-1.0).

Why: 217 of v3's 284 IF dev failures are on constraint types it never trained on, and the IF-RLVR paper finds
more and more varied training constraints is what generalizes. This source adds 29 IFBench-train types with
upstream checker code (vendored in pipeline.constraints.ifevalg). We keep its prompts and constraints, and
re-answer with our own teacher under strict checking (the dataset ships no responses).

pool:    source filter -> drop held-out types -> split base task from constraint text -> dedup:
         1. exact (normalised base task), 2. near-duplicate within the pool (MinHash LSH on word 5-grams),
         3. eval sets (MATH test, IFEval, MBPP all splits, HumanEval) on the base task,
         4. our own IF dev sets, discovery probes and existing IF base tasks (keeps dev and probes clean).
         Every drop is counted in data/if_rlvr_prompts.jsonl.report.json.
respond: teacher answers, kept only if all constraints pass (upstream IFEval-style strict check).

Usage:
    ALLOW_TEST_OVERLAP_CHECK=1 python -m pipeline.generate.if_rlvr pool
    python -m pipeline.generate.if_rlvr respond --teacher codex --limit 200
"""

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
from collections import Counter

from pipeline import jsonl, leakage, teachers
from pipeline.config import DATA, GPT_MODEL, SEED
from pipeline.constraints import ifevalg
from pipeline.constraints.sample import STANDARD_IDS
from pipeline.generate import ifeval as gif
from pipeline.teachers.runner import run_many

DATASET = "allenai/IF_multi_constraints_upto5"
PROMPTS = DATA / "if_rlvr_prompts.jsonl"
# Jailbreak / safety prompts: teachers refuse or hedge, and they teach nothing the graders reward.
DROP_SOURCES = ("wildjailbreak", "wildguardmix")
# Closest to our dev-only types (x:paragraph_sentences, x:nth_sentence_start): never trained on, so the dev
# "unseen types" score stays a real generalization measure.
HELD_OUT = frozenset({"count:counting_composition", "first_word:first_word_sent",
                      "keywords:keyword_specific_position"})
NEAR_DUP_JACCARD = 0.6
_WORD = re.compile(r"\w+", re.UNICODE)


def norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def source_of(key: str) -> str:
    """Dataset key "<source>_<n>" -> source; bare ids (no underscore) are grouped as "other"."""
    return key.rsplit("_", 1)[0] if "_" in key else "other"


def base_task(prompt: str, constraint_field: str) -> str | None:
    """The prompt without its appended constraint sentences; None if a constraint text is not found."""
    task = prompt
    for c in (t.strip() for t in constraint_field.split("\t") if t.strip()):
        if c not in task:
            return None
        task = task.replace(c, " ")
    task = re.sub(r"[ \t]+", " ", task).strip()
    return task or None


# ---------------------------------------------------------------- near-duplicate detection (MinHash LSH)

_PERMS, _BANDS = 64, 16  # 16 bands x 4 rows: pairs above ~0.6 Jaccard collide with high probability
_MASK = (1 << 61) - 1
_rng = random.Random(SEED)
_COEF = [(_rng.randrange(1, _MASK), _rng.randrange(0, _MASK)) for _ in range(_PERMS)]


def shingles(text: str, n: int = 5) -> set[int]:
    toks = norm(text).split()
    grams = [" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))]
    return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams}


def minhash(sh: set[int]) -> tuple[int, ...]:
    return tuple(min((a * x + b) & _MASK for x in sh) for a, b in _COEF)


class NearDup:
    """Incremental near-duplicate index: add() returns False if a kept text is already that similar."""

    def __init__(self, threshold: float = NEAR_DUP_JACCARD):
        self.threshold, self.buckets, self.kept = threshold, {}, []

    def add(self, text: str) -> bool:
        sh = shingles(text)
        sig = minhash(sh)
        rows = _PERMS // _BANDS
        keys = [(b, sig[b * rows:(b + 1) * rows]) for b in range(_BANDS)]
        for cand in {i for k in keys for i in self.buckets.get(k, ())}:
            other = self.kept[cand]
            if len(sh & other) / len(sh | other) >= self.threshold:
                return False
        idx = len(self.kept)
        self.kept.append(sh)
        for k in keys:
            self.buckets.setdefault(k, []).append(idx)
        return True


# ---------------------------------------------------------------- pool

def _own_texts() -> list[str]:
    """Our dev sets, probes and existing IF base tasks: nothing in the new pool may duplicate them."""
    texts = []
    for name in ("if_dev.jsonl", "if_dev_new.jsonl", "discover/probes_if.jsonl", "if_base_tasks.jsonl"):
        path = DATA / name
        if path.exists():
            for r in jsonl.read(path):
                texts += [r[k] for k in ("task", "prompt", "text") if isinstance(r.get(k), str)]
    return texts


def _humaneval_texts() -> list[str]:
    from datasets import load_dataset
    return list(load_dataset("openai/openai_humaneval", split="test")["prompt"])


def build_pool() -> dict:
    from datasets import load_dataset
    eval_filter = leakage.load_overlap_filter()
    if eval_filter is None:
        raise SystemExit("set ALLOW_TEST_OVERLAP_CHECK=1: the pool must be checked against the eval sets")
    humaneval = leakage.OverlapFilter(_humaneval_texts())
    own = leakage.OverlapFilter(_own_texts())
    rows = list(load_dataset(DATASET, split="train"))
    random.Random(SEED).shuffle(rows)  # dedup keeps the first copy: make "first" seed-random, not source order
    drops, kept, seen, near = Counter(), [], set(), NearDup()
    for r in rows:
        key = r["key"]
        if any(s in key for s in DROP_SOURCES):
            drops["source_safety"] += 1
            continue
        ids, kwargs = ifevalg.parse_ground_truth(r["ground_truth"])
        if set(ids) & HELD_OUT:
            drops["held_out_type"] += 1
            continue
        if any(i not in ifevalg.REGISTRY for i in ids):
            drops["unknown_type"] += 1
            continue
        prompt = r["messages"][0]["content"].strip()
        task = base_task(prompt, r["constraint"])
        if task is None or len(task.split()) < 4:
            drops["no_base_task"] += 1
            continue
        if norm(task) in seen:
            drops["dup_exact"] += 1
            continue
        # Base task only, as in select.py: constraint sentences are the checkers' template text, shared with
        # every IFEval prompt of that type, and would flag most rows as false positives.
        if eval_filter.is_leak(task) or humaneval.is_leak(task):
            drops["eval_overlap"] += 1
            continue
        if own.is_leak(task):
            drops["own_dev_probe_overlap"] += 1
            continue
        if not near.add(task):
            drops["dup_near"] += 1
            continue
        seen.add(norm(task))
        kept.append({"id": "rl-" + hashlib.sha1(key.encode()).hexdigest()[:12], "key": key,
                     "source": source_of(key), "task": task, "prompt": prompt,
                     "constraints": [{"id": i, "kwargs": k} for i, k in zip(ids, kwargs)],
                     "new_types": sorted(set(ids) - set(STANDARD_IDS))})
    jsonl.write(PROMPTS, kept)
    report = {"dataset": DATASET, "split": "train", "rows": len(rows), "kept": len(kept), "drops": dict(drops),
              "held_out_types": sorted(HELD_OUT), "near_dup_jaccard": NEAR_DUP_JACCARD,
              "sources_kept": dict(Counter(k["source"] for k in kept).most_common()),
              "new_type_counts": dict(Counter(t for k in kept for t in k["new_types"]).most_common())}
    (PROMPTS.parent / (PROMPTS.name + ".report.json")).write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- respond

TEACHER = "codex"


def pilot_items(limit: int | None, seed: int = SEED) -> list[dict]:
    """Prompts with at least one new type, balanced so every new type is covered (round-robin by type)."""
    rows = [r for r in jsonl.read(PROMPTS) if r["new_types"]]
    rng = random.Random(seed)
    rng.shuffle(rows)
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(rng.choice(r["new_types"]), []).append(r)
    out, taken = [], set()
    while by_type and (limit is None or len(out) < limit):
        for t in sorted(by_type):
            while by_type[t] and by_type[t][-1]["id"] in taken:
                by_type[t].pop()
            if not by_type[t]:
                del by_type[t]
                continue
            r = by_type[t].pop()
            taken.add(r["id"])
            out.append(dict(r, teacher=TEACHER))
            if limit is not None and len(out) >= limit:
                break
    return out


# Full run: prompts with only IFEval's own types (direct transfer to the graded set) next to prompts with new
# types (generalization to the held-out set). Standard-only quotas lean to 2-3 constraints: single constraints
# pass 94% of the time and teach little, and v1's own IF data already covers them.
FULL_STANDARD = {1: 1200, 2: 2000, 3: 1300, 4: 420, 5: 80}
FULL_NEW = 7000


def full_items(limit: int | None, seed: int = SEED) -> list[dict]:
    """Pilot items first (already answered, so a resume skips them), then the full quotas, interleaved."""
    new = pilot_items(FULL_NEW, seed)
    rng = random.Random(seed + 1)
    std = [r for r in jsonl.read(PROMPTS) if not r["new_types"]]
    rng.shuffle(std)
    picked = []
    for n, quota in FULL_STANDARD.items():
        picked += [r for r in std if len(r["constraints"]) == n][:quota]
    rng.shuffle(picked)
    picked = [dict(r, teacher=TEACHER) for r in picked]
    head, rest = new[:200], new[200:]
    out = list(head)
    for i in range(max(len(rest), len(picked))):  # interleave so a stopped run still has both kinds
        out += rest[i:i + 1] + picked[i:i + 1]
    return out[:limit] if limit else out


ATTEMPTS = gif.ATTEMPTS
FEEDBACK = """{prompt}

---
Your previous answer to the request above was:
<<<
{response}
>>>
It broke these instructions:
{broken}
Write a new, complete answer to the request above that follows every instruction. Output only the answer."""


def describe(iid: str, kwargs: dict | None) -> str:
    """The checker's own instruction text, used to tell the teacher what its last answer broke."""
    checker = ifevalg.REGISTRY[iid](iid)
    return checker.build_description(**{k: v for k, v in (kwargs or {}).items() if v})


async def respond(item: dict) -> list[dict] | None:
    """Up to ATTEMPTS teacher answers; after a miss the next call is told which instructions it broke.

    Only the original prompt and the final passing answer are stored, so the feedback never reaches training.
    Strict checking means extra attempts add only fully passing answers; `attempts` is kept per row so late
    passes can be spot-checked for checker-gaming.
    """
    complete = teachers.get(TEACHER)
    ids = [c["id"] for c in item["constraints"]]
    kwargs = [c["kwargs"] for c in item["constraints"]]
    got, query = False, item["prompt"]
    for attempt in range(ATTEMPTS):
        response = await complete(query, system=gif.SYSTEM, max_tokens=gif.RESPONSE_MAX_TOKENS)
        if not response:
            continue
        got = True
        results = ifevalg.check(response, ids, kwargs, item["prompt"])
        too_long = len(response.split()) > gif.MAX_RESPONSE_WORDS
        if all(results) and not too_long:
            return [{"id": f"if-{item['id']}", "task": item["task"], "prompt": item["prompt"],
                     "constraints": item["constraints"], "response": response, "teacher": TEACHER,
                     "model": GPT_MODEL if TEACHER in ("codex", "gpt") else None,
                     "verifier": "ifevalg", "source": item["source"], "new_types": item["new_types"],
                     "attempts": attempt + 1, "style": "if_rlvr"}]
        broken = [describe(i, k) for i, k, ok in zip(ids, kwargs, results) if not ok]
        if too_long:
            broken.append(f"Keep the answer under {gif.MAX_RESPONSE_WORDS} words.")
        query = FEEDBACK.format(prompt=item["prompt"], response=response, broken="\n".join(f"- {b}" for b in broken))
    return [] if got else None


def retry_items(limit: int | None) -> list[dict]:
    """Pilot items that were attempted but not kept."""
    out = DATA / f"if_rlvr.{TEACHER}.jsonl"
    done = jsonl.done_ids(out)
    kept = {r["id"][3:] for r in jsonl.read(out)}
    return [i for i in pilot_items(limit) if i["id"] in done and i["id"] not in kept]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pool", "respond", "retry", "full"])
    p.add_argument("--teacher", choices=["codex", "claude", "gpt"], default="codex")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--attempts", type=int, default=gif.ATTEMPTS)
    args = p.parse_args()
    if args.command == "pool":
        print(json.dumps(build_pool(), indent=2))
        return
    global TEACHER, ATTEMPTS
    TEACHER, ATTEMPTS = args.teacher, args.attempts
    teachers.check_configured([TEACHER])
    out = DATA / f"if_rlvr.{TEACHER}.jsonl"
    items = full_items(args.limit) if args.command == "full" else pilot_items(args.limit)
    if args.command == "retry":  # failed items again, with feedback, into a separate file
        items = retry_items(args.limit)
        out = DATA / f"if_rlvr.{TEACHER}.retry.jsonl"
    try:
        asyncio.run(run_many(items, respond, out, args.concurrency))
    finally:
        teachers.write_usage(out)


if __name__ == "__main__":
    main()

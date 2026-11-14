"""Cell-targeted training data for the weaknesses found by discovery round 1 (reports/discovery_round1.md).

Each item is one weak cell: Opus writes 8 new problems or tasks for it (the same prompts as the probes), and
each one then gets a training answer in the v2 formats:
- math-cells: a long self-checking solution (v2_teacher.math_long) that must agree with the author's answer;
- code-cells: the reference must pass all its tests, then a plan-first answer (v2_teacher.code_plan).
Candidates that overlap a public test set or a discovery probe are dropped, so the probes stay a clean
measurement of whether the weakness closed.

Usage:
    python -m pipeline.generate.v2_cells math-cells --limit 12
    python -m pipeline.generate.v2_cells code-cells --limit 12
"""

import argparse
import asyncio
import logging
import random

from pipeline import jsonl, teachers
from pipeline.config import DATA, SEED
from pipeline.discover.probes import CODE_WRITE, MATH_WRITE, _json_list, probes_path
from pipeline.discover.taxonomy import describe, facets
from pipeline.generate.v2_teacher import code_plan, math_long
from pipeline.leakage import OverlapFilter, load_overlap_filter
from pipeline.teachers.runner import run_many
from pipeline.verify import code as vc

MATH_CELLS = DATA / "v2_math_cells.jsonl"
CODE_CELLS = DATA / "v2_code_cells.jsonl"
PER_CALL = 8
N_CELLS = 150  # x 8 candidates per task

# Weak facet values from round 1 (v1 failure rate on probes in brackets). Each cell anchors one of them.
WEAK = {
    "math": {
        "answer_type": ["complex number in a+bi form or polar/exponential form",  # 81%
                        "numeral written in a non-decimal base (e.g. 1011_2, 4A_{11})",  # 78%
                        "angle in degrees", "radical expression", "ordered pair or tuple"],  # 69, 65, 60%
        "technique": ["complementary counting", "functional equations",  # 88, 88%
                      "Chinese remainder theorem, modular inverses, and orders / repeating decimal periods",  # 75%
                      "inclusion–exclusion and stars-and-bars / distributions with restrictions", "casework"],  # 72, 67%
        "subject": ["Precalculus", "Counting & Probability"],  # 65, 64%
    },
    "code": {
        "prompt_style": ["ambiguous wording whose intended meaning (1- vs 0-indexing, inclusive vs exclusive bounds, "
                         "nth term convention) is only recoverable from the example asserts"],  # 60%
        "input_regime": ["deeply nested structures",  # 78%
                         "date/time inputs as strings in varied formats (YYYY-MM-DD, DD-MM-YYYY, 12h clock) including "
                         "leap years and month boundaries",  # 75%
                         "ties and equal keys where stable ordering or a specific tie-break (first occurrence, "
                         "smallest value) is required"],  # 54%
        "output_contract": ["nested list or list of lists with a specific inner order (e.g. grouped items, chunks, "
                            "pairs)", "dictionary"],  # 65, 53%
        "topic": ["roman numerals and number words", "math with fractions and gcd", "regular expressions",
                  "bit manipulation", "classic DP problems with named formulations (min cost path in matrix, longest "
                  "common/increasing subsequence, coin change ways, max sum with no adjacent elements, tiling counts)"],
    },
}
MATH_LEVELS = ["3", "4", "4", "5", "5"]


def cells(task: str, n: int) -> list[dict]:
    """n cells, cycling through the weak values. Only the weak value is fixed (plus the level for math and
    the prompt style for code); the writer picks the other facets so that each problem is natural."""
    rng = random.Random(f"{SEED}-v2cells-{task}")
    anchors = [(f, v) for f, vals in WEAK[task].items() for v in vals]
    styles = facets(task).get("prompt_style", [])
    out = []
    for i in range(n):
        f, v = anchors[i % len(anchors)]
        cell = {"level": rng.choice(MATH_LEVELS)} if task == "math" else {"prompt_style": rng.choice(styles)}
        cell[f] = v
        out.append({"id": f"{task[0]}c-{i:03d}", "cell": cell})
    return out


def _probe_filter() -> OverlapFilter:
    texts = [p.get("problem") or p.get("text") for t in ("math", "code") if probes_path(t).exists()
             for p in jsonl.read(probes_path(t)) if not p.get("empty")]
    return OverlapFilter([t for t in texts if t])


async def gen(task: str, item: dict, filters: list) -> list[dict] | None:
    complete = teachers.get("claude")
    prompt = (MATH_WRITE if task == "math" else CODE_WRITE).format(n=PER_CALL, cell=describe(task, item["cell"]))
    cands = _json_list(await complete(prompt, max_tokens=16000), latex=task == "math")
    if not cands:
        return None
    key = "problem" if task == "math" else "text"
    need = ("problem", "answer") if task == "math" else ("text", "code", "tests", "hidden_tests")

    async def one(k: int, c) -> list[dict]:
        if not isinstance(c, dict) or not all(c.get(f) for f in need) or any(f.is_leak(c[key]) for f in filters):
            return []
        cid = f"{item['id']}-{k}"
        if task == "math":
            return await math_long({"id": cid, "problem": c["problem"], "target": str(c["answer"]),
                                    "level": int(item["cell"]["level"]), "source": "v2_cells",
                                    "subject": item["cell"].get("subject")}) or []
        tests, hidden = [str(t) for t in c["tests"]], [str(t) for t in c["hidden_tests"]]
        if not await asyncio.to_thread(vc.run_tests, c["code"], tests + hidden):
            return []
        rows = await code_plan({"id": cid, "text": c["text"], "tests": tests, "hidden_tests": hidden,
                                "category": item["cell"].get("topic")}) or []
        return [dict(r, cell=item["cell"]) for r in rows]

    got = await asyncio.gather(*(one(k, c) for k, c in enumerate(cands[:PER_CALL])))
    return [r for rows in got for r in rows]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["math-cells", "code-cells"])
    p.add_argument("--limit", type=int, default=N_CELLS)
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    teachers.check_configured(["claude"])
    task = a.command.split("-")[0]
    out = MATH_CELLS if task == "math" else CODE_CELLS
    filters = [f for f in (load_overlap_filter(), _probe_filter()) if f is not None]
    if len(filters) < 2:
        logging.warning("test-overlap filter disabled (set ALLOW_TEST_OVERLAP_CHECK=1)")

    async def fn(item):
        return await gen(task, item, filters)

    try:
        asyncio.run(run_many(cells(task, a.limit), fn, out, a.concurrency))
    finally:
        teachers.write_usage(out)


if __name__ == "__main__":
    main()

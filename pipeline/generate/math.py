"""Math pools: verified solutions to MATH train problems and to new teacher-written problems.

Usage:
    python -m pipeline.generate.math dev
    python -m pipeline.generate.math solve-train --teachers claude gpt
    python -m pipeline.generate.math new-problems --teachers claude gpt --calls 8000
    python -m pipeline.generate.math solve-new --teachers claude gpt
"""

import argparse
import asyncio
import logging
import random

from datasets import load_dataset

from pipeline import jsonl, teachers
from pipeline.config import DATA, SEED, is_dev, is_vault
from pipeline.teachers.runner import run_many
from pipeline.verify import math as vm

MATH_DATASET = "DigitalLearningGmbH/MATH-lighteval"
DEV_PER_MILLE = 67  # about 500 of 7,500

TRAIN_SOLUTIONS = DATA / "math_train_solutions.jsonl"
NEW_PROBLEMS = DATA / "math_new_problems.jsonl"
NEW_SOLUTIONS = DATA / "math_new_solutions.jsonl"
DEV = DATA / "math_dev.jsonl"

SOLVE_PROMPT = """Solve the following competition math problem.

Write a concise step-by-step solution that a strong high-school student could follow. Show every \
necessary step and skip nothing essential, but add no filler, no restating of the problem, and no \
commentary. Aim for under 300 words.

Finish with a final line of the form
ANSWER: <answer>
where <answer> is only the final answer in simplest form (LaTeX where needed), with no \\boxed, \
no units, and no explanation.

Problem:
{problem}"""

WRITE_PROMPT = """Below are three competition math problems on {subject}, difficulty level {level} \
on a 1 (easiest) to 5 (hardest) scale.

{seeds}

Write 5 new, original problems in the same subject, style and difficulty. Requirements:
- Each problem must have exactly one final answer (a number, expression, interval, or ordered tuple).
- Do not reuse the numbers, setups or wording of the examples; vary the underlying idea.
- Each problem must be self-contained and solvable without a calculator.
- Use LaTeX for mathematics.

Return only a JSON array of 5 strings, one problem per string, with no solutions."""


def load_train() -> list[dict]:
    rows = load_dataset(MATH_DATASET, split="train")
    out = []
    for idx, row in enumerate(rows):
        target = vm.boxed_target(row["solution"])
        if not target:
            continue
        item_id = f"mt-{idx}"
        out.append({
            "id": item_id,
            "problem": row["problem"],
            "target": target,
            "level": int(str(row["level"]).split()[-1]) if str(row["level"])[-1].isdigit() else 3,
            "subject": row["type"],
            "dev": is_dev(item_id, DEV_PER_MILLE),
        })
    return out


def split_solution(text: str) -> tuple[str, str] | None:
    answer = vm.answer_of(text or "")
    body = vm.body_of(text or "")
    if not answer or len(body.split()) < 3:
        return None
    return body, answer


async def _solve(problem: str, teacher: str, effort: str = "medium") -> tuple[bool, tuple[str, str] | None]:
    """Return (responded, parsed). responded is False when the teacher gave no text at all."""
    text = await teachers.get(teacher)(SOLVE_PROMPT.format(problem=problem), effort=effort)
    return bool(text), split_solution(text)


def _row(item: dict, teacher: str, body: str, target: str, source: str) -> dict:
    return {
        "id": f"{item['id']}-{teacher}", "pid": item["id"], "problem": item["problem"],
        "target": target, "solution": body, "teacher": teacher,
        "level": item["level"], "subject": item["subject"], "source": source,
    }


async def solve_known(item: dict, teacher_names: list[str], source: str) -> list[dict] | None:
    """Solve a problem with a known answer; keep each teacher solution that matches it.

    Returns None (retry) only when no teacher responded at all.
    """
    rows = []
    any_response = False
    for name in teacher_names:
        responded, solved = await _solve(item["problem"], name)
        any_response = any_response or responded
        if solved and await vm.equivalent(solved[1], item["target"]):
            rows.append(_row(item, name, solved[0], item["target"], source))
    return rows if any_response else None


async def solve_unknown(item: dict, teacher_names: list[str]) -> list[dict] | None:
    """Solve a new problem. Two teachers agree -> accept. Otherwise a third, high-effort solve decides.

    Returns None (retry) only when every solve call produced no response.
    """
    first, second = teacher_names[0], teacher_names[-1]
    second_label = f"{second}-2" if second == first else second
    ra, a = await _solve(item["problem"], first)
    rb, b = await _solve(item["problem"], second)
    any_response = ra or rb
    solved = [(first, a), (second_label, b)]
    if a and b and await vm.equivalent(a[1], b[1]):
        target = a[1]
    else:
        rc, c = await _solve(item["problem"], first, effort="high")
        any_response = any_response or rc
        solved.append((f"{first}-tiebreak", c))
        target = None
        for cand in (a, b):
            if cand and c and await vm.equivalent(cand[1], c[1]):
                target = c[1]
                break
        if target is None:
            return [] if any_response else None
    rows = []
    for name, sol in solved:
        if sol and await vm.equivalent(sol[1], target):
            rows.append(_row(item, name, sol[0], target, "math_new"))
    return rows


async def write_problems(item: dict) -> list[dict] | None:
    seeds = "\n\n".join(f"Example {i + 1}: {p}" for i, p in enumerate(item["seeds"]))
    prompt = WRITE_PROMPT.format(subject=item["subject"], level=item["level"], seeds=seeds)
    reply = await teachers.get(item["author"])(prompt)
    if not reply:
        return None  # no response: retry on rerun
    problems = teachers.parse_json(reply)
    if not isinstance(problems, list):
        return []
    return [
        {"id": f"{item['id']}-{j}", "problem": p.strip(), "level": item["level"],
         "subject": item["subject"], "author": item["author"]}
        for j, p in enumerate(problems) if isinstance(p, str) and len(p.strip()) > 20
    ]


def _write_calls(train: list[dict], names: list[str], calls: int) -> list[dict]:
    rng = random.Random(SEED)
    by_cell: dict[tuple, list[dict]] = {}
    for row in train:
        by_cell.setdefault((row["subject"], row["level"]), []).append(row)
    cells = [c for c, rows in by_cell.items() if len(rows) >= 3]
    # Harder cells get more calls: weight = level.
    weights = [c[1] for c in cells]
    out = []
    for i in range(calls):
        subject, level = rng.choices(cells, weights=weights)[0]
        seeds = rng.sample(by_cell[(subject, level)], 3)
        out.append({"id": f"mn-{i}", "subject": subject, "level": level,
                    "seeds": [s["problem"] for s in seeds], "author": names[i % len(names)]})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["dev", "solve-train", "new-problems", "solve-new"])
    p.add_argument("--teachers", nargs="+", default=["claude", "gpt"])
    p.add_argument("--calls", type=int, default=8000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    if args.command == "dev":
        dev = [
            {k: r[k] for k in ("id", "problem", "target", "level", "subject")} | {"vault": is_vault(r["id"])}
            for r in load_train() if r["dev"]
        ]
        jsonl.write(DEV, dev)
        print(f"wrote {len(dev)} dev problems to {DEV}")
        return

    train = [r for r in load_train() if not r["dev"]]
    if args.command == "solve-train":
        items = train[: args.limit]
        fn = lambda it: solve_known(it, args.teachers, "math_train")  # noqa: E731
        asyncio.run(run_many(items, fn, TRAIN_SOLUTIONS, args.concurrency))
    elif args.command == "new-problems":
        items = _write_calls(train, args.teachers, args.calls)[: args.limit]
        asyncio.run(run_many(items, write_problems, NEW_PROBLEMS, args.concurrency))
    else:
        items = jsonl.read(NEW_PROBLEMS)[: args.limit]
        fn = lambda it: solve_unknown(it, args.teachers)  # noqa: E731
        asyncio.run(run_many(items, fn, NEW_SOLUTIONS, args.concurrency))


if __name__ == "__main__":
    main()

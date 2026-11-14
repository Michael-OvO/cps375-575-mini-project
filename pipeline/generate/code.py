"""Code pool: short function-writing tasks with executed tests.

Usage:
    python -m pipeline.generate.code write --teachers claude gpt --calls-per-category 40
    python -m pipeline.generate.code hidden --teachers claude gpt
"""

import argparse
import asyncio
import hashlib
import logging
import re

from pipeline import jsonl, teachers
from pipeline.config import DATA
from pipeline.teachers.runner import run_many
from pipeline.verify import code as vc

log = logging.getLogger(__name__)

RAW = DATA / "code_tasks_raw.jsonl"
TASKS = DATA / "code_tasks.jsonl"

CATEGORIES = [
    "string manipulation", "string searching and counting", "string formatting and parsing",
    "list filtering", "list transformation", "list aggregation", "sorting with custom keys",
    "searching in sorted lists", "nested lists and matrices", "tuples", "lists of tuples",
    "dictionaries", "dictionaries of lists", "sets", "number theory", "primes and divisors",
    "digits of integers", "arithmetic and rounding", "sequences and series", "combinatorics",
    "geometry formulas", "bit manipulation", "base conversion", "recursion",
    "simple dynamic programming", "two-pointer techniques", "sliding windows", "stacks and queues",
    "regular expressions", "character classification", "palindromes and anagrams",
    "dates and times arithmetic", "intervals and ranges", "frequency counting", "removing duplicates",
    "merging and zipping collections", "boolean checks and validation", "math with fractions and gcd",
    "text tokenizing and word statistics", "heap and priority selection",
]

# Every write call gets its own (data shape, real-world framing) pair plus a call index, so no two
# calls in a category send the same prompt. 6 x 9 = 54 distinct pairs per category.
SHAPES = [
    "inputs that are short lists or strings, with emphasis on empty and single-element cases",
    "inputs with repeated values, ties or duplicates that the function must handle",
    "inputs mixing negative numbers, zero and positive numbers",
    "nested or paired data such as lists of tuples, dicts of lists or rows of a table",
    "typical input sizes of a few dozen items, where an off-by-one error would show",
    "boundary conditions: first and last positions, minimum and maximum values",
]
FRAMINGS = [
    "a store inventory", "student grades", "server log lines", "text processing",
    "scheduling and calendars", "geometry and measurements", "personal finance",
    "sports scores and rankings", "sensor readings",
]

WRITE_PROMPT = """Write 5 short Python programming tasks in the category: {category}.

Each task is a one-sentence description beginning "Write a function to" or "Write a python function \
to", a correct reference solution, and exactly 3 assert statements that call the function.

Sub-focus for this batch (batch {call} for this category): {shape}. Where it fits naturally, frame \
the tasks around {framing}.

Requirements:
- Difficulty: entry-level to intermediate; solvable in under 20 lines.
- Standard library only. No input(), no printing, no files, no randomness, no network.
{style_rule}
- Avoid the most common textbook exercises for this category (the first few anyone would think \
of); write less obvious tasks instead.
- The asserts must be deterministic and compare with ==.
- Make the 5 tasks clearly different from each other.

Return only a JSON array of 5 objects with keys "text", "code" and "tests" (a list of 3 strings)."""

DETAILED_RULE = """- The description must be short and natural, like a textbook exercise. It does not name the \
function or its parameters: those are fixed by the tests."""

TERSE_RULE = """- Each description is a single sentence of at most 15 words that names what to compute and \
nothing else: no edge-case clauses, no input-format details, no "returning X for Y" clauses. Leave \
the details to the tests: the 3 asserts must carry them, and at least one should exercise an edge \
case (such as an empty or single-element input) where that makes sense. The description does not \
name the function or its parameters: those are fixed by the tests."""

TERSE_FRACTION = 0.7
MAX_TERSE_WORDS = 18  # the prompt asks for 15; small tolerance

HIDDEN_PROMPT = """A Python function is described as follows:

{text}

Its signature is:
{signature}

These example tests show how it is called:
{tests}

Write 5 additional assert statements that a correct implementation must pass. Cover edge cases the \
examples do not. Use only what the description and examples make unambiguous: do not test \
behaviour the description leaves open. Where the description is brief, the example tests define the \
intended behaviour. Each assert must compare with ==.

Return only a JSON array of 5 strings."""


MAX_HIDDEN = 8
_NAME = re.compile(r"def\s+(\w+)")


def function_name(code: str) -> str | None:
    sig = vc.signature_of(code)
    match = _NAME.match(sig) if sig else None
    return match.group(1) if match else None


def calls(test, name: str | None) -> bool:
    """An assert counts only if it calls the solution's function."""
    return (bool(name) and isinstance(test, str) and test.startswith("assert ")
            and re.search(rf"\b{re.escape(name)}\s*\(", test) is not None)


def _valid(task) -> bool:
    return (
        isinstance(task, dict)
        and isinstance(task.get("text"), str)
        and isinstance(task.get("code"), str)
        and isinstance(task.get("tests"), list)
        and len(task["tests"]) == 3
        and all(isinstance(t, str) and t.startswith("assert ") for t in task["tests"])
    )


def style_of(call_id: str) -> str:
    """Deterministic per-call description style: ~70% terse, ~30% detailed."""
    bucket = int(hashlib.md5(call_id.encode()).hexdigest(), 16) % 1000
    return "terse" if bucket < TERSE_FRACTION * 1000 else "detailed"


def write_prompt(item: dict) -> str:
    rule = TERSE_RULE if item.get("style") == "terse" else DETAILED_RULE
    return WRITE_PROMPT.format(category=item["category"], call=item["call"] + 1,
                               shape=item["shape"], framing=item["framing"], style_rule=rule)


def write_items(teacher_names: list[str], calls_per_category: int) -> list[dict]:
    """One item per (category, call), interleaved across categories so a --limit pilot covers many."""
    return [
        {"id": f"ct-{ci}-{k}", "category": cat, "author": teacher_names[(ci + k) % len(teacher_names)],
         "call": k, "style": style_of(f"ct-{ci}-{k}"), "shape": SHAPES[k % len(SHAPES)],
         "framing": FRAMINGS[(k // len(SHAPES) + ci) % len(FRAMINGS)]}
        for k in range(calls_per_category) for ci, cat in enumerate(CATEGORIES)
    ]


async def write_tasks(item: dict) -> list[dict] | None:
    reply = await teachers.get(item["author"])(write_prompt(item))
    if reply is None:  # refusal, truncation or empty: retry on rerun
        return None
    tasks = teachers.parse_json(reply, latex=False)  # strings hold code: \n is a newline
    if not isinstance(tasks, list):
        return []
    rows = []
    too_long = 0
    style = item.get("style", "detailed")
    for j, task in enumerate(tasks):
        if not _valid(task):
            continue
        if style == "terse" and len(task["text"].split()) > MAX_TERSE_WORDS:
            too_long += 1
            continue
        code = vc.extract_code(task["code"]).strip()
        name = function_name(code)
        if not (name and all(calls(t, name) for t in task["tests"])):
            continue  # an assert that never calls the function tests nothing
        if await asyncio.to_thread(vc.run_tests, code, task["tests"]):
            rows.append({"id": f"{item['id']}-{j}", "text": task["text"].strip(), "code": code,
                         "tests": task["tests"], "category": item["category"], "author": item["author"],
                         "style": style})
    if too_long:
        log.info("%s: dropped %d terse task(s) over %d words", item["id"], too_long, MAX_TERSE_WORDS)
    return rows


async def add_hidden(task: dict, teacher_names: list[str]) -> list[dict] | None:
    others = [n for n in teacher_names if n != task["author"]] or teacher_names
    prompt = HIDDEN_PROMPT.format(
        text=task["text"], signature=vc.signature_of(task["code"]), tests="\n".join(task["tests"])
    )
    reply = await teachers.get(others[0])(prompt)
    if reply is None:  # refusal, truncation or empty: retry on rerun
        return None
    hidden = teachers.parse_json(reply, latex=False)
    if not isinstance(hidden, list):
        return []
    name = function_name(task["code"])
    hidden = list(dict.fromkeys(h.strip() for h in hidden if calls(h, name)))[:MAX_HIDDEN]
    if len(hidden) < 3:
        return []
    if not await asyncio.to_thread(vc.run_tests, task["code"], hidden):
        return []
    return [dict(task, hidden_tests=hidden)]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["write", "hidden"])
    p.add_argument("--teachers", nargs="+", default=["claude"])
    p.add_argument("--calls-per-category", type=int, default=40)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()
    teachers.check_configured(args.teachers)

    out = RAW if args.command == "write" else TASKS
    try:
        if args.command == "write":
            items = write_items(args.teachers, args.calls_per_category)[: args.limit]
            asyncio.run(run_many(items, write_tasks, out, args.concurrency))
        else:
            items = jsonl.read(RAW)[: args.limit]
            fn = lambda t: add_hidden(t, args.teachers)  # noqa: E731
            asyncio.run(run_many(items, fn, out, args.concurrency))
    finally:
        teachers.write_usage(out)


if __name__ == "__main__":
    main()

"""v2 teacher data (Claude via the CLI backend), aimed at gaps self-distillation cannot fill.

math-long:  hard problems where no base-model sample was kept; full reasoning with a check, verified
            against the known answer.
code-plan:  short plan that walks the given asserts, then one ```python block; verified by given and
            hidden tests.
if-count:   prompts that always include one weak counting constraint; strict pass required.

Usage:
    python -m pipeline.generate.v2_teacher math-long --limit 100
    python -m pipeline.generate.v2_teacher code-plan --limit 100
    python -m pipeline.generate.v2_teacher if-count --limit 100
"""

import argparse
import asyncio
import logging
import random
import re

from pipeline import jsonl, teachers
from pipeline.config import DATA, SEED
from pipeline.constraints.sample import _build, group_of, render_prompt, sample_constraints
from pipeline.generate import ifeval as gif
from pipeline.generate.math import split_solution
from pipeline.self_distill import code_items, if_items, math_items_v2
from pipeline.teachers.runner import run_many
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

MATH_LONG = DATA / "v2_math_long.jsonl"
CODE_PLAN = DATA / "v2_code_plan.jsonl"
IF_COUNT = DATA / "v2_if_count.jsonl"
IF_NEW = DATA / "v2_if_new.jsonl"

MATH_LONG_PROMPT = """Solve the following competition math problem.

Write a complete step-by-step solution in the way a careful strong student reasons on paper:
- Set up the problem, then work one step per line, writing out intermediate arithmetic and algebra \
explicitly. Never skip a step a student would need to compute.
- Do the full case analysis where there are cases.
- Before the final answer, check the result with a quick independent verification (substitute back, \
test a small case, or recompute another way) and fix any error you find.
- The problem has one definite answer: always finish with that single value, never "no solution" and \
never an option letter.
Use as many words as the problem needs, typically 200 to 900; never pad.

Finish with a final line of the form
ANSWER: <answer>
where <answer> is only the final answer in simplest form (LaTeX where needed), with no \\boxed, \
no units, and no explanation.

Write all mathematics in LaTeX inside $...$, and use only ASCII characters otherwise (no Unicode \
math symbols such as −, ×, ÷, ≤).

Problem:
{problem}"""

CODE_PLAN_PROMPT = """You will write a Python function for the task below. Your function must pass these tests:

{tests}

Task: {text}

Respond in exactly this form:
1. A short plan in plain prose (2 to 6 sentences, under 120 words): the approach, the edge cases \
that matter, and a quick check of the approach against at least one of the given asserts, computed \
by hand. Do not put any code block or backticks in the plan.
2. Then one fenced code block that starts with ```python and contains only the function and any \
standard-library imports it needs. No tests, no example usage, no prints, and nothing after the block."""

# Counting families the student fails most (v1 error analysis).
WEAK_COUNT_IDS = (
    "x:exact_words", "x:exact_sentences", "x:keyword_exact", "keywords:letter_frequency",
    "keywords:frequency", "length_constraints:number_words", "length_constraints:number_sentences",
    "change_case:capital_word_frequency", "x:min_sentence_words",
)
_FENCE = re.compile(r"```python\n")


# ---------------------------------------------------------------- student-token length

_TOK = None


def student_tokens(text: str) -> int:
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        from pipeline.config import STUDENT
        _TOK = AutoTokenizer.from_pretrained(STUDENT)
    return len(_TOK(text)["input_ids"])


# ---------------------------------------------------------------- math-long

MATH_LONG_MIN_TOKENS, MATH_LONG_MAX_TOKENS = 250, 2500
WEAK_SUBJECTS = ("Geometry", "Intermediate Algebra", "Precalculus")  # v1 dev: 44, 52, 60
_GIVE_UP = re.compile(r"(?i)no (real )?solution|no such|does not exist|cannot be determined|"
                      r"^\(?[A-E]\)?$|\\textbf\{\([A-E]\)\}")


def math_long_items(limit: int | None, correct_l5: int = 1000) -> list[dict]:
    """v1-failed hard problems (weak subjects first), then v1-correct level-5 problems.

    Order comes from failure mining (data/v1_mined_math.jsonl) on training-side problems only.
    """
    mined = {r["id"]: r for r in jsonl.read(DATA / "v1_mined_math.jsonl")}
    pool = {i["id"]: i for i in math_items_v2(4, 10**6) if i["id"] in mined}
    subject = {r.get("pid", r["id"]): r.get("subject") for r in jsonl.read(DATA / "math_train_solutions.jsonl")}
    subject.update({r.get("pid", r["id"]): r.get("subject") for r in jsonl.read(DATA / "math_new_solutions.jsonl")})
    failed = [i for i in pool.values() if not mined[i["id"]]["correct"]]
    failed.sort(key=lambda i: (subject.get(i["id"]) not in WEAK_SUBJECTS, -(i["level"] or 0)))
    right_l5 = [i for i in pool.values() if mined[i["id"]]["correct"] and i["level"] == 5][:correct_l5]
    items = [dict(i, subject=subject.get(i["id"]), v1_correct=False, teacher="claude") for i in failed]
    items += [dict(i, subject=subject.get(i["id"]), v1_correct=True, teacher="claude") for i in right_l5]
    return items[:limit]


async def math_long(item: dict) -> list[dict] | None:
    text = await teachers.get("claude")(MATH_LONG_PROMPT.format(problem=item["problem"]), effort="high")
    if not text:
        return None
    parsed = split_solution(text)
    if not parsed or not await vm.equivalent(parsed[1], item["target"]):
        return []
    body, answer = parsed
    if _GIVE_UP.search(answer):
        return []
    ntok = student_tokens(f"{body}\n\nANSWER: {answer}")
    if not MATH_LONG_MIN_TOKENS <= ntok <= MATH_LONG_MAX_TOKENS:
        return []
    return [{"id": f"{item['id']}-long", "pid": item["id"], "problem": item["problem"], "target": item["target"],
             "answer": answer, "solution": body, "level": item["level"], "source": item["source"],
             "subject": item.get("subject"), "v1_correct": item.get("v1_correct"), "student_tokens": ntok,
             "style": "long", "teacher": "claude"}]


# ---------------------------------------------------------------- code-plan

WEAK_TOPICS = ("dates and times arithmetic", "sorting with custom keys", "recursion", "simple dynamic programming",
               "arithmetic and rounding", "math with fractions and gcd", "two-pointer techniques")  # v1 dev 33-56%


def code_plan_items(limit: int | None) -> list[dict]:
    """v1-failed training tasks (weak topics first), then the remaining weak-topic tasks."""
    mined = {r["id"]: r for r in jsonl.read(DATA / "v1_mined_code.jsonl")}
    tasks = [t for t in code_items() if t["id"] in mined]
    failed = sorted((t for t in tasks if not mined[t["id"]]["correct"]),
                    key=lambda t: t.get("category") not in WEAK_TOPICS)
    weak_rest = [t for t in tasks if mined[t["id"]]["correct"] and t.get("category") in WEAK_TOPICS]
    return [dict(t, v1_correct=t in weak_rest, teacher="claude") for t in failed + weak_rest][:limit]


def normalize_plan(plan: str) -> str:
    """One plan format for every example: drop list numbering and a leading "Plan:" label."""
    return re.sub(r"^\s*(?:1\.\s*)?(?:\**plan\**:\**\s*)?", "", plan.strip(), flags=re.IGNORECASE).strip()


async def code_plan(item: dict) -> list[dict] | None:
    prompt = CODE_PLAN_PROMPT.format(tests="\n".join(item["tests"]), text=item["text"])
    text = await teachers.get("claude")(prompt, effort="medium")
    if not text:
        return None
    text = text.strip()
    if len(_FENCE.findall(text)) != 1 or not text.endswith("```"):
        return []
    plan = normalize_plan(text.split("```python")[0])
    if not (5 <= len(plan.split()) <= 160) or "`" in plan:
        return []
    text = plan + "\n\n```python" + text.split("```python", 1)[1]
    tests = item["tests"] + item.get("hidden_tests", [])
    if not await asyncio.to_thread(vc.run_tests, vc.extract_code(text), tests, timeout=10.0):
        return []
    if student_tokens(text) > 1500:
        return []
    return [{"id": f"{item['id']}-plan", "task_id": item["id"], "text": item["text"], "tests": item["tests"],
             "hidden_tests": item.get("hidden_tests", []), "category": item.get("category"),
             "response": text, "style": "plan", "teacher": "claude"}]


# ---------------------------------------------------------------- if-count

EXACT_WORDS_MAX = 60  # larger exact word counts are not learnable; longer lengths use at-least/at-most bounds
# Discovery round 1: v1 fails 86-94% of probes with 4-5 rules vs 48% with 1-2, and more often when the rules
# come before the task (77% vs 68%). So 2-4 rules besides the first, and rules first in 40% of prompts.
EXTRA_RULES = [1, 2, 2, 3]
RULES_FIRST = 0.4


def render(task: str, cons: list[dict], rng: random.Random) -> str:
    if rng.random() < RULES_FIRST:
        return " ".join(c["text"] for c in cons) + "\n\n" + task.strip()
    return render_prompt(task, cons)


def _weak_rule(rng: random.Random, task: str) -> dict:
    for _ in range(20):
        c = _build(rng.choice(WEAK_COUNT_IDS), rng, task)
        if c["id"] != "x:exact_words" or c["kwargs"]["n"] <= EXACT_WORDS_MAX:
            return c
    return _build("length_constraints:number_words", rng, task)


def if_count_items(limit: int | None, skip: int = 4000) -> list[dict]:
    """Unused base tasks (after the ones reserved for self-distillation), each with one weak counting rule.

    Extra rules: EXTRA_RULES more; 15% also ask for a non-English response (repetition loops on public
    IFEval were often non-English).
    """
    items = []
    for it in if_items()[skip:]:
        rng = random.Random(f"ifc-{it['id']}")
        weak = _weak_rule(rng, it["task"])
        rest = [c for c in sample_constraints(rng, it["task"], mode="train")
                if group_of(c["id"]) != group_of(weak["id"]) and group_of(c["id"]) != "wrap"
                and not (c["id"] == "x:exact_words" and c["kwargs"]["n"] > EXACT_WORDS_MAX)]
        cons = [weak] + rest[: rng.choice(EXTRA_RULES)]
        if rng.random() < 0.15 and all(group_of(c["id"]) != group_of("language:response_language") for c in cons):
            cons.append(_build("language:response_language", rng, it["task"]))
        rng.shuffle(cons)
        items.append({"id": f"ifc-{it['id'][3:]}", "task": it["task"], "prompt": render(it["task"], cons, rng),
                      "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in cons], "teacher": "claude"})
    return items[:limit]


async def if_count(item: dict) -> list[dict] | None:
    complete = teachers.get("claude")
    got = False
    for _ in range(gif.ATTEMPTS):
        response = await complete(item["prompt"], system=gif.SYSTEM, max_tokens=gif.RESPONSE_MAX_TOKENS)
        if not response:
            continue
        got = True
        if len(response.split()) <= gif.MAX_RESPONSE_WORDS and vi.passes(item["prompt"], item["constraints"], response):
            return [{"id": f"if-{item['id']}", "task": item["task"], "prompt": item["prompt"],
                     "constraints": item["constraints"], "response": response, "teacher": "claude",
                     "style": "count"}]
    return [] if got else None


# ---------------------------------------------------------------- if-new

def if_new_items(limit: int | None, skip: int = 6000) -> list[dict]:
    """Unused base tasks after if-count's share, each with one of the new (v2) training constraint types.

    Failure I4: held-out types did not improve with more examples of the same types, so v2 adds breadth.
    """
    from pipeline.constraints import extra
    new_train = sorted(i for i, e in extra.REGISTRY.items() if i not in extra.V1_IDS and not e.dev_only)
    items = []
    for it in if_items()[skip:]:
        rng = random.Random(f"ifn-{it['id']}")
        first = _build(rng.choice(new_train), rng, it["task"])
        from pipeline.constraints.sample import groups_of
        taken = groups_of(first["id"])
        rest = [c for c in sample_constraints(rng, it["task"], mode="train")
                if not (groups_of(c["id"]) & taken) and group_of(c["id"]) != "wrap"
                and not (c["id"] == "x:exact_words" and c["kwargs"]["n"] > EXACT_WORDS_MAX)]
        cons = [first] + rest[: rng.choice(EXTRA_RULES)]
        rng.shuffle(cons)
        items.append({"id": f"ifn-{it['id'][3:]}", "task": it["task"], "prompt": render(it["task"], cons, rng),
                      "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in cons], "teacher": "claude"})
    return items[:limit]


async def if_new(item: dict) -> list[dict] | None:
    rows = await if_count(item)
    if rows:
        rows[0]["style"] = "new_types"
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["math-long", "code-plan", "if-count", "if-new"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()
    teachers.check_configured(["claude"])
    plan = {"math-long": (math_long_items, math_long, MATH_LONG),
            "code-plan": (code_plan_items, code_plan, CODE_PLAN),
            "if-count": (if_count_items, if_count, IF_COUNT),
            "if-new": (if_new_items, if_new, IF_NEW)}[args.command]
    items_fn, fn, out = plan
    try:
        asyncio.run(run_many(items_fn(args.limit), fn, out, args.concurrency))
    finally:
        teachers.write_usage(out)


if __name__ == "__main__":
    random.seed(SEED)
    main()

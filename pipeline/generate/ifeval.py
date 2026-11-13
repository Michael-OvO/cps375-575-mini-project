"""Instruction-following pool: fresh tasks, composed constraints, strictly verified responses.

Usage:
    python -m pipeline.generate.ifeval base --teachers claude gpt
    python -m pipeline.generate.ifeval respond --teachers claude gpt --limit 14000
"""

import argparse
import asyncio
import logging
import random
import re

from pipeline import jsonl, teachers
from pipeline.config import DATA, SEED, is_dev
from pipeline.constraints.sample import render_prompt, sample_constraints
from pipeline.teachers.runner import run_many
from pipeline.verify import ifeval as vi

BASE = DATA / "if_base_tasks.jsonl"
RESPONSES = DATA / "if_responses.jsonl"
DEV_PER_MILLE = 80
ATTEMPTS = 3
# Teacher thinking counts against max_tokens, so both calls get the full budget.
RESPONSE_MAX_TOKENS = 16000
REWRITE_MAX_TOKENS = 16000

SYSTEM = (
    "Follow every instruction in the user's message exactly, including every formatting, length and "
    "wording requirement. Output only the requested content: no preamble, no closing remarks, and "
    "no commentary about the instructions."
)

TOPICS = [
    "urban gardening", "space exploration", "personal finance", "ancient Rome", "marine biology",
    "remote work", "classical music", "nutrition", "public transport", "renewable energy",
    "chess", "hiking", "small business marketing", "volcanoes", "bird watching", "home repair",
    "the printing press", "mental health at work", "coffee", "weather forecasting", "board games",
    "language learning", "city planning", "photography", "sleep", "recycling", "the Silk Road",
    "cybersecurity basics", "baking bread", "volunteering", "electric vehicles", "museums",
    "time management", "rivers", "job interviews", "astronomy for beginners", "tea ceremonies",
    "bicycles", "libraries", "the water cycle", "team sports", "podcasts", "architecture",
    "first aid", "honeybees", "travel on a budget", "the history of maps", "cooking for one",
    "study habits", "deserts", "customer service", "lighthouses", "pets", "the Olympic Games",
    "minimalism", "railways", "mountain ecosystems", "public speaking", "robots", "farmers markets",
]
FORMS = [
    "a short essay", "an email to a colleague", "a short story", "a product description",
    "an explanation for a beginner", "a speech", "a poem", "a review", "a letter to a friend",
    "a blog post", "a set of instructions", "a list of tips", "a dialogue between two people",
    "an advertisement", "a summary of the key ideas",
]

BASE_PROMPT = """Write 20 different requests that a person might send to a writing assistant. Every \
request must be about {topic} and ask for {form}.

Requirements:
- One or two sentences each, natural and specific, with varied angles and audiences.
- Do not include any formatting, length, wording or style requirements: only say what to write about.

Return only a JSON array of 20 strings."""

# Share of prompts whose wording is rewritten by the teacher, so the student does not
# learn only the fixed phrasing of each constraint. Verification still uses the stored kwargs.
REWRITE_SHARE = 0.3
REWRITE_PROMPT = """Rewrite the request below in different words, as a different person might \
phrase it. Keep every requirement, and keep every number, quoted word and quoted phrase exactly \
as written. Do not add, drop or soften any requirement. Return only the rewritten request.

{prompt}"""


def _kwarg_values(obj) -> list:
    """All ints, floats and strings inside constraint kwargs, recursively."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float, str)):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        return [v for x in obj for v in _kwarg_values(x)]
    return []


def _preserves(original: str, rewritten: str, constraints: list[dict]) -> bool:
    """Every number and string from the kwargs that the original prompt shows must survive."""
    for v in _kwarg_values([c.get("kwargs", {}) for c in constraints]):
        if isinstance(v, str):
            if v and v in original and v not in rewritten:
                return False
        else:
            pat = rf"(?<![\d.]){re.escape(str(v))}(?![\d])"
            if re.search(pat, original) and not re.search(pat, rewritten):
                return False
    return True


async def maybe_rewrite(prompt: str, constraints: list[dict], rng: random.Random, complete) -> str:
    """Reword the prompt for a share of items. Skipped when a constraint depends on the prompt text.

    Falls back to the original unless the rewrite has a plausible length and keeps every number
    and string from the constraint kwargs, so the stored kwargs still describe the prompt.
    """
    if rng.random() >= REWRITE_SHARE or any(c["id"] == "combination:repeat_prompt" for c in constraints):
        return prompt
    rewritten = await complete(REWRITE_PROMPT.format(prompt=prompt), max_tokens=REWRITE_MAX_TOKENS)
    if not rewritten or not (0.5 * len(prompt) < len(rewritten) <= 2 * len(prompt)):
        return prompt
    return rewritten if _preserves(prompt, rewritten, constraints) else prompt


async def write_base_tasks(item: dict) -> list[dict] | None:
    reply = await teachers.get(item["teacher"])(BASE_PROMPT.format(topic=item["topic"], form=item["form"]))
    if not reply:
        return None  # teacher returned nothing: retry on rerun
    tasks = teachers.parse_json(reply, latex=False)  # plain prose, no LaTeX
    if not isinstance(tasks, list):
        return []
    return [
        {"id": f"{item['id']}-{j}", "task": t.strip()}
        for j, t in enumerate(tasks) if isinstance(t, str) and len(t.strip()) > 15
    ]


async def respond(item: dict) -> list[dict] | None:
    rng = random.Random(item["seed"])
    constraints = sample_constraints(rng, item["task"], mode="train")
    stored = [{"id": c["id"], "kwargs": c["kwargs"]} for c in constraints]
    complete = teachers.get(item["teacher"])
    prompt = await maybe_rewrite(render_prompt(item["task"], constraints), stored, rng, complete)
    got_response = False
    for _ in range(ATTEMPTS):
        response = await complete(prompt, system=SYSTEM, max_tokens=RESPONSE_MAX_TOKENS)
        if not response:
            continue
        got_response = True
        if vi.passes(prompt, stored, response):
            # "task" is the base task before constraints, for later overlap checks.
            return [{"id": f"if-{item['id']}", "task": item["task"], "prompt": prompt,
                     "constraints": stored, "response": response, "teacher": item["teacher"]}]
    return [] if got_response else None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["base", "respond"])
    p.add_argument("--teachers", nargs="+", default=["claude", "gpt"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()
    teachers.check_configured(args.teachers)

    out = BASE if args.command == "base" else RESPONSES
    try:
        if args.command == "base":
            items = [
                {"id": f"bt-{ti}-{fi}", "topic": topic, "form": form,
                 "teacher": args.teachers[(ti + fi) % len(args.teachers)]}
                for ti, topic in enumerate(TOPICS) for fi, form in enumerate(FORMS)
            ][: args.limit]
            asyncio.run(run_many(items, write_base_tasks, out, args.concurrency))
        else:
            rng = random.Random(SEED)
            tasks = [t for t in jsonl.read(BASE) if not is_dev(t["id"], DEV_PER_MILLE)]
            rng.shuffle(tasks)
            items = [
                dict(t, teacher=args.teachers[i % len(args.teachers)], seed=rng.getrandbits(32))
                for i, t in enumerate(tasks[: args.limit])
            ]
            asyncio.run(run_many(items, respond, out, args.concurrency))
    finally:
        teachers.write_usage(out)


if __name__ == "__main__":
    main()

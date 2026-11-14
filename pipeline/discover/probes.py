"""Probe sets: fresh, verified items for one taxonomy cell. Measurement only, never trained on.

Per cell, one Opus call writes N candidate items; each is then verified and answered once more by an
independent Opus call, which gives the teacher pass rate (learnability):
- math: the author's answer and an independent solve must agree (the solve also is the teacher pass);
- code: the author's reference must pass all its tests; an independent solve from the evaluation prompt
  gives the teacher pass;
- IF: constraints come from our verifier registry (never from the model); the base task is written by Opus
  in the cell's genre, the request is rendered in the cell's wording and position; the teacher answers it.

Usage:
    python -m pipeline.discover.probes cells                     # write data/discover/cells_<task>.json
    python -m pipeline.discover.probes gen --task math --cells 3 # pilot: first 3 cells
"""

import argparse
import asyncio
import json
import logging
import random
import re

from pipeline import formats, jsonl, teachers
from pipeline.config import SEED
from pipeline.constraints import extra
from pipeline.constraints.sample import (DEV_ONLY_IDS, STANDARD_IDS, TRAIN_EXTRA_IDS, WRAP_CONFLICTS, _build,
                                         group_of, groups_of)
from pipeline.discover import coverage
from pipeline.discover.taxonomy import DISCOVER, FACETS, describe, facets, select_cells
from pipeline.generate import ifeval as gif
from pipeline.generate.v2_teacher import MATH_LONG_PROMPT
from pipeline.verify import code as vc
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm

PER_CELL = 12
CELLS_PER_TASK = 50
LANG_CODE = {"Spanish": "es", "German": "de", "Japanese": "ja", "Hindi": "hi", "Korean": "ko", "Russian": "ru",
             "Thai": "th", "Bengali": "bn"}
STAFF = {
    "math": "competition math problems with a single final answer (MATH-style)",
    "code": "Python function-writing tasks checked by unit tests (MBPP/HumanEval-style)",
    "if": "instructions with programmatically verifiable constraints (IFEval-style)",
}


def cells_path(task: str):
    return DISCOVER / f"cells_{task}.json"


def probes_path(task: str):
    return DISCOVER / f"probes_{task}.jsonl"


def write_cells() -> None:
    """Covering selection, plus one extra cell for every facet value the training data never contains."""
    DISCOVER.mkdir(parents=True, exist_ok=True)
    for task in FACETS:
        cov = json.loads((DISCOVER / f"coverage_{task}.json").read_text()) if (DISCOVER / f"coverage_{task}.json").exists() else {}
        gaps = {f: [v for v in vals if not cov.get(f, {}).get(v)] for f, vals in facets(task).items() if f in cov}
        gaps = {f: v for f, v in gaps.items() if v}
        cells = select_cells(task, CELLS_PER_TASK, must_include=gaps, valid=valid_if_cell if task == "if" else None)
        cells_path(task).write_text(json.dumps(cells, indent=1))
        print(task, len(cells), "cells; forced gap values:", sum(len(v) for v in gaps.values()))


# ---------------------------------------------------------------- math

MATH_WRITE = """Write {n} new competition math problems for a test of a student model. Every problem must fit this description:

{cell}

Requirements:
- Original problems in the style of AMC/AIME/MATH competition problems at the stated difficulty level \
(MATH levels 1 to 5; level 5 is the hardest MATH problems). Do not copy or lightly edit known problems.
- Each problem has exactly one correct final answer of the stated answer type, written in simplest form.
- No diagrams: describe any figure in words. No multiple choice.
- The {n} problems must differ from each other in setting and in the specific idea.

Return only a JSON list of {n} objects with keys "problem" (LaTeX in $...$) and "answer" (the final answer only, LaTeX, no \\boxed)."""

# ---------------------------------------------------------------- code

CODE_WRITE = """Write {n} new Python function-writing tasks for a test of a student model. Every task must fit this description:

{cell}

Prompt styles:
- "MBPP one-line description": text is one sentence like "Write a function to ..." naming nothing but the task.
- "HumanEval-style signature with docstring and examples": text is a Python function signature with a docstring \
that explains the task and shows one or two >>> examples; no body.
- "longer specification with explicit rules": text is a paragraph of 3 to 6 sentences giving precise rules and edge cases.

Requirements for each task:
- Original (not a known MBPP or HumanEval problem), solvable in under 30 lines with the standard library.
- "text": the task in the cell's prompt style; it must make the expected behaviour unambiguous, including the \
input regime and output contract described above.
- "code": a correct reference solution defining exactly the function the tests call.
- "tests": exactly 3 assert statements calling the function (shown to the student).
- "hidden_tests": 4 more assert statements, exercising the input regime above (not shown).
- The {n} tasks must differ from each other.

Return only a JSON list of {n} objects with keys "text", "code", "tests", "hidden_tests"."""

# ---------------------------------------------------------------- IF

IF_WRITE = """Write {n} different requests that a user might send to an assistant. Each request asks for {genre}. \
Vary the subject widely (science, daily life, history, work, hobbies, travel, technology, ...). Each request is \
one or two sentences, asks for the content only, and contains no formatting, length or style requirements.
Return only a JSON list of {n} strings."""

IF_REWORD = {
    "paraphrased in casual language": "Rewrite the request below as a casual, chatty user would type it.",
    "paraphrased formally with extra context": "Rewrite the request below formally, as a busy professional would, "
                                               "adding one sentence of context about why they need it.",
}
IF_MIX = ("Rewrite the request below as one natural message in which the requirements are woven into the "
          "description of the task instead of being listed at the end.")
IF_KEEP = (" Keep every requirement, and keep every number, quoted word and quoted phrase exactly as written. "
           "Keep the requirements where they are relative to the task unless told to weave them in. "
           "Do not add, drop or soften any requirement. Return only the rewritten request.\n\nRequest:\n{prompt}")

# verified against the exact prompt text, so it cannot appear in a reworded prompt
_NO_REWORD = {"combination:repeat_prompt"}
ALL_IDS = sorted(set(STANDARD_IDS) | set(TRAIN_EXTRA_IDS) | set(DEV_ONLY_IDS))
LANG_RULE = "language:response_language"
# Groups whose rules count English words, letters or case: unsatisfiable when the whole answer must be in
# another language (pilot: "words with double letters" in Japanese, English capitals in Hindi). Word and
# sentence counts break in languages written without spaces.
ENGLISH_ONLY_GROUPS = {"keyword", "letters", "doubles", "wordlen", "case", "repeat", "position", "start",
                       "length", "sentence"}


def english_only(cid: str, kwargs: dict | None = None) -> bool:
    if cid == LANG_RULE:
        return False
    words = [v for v in (kwargs or {}).values() if isinstance(v, str) and re.search(r"[A-Za-z]{2}", v)]
    return group_of(cid) in ENGLISH_ONLY_GROUPS or bool(words)


def valid_if_cell(cell: dict) -> bool:
    return cell["language"] == "English" or cell["family"] not in ENGLISH_ONLY_GROUPS


def reworded(cell: dict) -> bool:
    return cell["wording"] in IF_REWORD or cell["position"] == "constraints mixed into the task"


def count_regime(c: dict) -> str:
    kw = c.get("kwargs") or {}
    if any(k in kw for k in ("relation", "low", "high", "min", "max")) or any(
            str(v) in ("less than", "at least", "at most") for v in kw.values()):
        return "bound"
    if any(isinstance(v, int) and not isinstance(v, bool) for v in kw.values()):
        return "exact count"
    return "no count"


def if_constraints(rng: random.Random, cell: dict, task: str) -> list[dict] | None:
    """First rule from the cell's family (dev-only types included), a language rule if the cell is not
    English, then compatible training rules up to n_constraints. None if they cannot be combined."""
    k = int(cell["n_constraints"])
    lang = cell["language"]
    if not valid_if_cell(cell):
        return None
    banned = _NO_REWORD if reworded(cell) else set()
    english = lang == "English"
    first = [i for i in ALL_IDS if group_of(i) == cell["family"] and i not in banned and (english or not english_only(i))]
    if not first:
        return None
    chosen = [rng.choice(first)]
    if not english and LANG_RULE not in chosen:
        chosen.append(LANG_RULE)
    occupied = set().union(*(groups_of(c) for c in chosen))
    primary = {group_of(c) for c in chosen}
    pool = [i for i in STANDARD_IDS + TRAIN_EXTRA_IDS if i not in banned and (english or not english_only(i))]
    for _ in range(200):
        if len(chosen) >= k:
            break
        cid = rng.choice(pool)
        g = group_of(cid)
        if groups_of(cid) & occupied or (g == "wrap" and primary & WRAP_CONFLICTS) or (g in WRAP_CONFLICTS and "wrap" in primary):
            continue
        chosen.append(cid)
        occupied |= groups_of(cid)
        primary.add(g)
    if len(chosen) < k:
        return None
    built = [_language_rule(LANG_CODE[lang]) if cid == LANG_RULE and lang in LANG_CODE else _build(cid, rng, task)
             for cid in chosen]
    # sampled kwargs can still carry English words (e.g. a required end phrase)
    if not english and any(english_only(c["id"], c["kwargs"]) for c in built):
        return None
    return built


def _language_rule(code: str) -> dict:
    from instruction_following_eval import instructions_registry
    inst = instructions_registry.INSTRUCTION_DICT[LANG_RULE](LANG_RULE)
    text = inst.build_description(language=code)
    return {"id": LANG_RULE, "kwargs": inst.get_instruction_args(), "text": text}


def render(task: str, cons: list[dict], position: str) -> str:
    rules = " ".join(c["text"] for c in cons)
    if position == "constraints before the task":
        return f"{rules}\n\n{task.strip()}"
    return f"{task.strip()} {rules}"


async def reword(prompt: str, cons: list[dict], cell: dict, complete) -> str | None:
    """Apply the cell's wording and 'mixed' position; None when the rewrite loses a requirement."""
    asks = []
    if cell["wording"] in IF_REWORD:
        asks.append(IF_REWORD[cell["wording"]])
    if cell["position"] == "constraints mixed into the task":
        asks.append(IF_MIX)
    if not asks:
        return prompt
    out = await complete(" ".join(asks) + IF_KEEP.format(prompt=prompt), max_tokens=gif.REWRITE_MAX_TOKENS)
    if not out or not (0.5 * len(prompt) < len(out) <= 2.5 * len(prompt)):
        return None
    return out.strip() if gif._preserves(prompt, out, cons) else None


# ---------------------------------------------------------------- per cell

def _json_list(text: str | None, latex: bool = False) -> list:
    if not text:
        return []
    try:
        got = teachers.parse_json(text, latex=latex)
    except Exception:
        return []
    return got if isinstance(got, list) else []


async def gen_cell(task: str, cell: dict, n: int, leak) -> list[dict] | None:
    complete = teachers.get("claude")
    cid = cell["cell_id"]
    spec = describe(task, cell)
    rows: list[dict] = []
    if task == "math":
        cands = _json_list(await complete(MATH_WRITE.format(n=n, cell=spec), max_tokens=16000), latex=True)
        if not cands:
            return None

        async def check(i, c):
            if not isinstance(c, dict) or not c.get("problem") or not c.get("answer"):
                return None
            if leak and leak.is_leak(c["problem"]):
                return None
            sol = await complete(MATH_LONG_PROMPT.format(problem=c["problem"]), max_tokens=16000)
            ans = vm.answer_of(sol or "")
            agree = bool(ans) and await vm.equivalent(ans, str(c["answer"]))
            return {"id": f"{cid}-p{i:02d}", "cell_id": cid, "task": task, "problem": c["problem"],
                    "target": str(c["answer"]), "teacher_pass": agree, "teacher_answer": ans, "cell": cell}
        got = await asyncio.gather(*(check(i, c) for i, c in enumerate(cands[:n])))
        rows = [g for g in got if g]
    elif task == "code":
        cands = _json_list(await complete(CODE_WRITE.format(n=n, cell=spec), max_tokens=16000))
        if not cands:
            return None

        async def check(i, c):
            if not isinstance(c, dict) or not all(c.get(k) for k in ("text", "code", "tests", "hidden_tests")):
                return None
            tests, hidden = [str(t) for t in c["tests"]], [str(t) for t in c["hidden_tests"]]
            if leak and leak.is_leak(c["text"]):
                return None
            if not await asyncio.to_thread(vc.run_tests, c["code"], tests + hidden):
                return None  # broken reference or tests: not a valid probe
            sol = await complete(formats.code_prompt(c["text"], tests), max_tokens=16000)
            ok = bool(sol) and await asyncio.to_thread(vc.run_tests, vc.extract_code(sol), tests + hidden)
            return {"id": f"{cid}-p{i:02d}", "cell_id": cid, "task": task, "text": c["text"], "tests": tests,
                    "hidden_tests": hidden, "reference": c["code"], "teacher_pass": ok, "cell": cell}
        got = await asyncio.gather(*(check(i, c) for i, c in enumerate(cands[:n])))
        rows = [g for g in got if g]
    else:
        bases = [b for b in _json_list(await complete(IF_WRITE.format(n=n, genre=cell["genre"]), max_tokens=8000))
                 if isinstance(b, str) and b.strip()]
        if not bases:
            return None

        async def check(i, base):
            # few rules are language-neutral, so non-English cells need several draws
            cons = next((c for j in range(30) if (c := if_constraints(random.Random(f"{SEED}-{cid}-{i}-{j}"), cell, base))), None)
            if not cons:
                return None
            prompt = await reword(render(base, cons, cell["position"]), cons, cell, complete)
            if not prompt or (leak and leak.is_leak(base)):
                return None
            stored = [{"id": c["id"], "kwargs": c["kwargs"]} for c in cons]
            resp = await complete(prompt, system=gif.SYSTEM, max_tokens=gif.RESPONSE_MAX_TOKENS)
            ok = bool(resp) and vi.passes(prompt, stored, resp)
            return {"id": f"{cid}-p{i:02d}", "cell_id": cid, "task": task, "base": base, "prompt": prompt,
                    "constraints": stored, "regimes": [count_regime(c) for c in cons],
                    "dev_only": [c["id"] in DEV_ONLY_IDS for c in cons], "teacher_pass": ok, "cell": cell}
        got = await asyncio.gather(*(check(i, b) for i, b in enumerate(bases[:n])))
        rows = [g for g in got if g]
    # a cell row carries the done marker even when few probes survive
    return rows or [{"id": f"{cid}-empty", "cell_id": cid, "task": task, "empty": True, "cell": cell}]


def generate(task: str, n_cells: int | None, start: int, per_cell: int, concurrency: int) -> None:
    from pipeline.leakage import load_overlap_filter
    from pipeline.teachers.runner import run_many
    teachers.check_configured(["claude"])
    cells = json.loads(cells_path(task).read_text())[start:]
    cells = cells[:n_cells] if n_cells else cells
    leak = load_overlap_filter()
    if leak is None:
        logging.warning("overlap filter disabled (set ALLOW_TEST_OVERLAP_CHECK=1)")

    async def fn(item):
        return await gen_cell(task, item["cell"], per_cell, leak)

    items = [{"id": c["cell_id"], "cell": c} for c in cells]
    out = probes_path(task)
    try:
        asyncio.run(run_many(items, fn, out, concurrency))
    finally:
        teachers.write_usage(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["cells", "gen"])
    p.add_argument("--task", choices=list(FACETS))
    p.add_argument("--cells", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--per_cell", type=int, default=PER_CELL)
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    if a.command == "cells":
        coverage.build()
        write_cells()
    else:
        generate(a.task, a.cells, a.start, a.per_cell, a.concurrency)


if __name__ == "__main__":
    random.seed(SEED)
    main()

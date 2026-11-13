"""Math answer extraction and equivalence, matching evaluation/math_scorer.py."""

import asyncio
import re

from inspect_evals.math.utils import (
    extract_answer,
    is_equiv,
    is_equiv_sympy,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from math_verify import parse, verify

_ANSWER_LINE = re.compile(r"(?i)\n?[^\n]*ANSWER\s*:\s*[^\n]+\s*\Z")


def answer_of(text: str) -> str:
    return extract_answer(text).strip()


def body_of(text: str) -> str:
    return _ANSWER_LINE.sub("", text.rstrip()).rstrip()


async def equivalent(answer: str, target: str) -> bool:
    if not answer:
        return False
    try:
        if verify(parse(f"${target}$"), parse(f"${answer}$")):
            return True
    except Exception:
        pass
    try:
        if await is_equiv(answer, target):
            return True
        return bool(
            await is_equiv_sympy(
                await normalize_final_answer(answer), await normalize_final_answer(target)
            )
        )
    except Exception:
        return False


def is_correct(answer: str, target: str) -> bool:
    return asyncio.run(equivalent(answer, target))


def boxed_target(solution: str) -> str | None:
    boxed = last_boxed_only_string(solution)
    return remove_boxed(boxed) if boxed else None

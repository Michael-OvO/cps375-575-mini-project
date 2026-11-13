"""Fallback teacher hosted on Tinker. Costs Tinker budget; use only if outside teachers are not allowed."""

from pipeline.config import TINKER_TEACHER
from pipeline.student import Sampler

_sampler: Sampler | None = None


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 8000
) -> str | None:
    global _sampler
    if _sampler is None:
        _sampler = Sampler(TINKER_TEACHER)
    text = f"{system}\n\n{prompt}" if system else prompt
    answers = await _sampler.answer(text, max_tokens=max_tokens, temperature=0.7)
    return answers[0].strip() or None

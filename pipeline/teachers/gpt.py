"""GPT teacher through the OpenAI Responses API. Model: config.GPT_MODEL (OPENAI_TEACHER_MODEL overrides).

`effort` is passed as the reasoning effort (low / medium / high / xhigh / max), the same names the Claude
teachers take. Reasoning tokens are billed as output and counted in output_tokens.
"""

from openai import AsyncOpenAI

from pipeline.config import GPT_MODEL
from pipeline.teachers import new_usage, record_usage

_client: AsyncOpenAI | None = None
USAGE = new_usage()  # calls, tokens and served model ids, for this process


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(max_retries=6, timeout=900)
    return _client


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 16000
) -> str | None:
    kwargs = dict(model=GPT_MODEL, input=prompt, max_output_tokens=max_tokens,
                  reasoning={"effort": effort}, store=False)
    if system:
        kwargs["instructions"] = system
    response = await _get_client().responses.create(**kwargs)
    record_usage(USAGE, response)
    if getattr(response, "status", "completed") != "completed":
        return None  # incomplete (hit max_output_tokens) or refused: same as Claude's max_tokens / refusal
    text = (response.output_text or "").strip()
    return text or None

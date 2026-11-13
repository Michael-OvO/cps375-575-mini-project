"""GPT teacher. OpenAI SDK only. The model id comes from OPENAI_TEACHER_MODEL."""

from openai import AsyncOpenAI

from pipeline.config import GPT_MODEL

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(max_retries=6)
    return _client


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 16000
) -> str | None:
    if not GPT_MODEL:
        raise RuntimeError("Set OPENAI_TEACHER_MODEL to the GPT model id to use as a teacher.")
    kwargs = dict(model=GPT_MODEL, input=prompt, max_output_tokens=max_tokens)
    if system:
        kwargs["instructions"] = system
    response = await _get_client().responses.create(**kwargs)
    if getattr(response, "status", "completed") != "completed":
        return None
    text = (response.output_text or "").strip()
    return text or None

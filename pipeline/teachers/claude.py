"""Claude teacher. Anthropic SDK only."""

import anthropic

from pipeline.config import CLAUDE_MODEL

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(max_retries=6)
    return _client


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 16000
) -> str | None:
    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    response = await _get_client().beta.messages.create(**kwargs)
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text or None

"""Claude teacher through the locally installed Claude Code CLI (headless print mode)."""

import asyncio
import json
import os
import re
import tempfile

from pipeline.config import CLAUDE_MODEL
from pipeline.teachers import TeacherError, TeacherUnavailable, new_usage

# calls, tokens, served model ids, list-price cost (USD); failed_calls counts error results
# (their tokens and cost are included in the totals above).
USAGE = {**new_usage(), "cost_usd": 0.0, "failed_calls": 0}
DEFAULT_SYSTEM = ("You are a careful assistant. Follow the user's instructions exactly "
                  "and output only what is asked for.")
# Only clear usage-limit / rate-limit / not-signed-in messages; applied to error results only.
_LIMIT = re.compile(r"usage limit|rate limit|limit reached|not logged in|please run /login",
                    re.IGNORECASE)
# Removed from the child's environment so the CLI uses the owner's subscription login, not an API key.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE")


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}


async def _reap(proc) -> None:
    """Kill the process (if still running) and wait for it, even if we are being cancelled."""
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await asyncio.shield(proc.wait())


async def _run_cli(argv: list[str], prompt: str, timeout: float) -> tuple[int, str, str]:
    """Run the CLI in a fresh empty directory (no project files or CLAUDE.md), prompt on stdin."""
    with tempfile.TemporaryDirectory(prefix="claude_teacher_") as cwd:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env=_child_env(), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout)
        except asyncio.TimeoutError:
            await _reap(proc)
            raise TeacherError(f"claude CLI timed out after {timeout:g}s") from None
        except BaseException:  # cancellation or anything else: never leave the child running
            await _reap(proc)
            raise
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _record(data: dict, failed: bool = False) -> None:
    usage = data.get("usage") or {}
    USAGE["calls"] += 1
    USAGE["failed_calls"] += int(failed)
    USAGE["input_tokens"] += int(usage.get("input_tokens") or 0)
    USAGE["output_tokens"] += int(usage.get("output_tokens") or 0)
    USAGE["cost_usd"] += float(data.get("total_cost_usd") or 0.0)
    for model in data.get("modelUsage") or {"unknown": None}:
        USAGE["models"][model] = USAGE["models"].get(model, 0) + 1


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 16000,
    timeout: float = 1800,
) -> str | None:
    # max_tokens is accepted for interface parity only: the CLI has no such flag, so it is not passed.
    argv = ["claude", "-p", "--system-prompt", system or DEFAULT_SYSTEM, "--tools", "",
            "--output-format", "json", "--model", CLAUDE_MODEL, "--effort", effort,
            "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config",
            "--setting-sources", ""]
    code, out, err = await _run_cli(argv, prompt, timeout)
    try:
        data = json.loads(out)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        data = None
    if data is None:
        USAGE["failed_calls"] += 1
        detail = (err.strip() or out.strip())[:500]
        cls = TeacherUnavailable if _LIMIT.search(detail) else TeacherError
        raise cls(f"claude CLI exit {code}, no JSON result: {detail}")
    result = data.get("result")
    result = result if isinstance(result, str) else ""
    status = data.get("api_error_status")
    subtype = data.get("subtype")
    if code != 0 or data.get("is_error") or (subtype is not None and subtype != "success"):
        _record(data, failed=True)
        detail = (f"claude CLI exit {code}, subtype={subtype}, status={status}: "
                  f"{(result or err).strip()[:500]}")
        unavailable = status in (401, 403, 429) or _LIMIT.search(f"{result}\n{err}")
        raise (TeacherUnavailable if unavailable else TeacherError)(detail)
    _record(data)
    if data.get("stop_reason") in ("refusal", "max_tokens"):
        return None
    return result.strip() or None

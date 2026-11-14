"""Claude teacher through the locally installed Claude Code CLI (headless print mode)."""

import asyncio
import json
import re
import tempfile

from pipeline.config import CLAUDE_MODEL
from pipeline.teachers import TeacherError, TeacherUnavailable, new_usage

USAGE = {**new_usage(), "cost_usd": 0.0}  # calls, tokens, served model ids, list-price cost (USD)
DEFAULT_SYSTEM = ("You are a careful assistant. Follow the user's instructions exactly "
                  "and output only what is asked for.")
_LIMIT = re.compile(r"usage limit|rate limit|limit reached|not logged in|please run /login|\blogin\b|"
                    r"invalid api key|authentication", re.IGNORECASE)


async def _run_cli(argv: list[str], prompt: str, timeout: float) -> tuple[int, str, str]:
    """Run the CLI in a fresh empty directory (no project files or CLAUDE.md), prompt on stdin."""
    with tempfile.TemporaryDirectory(prefix="claude_teacher_") as cwd:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TeacherError(f"claude CLI timed out after {timeout:g}s") from None
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            raise
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _record(data: dict) -> None:
    usage = data.get("usage") or {}
    USAGE["calls"] += 1
    USAGE["input_tokens"] += int(usage.get("input_tokens") or 0)
    USAGE["output_tokens"] += int(usage.get("output_tokens") or 0)
    USAGE["cost_usd"] += float(data.get("total_cost_usd") or 0.0)
    models = data.get("modelUsage") or {"unknown": None}
    for model in models:
        USAGE["models"][model] = USAGE["models"].get(model, 0) + 1


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 16000,
    timeout: float = 900,
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
        detail = (err.strip() or out.strip())[:500]
        cls = TeacherUnavailable if _LIMIT.search(detail) else TeacherError
        raise cls(f"claude CLI exit {code}, no JSON result: {detail}")
    result = data.get("result")
    result = result if isinstance(result, str) else ""
    status = data.get("api_error_status")
    if code != 0 or data.get("is_error"):
        detail = f"claude CLI exit {code}, subtype={data.get('subtype')}, status={status}: " \
                 f"{(result or err).strip()[:500]}"
        unavailable = status in (401, 403, 429) or _LIMIT.search(f"{result}\n{err}")
        raise (TeacherUnavailable if unavailable else TeacherError)(detail)
    _record(data)
    if data.get("stop_reason") in ("refusal", "max_tokens"):
        return None
    return result.strip() or None

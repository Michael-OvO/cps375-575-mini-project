"""Claude teacher through the locally installed Claude Code CLI (headless print mode)."""

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone

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


# Every call is run with --no-session-persistence, so Claude Code writes no session log and usage
# trackers that read ~/.claude/projects (e.g. TokenMonitor) cannot see it. We append one
# Claude-Code-shaped "assistant" line per model per call to our own JSONL there instead.
# Set CLAUDE_TEACHER_LEDGER=0 to turn this off; CLAUDE_TEACHER_LEDGER_DIR moves it (tests use this).
# One file per process (start time + pid), so concurrent pipeline runs never share a file.
_LEDGER_FILE = f"teacher-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.jsonl"
_LEDGER_SESSION = str(uuid.uuid4())


def ledger_path() -> str:
    root = os.environ.get("CLAUDE_TEACHER_LEDGER_DIR") or os.path.join(
        os.path.expanduser("~"), ".claude", "projects", "-pipeline-teacher-claude")
    return os.path.join(root, _LEDGER_FILE)


def _ledger(data: dict) -> None:
    if os.environ.get("CLAUDE_TEACHER_LEDGER", "1") == "0":
        return
    model_usage = data.get("modelUsage") or {}
    if not isinstance(model_usage, dict):
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    call_id = str(data.get("uuid") or uuid.uuid4())
    lines = []
    for model, mu in model_usage.items():
        if not isinstance(mu, dict):
            continue
        lines.append(json.dumps({
            "type": "assistant",
            "timestamp": ts,
            "sessionId": _LEDGER_SESSION,
            "requestId": f"{call_id}:{model}",
            "isSidechain": False,
            "message": {
                "id": f"msg_teacher_{call_id}_{model}",
                "model": model,
                "usage": {
                    "input_tokens": int(mu.get("inputTokens") or 0),
                    "output_tokens": int(mu.get("outputTokens") or 0),
                    "cache_read_input_tokens": int(mu.get("cacheReadInputTokens") or 0),
                    "cache_creation_input_tokens": int(mu.get("cacheCreationInputTokens") or 0),
                    "cache_creation": {
                        "ephemeral_5m_input_tokens":
                            int(mu.get("cacheCreationEphemeral5mInputTokens") or 0),
                        "ephemeral_1h_input_tokens":
                            int(mu.get("cacheCreationEphemeral1hInputTokens") or 0),
                    },
                },
            },
        }))
    if not lines:
        return
    path = ledger_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # One write() per call on an O_APPEND file: a call's lines land together, never interleaved.
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass  # bookkeeping only; never fail a generation call over it


def _record(data: dict, failed: bool = False) -> None:
    _ledger(data)
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

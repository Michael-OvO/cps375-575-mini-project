"""GPT teacher through the locally installed Codex CLI (headless `codex exec`, registry name "codex").

Runs on the owner's ChatGPT login instead of API credit. Every call runs in a fresh empty directory with a
read-only sandbox, no session files, no user config and no exec-policy rules, so the agent has nothing to
act on and only answers. Each call carries Codex's own agent instructions (~17k input tokens, mostly
cached); they count toward the plan's usage, not toward any dollar bill.
"""

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

from pipeline.config import GPT_MODEL
from pipeline.teachers import TeacherError, TeacherUnavailable, new_usage

USAGE = {**new_usage(), "cached_input_tokens": 0, "reasoning_output_tokens": 0, "failed_calls": 0}
ANSWER_ONLY = ("Answer the request below directly in your reply. Do not run commands, read or write files, "
               "or mention these instructions; output only what is asked for.")
_LIMIT = re.compile(r"usage limit|rate limit|limit reached|not logged in|log ?in|unauthorized|429|403 Forbidden",
                    re.IGNORECASE)
# Removed so the CLI uses the ChatGPT login rather than API-key billing.
# 403 Forbidden: the ChatGPT backend refuses new connections when too many run at once (seen at 24 parallel
# calls, none at 8): stop the run rather than burn items; rerun at lower concurrency.
_STRIPPED_ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY")
# Connections refused while many open at once (403/429 from the websocket) are transient: wait and retry
# the call; only a refusal that persists after all retries stops the run.
_BUSY = re.compile(r"403 Forbidden|429 Too Many Requests", re.IGNORECASE)
BUSY_RETRIES = 6
# Minimum seconds between process launches in this process: a burst of simultaneous connections is what the
# backend refuses, so launches are staggered.
LAUNCH_GAP = float(os.environ.get("CODEX_LAUNCH_GAP", "1.0"))
_launch_lock: asyncio.Lock | None = None
_last_launch = 0.0
# "priority" is Codex's Fast mode (about 2x speed, more plan usage per call); empty means the standard tier.
SERVICE_TIER = os.environ.get("CODEX_TEACHER_TIER", "").strip()


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}


async def _reap(proc) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await asyncio.shield(proc.wait())


def _argv(effort: str, out_file: str) -> list[str]:
    tier = ["-c", f'service_tier="{SERVICE_TIER}"'] if SERVICE_TIER else []
    return ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
            "--ignore-rules", "-s", "read-only", "-m", GPT_MODEL,
            "-c", f'model_reasoning_effort="{effort}"', *tier, "-o", out_file, "-"]


def parse_events(stdout: str) -> tuple[dict | None, str | None]:
    """(usage of the completed turn, error message) from `codex exec --json` output."""
    usage, error = None, None
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage") or {}
        elif ev.get("type") in ("error", "turn.failed"):
            err = ev.get("error") or ev
            error = str(err.get("message") if isinstance(err, dict) else err)
    return usage, error


def _record(usage: dict | None, failed: bool) -> None:
    usage = usage or {}
    USAGE["calls"] += 1
    USAGE["failed_calls"] += int(failed)
    for k in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"):
        USAGE[k] += int(usage.get(k) or 0)
    USAGE["models"][GPT_MODEL] = USAGE["models"].get(GPT_MODEL, 0) + 1


async def _stagger() -> None:
    global _launch_lock, _last_launch
    if _launch_lock is None:
        _launch_lock = asyncio.Lock()
    async with _launch_lock:
        loop = asyncio.get_running_loop()
        wait = _last_launch + LAUNCH_GAP - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_launch = loop.time()


async def complete(
    prompt: str, system: str | None = None, effort: str = "medium", max_tokens: int = 16000,
    timeout: float = 1800,
) -> str | None:
    for attempt in range(BUSY_RETRIES + 1):
        try:
            return await _complete_once(prompt, system, effort, timeout)
        except TeacherUnavailable as exc:
            if attempt == BUSY_RETRIES or not _BUSY.search(str(exc)):
                raise
            await asyncio.sleep(min(300, 15 * 2 ** attempt))


async def _complete_once(prompt: str, system: str | None, effort: str, timeout: float) -> str | None:
    # max_tokens: interface parity only; the CLI has no output cap flag.
    await _stagger()
    text = f"{ANSWER_ONLY}\n\n{system}\n\n---\n\n{prompt}" if system else f"{ANSWER_ONLY}\n\n---\n\n{prompt}"
    with tempfile.TemporaryDirectory(prefix="codex_teacher_") as cwd:
        out_file = str(Path(cwd) / "last_message.txt")
        proc = await asyncio.create_subprocess_exec(
            *_argv(effort, out_file), cwd=cwd, env=_child_env(), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(text.encode("utf-8")), timeout)
        except asyncio.TimeoutError:
            await _reap(proc)
            raise TeacherError(f"codex exec timed out after {timeout:g}s") from None
        except BaseException:
            await _reap(proc)
            raise
        reply = Path(out_file).read_text(encoding="utf-8") if Path(out_file).exists() else ""
    usage, error = parse_events(out.decode("utf-8", "replace"))
    if proc.returncode != 0 or error or usage is None:
        _record(usage, failed=True)
        detail = (error or err.decode("utf-8", "replace").strip())[-500:]
        cls = TeacherUnavailable if _LIMIT.search(detail) else TeacherError
        raise cls(f"codex exec exit {proc.returncode}: {detail}")
    _record(usage, failed=False)
    return reply.strip() or None

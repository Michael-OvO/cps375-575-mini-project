import asyncio
import json

import pytest

from pipeline import teachers
from pipeline.teachers import claude_code


class FakeProc:
    def __init__(self, out, err="", code=0, hang=False):
        self._out, self._err, self.returncode, self.hang = out, err, code, hang
        self.stdin_data = None
        self.killed = False

    async def communicate(self, data=None):
        self.stdin_data = data
        if self.hang:
            await asyncio.sleep(3600)
        return self._out.encode(), self._err.encode()

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


def good(result="hello", **over):
    data = {"result": result, "is_error": False, "subtype": "success", "stop_reason": "end_turn",
            "api_error_status": None, "total_cost_usd": 0.125,
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "modelUsage": {"claude-fable-5-2": {"inputTokens": 10, "outputTokens": 20, "costUSD": 0.125}}}
    data.update(over)
    return json.dumps(data)


@pytest.fixture
def cli(monkeypatch):
    state = {"proc": None, "argv": None, "kwargs": None}
    monkeypatch.setattr(claude_code, "USAGE", {**teachers.new_usage(), "cost_usd": 0.0, "failed_calls": 0})

    def install(proc):
        async def fake_exec(*argv, **kwargs):
            state.update(argv=list(argv), kwargs=kwargs)
            return proc
        state["proc"] = proc
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        return state
    state["install"] = install
    return state


def run(cli, proc, *a, **kw):
    cli["install"](proc)
    return asyncio.run(claude_code.complete(*a, **kw))


def test_success_records_usage_cost_and_model(cli):
    assert run(cli, FakeProc(good("  hello \n")), "p") == "hello"
    run(cli, FakeProc(good()), "p")
    snap = teachers.usage_snapshot()["claude"]
    assert snap == {"calls": 2, "input_tokens": 20, "output_tokens": 40, "cost_usd": 0.25,
                    "failed_calls": 0, "models": {"claude-fable-5-2": 2}}


def test_argv_stdin_and_cwd(cli):
    run(cli, FakeProc(good()), "a long prompt", system="SYS", effort="high", max_tokens=5)
    argv = cli["argv"]
    assert argv[0] == "claude" and "a long prompt" not in argv
    assert cli["proc"].stdin_data == b"a long prompt"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1] == "SYS"
    assert argv[argv.index("--model") + 1] == claude_code.CLAUDE_MODEL
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "5" not in argv and "--max-tokens" not in argv
    assert "--output-format" in argv and "--no-session-persistence" in argv
    assert cli["kwargs"]["cwd"] and "claude_teacher_" in cli["kwargs"]["cwd"]


def test_default_system_prompt(cli):
    run(cli, FakeProc(good()), "p")
    argv = cli["argv"]
    assert argv[argv.index("--system-prompt") + 1] == claude_code.DEFAULT_SYSTEM


@pytest.mark.parametrize("payload", [
    good(""), good("   "), good("x", stop_reason="refusal"), good("x", stop_reason="max_tokens")])
def test_content_outcomes_return_none(cli, payload):
    assert run(cli, FakeProc(payload), "p") is None


def test_errors_raise_teacher_error(cli):
    for proc in (FakeProc("", "boom", code=1), FakeProc("not json"),
                 FakeProc(good("oops", is_error=True, subtype="error_during_execution"))):
        with pytest.raises(teachers.TeacherError) as err:
            run(cli, proc, "p")
        assert not isinstance(err.value, teachers.TeacherUnavailable)


def test_limit_and_auth_raise_unavailable(cli):
    cases = [
        FakeProc(good("Claude AI usage limit reached", is_error=True)),
        FakeProc(good("x", is_error=True, api_error_status=429)),
        FakeProc(good("x", is_error=True, api_error_status=401)),
        FakeProc(good("x", is_error=True, api_error_status=403)),
        FakeProc("", "Error: Not logged in. Please run /login", code=1),
        FakeProc("", "rate limit exceeded", code=1),
    ]
    for proc in cases:
        with pytest.raises(teachers.TeacherUnavailable):
            run(cli, proc, "p")
    with pytest.raises(teachers.TeacherUnavailable, match="usage limit reached"):
        run(cli, FakeProc(good("Claude AI usage limit reached", is_error=True)), "p")


def test_timeout_kills_process(cli):
    proc = FakeProc("", hang=True)
    with pytest.raises(teachers.TeacherError, match="timed out"):
        run(cli, proc, "p", timeout=0.05)
    assert proc.killed


def test_env_strips_api_credentials(cli, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("HOME", "/home/someone")
    run(cli, FakeProc(good()), "p")
    env = cli["kwargs"]["env"]
    assert not any(k.startswith("ANTHROPIC_") for k in env if k in claude_code._STRIPPED_ENV)
    assert "PATH" in env and env["HOME"] == "/home/someone"


def test_cancellation_kills_and_awaits_process(cli):
    proc = FakeProc("", hang=True)
    waited = []
    orig_wait = proc.wait

    async def wait():
        waited.append(True)
        return await orig_wait()
    proc.wait = wait
    cli["install"](proc)

    async def main():
        task = asyncio.ensure_future(claude_code.complete("p"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert proc.killed and waited


def test_kill_tolerates_vanished_process(cli):
    proc = FakeProc("", hang=True)

    def kill():
        raise ProcessLookupError
    proc.kill = kill
    with pytest.raises(teachers.TeacherError, match="timed out"):
        run(cli, proc, "p", timeout=0.05)


def test_non_success_subtype_is_error_even_without_is_error(cli):
    with pytest.raises(teachers.TeacherError, match="error_max_turns"):
        run(cli, FakeProc(good("partial", subtype="error_max_turns")), "p")


def test_success_text_mentioning_limits_is_returned(cli):
    text = "Explains the rate limit and the login flow."
    assert run(cli, FakeProc(good(text)), "p") == text


def test_bare_login_or_authentication_is_not_unavailable(cli):
    for msg in ("login page crashed", "authentication module failed"):
        with pytest.raises(teachers.TeacherError) as err:
            run(cli, FakeProc(good(msg, is_error=True)), "p")
        assert not isinstance(err.value, teachers.TeacherUnavailable)


def test_failed_calls_and_error_spend_are_recorded(cli):
    with pytest.raises(teachers.TeacherError):
        run(cli, FakeProc(good("oops", is_error=True)), "p")
    with pytest.raises(teachers.TeacherError):
        run(cli, FakeProc("", "boom", code=1), "p")  # no JSON: counted failed, no tokens
    snap = teachers.usage_snapshot()["claude"]
    assert snap["failed_calls"] == 2 and snap["calls"] == 1
    assert snap["input_tokens"] == 10 and snap["cost_usd"] == 0.125


def test_failed_calls_merge_on_write(monkeypatch, tmp_path):
    out = tmp_path / "p.jsonl"
    snap = {"claude": {"calls": 1, "failed_calls": 1, "input_tokens": 1, "output_tokens": 1,
                       "cost_usd": 0.5, "models": {"m": 1}}}
    monkeypatch.setattr(teachers, "usage_snapshot", lambda: snap)
    teachers.write_usage(out)
    total = teachers.write_usage(out)
    assert total["claude"]["failed_calls"] == 2 and total["claude"]["cost_usd"] == 1.0

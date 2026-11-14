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
    monkeypatch.setattr(claude_code, "USAGE", {**teachers.new_usage(), "cost_usd": 0.0})

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
                    "models": {"claude-fable-5-2": 2}}


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

import asyncio

from pipeline import jsonl, teachers
from pipeline.teachers import runner


def test_parse_json():
    assert teachers.parse_json('Here:\n```json\n["a", "b"]\n```') == ["a", "b"]
    assert teachers.parse_json('{"x": 1} trailing') == {"x": 1}
    assert teachers.parse_json("no json") is None


def test_run_many_resumes(tmp_path, caplog):
    out = tmp_path / "o.jsonl"
    calls = []

    async def fn(item):
        calls.append(item["id"])
        if item["id"] == "b":
            return None  # could not process: must be retried
        if item["id"] == "e":
            return []  # processed, no rows: done
        return [{"id": item["id"], "v": 1}]

    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "e"}]
    with caplog.at_level("WARNING"):
        asyncio.run(runner.run_many(items, fn, out, concurrency=2))
    assert sum("1 item(s) returned None" in r.message for r in caplog.records) == 1
    assert sorted(r["id"] for r in jsonl.read(out)) == ["a", "c"]
    assert jsonl.done_ids(out) == {"a", "c", "e"}
    asyncio.run(runner.run_many(items, fn, out, concurrency=2))
    assert sorted(calls) == ["a", "b", "b", "c", "e"]  # only b re-run


def test_run_many_crash_window_no_duplicates(tmp_path):
    out = tmp_path / "o.jsonl"
    jsonl.append(out, {"id": "a", "v": 1})  # rows written, done marker never was

    async def fn(item):
        return [{"id": item["id"], "v": 1}]

    asyncio.run(runner.run_many([{"id": "a"}, {"id": "b"}], fn, out))
    ids = [r["id"] for r in jsonl.read(out)]
    assert sorted(ids) == ["a", "b"]
    assert jsonl.done_ids(out) == {"a", "b"}


def test_run_many_exception_is_retried(tmp_path):
    out = tmp_path / "o.jsonl"
    attempts = []

    async def flaky(item):
        attempts.append(item["id"])
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return [{"id": item["id"]}]

    asyncio.run(runner.run_many([{"id": "a"}], flaky, out))
    assert jsonl.done_ids(out) == set()
    assert jsonl.read(out) == []
    asyncio.run(runner.run_many([{"id": "a"}], flaky, out))
    assert jsonl.done_ids(out) == {"a"}
    assert [r["id"] for r in jsonl.read(out)] == ["a"]


def test_registry_names():
    for name in ("claude", "claude_api", "gpt", "tinker"):
        assert callable(teachers.get(name))
    from pipeline.teachers import claude, claude_code
    assert teachers.get("claude") is claude_code.complete
    assert teachers.get("claude_api") is claude.complete


def test_parse_json_repairs_single_backslash_latex():
    reply = r'["\frac{1}{2} \times \sqrt{2}", "\\frac{3}{4}"]'
    assert teachers.parse_json(reply) == [r"\frac{1}{2} \times \sqrt{2}", r"\frac{3}{4}"]
    assert teachers.parse_json(r'["$\nu \neq \rho \beta$"]') == [r"$\nu \neq \rho \beta$"]
    assert teachers.parse_json(r'["Find \frac{1}{2}.\nThen \times 2."]') == ["Find \\frac{1}{2}.\nThen \\times 2."]
    # Valid escapes are kept: \n before a non-letter, \" and é.
    assert teachers.parse_json(r'["a\n1", "say \"hi\"", "café"]') == ["a\n1", 'say "hi"', "café"]


def test_parse_json_code_mode_keeps_newlines():
    reply = r'[{"code": "import re\ndef f(s):\n\treturn re.findall(\"\d+\", s)"}]'
    assert teachers.parse_json(reply, latex=False) == [
        {"code": 'import re\ndef f(s):\n\treturn re.findall("\\d+", s)'}]


def test_parse_json_never_returns_inner_bracket_of_failed_parse():
    assert teachers.parse_json('[{"a": 1}, {"b": oops}]') is None
    assert teachers.parse_json('["one", ["two"], "trunc') is None  # truncated outer array
    # Only a value starting at the first bracket counts: a bracket in the prose makes the candidate fail.
    assert teachers.parse_json('See [the list below]: ["x", "y"]') is None
    assert teachers.parse_json('Here are the problems: ["x", "y"] Done.') == ["x", "y"]
    assert teachers.parse_json('Sure [note]\n```json\n["x"]\n```') == ["x"]  # fenced block first


def test_parse_json_malformed_quote_returns_none():
    assert teachers.parse_json('["a"b", "]x [0, 1]"]') is None


def test_parse_json_latex_n_commands_vs_real_newlines():
    assert teachers.parse_json(r'["Line one.\nLine two has x."]') == ["Line one.\nLine two has x."]
    assert teachers.parse_json(r'["\\frac{1}{2}\nThen find $x$."]') == ["\\frac{1}{2}\nThen find $x$."]
    assert teachers.parse_json(r'["Given this.\n\nFind $y$."]') == ["Given this.\n\nFind $y$."]
    reply = r'["$\nu$", "$x \neq 0$", "$\nabla f$", "$a \not\in B$", "a\newline b", "$\notin$"]'
    assert teachers.parse_json(reply) == [r"$\nu$", r"$x \neq 0$", r"$\nabla f$", r"$a \not\in B$",
                                         r"a\newline b", r"$\notin$"]
    assert teachers.parse_json(r'["end \nu"]') == [r"end \nu"]  # command at end of string
    assert teachers.parse_json(r'["$\rho$ and $\beta$"]') == [r"$\rho$ and $\beta$"]


def test_has_control_chars():
    assert teachers.has_control_chars("a\x0cb") and teachers.has_control_chars("a\tb")
    assert not teachers.has_control_chars("line one\nline two\r\n")
    assert not teachers.has_control_chars("a\tb", allow_tab=True)
    assert teachers.has_control_chars("a\x0cb", allow_tab=True)


def _clear_teacher_env(monkeypatch, tmp_path):
    for var in ("OPENAI_API_KEY", "OPENAI_TEACHER_MODEL", "TINKER_API_KEY",
                "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(teachers.Path, "home", lambda: tmp_path)  # no ~/.config/anthropic


def test_check_configured_lists_everything_missing(monkeypatch, tmp_path):
    import pytest
    _clear_teacher_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as err:
        teachers.check_configured(["claude_api", "gpt", "tinker"])
    msg = str(err.value)
    for needle in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TINKER_API_KEY"):
        assert needle in msg


def test_check_configured_accepts_each_credential_form(monkeypatch, tmp_path):
    import pytest
    _clear_teacher_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        teachers.check_configured(["gpt"])
    monkeypatch.setenv("OPENAI_API_KEY", "x")  # the model id has a default (config.GPT_MODEL)
    monkeypatch.setenv("TINKER_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    teachers.check_configured(["gpt", "tinker", "claude_api"])
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    (tmp_path / ".config" / "anthropic").mkdir(parents=True)
    teachers.check_configured(["claude_api"])


def test_check_configured_claude_needs_cli_on_path(monkeypatch):
    import pytest
    monkeypatch.setattr(teachers.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="claude: the `claude`"):
        teachers.check_configured(["claude"])
    monkeypatch.setattr(teachers.shutil, "which", lambda n: "/usr/bin/claude")
    teachers.check_configured(["claude"])


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Resp:
    def __init__(self, model, i, o, text="hi"):
        self.model, self.usage = model, _Usage(i, o)
        self.stop_reason, self.status, self.output_text = "end_turn", "completed", text
        self.content = [type("B", (), {"type": "text", "text": text})()]


def test_claude_and_gpt_record_usage_and_served_model(monkeypatch):
    from pipeline.teachers import claude, gpt

    class FakeAnthropic:
        class beta:
            class messages:
                @staticmethod
                async def create(**kw):
                    return _Resp("claude-fable-5-2", 10, 20)

    class FakeOpenAI:
        class responses:
            @staticmethod
            async def create(**kw):
                sent.update(kw)
                return _Resp("gpt-served-id", 3, 4)

    sent = {}
    monkeypatch.setattr(claude, "USAGE", teachers.new_usage())
    monkeypatch.setattr(gpt, "USAGE", teachers.new_usage())
    monkeypatch.setattr(claude, "_get_client", lambda: FakeAnthropic)
    monkeypatch.setattr(gpt, "_get_client", lambda: FakeOpenAI)
    monkeypatch.setattr(gpt, "GPT_MODEL", "gpt-requested")
    assert asyncio.run(claude.complete("p")) == "hi"
    assert asyncio.run(claude.complete("p")) == "hi"
    assert asyncio.run(gpt.complete("p", effort="high")) == "hi"
    assert sent["reasoning"] == {"effort": "high"} and sent["model"] == "gpt-requested"
    snap = teachers.usage_snapshot()
    assert snap["claude_api"] == {"calls": 2, "input_tokens": 20, "output_tokens": 40,
                              "models": {"claude-fable-5-2": 2}}
    assert snap["gpt"] == {"calls": 1, "input_tokens": 3, "output_tokens": 4, "models": {"gpt-served-id": 1}}
    assert set(snap) == {"claude_api", "gpt"} | ({"claude"} if "claude" in snap else set())
    assert "cost_usd" not in snap["claude_api"] and "cost_usd" not in snap["gpt"]


def test_record_usage_tolerates_missing_usage():
    counter = teachers.new_usage()
    teachers.record_usage(counter, type("R", (), {"usage": None, "model": None})())
    assert counter == {"calls": 1, "input_tokens": 0, "output_tokens": 0, "models": {"unknown": 1}}


def test_write_usage_sums_with_existing_file(monkeypatch, tmp_path, capsys):
    import json
    out = tmp_path / "pool.jsonl"
    snap = {"claude": {"calls": 2, "input_tokens": 5, "output_tokens": 7, "models": {"m1": 2}}}
    monkeypatch.setattr(teachers, "usage_snapshot", lambda: snap)
    teachers.write_usage(out)
    snap = {"claude": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "models": {"m1": 1, "m2": 0}},
            "gpt": {"calls": 1, "input_tokens": 2, "output_tokens": 3, "models": {"g": 1}}}
    monkeypatch.setattr(teachers, "usage_snapshot", lambda: snap)
    total = teachers.write_usage(out)
    assert total == json.loads((tmp_path / "pool.jsonl.usage.json").read_text())
    assert total["claude"] == {"calls": 3, "input_tokens": 6, "output_tokens": 8, "models": {"m1": 3, "m2": 0}}
    assert total["gpt"]["calls"] == 1
    assert "teacher usage" in capsys.readouterr().out


def test_write_usage_sums_cost_usd(monkeypatch, tmp_path):
    out = tmp_path / "pool.jsonl"
    one = {"claude": {"calls": 1, "input_tokens": 1, "output_tokens": 2, "cost_usd": 0.25, "models": {"m": 1}}}
    monkeypatch.setattr(teachers, "usage_snapshot", lambda: one)
    teachers.write_usage(out)
    total = teachers.write_usage(out)
    assert total["claude"]["cost_usd"] == 0.5 and total["claude"]["calls"] == 2
    old = {"claude": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "models": {"m": 1}}}  # no cost field
    (tmp_path / "x.jsonl.usage.json").write_text(__import__("json").dumps(old))
    assert teachers.write_usage(tmp_path / "x.jsonl")["claude"]["cost_usd"] == 0.25


def test_run_many_teacher_unavailable_stops_and_resumes(tmp_path, caplog):
    out = tmp_path / "o.jsonl"
    started = []
    broken = {"on": True}

    async def fn(item):
        started.append(item["id"])
        if broken["on"] and item["id"] == "c":
            raise teachers.TeacherUnavailable("usage limit reached")
        await asyncio.sleep(0)
        return [{"id": item["id"]}]

    items = [{"id": str(i)} for i in "abcdefgh"]
    with caplog.at_level("ERROR"):
        asyncio.run(runner.run_many(items, fn, out, concurrency=1))  # returns normally
    assert started == ["a", "b", "c"]
    assert jsonl.done_ids(out) == {"a", "b"}
    assert sorted(r["id"] for r in jsonl.read(out)) == ["a", "b"]
    errors = [r.message for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "usage limit reached" in errors[0] and "5 of 8" in errors[0]
    broken["on"] = False
    asyncio.run(runner.run_many(items, fn, out, concurrency=1))
    assert jsonl.done_ids(out) == {i["id"] for i in items}
    assert started[3:] == ["c", "d", "e", "f", "g", "h"]


def test_run_many_unavailable_lets_in_flight_finish(tmp_path):
    out = tmp_path / "o.jsonl"

    async def fn(item):
        if item["id"] == "a":
            await asyncio.sleep(0.01)  # b is already in flight when this raises
            raise teachers.TeacherUnavailable("429")
        await asyncio.sleep(0.05)
        return [{"id": item["id"]}]

    asyncio.run(runner.run_many([{"id": "a"}, {"id": "b"}, {"id": "c"}], fn, out, concurrency=2))
    assert jsonl.done_ids(out) == {"b"}  # b was in flight and finished; c never started


def test_run_many_stops_after_25_consecutive_failures(tmp_path, caplog):
    out = tmp_path / "o.jsonl"
    started = []

    async def fn(item):
        started.append(item["id"])
        raise ValueError("bad")

    items = [{"id": str(i)} for i in range(40)]
    with caplog.at_level("ERROR"):
        asyncio.run(runner.run_many(items, fn, out, concurrency=1))
    assert len(started) == 25 and jsonl.done_ids(out) == set()
    assert any("25 consecutive" in r.message for r in caplog.records)


def test_run_many_success_resets_failure_counter(tmp_path):
    out = tmp_path / "o.jsonl"

    async def fn(item):
        n = int(item["id"])
        if n % 25 == 24:  # every 25th item succeeds: never 25 failures in a row
            return [{"id": item["id"]}]
        raise ValueError("bad")

    items = [{"id": str(i)} for i in range(100)]
    asyncio.run(runner.run_many(items, fn, out, concurrency=1))
    assert len(jsonl.done_ids(out)) == 4  # ran to completion


def test_run_many_none_results_count_as_failures(tmp_path):
    out = tmp_path / "o.jsonl"
    started = []

    async def fn(item):
        started.append(item["id"])
        return None

    asyncio.run(runner.run_many([{"id": str(i)} for i in range(40)], fn, out, concurrency=1))
    assert len(started) == 25 and jsonl.done_ids(out) == set()


def test_run_many_list_result_resets_none_counter(tmp_path):
    out = tmp_path / "o.jsonl"

    async def fn(item):
        return [] if int(item["id"]) % 25 == 24 else None

    asyncio.run(runner.run_many([{"id": str(i)} for i in range(100)], fn, out, concurrency=1))
    assert len(jsonl.done_ids(out)) == 4  # ran to completion


def test_codex_parse_events_reads_usage_and_errors():
    from pipeline.teachers import codex_cli
    ok = ('{"type":"thread.started","thread_id":"t"}\n{"type":"item.completed","item":{"type":"agent_message",'
          '"text":"391"}}\n{"type":"turn.completed","usage":{"input_tokens":17410,"output_tokens":5}}\n')
    assert codex_cli.parse_events(ok) == ({"input_tokens": 17410, "output_tokens": 5}, None)
    bad = '{"type":"error","message":"You have hit your usage limit"}\nnot json\n'
    assert codex_cli.parse_events(bad) == (None, "You have hit your usage limit")
    argv = codex_cli._argv("high", "/x/out.txt")
    assert {"-s", "read-only", "--ephemeral", "--ignore-user-config"} <= set(argv)
    assert 'model_reasoning_effort="high"' in argv and argv[-1] == "-"
    assert "OPENAI_API_KEY" in codex_cli._STRIPPED_ENV


def test_codex_registered():
    assert teachers.get("codex").__module__ == "pipeline.teachers.codex_cli"

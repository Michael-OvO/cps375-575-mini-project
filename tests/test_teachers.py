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
    for name in ("claude", "gpt", "tinker"):
        assert callable(teachers.get(name))


def test_parse_json_repairs_single_backslash_latex():
    reply = r'["\frac{1}{2} \times \sqrt{2}", "\\frac{3}{4}"]'
    assert teachers.parse_json(reply) == [r"\frac{1}{2} \times \sqrt{2}", r"\frac{3}{4}"]
    assert teachers.parse_json(r'["$\nu \neq \rho \beta$"]') == [r"$\nu \neq \rho \beta$"]
    # Valid escapes are kept: \n before a non-letter, \" and é.
    assert teachers.parse_json(r'["a\n1", "say \"hi\"", "café"]') == ["a\n1", 'say "hi"', "café"]


def test_parse_json_code_mode_keeps_newlines():
    reply = r'[{"code": "import re\ndef f(s):\n\treturn re.findall(\"\d+\", s)"}]'
    assert teachers.parse_json(reply, latex=False) == [
        {"code": 'import re\ndef f(s):\n\treturn re.findall("\\d+", s)'}]


def test_parse_json_never_returns_inner_bracket_of_failed_parse():
    assert teachers.parse_json('[{"a": 1}, {"b": oops}]') is None
    assert teachers.parse_json('["one", ["two"], "trunc') is None  # truncated outer array
    assert teachers.parse_json('See [the list below]: ["x", "y"]') == ["x", "y"]


def test_has_control_chars():
    assert teachers.has_control_chars("a\x0cb") and teachers.has_control_chars("a\tb")
    assert not teachers.has_control_chars("line one\nline two\r\n")


def _clear_teacher_env(monkeypatch, tmp_path):
    for var in ("OPENAI_API_KEY", "OPENAI_TEACHER_MODEL", "TINKER_API_KEY",
                "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(teachers.Path, "home", lambda: tmp_path)  # no ~/.config/anthropic


def test_check_configured_lists_everything_missing(monkeypatch, tmp_path):
    import pytest
    _clear_teacher_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as err:
        teachers.check_configured(["claude", "gpt", "tinker"])
    msg = str(err.value)
    for needle in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_TEACHER_MODEL", "TINKER_API_KEY"):
        assert needle in msg


def test_check_configured_accepts_each_credential_form(monkeypatch, tmp_path):
    import pytest
    _clear_teacher_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_TEACHER_MODEL", "  ")
    with pytest.raises(RuntimeError, match="OPENAI_TEACHER_MODEL"):
        teachers.check_configured(["gpt"])
    monkeypatch.setenv("OPENAI_TEACHER_MODEL", "some-model")
    monkeypatch.setenv("TINKER_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    teachers.check_configured(["gpt", "tinker", "claude"])
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    (tmp_path / ".config" / "anthropic").mkdir(parents=True)
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
                return _Resp("gpt-served-id", 3, 4)

    monkeypatch.setattr(claude, "USAGE", teachers.new_usage())
    monkeypatch.setattr(gpt, "USAGE", teachers.new_usage())
    monkeypatch.setattr(claude, "_get_client", lambda: FakeAnthropic)
    monkeypatch.setattr(gpt, "_get_client", lambda: FakeOpenAI)
    monkeypatch.setattr(gpt, "GPT_MODEL", "gpt-requested")
    assert asyncio.run(claude.complete("p")) == "hi"
    assert asyncio.run(claude.complete("p")) == "hi"
    assert asyncio.run(gpt.complete("p")) == "hi"
    snap = teachers.usage_snapshot()
    assert snap["claude"] == {"calls": 2, "input_tokens": 20, "output_tokens": 40,
                              "models": {"claude-fable-5-2": 2}}
    assert snap["gpt"] == {"calls": 1, "input_tokens": 3, "output_tokens": 4, "models": {"gpt-served-id": 1}}


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

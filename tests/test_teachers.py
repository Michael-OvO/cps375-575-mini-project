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

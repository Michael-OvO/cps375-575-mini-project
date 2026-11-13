import asyncio

from pipeline import jsonl, teachers
from pipeline.teachers import runner


def test_parse_json():
    assert teachers.parse_json('Here:\n```json\n["a", "b"]\n```') == ["a", "b"]
    assert teachers.parse_json('{"x": 1} trailing') == {"x": 1}
    assert teachers.parse_json("no json") is None


def test_run_many_resumes(tmp_path):
    out = tmp_path / "o.jsonl"
    calls = []

    async def fn(item):
        calls.append(item["id"])
        if item["id"] == "b":
            return None
        return [{"id": item["id"], "v": 1}]

    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    asyncio.run(runner.run_many(items, fn, out, concurrency=2))
    assert sorted(r["id"] for r in jsonl.read(out)) == ["a", "c"]
    asyncio.run(runner.run_many(items, fn, out, concurrency=2))
    assert sorted(calls) == ["a", "b", "c"]  # nothing re-run


def test_registry_names():
    for name in ("claude", "gpt", "tinker"):
        assert callable(teachers.get(name))

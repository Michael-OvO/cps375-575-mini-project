import asyncio
import json

from pipeline.generate import code as gc

TASK = {
    "text": "Write a function to double a number.",
    "code": "def double(n):\n    return n * 2",
    "tests": ["assert double(2) == 4", "assert double(0) == 0", "assert double(-1) == -2"],
}


def test_write_tasks_filters_failing(monkeypatch):
    bad = dict(TASK, code="def double(n):\n    return n + 2")

    async def fake(prompt, **kw):
        return json.dumps([TASK, bad, {"text": "broken"}])

    monkeypatch.setattr(gc.teachers, "get", lambda name: fake)
    rows = asyncio.run(gc.write_tasks({"id": "ct-0", "category": "arithmetic", "author": "claude"}))
    assert len(rows) == 1 and rows[0]["id"] == "ct-0-0" and rows[0]["code"] == TASK["code"]


def test_add_hidden_requires_all_pass(monkeypatch):
    task = dict(TASK, id="ct-0-0", category="arithmetic", author="claude")

    async def good(prompt, **kw):
        return json.dumps(["assert double(5) == 10", "assert double(10) == 20", "assert double(1) == 2",
                           "assert double(-3) == -6", "assert double(7) == 14"])

    async def wrong(prompt, **kw):
        return json.dumps(["assert double(5) == 11"] * 5)

    monkeypatch.setattr(gc.teachers, "get", lambda name: good)
    rows = asyncio.run(gc.add_hidden(task, ["claude", "gpt"]))
    assert len(rows) == 1 and len(rows[0]["hidden_tests"]) == 5
    monkeypatch.setattr(gc.teachers, "get", lambda name: wrong)
    assert asyncio.run(gc.add_hidden(task, ["claude", "gpt"])) == []


def test_write_tasks_returns_none_when_teacher_returns_nothing(monkeypatch):
    async def nothing(prompt, **kw):
        return None

    monkeypatch.setattr(gc.teachers, "get", lambda name: nothing)
    assert asyncio.run(gc.write_tasks({"id": "ct-0", "category": "arithmetic", "author": "claude"})) is None


def test_write_tasks_returns_empty_on_unparseable_reply(monkeypatch):
    async def junk(prompt, **kw):
        return "not json at all"

    monkeypatch.setattr(gc.teachers, "get", lambda name: junk)
    assert asyncio.run(gc.write_tasks({"id": "ct-0", "category": "arithmetic", "author": "claude"})) == []


def test_add_hidden_returns_none_when_teacher_returns_nothing(monkeypatch):
    task = dict(TASK, id="ct-0-0", category="arithmetic", author="claude")

    async def nothing(prompt, **kw):
        return None

    monkeypatch.setattr(gc.teachers, "get", lambda name: nothing)
    assert asyncio.run(gc.add_hidden(task, ["claude", "gpt"])) is None

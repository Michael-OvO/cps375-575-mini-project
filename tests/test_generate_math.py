import asyncio

from pipeline.generate import math as gm


def test_split_solution():
    assert gm.split_solution("Add the two numbers.\nANSWER: 5") == ("Add the two numbers.", "5")
    assert gm.split_solution("no answer line") is None
    assert gm.split_solution("ANSWER: 5") is None  # no working shown


def test_solve_one_keeps_only_correct(monkeypatch):
    async def right(prompt, **kw):
        return "Two plus three is five.\nANSWER: 5"

    async def wrong(prompt, **kw):
        return "This is a guess.\nANSWER: 6"

    item = {"id": "mt-1", "problem": "2+3?", "target": "5", "level": 1, "subject": "Algebra"}
    monkeypatch.setattr(gm.teachers, "get", lambda name: right if name == "claude" else wrong)
    rows = asyncio.run(gm.solve_known(item, ["claude", "gpt"], source="math_train"))
    assert [r["teacher"] for r in rows] == ["claude"]
    assert rows[0]["solution"] == "Two plus three is five." and rows[0]["pid"] == "mt-1"


def test_solve_unknown_uses_tiebreak(monkeypatch):
    answers = {"claude": ["First we add them.\nANSWER: 7", "Adding again carefully gives this.\nANSWER: 7"],
               "gpt": ["We multiply the terms.\nANSWER: 8"]}

    def fake_get(name):
        async def fn(prompt, **kw):
            return answers[name].pop(0)
        return fn

    monkeypatch.setattr(gm.teachers, "get", fake_get)
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    rows = asyncio.run(gm.solve_unknown(item, ["claude", "gpt"]))
    assert {r["target"] for r in rows} == {"7"}
    assert all(r["source"] == "math_new" for r in rows)


def _fixed(replies):
    """get() stub: each teacher name maps to a list of replies consumed in order."""
    def fake_get(name):
        async def fn(prompt, **kw):
            return replies[name].pop(0)
        return fn
    return fake_get


def test_solve_known_none_when_no_teacher_responds(monkeypatch):
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [None], "gpt": [None]}))
    item = {"id": "mt-1", "problem": "2+3?", "target": "5", "level": 1, "subject": "Algebra"}
    assert asyncio.run(gm.solve_known(item, ["claude", "gpt"], source="math_train")) is None


def test_solve_known_empty_when_wrong_and_other_none(monkeypatch):
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": ["A guess.\nANSWER: 6"], "gpt": [None]}))
    item = {"id": "mt-1", "problem": "2+3?", "target": "5", "level": 1, "subject": "Algebra"}
    assert asyncio.run(gm.solve_known(item, ["claude", "gpt"], source="math_train")) == []


def test_write_problems(monkeypatch):
    item = {"id": "mn-0", "subject": "Algebra", "level": 2, "seeds": ["a", "b", "c"], "author": "claude"}
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [None]}))
    assert asyncio.run(gm.write_problems(item)) is None
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": ["not json at all"]}))
    assert asyncio.run(gm.write_problems(item)) == []
    reply = '["What is the value of $2+3$ when computed twice over?", "Find $x$ if $x+1=4$ and $x$ is real."]'
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [reply]}))
    rows = asyncio.run(gm.write_problems(item))
    assert [r["id"] for r in rows] == ["mn-0-0", "mn-0-1"]


def test_solve_unknown_single_teacher_distinct_ids(monkeypatch):
    monkeypatch.setattr(gm.teachers, "get", _fixed({"tinker": [
        "First we add them.\nANSWER: 7", "Adding again carefully gives this.\nANSWER: 7"]}))
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    rows = asyncio.run(gm.solve_unknown(item, ["tinker"]))
    ids = [r["id"] for r in rows]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert {r["teacher"] for r in rows} == {"tinker", "tinker-2"}


def test_solve_unknown_tiebreak_label_and_no_answer_line(monkeypatch):
    monkeypatch.setattr(gm.teachers, "get", _fixed({
        "claude": ["First we add them.\nANSWER: 7", "Adding again carefully gives this.\nANSWER: 7"],
        "gpt": ["We multiply the terms.\nANSWER: 8"]}))
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    rows = asyncio.run(gm.solve_unknown(item, ["claude", "gpt"]))
    assert {r["teacher"] for r in rows} == {"claude", "claude-tiebreak"}
    assert all("ANSWER:" not in r["solution"] for r in rows)


def test_solve_unknown_none_when_nothing_responds(monkeypatch):
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [None, None], "gpt": [None]}))
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    assert asyncio.run(gm.solve_unknown(item, ["claude", "gpt"])) is None

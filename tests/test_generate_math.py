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

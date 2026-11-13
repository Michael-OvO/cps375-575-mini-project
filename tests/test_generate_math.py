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


def test_write_problems_keeps_latex_and_rejects_control_chars(monkeypatch):
    item = {"id": "mn-0", "subject": "Algebra", "level": 2, "seeds": ["a", "b", "c"], "author": "claude"}
    reply = r'["Compute $\frac{1}{2} \times \sqrt{2}$ exactly, in simplest radical form.", ' \
            r'"Find $x$ such that $x \neq 0$ and $x^2 = \frac{9}{4}$ holds."]'
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [reply]}))
    rows = asyncio.run(gm.write_problems(item))
    assert [r["problem"] for r in rows] == [
        r"Compute $\frac{1}{2} \times \sqrt{2}$ exactly, in simplest radical form.",
        r"Find $x$ such that $x \neq 0$ and $x^2 = \frac{9}{4}$ holds."]
    bad = '["What is the value of \\u000crac{1}{2} plus one, written as a fraction?"]'
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [bad]}))
    assert asyncio.run(gm.write_problems(item)) == []


def test_write_prompt_asks_for_doubled_backslashes():
    text = gm.WRITE_PROMPT.format(subject="Algebra", level=3, seeds="")
    assert r"\\frac{1}{2}" in text and "double every backslash" in text


def test_solve_known_items_one_per_problem_and_teacher(monkeypatch, tmp_path):
    from pipeline import jsonl
    from pipeline.teachers import runner
    train = [{"id": "mt-1", "problem": "2+3?", "target": "5", "level": 1, "subject": "Algebra"},
             {"id": "mt-2", "problem": "1+4?", "target": "5", "level": 1, "subject": "Algebra"}]
    items = gm.solve_known_items(train, ["claude", "gpt"])
    assert [it["id"] for it in items] == ["mt-1-claude", "mt-1-gpt", "mt-2-claude", "mt-2-gpt"]

    async def claude(prompt, **kw):
        return "Two plus three is five.\nANSWER: 5"

    async def gpt(prompt, **kw):
        raise RuntimeError("gpt down")

    monkeypatch.setattr(gm.teachers, "get", lambda name: claude if name == "claude" else gpt)
    out = tmp_path / "sol.jsonl"
    asyncio.run(runner.run_many(items, lambda it: gm.solve_known_one(it, "math_train"), out))
    rows = jsonl.read(out)
    assert sorted(r["id"] for r in rows) == ["mt-1-claude", "mt-2-claude"]  # claude's paid work kept
    assert {r["pid"] for r in rows} == {"mt-1", "mt-2"} and all(r["teacher"] == "claude" for r in rows)
    assert jsonl.done_ids(out) == {"mt-1-claude", "mt-2-claude"}  # gpt items retried on rerun


def test_solve_known_one_none_without_response(monkeypatch):
    monkeypatch.setattr(gm.teachers, "get", _fixed({"gpt": [None]}))
    item = gm.solve_known_items([{"id": "mt-1", "problem": "?", "target": "5", "level": 1,
                                  "subject": "Algebra"}], ["gpt"])[0]
    assert asyncio.run(gm.solve_known_one(item, "math_train")) is None


def test_solve_unknown_retries_without_tiebreak_when_a_first_solve_raises(monkeypatch):
    calls = []

    def fake_get(name):
        async def fn(prompt, **kw):
            calls.append((name, kw.get("effort", "medium")))
            if name == "gpt":
                raise RuntimeError("boom")
            return "First we add them.\nANSWER: 7"
        return fn

    monkeypatch.setattr(gm.teachers, "get", fake_get)
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    assert asyncio.run(gm.solve_unknown(item, ["claude", "gpt"])) is None
    assert sorted(calls) == [("claude", "medium"), ("gpt", "medium")]  # no high-effort tie-break


def test_split_solution_cleans_answer():
    body = "We compute it step by step."
    assert gm.split_solution(f"{body}\nANSWER: 42**") == (body, "42")
    assert gm.split_solution(f"{body}\nANSWER: 42.") == (body, "42")
    assert gm.split_solution(f"{body}\nANSWER: $\\frac{{1}}{{2}}$") == (body, "\\frac{1}{2}")
    assert gm.split_solution(f"{body}\nANSWER: **$3\\sqrt{{2}}$.**") == (body, "3\\sqrt{2}")
    assert gm.split_solution(f"{body}\nANSWER: \\(x+1\\)") == (body, "x+1")
    assert gm.split_solution(f"{body}\nANSWER: $a$ or $b$") == (body, "$a$ or $b$")
    assert gm.split_solution(f"{body}\nANSWER: $$") is None
    assert gm.split_solution(f"{body}\nANSWER: **.**") is None

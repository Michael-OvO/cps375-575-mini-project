import asyncio

from pipeline.generate import math as gm


def test_split_solution():
    assert gm.split_solution("Add the two numbers.\nANSWER: 5") == ("Add the two numbers.", "5")
    assert gm.split_solution("no answer line") is None
    assert gm.split_solution("ANSWER: 5") is None  # no working shown
    assert gm.split_solution("$x=2$ so $2x=4$.\nANSWER: 4") == ("$x=2$ so $2x=4$.", "4")
    assert gm.split_solution("So $x$.\nANSWER: 4") is None  # trivial body


def test_solve_prompt_asks_for_latex_and_keeps_answer_line():
    assert "inside $...$" in gm.SOLVE_PROMPT and "ASCII" in gm.SOLVE_PROMPT
    assert "ANSWER: <answer>" in gm.SOLVE_PROMPT


def test_rows_carry_teachers_own_answer(monkeypatch):
    item = {"id": "mt-1", "problem": "?", "target": "\\!\\sqrt{21}", "level": 1, "subject": "Algebra"}
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": ["Root of twenty-one.\nANSWER: $\\sqrt{21}$"]}))
    rows = asyncio.run(gm.solve_known(item, ["claude"], source="math_train"))
    assert rows[0]["target"] == "\\!\\sqrt{21}" and rows[0]["answer"] == "\\sqrt{21}"
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": ["Root of twenty-one.\nANSWER: \\sqrt{21}"]}))
    one = gm.solve_known_items([item], ["claude"])[0]
    rows = asyncio.run(gm.solve_known_one(one, "math_train"))
    assert rows[0]["answer"] == "\\sqrt{21}" and rows[0]["target"] == "\\!\\sqrt{21}"
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": ["First we add them.\nANSWER: $\\dfrac12$",
                                                                 "Adding again carefully.\nANSWER: \\frac12"]}))
    new = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    rows = asyncio.run(gm.solve_unknown(new, ["claude"]))
    assert {r["answer"] for r in rows} == {"\\dfrac12", "\\frac12"} and len(rows) == 2


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
    import pytest
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(gm.solve_unknown(item, ["claude", "gpt"]))
    assert sorted(calls) == [("claude", "medium"), ("gpt", "medium")]  # no high-effort tie-break


def test_solve_unknown_reraises_unavailable_in_preference(monkeypatch):
    import pytest
    from pipeline.teachers import TeacherUnavailable

    def fake_get(name):
        async def fn(prompt, **kw):
            raise RuntimeError("other") if name == "claude" else TeacherUnavailable("429")
        return fn

    monkeypatch.setattr(gm.teachers, "get", fake_get)
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    with pytest.raises(TeacherUnavailable):
        asyncio.run(gm.solve_unknown(item, ["claude", "gpt"]))


def _new_items(n):
    return [{"id": f"mn-{i}", "problem": "?", "level": 3, "subject": "Algebra"} for i in range(n)]


def test_run_many_solve_new_unavailable_stops_run(monkeypatch, tmp_path):
    from pipeline import jsonl
    from pipeline.teachers import TeacherUnavailable, runner
    calls = []

    def fake_get(name):
        async def fn(prompt, **kw):
            calls.append(prompt)
            raise TeacherUnavailable("usage limit reached")
        return fn

    monkeypatch.setattr(gm.teachers, "get", fake_get)
    out = tmp_path / "o.jsonl"
    asyncio.run(runner.run_many(_new_items(10), lambda it: gm.solve_unknown(it, ["claude"]), out, concurrency=1))
    assert len(calls) == 2  # only the first item's two first solves ran
    assert jsonl.done_ids(out) == set()


def test_run_many_solve_new_ordinary_errors_trip_breaker(monkeypatch, tmp_path):
    from pipeline.teachers import TeacherError, runner
    calls = []

    def fake_get(name):
        async def fn(prompt, **kw):
            calls.append(prompt)
            raise TeacherError("bad")
        return fn

    monkeypatch.setattr(gm.teachers, "get", fake_get)
    out = tmp_path / "o.jsonl"
    asyncio.run(runner.run_many(_new_items(40), lambda it: gm.solve_unknown(it, ["claude"]), out, concurrency=1))
    assert len(calls) == 50  # 25 items x 2 first solves, then the breaker stopped the run


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


def test_write_problems_rejects_tabs(monkeypatch):
    item = {"id": "mn-0", "subject": "Algebra", "level": 2, "seeds": ["a", "b", "c"], "author": "claude"}
    reply = '["Find the value of x\\t when x plus three equals ten exactly.", "Find y if y + 1 = 4 and y is real."]'
    monkeypatch.setattr(gm.teachers, "get", _fixed({"claude": [reply]}))
    rows = asyncio.run(gm.write_problems(item))
    assert [r["problem"] for r in rows] == ["Find y if y + 1 = 4 and y is real."]


def test_claude_only_defaults_and_single_teacher_solve_train(monkeypatch):
    import sys
    from pipeline.generate import code as gc, ifeval as gi
    for mod in (gm, gc, gi):
        monkeypatch.setattr(sys, "argv", [mod.__name__, {gm: "dev", gc: "write", gi: "base"}[mod]])
        import argparse
        captured = {}
        orig = argparse.ArgumentParser.parse_args

        def spy(self, *a, **k):
            ns = orig(self, *a, **k)
            captured["t"] = ns.teachers
            raise SystemExit
        argparse.ArgumentParser.parse_args = spy
        try:
            try:
                mod.main()
            except SystemExit:
                pass
        finally:
            argparse.ArgumentParser.parse_args = orig
        assert captured["t"] == ["claude"]
    train = [{"id": "mt-1", "problem": "p", "target": "5", "level": 1, "subject": "Algebra"},
             {"id": "mt-2", "problem": "p", "target": "5", "level": 1, "subject": "Algebra"}]
    items = gm.solve_known_items(train, ["claude"])
    assert [it["id"] for it in items] == ["mt-1-claude", "mt-2-claude"]


def test_solve_unknown_single_teacher_separate_calls_and_tiebreak(monkeypatch):
    calls = []

    def fake_get(name):
        replies = iter(["We add them up.\nANSWER: 7", "Another way to add.\nANSWER: 8",
                        "Careful high effort pass.\nANSWER: 8"])

        async def f(prompt, **kw):
            calls.append((name, kw.get("effort", "medium")))
            return next(replies)
        return f
    shared = fake_get("claude")
    monkeypatch.setattr(gm.teachers, "get", lambda name: shared)
    item = {"id": "mn-0-0", "problem": "?", "level": 3, "subject": "Algebra"}
    rows = asyncio.run(gm.solve_unknown(item, ["claude"]))
    assert calls == [("claude", "medium"), ("claude", "medium"), ("claude", "high")]  # two independent first solves
    assert {r["teacher"] for r in rows} == {"claude-2", "claude-tiebreak"}
    assert len({r["id"] for r in rows}) == 2
    # all three differ -> dropped
    replies = iter(["a b c d e f g h i j\nANSWER: 1", "a b c d e f g h i j\nANSWER: 2", "a b c d e f g h i j\nANSWER: 3"])

    async def differ(prompt, **kw):
        return next(replies)
    monkeypatch.setattr(gm.teachers, "get", lambda name: differ)
    assert asyncio.run(gm.solve_unknown(item, ["claude"])) == []
    # agreement accepts with labels claude and claude-2, no tie-break call
    replies = iter(["a b c d e f g h i j\nANSWER: 4", "e f g h i j k l m n\nANSWER: 4"])
    rows = asyncio.run(gm.solve_unknown(item, ["claude"]))
    assert {r["teacher"] for r in rows} == {"claude", "claude-2"}

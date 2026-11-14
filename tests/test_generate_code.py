import asyncio
import json

from pipeline.generate import code as gc

def _item():
    return dict(gc.write_items(["claude"], 1)[0], id="ct-0", category="arithmetic")


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
    rows = asyncio.run(gc.write_tasks(_item()))
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
    assert asyncio.run(gc.write_tasks(_item())) is None


def test_write_tasks_returns_empty_on_unparseable_reply(monkeypatch):
    async def junk(prompt, **kw):
        return "not json at all"

    monkeypatch.setattr(gc.teachers, "get", lambda name: junk)
    assert asyncio.run(gc.write_tasks(_item())) == []


def test_add_hidden_returns_none_when_teacher_returns_nothing(monkeypatch):
    task = dict(TASK, id="ct-0-0", category="arithmetic", author="claude")

    async def nothing(prompt, **kw):
        return None

    monkeypatch.setattr(gc.teachers, "get", lambda name: nothing)
    assert asyncio.run(gc.add_hidden(task, ["claude", "gpt"])) is None


def test_write_prompts_all_differ_and_items_interleave():
    items = gc.write_items(["claude", "gpt"], 45)
    assert len({it["id"] for it in items}) == len(items) == 45 * len(gc.CATEGORIES)
    for cat in gc.CATEGORIES:
        prompts = [gc.write_prompt(it) for it in items if it["category"] == cat]
        assert len(set(prompts)) == 45
        assert len({(it["shape"], it["framing"]) for it in items if it["category"] == cat}) == 45
    pilot = items[:len(gc.CATEGORIES)]
    assert {it["category"] for it in pilot} == set(gc.CATEGORIES)
    text = gc.write_prompt(items[0])
    assert '"Write a function to"' in text and "Write a python function" in text
    assert "most common textbook exercises" in text and "exactly 3 assert statements" in text


def test_write_tasks_rejects_asserts_that_do_not_call_the_function(monkeypatch):
    vacuous = dict(TASK, tests=["assert double(2) == 4", "assert double(0) == 0", "assert True == True"])

    async def fake(prompt, **kw):
        return json.dumps([vacuous, TASK])

    monkeypatch.setattr(gc.teachers, "get", lambda name: fake)
    rows = asyncio.run(gc.write_tasks(_item()))
    assert [r["id"] for r in rows] == ["ct-0-1"]


def test_add_hidden_filters_dedupes_and_caps(monkeypatch):
    task = dict(TASK, id="ct-0-0", category="arithmetic", author="claude")
    real = [f"assert double({i}) == {2 * i}" for i in range(1, 11)]

    async def reply(prompt, **kw):
        return json.dumps(["assert 1 == 1", "assert len([1]) == 1", real[0], real[0]] + real[1:])

    monkeypatch.setattr(gc.teachers, "get", lambda name: reply)
    rows = asyncio.run(gc.add_hidden(task, ["claude", "gpt"]))
    assert rows[0]["hidden_tests"] == real[:8]

    async def too_few(prompt, **kw):
        return json.dumps(["assert 1 == 1", "assert True", real[0], real[0], real[1]])

    monkeypatch.setattr(gc.teachers, "get", lambda name: too_few)
    assert asyncio.run(gc.add_hidden(task, ["claude", "gpt"])) == []


def test_add_hidden_single_teacher_prompt_has_no_solution(monkeypatch, caplog):
    task = dict(TASK, id="ct-0-0", category="arithmetic", author="claude")
    seen = []

    async def good(prompt, **kw):
        seen.append(prompt)
        return json.dumps(["assert double(5) == 10", "assert double(10) == 20", "assert double(1) == 2"])

    names = []
    monkeypatch.setattr(gc.teachers, "get", lambda name: (names.append(name), good)[1])
    with caplog.at_level("WARNING"):
        rows = asyncio.run(gc.add_hidden(task, ["claude"]))
    assert names == ["claude"] and len(rows) == 1
    assert "return n * 2" not in seen[0] and "def double(n):" in seen[0]  # signature only, no body
    assert TASK["text"] in seen[0] and TASK["tests"][0] in seen[0]
    assert not caplog.records


def _terse_item():
    return dict(_item(), style="terse")


def _detailed_item():
    return dict(_item(), style="detailed")


def test_style_split_is_deterministic_and_mixed():
    items = gc.write_items(["claude", "gpt"], 40)
    again = gc.write_items(["claude", "gpt"], 40)
    assert [it["style"] for it in items] == [it["style"] for it in again]
    for style, target in (("plain", 0.4), ("terse", 0.3), ("detailed", 0.3)):
        share = sum(it["style"] == style for it in items) / len(items)
        assert abs(share - target) <= 0.10, (style, share)
    assert {it["style"] for it in items[:40]} == {"plain", "terse", "detailed"}


def test_prompt_style_instructions():
    terse = gc.write_prompt(_terse_item())
    detailed = gc.write_prompt(_detailed_item())
    assert "at most 15 words" in terse and "Leave the details to the tests" in terse
    assert "15 words" not in detailed and "at most" not in detailed
    for text in (terse, detailed):
        assert "most common textbook exercises" in text and "exactly 3 assert statements" in text


def test_terse_length_enforced_and_style_stored(monkeypatch, caplog):
    long = dict(TASK, text=" ".join(["word"] * 25))
    short = dict(TASK, text=" ".join(["word"] * 12))

    async def fake(prompt, **kw):
        return json.dumps([long, short])

    monkeypatch.setattr(gc.teachers, "get", lambda name: fake)
    with caplog.at_level("INFO"):
        rows = asyncio.run(gc.write_tasks(_terse_item()))
    assert [r["id"] for r in rows] == ["ct-0-1"] and rows[0]["style"] == "terse"
    assert rows[0]["text"] == short["text"]
    assert "dropped 1 terse" in caplog.text
    rows = asyncio.run(gc.write_tasks(_detailed_item()))
    assert len(rows) == 2 and all(r["style"] == "detailed" for r in rows)


def _plain_item(call=0, category="string manipulation"):
    return dict(_item(), style="plain", category=category, call=call)


def test_every_category_has_subtopics():
    assert set(gc.SUBTOPICS) == set(gc.CATEGORIES)
    assert all(4 <= len(v) <= 6 for v in gc.SUBTOPICS.values())


def test_plain_prompt_has_subtopic_and_no_framing():
    for call in range(7):
        text = gc.write_prompt(_plain_item(call))
        topic = gc.SUBTOPICS["string manipulation"][call % len(gc.SUBTOPICS["string manipulation"])]
        assert topic in text and "at most 15 words" in text and "exactly 3 assert statements" in text
        assert "differ from each other" in text
        assert "textbook" not in text and "frame" not in text.lower()
        assert not any(f in text for f in gc.FRAMINGS)
    assert len({gc.write_prompt(_plain_item(c)) for c in range(40)}) == 40
    framed = gc.write_prompt(_terse_item())
    assert "textbook" in framed and "frame" in framed


def test_plain_length_enforced_and_style_stored(monkeypatch):
    long = dict(TASK, text=" ".join(["word"] * 25))
    short = dict(TASK, text=" ".join(["word"] * 12))

    async def fake(prompt, **kw):
        return json.dumps([long, short])

    monkeypatch.setattr(gc.teachers, "get", lambda name: fake)
    rows = asyncio.run(gc.write_tasks(_plain_item()))
    assert [r["id"] for r in rows] == ["ct-0-1"] and rows[0]["style"] == "plain"

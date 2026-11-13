import asyncio

from pipeline.generate import ifeval as gi


def test_respond_keeps_only_passing(monkeypatch):
    fixed = [{"id": "x:exact_words", "kwargs": {"n": 3}, "text": "Your response must contain exactly 3 words."}]
    monkeypatch.setattr(gi, "sample_constraints", lambda rng, task, mode="train": fixed)
    monkeypatch.setattr(gi, "REWRITE_SHARE", 0.0)
    replies = iter(["too many words here", "just three words"])

    async def fake(prompt, **kw):
        assert kw["system"] == gi.SYSTEM
        return next(replies)

    monkeypatch.setattr(gi.teachers, "get", lambda name: fake)
    rows = asyncio.run(gi.respond({"id": "bt-0-0", "task": "Describe tea.", "teacher": "claude", "seed": 1}))
    assert len(rows) == 1
    assert rows[0]["response"] == "just three words"
    assert rows[0]["constraints"] == [{"id": "x:exact_words", "kwargs": {"n": 3}}]
    assert rows[0]["prompt"].startswith("Describe tea.")


def test_respond_gives_up_after_three(monkeypatch):
    fixed = [{"id": "x:exact_words", "kwargs": {"n": 3}, "text": "Your response must contain exactly 3 words."}]
    monkeypatch.setattr(gi, "sample_constraints", lambda rng, task, mode="train": fixed)
    monkeypatch.setattr(gi, "REWRITE_SHARE", 0.0)

    async def fake(prompt, **kw):
        return "always far too many words"

    monkeypatch.setattr(gi.teachers, "get", lambda name: fake)
    assert asyncio.run(gi.respond({"id": "bt-0-1", "task": "Describe tea.", "teacher": "gpt", "seed": 2})) == []


def _stub_constraints(monkeypatch):
    fixed = [{"id": "x:exact_words", "kwargs": {"n": 3}, "text": "Your response must contain exactly 3 words."}]
    monkeypatch.setattr(gi, "sample_constraints", lambda rng, task, mode="train": fixed)
    monkeypatch.setattr(gi, "REWRITE_SHARE", 0.0)


def test_respond_returns_none_when_teacher_always_returns_nothing(monkeypatch):
    _stub_constraints(monkeypatch)

    async def fake(prompt, **kw):
        return None

    monkeypatch.setattr(gi.teachers, "get", lambda name: fake)
    assert asyncio.run(gi.respond({"id": "bt-0-2", "task": "Describe tea.", "teacher": "claude", "seed": 3})) is None


def test_respond_returns_empty_when_some_attempt_responded_but_none_passed(monkeypatch):
    _stub_constraints(monkeypatch)
    replies = iter([None, "far too many words here", None])

    async def fake(prompt, **kw):
        return next(replies)

    monkeypatch.setattr(gi.teachers, "get", lambda name: fake)
    assert asyncio.run(gi.respond({"id": "bt-0-3", "task": "Describe tea.", "teacher": "claude", "seed": 4})) == []


def _base_item():
    return {"id": "bt-1-1", "topic": "tea", "form": "a poem", "teacher": "claude"}


def test_write_base_tasks_none_when_teacher_returns_nothing(monkeypatch):
    for empty in (None, ""):
        async def fake(prompt, _e=empty, **kw):
            return _e

        monkeypatch.setattr(gi.teachers, "get", lambda name, f=fake: f)
        assert asyncio.run(gi.write_base_tasks(_base_item())) is None


def test_write_base_tasks_empty_list_on_unusable_reply(monkeypatch):
    async def fake(prompt, **kw):
        return "I cannot help with that."

    monkeypatch.setattr(gi.teachers, "get", lambda name: fake)
    assert asyncio.run(gi.write_base_tasks(_base_item())) == []


def test_write_base_tasks_parses_list(monkeypatch):
    async def fake(prompt, **kw):
        return '["Write a poem about steeping tea slowly.", "short", 5]'

    monkeypatch.setattr(gi.teachers, "get", lambda name: fake)
    assert asyncio.run(gi.write_base_tasks(_base_item())) == [
        {"id": "bt-1-1-0", "task": "Write a poem about steeping tea slowly."}
    ]


def _rewrite(monkeypatch, prompt, constraints, reply):
    monkeypatch.setattr(gi, "REWRITE_SHARE", 1.0)

    async def fake(p, **kw):
        return reply

    import random
    return asyncio.run(gi.maybe_rewrite(prompt, constraints, random.Random(0), fake))


_KW = [{"id": "keywords:include", "kwargs": {"keywords": ["tea", "steam"], "n": 120}}]
_PROMPT = "Write a short note about brewing. Include the words 'tea' and 'steam'. Use at least 120 words."


def test_rewrite_kept_when_values_preserved(monkeypatch):
    new = "Please draft a brief note on brewing; it has to use 'tea' and 'steam' and run to 120 words or more."
    assert _rewrite(monkeypatch, _PROMPT, _KW, new) == new


def test_rewrite_falls_back_when_number_changed(monkeypatch):
    new = "Please draft a brief note on brewing; it has to use 'tea' and 'steam' and run to 150 words or more."
    assert _rewrite(monkeypatch, _PROMPT, _KW, new) == _PROMPT


def test_rewrite_falls_back_when_keyword_dropped(monkeypatch):
    new = "Please draft a brief note on brewing; it has to use the word 'tea' and run to 120 words or more."
    assert _rewrite(monkeypatch, _PROMPT, _KW, new) == _PROMPT


def test_rewrite_falls_back_when_too_long(monkeypatch):
    new = "tea steam 120 " + "padding " * 60
    assert _rewrite(monkeypatch, _PROMPT, _KW, new) == _PROMPT


def test_rewrite_never_for_repeat_prompt(monkeypatch):
    cs = [{"id": "combination:repeat_prompt", "kwargs": {"prompt_to_repeat": _PROMPT}}]
    assert _rewrite(monkeypatch, _PROMPT, cs, "A completely different rewording of the request here.") == _PROMPT

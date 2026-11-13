import random

from pipeline.constraints import extra, sample
from pipeline.verify import ifeval as vi


def test_check_standard_and_extra():
    cons = [
        {"id": "punctuation:no_comma", "kwargs": {}},
        {"id": "x:exact_words", "kwargs": {"n": 3}},
    ]
    assert vi.check("p", cons, "one two three") == [True, True]
    assert vi.check("p", cons, "one, two three") == [False, True]
    assert vi.passes("p", cons, "one two three")
    assert vi.check("p", cons, "   ") == [False, False]


def test_loose_scoring_strips_first_line():
    cons = [{"id": "change_case:english_lowercase", "kwargs": {}}]
    s = vi.score("p", cons, "Sure! Here it is:\nthis is a simple sentence written entirely in "
                            "lowercase letters for the test.")
    assert s == {"n": 1, "inst_strict": 0, "inst_loose": 1, "prompt_strict": False, "prompt_loose": True}


def test_sampler_modes():
    rng = random.Random(3)
    for _ in range(300):
        cons = sample.sample_constraints(rng, "Write about tea.", mode="train")
        assert 1 <= len(cons) <= 4
        assert all(not extra.REGISTRY[c["id"]].dev_only for c in cons if c["id"] in extra.REGISTRY)
        groups = [sample.group_of(c["id"]) for c in cons]
        assert len(groups) == len(set(groups))
        prompt = sample.render_prompt("Write about tea.", cons)
        assert prompt.startswith("Write about tea.") and all(c["text"] in prompt for c in cons)
        vi.check(prompt, cons, "placeholder response")  # must not raise
    for _ in range(100):
        cons = sample.sample_constraints(rng, "Write about tea.", mode="dev_unseen")
        assert any(c["id"] in extra.REGISTRY and extra.REGISTRY[c["id"]].dev_only for c in cons)

import random

import pytest

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


def test_wrap_never_combined_with_start_or_end():
    rng = random.Random(11)
    for mode in ("train", "dev_seen", "dev_unseen"):
        for _ in range(3000):
            cons = sample.sample_constraints(rng, "Write about tea.", mode=mode)
            groups = {sample.group_of(c["id"]) for c in cons}
            assert not ("wrap" in groups and groups & {"start", "end"}), (mode, [c["id"] for c in cons])
            if mode == "dev_unseen":
                assert any(c["id"] in extra.REGISTRY and extra.REGISTRY[c["id"]].dev_only for c in cons)


def test_route_order():
    assert vi.route("x:xml_wrap") == "extra"
    assert vi.route("keywords:frequency") == "official"
    assert vi.route("letters:letter_counting2") == "ifevalg"
    with pytest.raises(KeyError):
        vi.route("nope:nothing")


def test_ifevalg_id_scores_without_keyerror():
    c = [{"id": "letters:letter_counting2", "kwargs": {"letter": "s", "let_frequency": 3, "let_relation": "at least"}}]
    assert vi.check("p", c, "sss is fun") == [True]
    assert vi.check("p", c, "no") == [False]


def test_official_none_kwargs_are_deterministic():
    c = [{"id": "length_constraints:number_words", "kwargs": {"num_words": 5, "relation": "at least",
                                                             "keyword": None}}]
    results = set()
    for seed in range(20):
        random.seed(seed)
        results.add(tuple(vi.check("p", c, "one two three four five six")))
    assert results == {(True,)}

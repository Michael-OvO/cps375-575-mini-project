"""The 29 constraint types added after v1, the sampler's group blocking and the new dev builder."""

import random

import pytest

from pipeline import dev
from pipeline.constraints import extra, sample

NEW = [e for e in extra._ALL if e.id not in extra.V1_IDS]
R = extra.REGISTRY

P3 = "Tea is warm. It calms the mind.\n\nCoffee is strong. It wakes you up.\n\nWater is plain. It is best."

# id -> (kwargs, passing response, failing response)
CASES = {
    "x:nth_sentence_start": ({"n": 2, "word": "river"},
                             "We walked. River water was cold. Then home.",
                             "We walked. The river was cold. Then home."),
    "x:line_end_char": ({"ch": "~"}, "First line~\nSecond line~", "First line~\nSecond line."),
    "x:paragraph_acrostic": ({"word": "SUN"},
                             "Some tea.\n\nUsually warm.\n\nNever cold.",
                             "Some tea.\n\nNever cold.\n\nUsually warm."),
    "x:paragraph_end_word": ({"word": "river"},
                             "We walked to the river.\n\nThen we swam in the river!",
                             "We walked to the river.\n\nThen we went home."),
    "x:distinct_sentence_starts": ({}, "Tea is warm. Coffee is strong. Water is plain.",
                                   "Tea is warm. Tea is calming. Water is plain."),
    "x:final_paragraph_one_sentence": ({}, "One. Two.\n\nThe end.", "One. Two.\n\nThe end. Really."),
    "x:colon_count": ({"n": 2}, "Note: one. Also: two.", "Note: one."),
    "x:em_dash_count": ({"n": 1}, "Tea — the best drink.", "Tea - the best drink."),
    "x:exclaim_count": ({"n": 2}, "Wow! Great!", "Wow! Great! Yes!"),
    "x:ellipsis_count": ({"n": 2}, "Well... maybe... yes.", "Well.... maybe yes."),
    "x:comma_range": ({"lo": 3, "hi": 5}, "a, b, c, d", "a, b"),
    "x:paragraph_sentences": ({"n": 2}, P3, P3 + " Extra one."),
    "x:bullet_item_max_words": ({"n": 3, "m": 5},
                                "Items:\n- one two\n- three four\n- five six",
                                "Items:\n- one two\n- three four five six seven\n- eight"),
    "x:paragraphs_min_words": ({"n": 2, "m": 3}, "one two three\n\nfour five six",
                               "one two three\n\nfour five"),
    "x:phrase_repeat": ({"phrase": "in the end", "n": 2},
                        "In the end we won. It worked in the end.", "In the end we won."),
    "x:keywords_alpha": ({"a": "river", "b": "anchor", "c": "meadow"},
                         "An anchor, a meadow, a river.", "A river, an anchor, a meadow."),
    "x:keyword_not_first_paragraph": ({"word": "river"}, "We walked.\n\nThe river ran.",
                                      "The river ran.\n\nWe walked by the river."),
    "x:keyword_every_paragraph": ({"word": "river"}, "A river.\n\nThe River again.",
                                  "A river.\n\nNothing here."),
    "x:word_repeat_cap": ({"n": 2}, "Tea is good, tea is warm, and a cup is a cup.",
                          "Cups, cups and more cups."),
    "x:all_caps_count": ({"n": 2}, "This is VERY good and REALLY warm, I think.",
                         "This is VERY good, I think."),
    "x:initial_letter_count": ({"letter": "s", "n": 3}, "Sam sees seven ships.", "Sam sees boats."),
    "x:long_words": ({"n": 2, "L": 10}, "An extraordinary and unbelievable tale.",
                     "An extraordinary tale, extraordinary indeed."),
    "x:double_letter_words": ({"n": 3}, "Coffee, tea and a good book.", "Coffee and coffee, tea."),
    "x:lipogram": ({"letter": "q"}, "A calm answer.", "A Quiet answer."),
    "x:key_value_lines": ({"n": 3}, "Facts:\nName: Tea\n- **Origin:** China\n2. Colour: green\nDone.",
                          "Facts:\nName: Tea\nOrigin: China"),
    "x:bracket_tags": ({"n": 2}, "[TIP] Steep it. [NOTE] Drink it.", "[TIP] Steep it. [note] Drink it."),
    "x:closing_label_line": ({"label": "Bottom line", "n": 3},
                             "Tea is great.\n**Bottom line:** drink more tea.",
                             "Tea is great.\nBottom line: you should really drink much more tea."),
    "x:score_line": ({"label": "Rating"}, "Good tea.\n**Rating:** 8/10", "Good tea. Rating: 8/10 overall."),
    "x:step_labels": ({"n": 3}, "Step 1: boil.\nStep 2: steep.\nstep 3: pour.",
                      "Step 1: boil.\nStep 3: pour."),
}


def test_new_type_count_and_dev_only_split():
    assert len(NEW) == 29 and set(CASES) == {e.id for e in NEW}
    dev_new = sorted(e.id for e in NEW if e.dev_only)
    assert dev_new == sample.NEW_DEV_ONLY_IDS and len(dev_new) == 6
    assert len({R[i].group for i in dev_new}) == 6
    assert sample.V1_DEV_ONLY_IDS == ["x:hashtags", "x:line_max_words", "x:min_chars", "x:open_question",
                                      "x:quote_pairs", "x:semicolons"]
    assert not set(dev_new) & set(sample.TRAIN_EXTRA_IDS)


@pytest.mark.parametrize("cid", sorted(CASES))
def test_pass_and_fail(cid):
    kwargs, good, bad = CASES[cid]
    assert R[cid].check(good, kwargs) is True, cid
    assert R[cid].check(bad, kwargs) is False, cid


@pytest.mark.parametrize("cid", sorted(CASES))
def test_render_mentions_params_and_checks_never_raise(cid):
    spec = R[cid]
    for seed in range(50):
        kwargs = spec.sample(random.Random(seed))
        text = spec.describe(kwargs)
        assert isinstance(text, str) and len(text) > 10
        for v in kwargs.values():
            assert str(v) in text, (cid, kwargs, text)
        for resp in ["", "   ", "x", "# Title\n\n- a\n- b\n\n| a |\n|---|", "...!!!::——", "\n\n\n"]:
            assert isinstance(spec.check(resp, kwargs), bool)
        assert spec.bind(kwargs).check_following("") is False


def test_markdown_handling():
    assert R["x:nth_sentence_start"].check("# Guide\nRiver walks are fun.", {"n": 2, "word": "river"})
    assert R["x:paragraph_acrostic"].check("## Sun\n\n**Oak** trees.\n\n1. Kites", {"word": "SOK"})
    assert R["x:ellipsis_count"].check("Wait…", {"n": 0}) and not R["x:ellipsis_count"].check("a... b", {"n": 0})
    assert R["x:bullet_item_max_words"].check("---\n- a\n- b\n- c\n---", {"n": 3, "m": 3})


def test_blocks_respected_and_wrap_rules():
    rng = random.Random(5)
    for mode in ("train", "dev_unseen_new"):
        for _ in range(3000):
            cons = sample.sample_constraints(rng, "Write about tea.", mode=mode)
            ids = [c["id"] for c in cons]
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    assert not sample.groups_of(a) & sample.groups_of(b), ids
            groups = {sample.group_of(c) for c in ids}
            assert not ("wrap" in groups and groups & sample.WRAP_CONFLICTS), ids
            if mode == "dev_unseen_new":
                assert ids[0] in sample.NEW_DEV_ONLY_IDS
                assert not set(ids[1:]) & set(sample.DEV_ONLY_IDS)
            else:
                assert not set(ids) & set(sample.DEV_ONLY_IDS)


def test_v1_dev_modes_never_use_new_types():
    rng = random.Random(9)
    for mode in ("dev_seen", "dev_unseen"):
        for _ in range(1000):
            ids = [c["id"] for c in sample.sample_constraints(rng, "Write about tea.", mode=mode)]
            assert all(i in extra.V1_IDS for i in ids if i.startswith("x:"))


def test_build_if_dev_new():
    tasks = [{"id": f"bt-0-{i}", "task": f"Write about topic {i}."} for i in range(300)]
    rows = dev.build_if_dev_new(tasks, n=200)
    assert len(rows) == 200
    assert rows == dev.build_if_dev_new(tasks, n=200)
    for row in rows:
        assert set(row) == {"id", "task", "prompt", "constraints", "kind", "vault"}
        assert row["kind"] == "unseen_new" and row["id"].startswith("ifdevnew-")
        assert row["constraints"][0]["id"] in sample.NEW_DEV_ONLY_IDS
        assert row["prompt"].startswith(row["task"])
    assert {r["constraints"][0]["id"] for r in rows} == set(sample.NEW_DEV_ONLY_IDS)

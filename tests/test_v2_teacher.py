from pipeline.generate import v2_teacher as t


def test_normalize_plan():
    assert t.normalize_plan("1. Plan: Sort first.") == "Sort first."
    assert t.normalize_plan("**Plan:** Use a set.") == "Use a set."
    assert t.normalize_plan("Planning ahead matters.") == "Planning ahead matters."
    assert t.normalize_plan("1. Sort the list.") == "Sort the list."


def test_give_up_answers_rejected():
    for bad in ("No such k exists", "\\text{no solution}", "(E)", "\\textbf{(A)}"):
        assert t._GIVE_UP.search(bad)
    for ok in ("17", "\\frac{1}{72}", "x^2 + 1", "(1,3)"):
        assert not t._GIVE_UP.search(ok)


def test_if_count_items_learnable_counts():
    items = t.if_count_items(300)
    assert all(any(c["id"] in t.WEAK_COUNT_IDS for c in it["constraints"]) for it in items)
    exact = [c["kwargs"]["n"] for it in items for c in it["constraints"] if c["id"] == "x:exact_words"]
    assert exact and max(exact) <= t.EXACT_WORDS_MAX

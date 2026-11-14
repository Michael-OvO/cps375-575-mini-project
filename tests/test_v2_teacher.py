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


def test_render_rules_first_share():
    import random
    from pipeline.generate import v2_teacher as v
    cons = [{"text": "Use no commas."}, {"text": "End with the word done."}]
    outs = [v.render("Write a poem.", cons, random.Random(i)) for i in range(400)]
    first = sum(o.startswith("Use no commas.") for o in outs) / len(outs)
    assert 0.3 < first < 0.5
    assert all("Write a poem." in o for o in outs)


def test_cell_targeted_items_cycle_weak_values():
    from pipeline.discover.taxonomy import facets
    from pipeline.generate import v2_cells as v
    for task in ("math", "code"):
        fx = facets(task)
        anchors = [(f, x) for f, vals in v.WEAK[task].items() for x in vals]
        assert all(x in fx[f] for f, x in anchors)  # weak values exist in the taxonomy
        items = v.cells(task, len(anchors))
        assert {(f, it["cell"].get(f)) for it, (f, _) in zip(items, anchors)} == set(anchors)
        assert len({it["id"] for it in items}) == len(items)

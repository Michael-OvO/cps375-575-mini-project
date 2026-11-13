import random

from pipeline.constraints import extra


def test_registry_shape():
    assert len(extra.REGISTRY) == 36
    assert sum(1 for e in extra.REGISTRY.values() if e.dev_only) == 6
    assert all(cid.startswith("x:") for cid in extra.REGISTRY)


def test_every_type_samples_and_describes():
    rng = random.Random(1)
    for spec in extra.REGISTRY.values():
        for _ in range(20):
            kwargs = spec.sample(rng)
            text = spec.describe(kwargs)
            assert isinstance(text, str) and len(text) > 10
            assert isinstance(spec.check("Some plain response text.", kwargs), bool)
            assert spec.bind(kwargs).check_following("") is False


def test_selected_checks():
    r = extra.REGISTRY
    assert r["x:exact_words"].check("one two three", {"n": 3})
    assert not r["x:exact_words"].check("one two", {"n": 3})
    assert r["x:exact_sentences"].check("A b. C d! E f?", {"n": 3})
    assert r["x:exact_paragraphs"].check("a\n\nb\n\nc", {"n": 3})
    assert r["x:numbered_list"].check("1. a\n2. b\n3. c", {"n": 3})
    assert not r["x:numbered_list"].check("1. a\n3. b", {"n": 2})
    assert r["x:no_digits"].check("none here", {})
    assert not r["x:no_digits"].check("route 66", {})
    assert r["x:keyword_exact"].check("Tea and more tea.", {"word": "tea", "n": 2})
    assert r["x:keyword_order"].check("river then bridge", {"first": "river", "second": "bridge"})
    assert not r["x:keyword_order"].check("bridge then river", {"first": "river", "second": "bridge"})
    assert r["x:xml_wrap"].check("<note>hi</note>", {"tag": "note"})
    assert r["x:csv_line"].check("a, b, c", {"n": 3})
    assert not r["x:csv_line"].check("a, b\nc", {"n": 3})
    assert r["x:table_rows"].check("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", {"n": 2})
    assert r["x:code_fence"].check("```\nhello\n```", {})
    assert r["x:hashtags"].check("go #fast and #far", {"n": 2})
    assert r["x:quote_pairs"].check('He said "yes" and "no".', {"n": 2})


def test_sentence_split_after_closing_quote():
    r = extra.REGISTRY
    assert r["x:exact_sentences"].check('He said "Go." Then she left.', {"n": 2})
    assert r["x:open_question"].check('"Why?" she asked. Fine.', {})

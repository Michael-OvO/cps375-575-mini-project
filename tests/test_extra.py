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


def test_sentences_split_on_newlines_and_strip_list_markers():
    text = "Here are the steps:\n1. Boil the water\n2) Add the leaves\n- Wait three minutes\n* Pour\n• Enjoy"
    assert extra.sentences(text) == ["Here are the steps:", "Boil the water", "Add the leaves",
                                     "Wait three minutes", "Pour", "Enjoy"]
    assert extra.sentences("# Tea Guide\n\n## Brewing\nSteep it. Drink it.") == [
        "Tea Guide", "Brewing", "Steep it.", "Drink it."]


def test_sentences_drop_fragments_without_word_characters():
    assert extra.sentences("Fine.\n---\n...\n* \nDone!") == ["Fine.", "Done!"]


def test_sentences_keep_abbreviations_before_lowercase():
    text = "Bring tools, e.g. a hammer. Use one, i.e. the big one. Ask Dr. smith. Cats vs. dogs is old."
    assert extra.sentences(text) == ["Bring tools, e.g. a hammer.", "Use one, i.e. the big one.",
                                     "Ask Dr. smith.", "Cats vs. dogs is old."]
    assert extra.sentences("Pack pens, paper, etc. and go. We left, etc. Then we ate.") == [
        "Pack pens, paper, etc. and go.", "We left, etc.", "Then we ate."]
    assert extra.sentences("Mr. and Mrs. jones came. Ms. lee too.") == [
        "Mr. and Mrs. jones came.", "Ms. lee too."]


def test_sentence_based_types_with_lists_and_abbreviations():
    r = extra.REGISTRY
    numbered = "1. Tea is calming\n2. Coffee is strong\n3. Water is best"
    assert r["x:exact_sentences"].check(numbered, {"n": 3})
    assert r["x:exact_sentences"].check("Use herbs, e.g. mint. It helps.", {"n": 2})
    # The list number is not a word: each item has exactly three words.
    assert r["x:max_sentence_words"].check(numbered, {"n": 3})
    assert r["x:min_sentence_words"].check(numbered, {"n": 3})
    assert not r["x:min_sentence_words"].check(numbered + "\n4. Yes", {"n": 3})
    assert r["x:keyword_first_sentence"].check("- The river runs\n- A bridge stands", {"word": "river"})
    assert not r["x:keyword_first_sentence"].check("- The river runs\n- A bridge stands", {"word": "bridge"})
    assert r["x:keyword_last_sentence"].check("- The river runs\n- A bridge stands", {"word": "bridge"})
    assert r["x:open_question"].check("# Why tea?\nBecause it is warm.", {})
    assert not r["x:open_question"].check("# Tea\nWhy tea? Because.", {})

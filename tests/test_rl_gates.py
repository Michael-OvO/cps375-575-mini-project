from pipeline.rl.gates import gate_reasons

KW = [{"id": "keywords:frequency", "kwargs": {"keyword": "tree", "frequency": 3, "relation": "at least"}}]


def test_clean_answer_passes():
    text = "Trees cool the street. A tree also shades the sidewalk. Every tree needs water in summer."
    assert gate_reasons(text, KW) == []


def test_label_line_fires():
    assert "label_line" in gate_reasons("Keyword: tree\nThe tree is tall and green today.", KW)


def test_placeholder_fires_unless_exempt():
    text = "Dear [name], the tree program starts on {date} at [place] for [group] on [day].\nIt is free.\nCome.\nBring tools.\nThanks."
    assert "placeholder" in gate_reasons(text, KW)
    ph = [{"id": "detectable_content:number_placeholders", "kwargs": {"num_placeholders": 2}}]
    assert "placeholder" not in gate_reasons(text, ph)


def test_repeated_sentence_fires_unless_copy_type():
    text = "Trees are good for cities. " * 3 + "That is all I have to say here."
    assert "repeated_sentence" in gate_reasons(text, KW)
    cp = [{"id": "combination:repeat_prompt", "kwargs": {"prompt_to_repeat": "x"}}]
    assert "repeated_sentence" not in gate_reasons(text, cp)


def test_duplicate_list_items_fire():
    text = "Benefits of trees:\n- shade for people\n- cleaner air today\n- shade for people"
    assert "duplicate_items" in gate_reasons(text, KW)


def test_self_evaluation_fires():
    assert "self_eval" in gate_reasons("Trees help a lot in every city we know.\n(Word count: 9)", KW)
    assert "self_eval" in gate_reasons("Trees help cities.\nI have followed all the instructions.", KW)


def test_too_short_fires_unless_letters_or_upper_bound():
    assert "too_short" in gate_reasons("Yes, tree.", KW)
    ub = [{"id": "length_constraints:number_words", "kwargs": {"num_words": 5, "relation": "less than"}}]
    assert "too_short" not in gate_reasons("Yes, tree.", ub)


def test_signoff_placeholder_is_clean():
    text = "Dear neighbors,\nThe tree pit program starts in May and needs volunteers.\nSign up at city hall.\n\nSincerely,\n[Your Name]"
    assert gate_reasons(text, KW) == []


def test_slot_word_placeholder_fires():
    assert "placeholder" in gate_reasons("Write about [Insert topic here] and the tree in your own words.", KW)


def test_four_distinct_body_placeholders_fire():
    text = "Hello [name], your tree is at [address] on [date] for [price].\nIt is a fine tree.\nWater it.\nThanks.\nBye now."
    assert "placeholder" in gate_reasons(text, KW)


def test_divider_lines_are_not_duplicate_items():
    assert "duplicate_items" not in gate_reasons("First part here.\n* * *\nSecond part there.\n* * *\nThird part ends.", KW)
    assert "duplicate_items" not in gate_reasons("A tree grows.\n- - -\nIt gives shade.\n- - -\nDone now.", KW)


def test_short_exemptions():
    assert "too_short" not in gate_reasons("Yes, tree.", [{"id": "letters:frequency", "kwargs": {}}])
    assert "too_short" not in gate_reasons("Yes, tree.", [{"id": "x:short_answer", "kwargs": {}}])
    am = [{"id": "length_constraints:number_words", "kwargs": {"relation": "at most", "num_words": 5}}]
    assert "too_short" not in gate_reasons("Yes, tree.", am)


def test_list_kwarg_brackets_exempt_and_bracket_tags():
    text = "Fill in the tree form: [Insert name here] must sign it before the street meeting."
    assert "placeholder" in gate_reasons(text, KW)
    lk = [{"id": "x:markers", "kwargs": {"markers": ["[Insert name here]", "{date}"]}}]
    assert "placeholder" not in gate_reasons(text, lk)
    bt = [{"id": "x:bracket_tags", "kwargs": {}}]
    assert "placeholder" not in gate_reasons("Use [Insert topic here] as the tag for the tree.", bt)


def test_short_response_does_not_skip_signoff_exclusion_failopen():
    one = "Hello [name], your tree is at [address] on [date] for [price]."
    assert "placeholder" in gate_reasons(one, KW)


def test_signoff_exclusion_only_for_long_responses():
    body = "\n".join(f"Line {i} about the tree pit program." for i in range(8))
    text = body + "\nBest,\n[Your Name]\n[Title]\n[Phone]\n[Email]"
    assert "placeholder" not in gate_reasons(text, KW)


def test_fill_word_and_lone_brace_do_not_trigger():
    assert "placeholder" not in gate_reasons("Ask [Fillmore] about the set {a} of trees on the block today.", KW)
    assert "placeholder" in gate_reasons("Use {x} and {y} to describe trees on the block today for the city.", KW)


def test_copy_types_exempt_from_duplicate_items_and_too_short():
    dup = "Benefits of trees:\n- shade for people\n- cleaner air today\n- shade for people"
    for cid in ("copy:copying_multiple", "combination:repeat_prompt", "combination:two_responses"):
        c = [{"id": cid, "kwargs": {}}]
        assert "duplicate_items" not in gate_reasons(dup, c)
        assert "too_short" not in gate_reasons("Yes, tree.", c)
    assert "duplicate_items" in gate_reasons(dup, KW)


def test_constrained_response_exempt_from_repeated_sentence():
    text = "My answer is yes. " * 3 + "The tree program works well here."
    assert "repeated_sentence" in gate_reasons(text, KW)
    cr = [{"id": "detectable_format:constrained_response", "kwargs": {}}]
    assert "repeated_sentence" not in gate_reasons(text, cr)

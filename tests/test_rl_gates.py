from pipeline.rl.gates import gate_reasons

KW = [{"id": "keywords:frequency", "kwargs": {"keyword": "tree", "frequency": 3, "relation": "at least"}}]


def test_clean_answer_passes():
    text = "Trees cool the street. A tree also shades the sidewalk. Every tree needs water in summer."
    assert gate_reasons(text, KW) == []


def test_label_line_fires():
    assert "label_line" in gate_reasons("Keyword: tree\nThe tree is tall and green today.", KW)


def test_placeholder_fires_unless_exempt():
    text = "Dear [name], the tree program starts on {date} for your street and block."
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

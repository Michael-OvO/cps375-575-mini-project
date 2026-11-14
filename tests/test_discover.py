import random

from pipeline.discover import coverage, measure, probes, taxonomy


def test_cells_cover_every_value():
    for task in taxonomy.FACETS:
        fx = taxonomy.facets(task, with_proposals=False)
        cells = taxonomy.select_cells(task, 10, with_proposals=False)
        for f, vals in fx.items():
            assert {c[f] for c in cells} == set(vals), (task, f)


def test_forced_values_get_extra_cells():
    base = taxonomy.select_cells("code", 10, with_proposals=False)
    forced = taxonomy.select_cells("code", 10, with_proposals=False, must_include={"prompt_style": ["longer specification with explicit rules"]})
    assert len(forced) == len(base) + 1
    assert forced[0]["prompt_style"] == "longer specification with explicit rules"


def test_if_cells_are_feasible():
    cells = taxonomy.select_cells("if", 10, with_proposals=False, valid=probes.valid_if_cell)
    assert all(probes.valid_if_cell(c) for c in cells)


def test_non_english_probe_has_no_english_word_rules():
    cell = {"family": "chars", "n_constraints": "4", "genre": "a poem", "language": "Hindi",
            "wording": "standard template wording", "position": "constraints after the task"}
    for seed in range(30):
        cons = probes.if_constraints(random.Random(seed), cell, "Write a poem about rain.")
        if cons is None:
            continue
        assert any(c["id"] == probes.LANG_RULE and c["kwargs"]["language"] == "hi" for c in cons)
        assert not any(probes.english_only(c["id"], c["kwargs"]) for c in cons)


def test_infeasible_if_cell_is_rejected():
    cell = {"family": "doubles", "n_constraints": "2", "genre": "an email to a colleague", "language": "Japanese",
            "wording": "standard template wording", "position": "constraints after the task"}
    assert not probes.valid_if_cell(cell)
    assert probes.if_constraints(random.Random(0), cell, "Write an email.") is None


def test_repeat_prompt_only_without_rewording():
    cell = {"family": "start", "n_constraints": "1", "genre": "a poem", "language": "English",
            "wording": "paraphrased in casual language", "position": "constraints after the task"}
    ids = {probes.if_constraints(random.Random(s), cell, "Write a poem.")[0]["id"] for s in range(200)}
    assert "combination:repeat_prompt" not in ids


def test_answer_type():
    assert coverage.answer_type("12") == "integer"
    assert coverage.answer_type("\\frac{3}{4}") == "fraction"
    assert coverage.answer_type("2\\sqrt{3}") == "radical expression"
    assert coverage.answer_type("(-\\infty, 2]") == "interval or inequality"
    assert coverage.answer_type("(1, 2)") == "ordered pair or tuple"
    assert coverage.answer_type("1, 3, 5") == "several values separated by commas"
    assert coverage.answer_type("30^\\circ") == "angle in degrees"
    assert coverage.answer_type("4\\pi") == "expression in pi"


def test_code_contract_and_regimes():
    assert coverage.code_contract(["assert f(1) == True"]) == "boolean"
    assert coverage.code_contract(["assert f(1) == (1, 2)"]) == "tuple"
    assert coverage.code_contract(["assert f(1) == 3", "assert f([]) == None"]) == "None or a value depending on input"
    assert coverage.code_contract(["assert f('a')"]) == "boolean"
    regimes = coverage.code_regimes(["assert f([]) == 0", "assert f([-1, 2.5]) == 1.5"])
    assert {"empty or degenerate inputs", "negative numbers and zero", "floating point values"} <= regimes


def test_wilson():
    lo, hi = measure.wilson(6, 12)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert measure.wilson(0, 0) == (0.0, 1.0)

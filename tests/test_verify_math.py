from pipeline.verify import math as vm


def test_answer_and_body():
    text = "Step one.\nStep two.\nANSWER: \\frac{1}{2}"
    assert vm.answer_of(text) == "\\frac{1}{2}"
    assert vm.body_of(text) == "Step one.\nStep two."
    assert vm.answer_of("no final line") == ""


def test_equivalence():
    assert vm.is_correct("42", "42")
    assert vm.is_correct("\\frac{1}{2}", "\\frac12")
    assert not vm.is_correct("3", "4")
    assert not vm.is_correct("", "4")


def test_boxed_target():
    assert vm.boxed_target("So the result is $\\boxed{7}$.") == "7"
    assert vm.boxed_target("no box here") is None


DRIFT_PAIRS = [
    ("7", "7"),
    ("\\frac{1}{2}", "\\frac12"),
    ("3", "4"),
    ("(1, 2)", "(1,2)"),
    ("[0, 1)", "[0,1)"),
    ("\\pi", "3.14"),
    ("0.5", "\\frac{1}{2}"),
    ("\\sqrt{2}", "\\sqrt{3}"),
]


def _scorer_verdict(answer, target):
    import asyncio
    from types import SimpleNamespace

    from inspect_ai.scorer import CORRECT, Target
    from evaluation.math_scorer import math_answer_match

    state = SimpleNamespace(output=SimpleNamespace(completion=f"work\nANSWER: {answer}"))
    score = asyncio.run(math_answer_match()(state, Target(target)))
    return score.value == CORRECT


def test_drift_guard_matches_eval_scorer():
    import asyncio

    for answer, target in DRIFT_PAIRS:
        ours = asyncio.run(vm.equivalent(answer, target))
        assert ours == _scorer_verdict(answer, target), (answer, target)

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

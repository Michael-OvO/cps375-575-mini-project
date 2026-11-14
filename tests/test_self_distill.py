import asyncio

from pipeline import self_distill as sd


def run(coro):
    return asyncio.run(coro)


def test_has_loop():
    assert sd.has_loop("the same line again. " * 60)
    assert not sd.has_loop("a short answer with no repetition at all")


def test_check_math():
    item = {"target": "17"}
    assert run(sd.check_math(item, "Reasoning.\nANSWER: 17", 500)) == (True, "kept")
    assert run(sd.check_math(item, "Reasoning.\nANSWER: 18", 500)) == (False, "wrong")
    assert run(sd.check_math(item, "Reasoning.\nANSWER: 17", 3500)) == (False, "too_long")
    assert run(sd.check_math(item, "x", 4095)) == (False, "truncated")


def test_check_code_requires_exactly_one_block_and_all_tests():
    item = {"tests": ["assert f(1) == 2"], "hidden_tests": ["assert f(5) == 6"]}
    good = "Plan: add one.\n```python\ndef f(x):\n    return x + 1\n```"
    two = "Draft:\n```python\ndef f(x):\n    return x\n```\nFinal:\n```python\ndef f(x):\n    return x + 1\n```"
    wrong = "```python\ndef f(x):\n    return 2\n```"
    assert run(sd.check_code(item, good, 100)) == (True, "kept")
    assert run(sd.check_code(item, two, 100)) == (False, "not_one_block")
    assert run(sd.check_code(item, wrong, 100)) == (False, "fails_tests")


def test_check_if():
    item = {"prompt": "Write something. Your entire response should be in English, and in all lowercase letters.",
            "constraints": [{"id": "change_case:english_lowercase", "kwargs": {}}]}
    assert run(sd.check_if(item, "all lowercase words here.", 10)) == (True, "kept")
    assert run(sd.check_if(item, "Not Lowercase.", 10)) == (False, "fails_constraints")

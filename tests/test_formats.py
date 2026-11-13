import random

from inspect_evals.math.math import USER_PROMPT_TEMPLATE
from inspect_evals.mbpp.mbpp import PROMPT_TEMPLATE

from pipeline import formats


def test_math_prompt_harness_is_exact():
    assert formats.math_prompt("What is 1+1?") == USER_PROMPT_TEMPLATE.format(prompt="What is 1+1?")


def test_math_target_ends_with_answer_line():
    t = formats.math_target("Add the numbers.\n", "2")
    assert t == "Add the numbers.\n\nANSWER: 2"


def test_code_prompt_harness_layout():
    p = formats.code_prompt("Write a function to add two numbers.", ["assert add(1, 2) == 3"])
    assert p.startswith(PROMPT_TEMPLATE + "\n\nFor example:\n\n## Prompt 1\n```python\n")
    assert "## Completion 3\n```python\n" in p
    assert p.endswith(
        "# Now, do it for the following task.\n\n## Prompt:\n```python\n"
        "Write a function to add two numbers.\n```\n\n## Test Case:\n```python\n"
        "assert add(1, 2) == 3\n```\n\n## Completion:\n"
    )


def test_code_target_is_single_fence():
    assert formats.code_target("def f():\n    return 1\n") == "```python\ndef f():\n    return 1\n```"


def test_paraphrases_appear_with_rng():
    rng = random.Random(0)
    prompts = {formats.math_prompt("P", rng) for _ in range(60)}
    assert len(prompts) > 1
    assert all("P" in p and "ANSWER" in p for p in prompts)
    code = {formats.code_prompt("T", ["assert f() == 1"], rng)[:40] for _ in range(60)}
    assert len(code) > 1


def test_every_fewshot_passes_its_own_tests():
    from pipeline.verify import code as vc
    assert len(formats.OWN_FEWSHOT) == 6
    for shot in formats.OWN_FEWSHOT:
        assert set(shot) == {"text", "tests", "code"} and len(shot["tests"]) == 3
        assert vc.run_tests(shot["code"], shot["tests"]), shot["text"]
        assert not vc.run_tests(shot["code"], shot["tests"] + ["assert False"])  # tests really execute


def test_fewshots_avoid_textbook_exercises():
    texts = " ".join(s["text"] for s in formats.OWN_FEWSHOT).lower()
    for old in ("vowel", "perfect square", "reverse the order of words"):
        assert old not in texts

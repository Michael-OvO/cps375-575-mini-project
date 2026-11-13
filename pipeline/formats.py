"""Prompt and target builders that mirror the evaluation harness."""

import random

from inspect_evals.math.math import USER_PROMPT_TEMPLATE
from inspect_evals.mbpp.mbpp import PROMPT_TEMPLATE

HARNESS_SHARE = 0.7

MATH_PARAPHRASES = [
    "Solve the problem below, showing your reasoning step by step. End with a final line "
    'written as "ANSWER: $ANSWER", where $ANSWER is your final answer.\n\n{prompt}',
    "{prompt}\n\nWork through this carefully. On the last line write "
    '"ANSWER: $ANSWER" (without quotes), giving only the final answer.',
    "Here is a math problem.\n\n{prompt}\n\nExplain your solution step by step, then finish "
    'with a line of the form "ANSWER: $ANSWER". Do not use \\boxed.',
    "Please solve this step by step:\n\n{prompt}\n\nThe final line of your response must be "
    '"ANSWER: $ANSWER", where $ANSWER is the answer to the problem.',
]

# Few-shot examples written for this project. Never replace these with MBPP items.
OWN_FEWSHOT = [
    {
        "text": "Write a function to return the total length of the strings in a list that start with a given prefix.",
        "tests": ['assert prefix_length_total(["apple", "apricot", "banana"], "ap") == 12',
                  'assert prefix_length_total(["cat", "dog"], "z") == 0',
                  'assert prefix_length_total(["", "pre", "prefix"], "pre") == 9'],
        "code": "def prefix_length_total(words, prefix):\n"
                "    return sum(len(w) for w in words if w.startswith(prefix))",
    },
    {
        "text": "Write a function to return the second largest distinct value in a list.",
        "tests": ["assert second_largest([4, 1, 7, 7, 3]) == 4", "assert second_largest([10, 9]) == 9",
                  "assert second_largest([5, 5, 2]) == 2"],
        "code": "def second_largest(nums):\n    values = sorted(set(nums), reverse=True)\n    return values[1]",
    },
    {
        "text": "Write a python function to check whether the digits of a non-negative integer are strictly "
                "increasing from left to right.",
        "tests": ["assert digits_increasing(1359) == True", "assert digits_increasing(1224) == False",
                  "assert digits_increasing(7) == True"],
        "code": "def digits_increasing(n):\n    digits = str(n)\n"
                "    return all(a < b for a, b in zip(digits, digits[1:]))",
    },
    {
        "text": "Write a function to merge two dictionaries, adding values for shared keys.",
        "tests": ["assert merge_add({'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}",
                  "assert merge_add({}, {'x': 1}) == {'x': 1}",
                  "assert merge_add({'k': 5}, {}) == {'k': 5}"],
        "code": "def merge_add(d1, d2):\n    result = dict(d1)\n    for key, value in d2.items():\n"
                "        result[key] = result.get(key, 0) + value\n    return result",
    },
    {
        "text": "Write a function to return the positions at which two lists of equal length hold different values.",
        "tests": ["assert diff_positions([1, 2, 3], [1, 5, 3]) == [1]",
                  'assert diff_positions(["a", "b"], ["a", "b"]) == []',
                  "assert diff_positions([0, 0, 0, 0], [1, 0, 2, 0]) == [0, 2]"],
        "code": "def diff_positions(a, b):\n    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]",
    },
    {
        "text": "Write a function to find the tuples whose elements sum to a given value.",
        "tests": ["assert tuples_with_sum([(1, 2), (3, 4), (0, 3)], 3) == [(1, 2), (0, 3)]",
                  "assert tuples_with_sum([(5, 5)], 9) == []",
                  "assert tuples_with_sum([], 1) == []"],
        "code": "def tuples_with_sum(pairs, total):\n    return [t for t in pairs if sum(t) == total]",
    },
]

CODE_PARAPHRASES = [
    "{text}\n\nYour function must pass these tests:\n\n```python\n{tests}\n```\n\n"
    "Reply with only the Python code in a single ```python block.",
    "Task: {text}\n\nTest cases:\n```python\n{tests}\n```\n\nWrite the Python function. "
    "Output just the function in a ```python code block, with no explanation.",
    "You are given a programming task and the assertions your solution must satisfy.\n\n"
    "{text}\n\n```python\n{tests}\n```\n\nReturn only the code, inside a ```python block. "
    "Use only the standard library.",
]


def math_prompt(problem: str, rng: random.Random | None = None) -> str:
    if rng is None or rng.random() < HARNESS_SHARE:
        return USER_PROMPT_TEMPLATE.format(prompt=problem)
    return rng.choice(MATH_PARAPHRASES).format(prompt=problem)


def math_target(solution: str, answer: str) -> str:
    return f"{solution.strip()}\n\nANSWER: {answer.strip()}"


def _fewshot_block(i: int, shot: dict) -> str:
    tests = "\n".join(shot["tests"])
    return (
        f"## Prompt {i}\n```python\n{shot['text']}\n```\n\n"
        f"## Test Case {i}\n```python\n{tests}\n```\n\n"
        f"## Completion {i}\n```python\n{shot['code']}\n```\n\n"
    )


def code_prompt(text: str, tests: list[str], rng: random.Random | None = None) -> str:
    tests_str = "\n".join(tests)
    if rng is not None and rng.random() >= HARNESS_SHARE:
        return rng.choice(CODE_PARAPHRASES).format(text=text, tests=tests_str)
    shots = OWN_FEWSHOT[:3] if rng is None else rng.sample(OWN_FEWSHOT, 3)
    out = PROMPT_TEMPLATE + "\n\nFor example:\n\n"
    for i, shot in enumerate(shots, start=1):
        out += _fewshot_block(i, shot)
    out += (
        "\n# Now, do it for the following task.\n\n"
        f"## Prompt:\n```python\n{text}\n```\n\n"
        f"## Test Case:\n```python\n{tests_str}\n```\n\n"
        "## Completion:\n"
    )
    return out


def code_target(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def to_messages(prompt: str, target: str) -> list[dict]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]

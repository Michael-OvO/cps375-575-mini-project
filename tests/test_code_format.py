import random

from pipeline import select
from pipeline.code_format import add_test_imports, inline_imports, modules_in_tests, vary_tests
from pipeline.verify import code as vc

HUMANEVAL_BODY = ":\n    "  # humaneval.extract_function_body keeps what follows the first def line


def body_only(code: str) -> str:
    return code[code.find(HUMANEVAL_BODY) + 2:]


def test_inline_imports_survive_body_only_harness():
    code = 'import heapq\n\ndef top(xs, k):\n    """Largest k."""\n    return heapq.nlargest(k, xs)'
    out = inline_imports(code)
    assert out.startswith("def top")
    assert '"""Largest k."""\n    import heapq' in out  # docstring stays first
    prompt = 'def top(xs, k):\n    """Largest k."""\n'
    assert vc.run_tests(prompt + body_only(out).split('"""Largest k."""\n', 1)[1], ["assert top([3, 1, 2], 2) == [3, 2]"])


def test_inline_imports_keeps_test_modules_on_top_and_rejects_constants():
    out = inline_imports("import math\n\ndef area(r):\n    return math.pi * r * r", {"math"})
    assert out.startswith("import math\n\ndef area") and "    import math" in out
    assert inline_imports("LIMIT = 3\n\ndef f():\n    return LIMIT") is None
    assert inline_imports("def f():\n    return 1") == "def f():\n    return 1"


def test_vary_tests_and_missing_imports():
    tests = ["assert area(1) == 3.1416", "assert area(0) == 0.0"]
    varied = vary_tests(tests, random.Random(0), isclose_share=1.0, nospace_share=0.0)
    assert varied[0] == "assert math.isclose(area(1), 3.1416, rel_tol=0.001)"
    assert modules_in_tests(varied) == {"math"}
    code = add_test_imports("def area(r):\n    return 3.1416 * r * r", varied)
    assert code.startswith("import math\n\n") and vc.run_tests(code, varied)
    assert vary_tests(["assert f(1) == 2"], random.Random(0), 0.0, 1.0) == ["assert f(1)==2"]


def test_harness_safe_code_drops_plan_and_checks_tests():
    row = {"id": "c1", "text": "t", "tests": ["assert add(1, 2) == 3"],
           "response": "Plan: add.\n\n```python\nimport operator\n\ndef add(a, b):\n    return operator.add(a, b)\n```"}
    out = select.harness_safe_code(row, random.Random(0))
    assert "response" not in out and out["code"].startswith("def add")
    bad = {**row, "response": "```python\ndef add(a, b):\n    return a - b\n```"}
    assert select.harness_safe_code(bad, random.Random(0)) is None

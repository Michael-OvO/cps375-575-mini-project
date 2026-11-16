from pipeline.generate import oci


def test_clean_code_keeps_functions_drops_usage_and_type_hints():
    out = ("Here:\n```python\nfrom typing import List\nimport math\n\n"
           "def area(rs: List[float]) -> float:\n    \"\"\"Sum of circle areas.\"\"\"\n"
           "    return sum(math.pi * r * r for r in rs)\n\nprint(area([1.0]))\n```")
    code = oci.clean_code(out)
    assert "print(" not in code and "typing" not in code and "List" not in code
    assert "def area(rs):" in code and "import math" in code


def test_clean_code_refuses_classes_and_multiple_blocks():
    assert oci.clean_code("```python\nclass A:\n    pass\n\ndef f():\n    return 1\n```") is None
    assert oci.clean_code("```python\ndef f():\n    return 1\n```\n```python\nf()\n```") is None


def test_clean_tests_needs_a_defined_call():
    tests = '["\\nassert f(1) == 2\\n", "assert g(1) == 2", "assert f(1) == 2", "print(f(1))"]'
    assert oci.clean_tests(tests, {"f"}) == ["assert f(1) == 2"]


def test_judged_perfect():
    assert oci.judged_perfect({"llm_judgement": '{"a": {"score": 5}, "b": {"score": 5}}'})
    assert not oci.judged_perfect({"llm_judgement": '{"a": {"score": 5}, "b": {"score": 4}}'})

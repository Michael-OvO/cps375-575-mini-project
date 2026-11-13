from pipeline.verify import code as vc


def test_extract_code():
    assert vc.extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1\n"
    assert vc.extract_code("def f(): return 1") == "def f(): return 1"


def test_run_tests_pass_fail_timeout():
    good = "def add(a, b):\n    return a + b\n"
    assert vc.run_tests(good, ["assert add(1, 2) == 3"])
    assert not vc.run_tests(good, ["assert add(1, 2) == 4"])
    assert not vc.run_tests("def add(a, b):\n    while True:\n        pass\n", ["assert add(1, 2) == 3"], timeout=1.0)
    assert not vc.run_tests("def add(a, b) return", ["assert add(1, 2) == 3"])


def test_signature_of():
    assert vc.signature_of("import math\n\ndef area(r):\n    return math.pi * r * r\n") == "def area(r):"
    assert vc.signature_of("x = 1") is None

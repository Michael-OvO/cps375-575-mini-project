import math

import pytest

from pipeline.rl.tests_split import Check, check, parse_value, split_assert, split_tests


@pytest.mark.parametrize("test, expr, kind, expected", [
    ("assert add(1, 2) == 3", "add(1, 2)", "eq", 3),
    ("assert 3 == add(1, 2)", "add(1, 2)", "eq", 3),
    ("assert f('x') == ['a', (1, 2.5)]", "f('x')", "eq", ["a", (1, 2.5)]),
    ("assert f() == {'k': {1, 2}}", "f()", "eq", {"k": {1, 2}}),
    ("assert f() == set()", "f()", "eq", set()),
    ("assert f() == -4", "f()", "eq", -4),
    ("assert len(f([1, 2])) == 2", "len(f([1, 2]))", "eq", 2),
    ("assert f() == float('inf')", "f()", "eq", math.inf),
    ("assert f() == None", "f()", "eq", None),
    ("assert is_prime(7) == True", "is_prime(7)", "eq", True),
    ("assert is_prime(7)", "is_prime(7)", "truthy", None),
    ("assert not is_prime(8)", "is_prime(8)", "falsy", None),
    ("assert Solution().ok(3)", "Solution().ok(3)", "truthy", None),
    ("assert f(1) is None", "f(1)", "is", None),
    ("assert f(1) is True", "f(1)", "is", True),
    ("assert f(1) is False", "f(1)", "is", False),
])
def test_supported_forms(test, expr, kind, expected):
    c = split_assert(test)
    assert c is not None and (c.expr, c.kind) == (expr, kind)
    assert c.expected == expected and type(c.expected) is type(expected)


def test_isclose_form_with_tolerances():
    c = split_assert("assert math.isclose(area(2), 12.566, rel_tol=1e-3)")
    assert (c.expr, c.kind, c.expected, c.rel_tol, c.abs_tol) == ("area(2)", "isclose", 12.566, 1e-3, 0.0)
    d = split_assert("assert math.isclose(1.5, mean([1, 2]), abs_tol=0.01)")
    assert (d.expr, d.expected, d.rel_tol, d.abs_tol) == ("mean([1, 2])", 1.5, 1e-09, 0.01)


def test_absdiff_form():
    c = split_assert("assert abs(mean([1, 2]) - 1.5) < 1e-9")
    assert (c.expr, c.kind, c.expected, c.tol, c.strict) == ("mean([1, 2])", "absdiff", 1.5, 1e-9, True)
    assert split_assert("assert abs(f(1) - -2) <= 0.5").strict is False


@pytest.mark.parametrize("test", [
    "assert f(x) == g(y)",
    "assert 1 == 1",
    "assert len(f(x)) > 3",
    "assert f(3) != 2",
    "assert f(1) == 2 and f(2) == 3",
    "assert f(1) == 2, 'message'",
    "assert f(1) == list(range(3))",
    "assert f(1) == 2 if True else False",
    "assert math.isclose(f(1), g(2))",
    "assert math.isclose(f(1), 2, foo=1)",
    "x = 1",
    "assert f(",
    # the passing value is guessable, so the expression must be a direct call to the solution
    "assert f(3) > 2",
    "assert not f(3) < 2",
    "assert isinstance(f(3), int)",
    "assert all(x > 0 for x in f(3))",
    "assert (f(3) > 2) == True",
    "assert isinstance(f(3), int) == 1",
    "assert len(f(3)) == 0",
    "assert f",
    "assert math.isfinite(f(3))",
    "assert f(1) is 3",
])
def test_unsupported_forms(test):
    assert split_assert(test) is None


def test_split_tests_none_if_any_unsupported():
    assert split_tests(["assert f(1) == 2", "assert f(2) > 1"]) is None
    assert [c.expr for c in split_tests(["assert f(1) == 2", "assert f(2)"])] == ["f(1)", "f(2)"]


@pytest.mark.parametrize("value", [
    0, -7, 10 ** 30, 1.5, -0.25, 1e300, "a\nb'c\"d", "", [1, [2, 3]], (1,), (), {"a": (1, 2), 3: None},
    {1, 2}, set(), frozenset({1}), frozenset(), None, True, False, b"x\x00", 2 + 3j,
    [{"k": [set(), (None, True)]}],
])
def test_parse_value_round_trips_repr(value):
    got = parse_value(repr(value))
    assert got == value and type(got) is type(value)


def test_parse_value_inf_nan():
    assert parse_value(repr(math.inf)) == math.inf and parse_value(repr(-math.inf)) == -math.inf
    assert math.isnan(parse_value(repr(math.nan)))
    assert parse_value(repr([math.inf, 1.0])) == [math.inf, 1.0]


@pytest.mark.parametrize("text", ["<__main__.O object at 0x1>", "Fraction(1, 2)", "ERR", "__import__('os')",
                                  "x", "[1, y]", "{[1]: 2}", "{{1}}", "-'a'", "1 +", "(" * 300])
def test_parse_value_rejects_non_literals(text):
    with pytest.raises(ValueError):
        parse_value(text)


def test_check_semantics_match_python():
    eq = split_assert("assert f() == 1")
    assert check(eq, 1) and check(eq, 1.0) and check(eq, True) and not check(eq, "1")
    tup = split_assert("assert f() == (1, 2)")
    assert check(tup, (1, 2)) and not check(tup, [1, 2])
    assert not check(split_assert("assert f() == float('nan')"), math.nan)
    t, n = split_assert("assert f(1)"), split_assert("assert not f(1)")
    assert check(t, [0]) and not check(t, []) and check(n, 0) and not check(n, "x")
    is_none = split_assert("assert f(1) is None")
    assert check(is_none, None) and not check(is_none, 0)
    assert check(split_assert("assert f(1) is True"), True) and not check(split_assert("assert f(1) is True"), 1)
    close = split_assert("assert math.isclose(f(1), 1.0, rel_tol=1e-3)")
    assert check(close, 1.0005) and not check(close, 1.01) and not check(close, "1.0")
    ad, adle = split_assert("assert abs(f(1) - 2) < 0.5"), split_assert("assert abs(f(1) - 2) <= 0.5")
    assert check(ad, 2.4) and not check(ad, 2.5) and check(adle, 2.5) and not check(ad, None)


def test_check_unknown_kind_fails():
    assert not check(Check(expr="f()", kind="weird"), 1)

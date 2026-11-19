"""Split code-test asserts into a model-dependent expression (evaluated in the sandboxed child) and an expected
value with a comparison (kept in the parent). Pure: no sandbox, no I/O.

A check whose passing value is guessable (truthiness, `is`, or `==` against True/False/None/0/1) must have a
direct call to the solution as its expression, e.g. `is_prime(7)`. Otherwise the expression would carry the
test's own logic (`f(3) > 2`, `all(...)`, `isinstance(...)`) into the child, and a lying child could report
the guessable value without computing anything."""

import ast
import builtins
import math
import sys
from dataclasses import dataclass

_NUM = (int, float, complex)
_NOT_SOLUTION = set(dir(builtins)) | set(sys.stdlib_module_names)


@dataclass(frozen=True)
class Check:
    expr: str
    kind: str  # eq | truthy | falsy | is | isclose | absdiff
    expected: object = None
    rel_tol: float = 1e-09
    abs_tol: float = 0.0
    tol: float = 0.0
    strict: bool = True


def _lit(node: ast.AST, names: bool):
    """Literal value of node: Python literals and containers, unary +/- on numbers, complex sums, set(),
    frozenset(literal), float('inf'|'-inf'|'nan'), and (names=True) bare inf/nan. Raises ValueError."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = [_lit(e, names) for e in node.elts]
        return {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise ValueError("dict unpacking")
        return {_lit(k, names): _lit(v, names) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _lit(node.operand, names)
        if type(v) not in _NUM:
            raise ValueError("unary on non-number")
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return ast.literal_eval(node)  # complex literals such as 1+2j; raises ValueError otherwise
    if isinstance(node, ast.Name) and names and node.id in ("inf", "nan"):
        return float(node.id)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords
            and len(node.args) <= 1):
        fn, args = node.func.id, node.args
        if fn in ("set", "frozenset"):
            return (set if fn == "set" else frozenset)(_lit(args[0], names) if args else ())
        if fn == "float" and args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            if args[0].value.strip().lower() in ("inf", "-inf", "+inf", "nan", "infinity", "-infinity"):
                return float(args[0].value)
    raise ValueError(f"not a literal: {ast.dump(node)[:80]}")


def literal(node: ast.AST):
    return _lit(node, names=False)


def parse_value(text: str):
    """Inverse of repr for the values a child reports. Raises ValueError for anything that is not a literal."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise ValueError(str(e)) from None
    try:
        return _lit(tree.body, names=True)
    except (RecursionError, MemoryError, TypeError) as e:  # TypeError: unhashable set/dict member
        raise ValueError(str(e)) from None


def _is_lit(node: ast.AST) -> bool:
    try:
        literal(node)
        return True
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False


def _direct_call(node: ast.AST) -> bool:
    """A call whose callee is rooted at a solution name (not a builtin or stdlib module): f(x), obj.m(x),
    Solution().m(x)."""
    if not isinstance(node, ast.Call):
        return False
    root = node.func
    while isinstance(root, (ast.Attribute, ast.Call, ast.Subscript)):
        root = root.func if isinstance(root, ast.Call) else root.value
    return isinstance(root, ast.Name) and root.id not in _NOT_SOLUTION


def _guessable(value) -> bool:
    return value is None or (type(value) in (bool, int, float) and value in (0, 1))


def _is_math_isclose(node: ast.AST) -> bool:
    f = node.func if isinstance(node, ast.Call) else None
    return (isinstance(f, ast.Attribute) and f.attr == "isclose" and isinstance(f.value, ast.Name)
            and f.value.id == "math")


def _num(node: ast.AST) -> float:
    v = literal(node)
    if type(v) not in (int, float):
        raise ValueError("tolerance not a real number")
    return v


def _split(e: ast.expr) -> Check | None:
    if isinstance(e, ast.Compare) and len(e.ops) == 1:
        left, op, right = e.left, e.ops[0], e.comparators[0]
        if isinstance(op, ast.Eq):
            if _is_lit(right) and not _is_lit(left):
                expr, expected = left, literal(right)
            elif _is_lit(left) and not _is_lit(right):
                expr, expected = right, literal(left)
            else:
                return None
            return _make(expr, "eq", expected=expected)
        if (isinstance(op, ast.Is) and isinstance(right, ast.Constant) and right.value in (True, False, None)
                and type(right.value) in (bool, type(None))):
            return _make(left, "is", expected=right.value)
        if (isinstance(op, (ast.Lt, ast.LtE)) and isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "abs" and len(left.args) == 1 and not left.keywords
                and isinstance(left.args[0], ast.BinOp) and isinstance(left.args[0].op, ast.Sub)):
            diff = left.args[0]
            if _is_lit(diff.right) and not _is_lit(diff.left):
                return _make(diff.left, "absdiff", expected=literal(diff.right), tol=_num(right),
                             strict=isinstance(op, ast.Lt))
        return None
    if _is_math_isclose(e):
        if len(e.args) != 2:
            return None
        a, b = e.args
        if _is_lit(b) and not _is_lit(a):
            expr, expected = a, literal(b)
        elif _is_lit(a) and not _is_lit(b):
            expr, expected = b, literal(a)
        else:
            return None
        tols = {}
        for kw in e.keywords:
            if kw.arg not in ("rel_tol", "abs_tol"):
                return None
            tols[kw.arg] = _num(kw.value)
        return _make(expr, "isclose", expected=expected, **tols)
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return _make(e.operand, "falsy")
    return _make(e, "truthy")


def _make(expr: ast.expr, kind: str, **kw) -> Check | None:
    if isinstance(expr, (ast.Compare, ast.BoolOp)) or (isinstance(expr, ast.UnaryOp)
                                                       and isinstance(expr.op, ast.Not)):
        return None
    guessable = kind in ("truthy", "falsy", "is") or (kind == "eq" and _guessable(kw.get("expected")))
    if guessable and not _direct_call(expr):
        return None
    return Check(expr=ast.unparse(expr), kind=kind, **kw)


def split_assert(test: str) -> Check | None:
    """One `assert ...` line (no message) to a Check, or None if the form is unsupported."""
    try:
        tree = ast.parse(test.strip())
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert) or tree.body[0].msg is not None:
        return None
    try:
        return _split(tree.body[0].test)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return None


def split_tests(tests: list[str]) -> list[Check] | None:
    checks = [split_assert(t) for t in tests]
    return None if any(c is None for c in checks) else checks


def check(c: Check, value) -> bool:
    """Same outcome as the assert would have in Python, given the expression's value."""
    try:
        if c.kind == "eq":
            return bool(value == c.expected)
        if c.kind == "truthy":
            return bool(value)
        if c.kind == "falsy":
            return not value
        if c.kind == "is":
            return value is c.expected
        if c.kind == "isclose":
            return math.isclose(value, c.expected, rel_tol=c.rel_tol, abs_tol=c.abs_tol)
        if c.kind == "absdiff":
            d = abs(value - c.expected)
            return bool(d < c.tol if c.strict else d <= c.tol)
    except Exception:
        return False
    return False

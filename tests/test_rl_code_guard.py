import pytest

from pipeline.rl.code_guard import violations

CLEAN = [
    "def f(kill, system):\n    return kill + system",
    "def f(o):\n    return getattr(o, 'system', None)",
    "def f(x):\n    return x\nif __name__ == '__main__':\n    f(1)",
    "class P:\n    def __init__(self, v):\n        self.v = v\n    def __repr__(self):\n        return str(self.v)\n"
    "    def __lt__(self, o):\n        return self.v < o.v",
    "def f(xs):\n    return sorted(xs, key=lambda x: x[1])",
    "import functools\ndef f(xs):\n    return sorted(xs, key=functools.cmp_to_key(lambda a, b: a - b))",
    "import heapq\ndef f(xs):\n    heapq.heapify(xs)\n    return heapq.heappop(xs)",
    "import re\nfrom collections import Counter\ndef f(s):\n    return Counter(s), re.compile('a')",
    "def f(o):\n    return getattr(o, 'name', None)",
    "import sys\ndef f(n):\n    return min(n, sys.maxsize)",
    "import os\ndef f(p):\n    return os.path.join(p, 'x')",
    "def find_exit(grid):\n    return len(grid)",
    "import math\ndef f(x):\n    return math.sqrt(x)",
    "def f(xs):\n    return sorted(xs, key=lambda x: (x != 0, x))",
    "def f(exit):\n    exit()",
    "def f(xs):\n    builtins = [x for x in xs]\n    builtins.append(1)\n    return builtins",
    "def f(xs):\n    builtins = []\n    builtins.x = 1\n    return builtins",
    "def f(x):\n    print = x\n    return print",
    "def f():\n    def quit():\n        return 1\n    return quit()",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
]

DIRTY = {
    "exec('x=1')": "call:exec",
    "def f():\n    return eval('1')": "call:eval",
    "def f(s):\n    return compile(s, 'a', 'exec')": "call:compile",
    "def f():\n    return globals()": "call:globals",
    "def f():\n    return vars()": "call:vars",
    "def f(g):\n    return g.__builtins__": "dunder:__builtins__",
    "def f():\n    return ().__class__.__base__.__subclasses__()": "dunder:__subclasses__",
    "def f(g):\n    return g.__code__": "dunder:__code__",
    "def f(g):\n    return g.__dict__": "dunder:__dict__",
    "def f(g):\n    return getattr(g, '__globals__')": "dunder:__globals__",
    "import pickle\ndef f():\n    pass": "import:pickle",
    "import types\ndef f():\n    pass": "import:types",
    "import traceback\ndef f():\n    pass": "import:traceback",
    "import operator\ndef f():\n    return operator.attrgetter('a')": "call:attrgetter",
    "from operator import methodcaller\ndef f():\n    pass": "call:methodcaller",
    "def f():\n    import sys\n    sys.exit(0)": "call:sys.exit",
    "from sys import exit\ndef f():\n    exit(0)": "from:sys.exit",
    "import sys as s\ndef f():\n    s.exit(0)": "call:sys.exit",
    "import os\ndef f():\n    os._exit(0)": "call:os._exit",
    "def f():\n    exit()": "call:exit",
    "def f():\n    quit()": "call:quit",
    "import subprocess\ndef f():\n    pass": "import:subprocess",
    "from unittest import mock\ndef f():\n    return mock.ANY": "from:unittest.mock",
    "from unittest.mock import ANY\ndef f():\n    return ANY": "import:unittest.mock",
    "import unittest.mock\ndef f():\n    pass": "import:unittest.mock",
    "import inspect\ndef f():\n    pass": "import:inspect",
    "import ctypes\ndef f():\n    pass": "import:ctypes",
    "import importlib\ndef f():\n    pass": "import:importlib",
    "import gc\ndef f():\n    pass": "import:gc",
    "import sys\ndef f():\n    return sys._getframe(1)": "call:sys._getframe",
    "def f():\n    try:\n        1/0\n    except Exception as e:\n        return e.__traceback__.tb_frame": "attr:tb_frame",
    "import sys\ndef f():\n    return getattr(sys, '_getframe')": "getattr:_getframe",
    "def f():\n    return __import__('subprocess')": "call:__import__",
    "class A:\n    def __eq__(self, o):\n        return True\ndef f():\n    return A()": "def:__eq__",
    "class A:\n    __ne__ = lambda self, o: False": "assign:__ne__",
    "AssertionError = Exception\ndef f():\n    pass": "assign:AssertionError",
    "import builtins\ndef f():\n    builtins.print = None": "assign:builtins.print",
    "import sys\ndef f():\n    return sys._current_frames()": "attr:_current_frames",
    "class A:\n    pass\nA.__eq__ = lambda s, o: True": "assign:__eq__",
    "def f(s, o):\n    return True\nA = type('A', (), {'__eq__': f})": "str:__eq__",
    "class A:\n    pass\nsetattr(A, '__ne__', lambda s, o: False)": "str:__ne__",
    "def print(*a):\n    pass": "def:print",
    "class AssertionError(Exception):\n    pass": "def:AssertionError",
    "import builtins\nbuiltins = None": "assign:builtins",
    "import builtins as b\nb.print = None": "assign:builtins.print",
    "import builtins\nsetattr(builtins, 'print', None)": "getattr:print",
    "import os\ndef f():\n    return getattr(os, 'system')": "getattr:system",
    "def f():\n    exit()\ndef g(exit):\n    pass": "call:exit",
    "import sys\ndef f():\n    return sys.exc_info()[2]": "attr:exc_info",
    "def f(t):\n    return getattr(t, 'tb_'+'frame')": "getattr:dynamic",
    "def f(t):\n    return object.__getattribute__(t, 'tb_frame')": "attr:__getattribute__",
    "def f(t, n):\n    return hasattr(t, n)": "getattr:dynamic",
    "def f(t):\n    return t.f_locals": "attr:f_locals",
}


@pytest.mark.parametrize("code", CLEAN)
def test_clean_code_has_no_violations(code):
    assert violations(code) == []


@pytest.mark.parametrize("code,tag", list(DIRTY.items()))
def test_hacks_are_flagged(code, tag):
    assert tag in violations(code)


def test_syntax_error_is_flagged():
    assert violations("def f(:\n") == ["syntax"]

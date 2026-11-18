import pytest

from pipeline.rl.code_guard import violations

CLEAN = [
    "import sys\ndef f(n):\n    return min(n, sys.maxsize)",
    "import os\ndef f(p):\n    return os.path.join(p, 'x')",
    "def find_exit(grid):\n    return len(grid)",
    "import math\ndef f(x):\n    return math.sqrt(x)",
    "def f(xs):\n    return sorted(xs, key=lambda x: (x != 0, x))",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
]

DIRTY = {
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
}


@pytest.mark.parametrize("code", CLEAN)
def test_clean_code_has_no_violations(code):
    assert violations(code) == []


@pytest.mark.parametrize("code,tag", list(DIRTY.items()))
def test_hacks_are_flagged(code, tag):
    assert tag in violations(code)


def test_syntax_error_is_flagged():
    assert violations("def f(:\n") == ["syntax"]

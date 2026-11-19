from pipeline.rl import rewards as R

TESTS = ["assert add(1, 2) == 3"]
HIDDEN = ["assert add(-1, 1) == 0"]
GOOD = "```python\ndef add(a, b):\n    return a + b\n```"


def test_correct_code_scores_one():
    s = R.code_reward(GOOD, TESTS, HIDDEN, truncated=False)
    assert s.reward == 1.0 and s.info["reason"] == "pass"


def test_truncated_scores_zero():
    assert R.code_reward(GOOD, TESTS, HIDDEN, truncated=True).info["reason"] == "truncated"


def test_hidden_failure_scores_zero_and_reports_visible_pass():
    code = "```python\ndef add(a, b):\n    return 3\n```"
    s = R.code_reward(code, TESTS, HIDDEN, truncated=False, check_visible=True)
    assert s.reward == 0.0 and s.info["visible_pass"] is True


def test_harness_unsafe_top_level_scores_zero():
    code = "```python\nimport math\ndef add(a, b):\n    return a + b\nprint(add(1, 2))\n```"
    assert R.code_reward(code, TESTS, HIDDEN, truncated=False).info["reason"] == "harness_unsafe"


def test_guard_blocks_always_equal():
    code = ("```python\nclass E:\n    def __eq__(self, o):\n        return True\n"
            "def add(a, b):\n    return E()\n```")
    s = R.code_reward(code, TESTS, HIDDEN, truncated=False)
    assert s.reward == 0.0 and s.info["reason"] == "guard"


def test_early_clean_exit_fails_even_without_guard():
    assert not R.passes("def add(a, b):\n    raise SystemExit(0)", TESTS)


def test_mock_module_loaded_fails_even_without_guard():
    code = "def add(a, b):\n    import unittest.mock as m\n    return m.ANY"
    assert not R.passes(code, TESTS)


def test_frame_reading_is_guarded():
    code = ("```python\ndef add(a, b):\n    import sys\n    fr = sys._getframe(1)\n"
            "    for c in fr.f_code.co_consts:\n        if isinstance(c, str) and len(c) == 32:\n"
            "            print(c)\n            raise SystemExit(0)\n```")
    assert R.code_reward(code, TESTS, HIDDEN, truncated=False).info["reason"] == "guard"


def test_prose_without_fence_scores_zero():
    assert R.code_reward("The answer is add.", TESTS, HIDDEN, truncated=False).reward == 0.0


def _reward(body):
    return R.code_reward(f"```python\n{body}\n```", TESTS, HIDDEN, truncated=False)




def test_early_exit_before_asserts_scores_zero():
    assert _reward("def add(a, b):\n    raise SystemExit(0)").reward == 0.0



def test_wrong_solution_scores_zero():
    assert _reward("def add(a, b):\n    return a - b").reward == 0.0


def test_traceback_frame_walk_is_guarded():
    body = ("def add(a, b):\n    try:\n        1 / 0\n    except Exception as e:\n"
            "        f = e.__traceback__.tb_frame.f_back\n        print(f.f_locals.get('token'))\n"
            "    raise SystemExit(0)")
    s = _reward(body)
    assert s.reward == 0.0 and s.info["reason"] == "guard"


def test_exec_string_frame_walk_is_guarded():
    body = ("def add(a, b):\n    exec(\"import sys\\nf=sys._get\" + \"frame(1)\\nprint(f.f_back.f_locals)\")\n"
            "    raise SystemExit(0)")
    s = _reward(body)
    assert s.reward == 0.0 and s.info["reason"] == "guard"


def test_eval_import_and_dunder_hacks_are_guarded():
    for body in ("def add(a, b):\n    eval(\"__imp\" + \"ort__('gc')\")\n    return a + b",
                 "def add(a, b):\n    add.__builtins__\n    return a + b",
                 "def add(a, b):\n    ().__class__.__base__.__subclasses__()\n    return a + b",
                 "def add(a, b):\n    import pickle\n    return a + b"):
        assert _reward(body).info["reason"] == "guard"


def test_honest_class_with_dunders_still_scores():
    body = ("class N:\n    def __init__(self, v):\n        self.v = v\n    def __repr__(self):\n"
            "        return str(self.v)\ndef add(a, b):\n    return N(a + b).v")
    assert _reward(body).reward == 1.0


def test_exit_inside_called_function_scores_zero():
    assert _reward("def add(a, b):\n    def g():\n        raise SystemExit(0)\n    g()").reward == 0.0


def test_exit_primitives_are_guarded():
    for body in ("def add(a, b):\n    import sys\n    sys.modules['os']._exit(0)",
                 "def add(a, b):\n    import os, sys\n    os.execv(sys.executable, [sys.executable])",
                 "def add(a, b):\n    import os, threading\n    threading.Thread(target=lambda: os._exit(0)).start()\n"
                 "    return a + b"):
        assert _reward(body).info["reason"] == "guard"


def test_overriding_builtins_exec_does_not_skip_tests():
    code = "import builtins\nbuiltins.exec = lambda *a, **k: None\ndef add(a, b):\n    return a - b"
    assert not R.passes(code, TESTS)


def test_junk_stdout_does_not_matter():
    body = "def add(a, b):\n    print('junk', 'x' * 50)\n    return a + b"
    assert _reward(body).reward == 1.0

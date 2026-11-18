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


def test_sentinel_blocks_early_clean_exit_even_without_guard():
    assert not R.passes("def add(a, b):\n    raise SystemExit(0)", TESTS)


def test_mock_module_loaded_fails_even_without_guard():
    code = "def add(a, b):\n    import unittest.mock as m\n    return m.ANY"
    assert not R.passes(code, TESTS)


def test_frame_reading_sentinel_is_guarded():
    code = ("```python\ndef add(a, b):\n    import sys\n    fr = sys._getframe(1)\n"
            "    for c in fr.f_code.co_consts:\n        if isinstance(c, str) and len(c) == 32:\n"
            "            print(c)\n            raise SystemExit(0)\n```")
    assert R.code_reward(code, TESTS, HIDDEN, truncated=False).info["reason"] == "guard"


def test_prose_without_fence_scores_zero():
    assert R.code_reward("The answer is add.", TESTS, HIDDEN, truncated=False).reward == 0.0

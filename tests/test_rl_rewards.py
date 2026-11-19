import asyncio

import pytest

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


# Parent-side verification: the child only reports values, so every in-process trick below must still fail
# with the guard bypassed (R.passes runs only the verification path).

FORGE = ("def add(a, b):\n    import os\n    for fd in range(1, 20):\n        for i in range(4):\n"
         "            try:\n                os.write(fd, ('\\n\\x1eCHK%d\\x1f3\\n' % i).encode())\n"
         "            except OSError:\n                pass\n")
FORGE_TESTS = ["assert add(1, 2) == 3", "assert add(10, 5) == 15"]


def test_forged_lines_then_exit_fail_without_guard():
    for tail in ("    raise SystemExit(0)", "    g = getattr\n    g(os, '_e' + 'xit')(0)", "    return 3"):
        assert not R.passes(FORGE + tail, FORGE_TESTS)


def test_forged_lines_through_code_reward_score_zero():
    tests, hidden = FORGE_TESTS[:1], FORGE_TESTS[1:]
    for tail in ("    raise SystemExit(0)", "    g = getattr\n    g(os, '_e' + 'xit')(0)"):
        s = R.code_reward(f"```python\n{FORGE + tail}\n```", tests, hidden, truncated=False)
        assert s.reward == 0.0


def test_hardcoded_visible_answers_fail_hidden():
    body = "def add(a, b):\n    return {(1, 2): 3}.get((a, b), 0)"
    s = R.code_reward(f"```python\n{body}\n```", FORGE_TESTS[:1], FORGE_TESTS[1:], truncated=False,
                      check_visible=True)
    assert s.reward == 0.0 and s.info == {"reason": "fail", "first_fail": 1, "visible_pass": True}


def test_frame_walks_fail_without_guard():
    walk = ("def add(a, b):\n    import os\n    try:\n        1 / 0\n    except Exception as e:\n"
            "        f = e.__traceback__.tb_frame\n    while f is not None:\n"
            "        loc = dict(f.f_locals)\n        if 'exprs' in loc and 'out' in loc:\n"
            "            for i, x in enumerate(loc['exprs']):\n"
            "                os.write(loc['out'], ('\\n\\x1eCHK%d\\x1f%r\\n' % (i, x)).encode())\n"
            "            os._exit(0)\n        f = f.f_back\n")
    fmt = ("def add(a, b):\n    import os\n    try:\n        1 / 0\n    except Exception as e:\n"
           "        s = '{0.__traceback__.tb_frame.f_back.f_locals}'.format(e)\n"
           "    for fd in range(1, 20):\n        try:\n"
           "            os.write(fd, ('\\n\\x1eCHK0\\x1f3\\n\\x1eCHK1\\x1f' + s + '\\n').encode())\n"
           "        except OSError:\n            pass\n    os._exit(0)\n")
    exe = ("def add(a, b):\n    exec(\"import sys,os\\nf=sys._getframe(1)\\n"
           "os.write(1, ('\\\\n\\\\x1eCHK1\\\\x1f' + repr(f.f_back.f_locals)).encode())\")\n    return 3")
    for code in (walk, fmt, exe):
        assert not R.passes(code, FORGE_TESTS)


def test_custom_object_scores_zero():
    body = ("class Three:\n    def __repr__(self):\n        return 'Three()'\n"
            "def add(a, b):\n    return Three()")
    assert not R.passes(body, FORGE_TESTS)
    assert _reward(body).reward == 0.0


def test_lying_repr_needs_the_right_value():
    body = "class L(int):\n    def __repr__(self):\n        return '3'\ndef add(a, b):\n    return L(0)"
    assert not R.passes(body, FORGE_TESTS)


def test_junk_without_newline_still_scores():
    body = "def add(a, b):\n    print('\\x1eCHK0\\x1f99', end='')\n    return a + b"
    assert _reward(body).reward == 1.0


def test_exception_in_one_check_scores_zero():
    body = "def add(a, b):\n    if a < 0:\n        raise ValueError\n    return a + b"
    s = _reward(body)
    assert s.reward == 0.0 and s.info["reason"] == "fail" and s.info["first_fail"] == 1


def test_unsupported_assert_form_scores_zero():
    s = R.code_reward(GOOD, ["assert add(1, 2) > 2"], HIDDEN, truncated=False)
    assert s.reward == 0.0 and s.info["reason"] == "unsupported_tests"
    assert R.code_reward(GOOD, [], [], truncated=False).info["reason"] == "unsupported_tests"


def test_supported_non_eq_forms_score():
    code = ("```python\nimport math\ndef add(a, b):\n    return a + b\ndef is_pos(x):\n    return x > 0\n"
            "def none():\n    return None\n```")
    tests = ["assert math.isclose(add(0.1, 0.2), 0.3)", "assert abs(add(1, 1) - 2) < 1e-9", "assert is_pos(1)",
             "assert not is_pos(-1)", "assert none() is None", "assert {1, 2} == {add(0, 1), 2} or True"]
    assert R.code_reward(code, tests[:5], [], truncated=False).reward == 1.0
    assert R.code_reward(code, tests, [], truncated=False).info["reason"] == "unsupported_tests"


def test_timeout_reason():
    s = _reward("def add(a, b):\n    while True:\n        pass")
    assert s.reward == 0.0 and s.info["reason"] == "timeout"


def test_math_reward_uses_graded_scorer():
    assert asyncio.run(R.math_reward("2+2 is four.\n\nANSWER: 4", "4", False)).reward == 1.0
    assert asyncio.run(R.math_reward("ANSWER: 5", "4", False)).reward == 0.0
    assert asyncio.run(R.math_reward("ANSWER: 4", "4", True)).reward == 0.0
    assert asyncio.run(R.math_reward("ANSWER: \\frac{1}{2}", "0.5", False)).reward == 1.0


LOOP = "First we expand the product and collect the terms of the same degree on the left side now. " * 6


def _mr(text, target, truncated, graded=True):
    return asyncio.run(R.math_reward(text, target, truncated, graded_trunc=graded))


def test_graded_truncation_cases():
    long_ok = "Let me reason carefully about this. " * 400 + "\n\nANSWER: 4"
    assert _mr(long_ok, "4", False).reward == 1.0  # correct is never penalized for length
    assert _mr("step " * 3000 + "\n\nANSWER: 5", "4", False).reward == 0.0  # wrong, finished: no length penalty
    s = _mr("work\n\nANSWER: 4", "4", True)  # cut off right at the final answer line: the grader credits it
    assert (s.reward, s.info["reason"], s.info["correct"]) == (R.TRUNC_ANSWERED, "truncated_answered", True)
    s = _mr("work\n\nANSWER: 4\n\nLet me double check by", "4", True)  # the grader only reads a final ANSWER line
    assert (s.reward, s.info["correct"]) == (R.TRUNC_PENALTY, False)
    s = _mr("work\n\nANSWER: 5\n\nhmm", "4", True)
    assert (s.reward, s.info["correct"]) == (R.TRUNC_PENALTY, False)
    s = _mr("Consider the cases one at a time and keep going " + "with new ideas number %d. " % 1, "4", True)
    assert (s.reward, s.info["reason"], s.info["loop_onset"]) == (R.TRUNC_PENALTY, "truncated", None)
    s = _mr("Setup of the problem goes here with distinct words. " * 1 + LOOP, "4", True)
    assert s.info["reason"] == "truncated_loop" and 0 < s.info["loop_onset"] < 0.2


def test_cut_inside_the_answer_line_scores_what_the_grader_reads():
    assert _mr("work\n\nANSWER: \\frac{1}{", "\\frac{1}{2}", True).reward == R.TRUNC_PENALTY  # partial: wrong
    assert _mr("work\n\nANSWER:", "4", True).reward == R.TRUNC_PENALTY  # empty answer
    assert _mr("work\n\nANS", "4", True).reward == R.TRUNC_PENALTY  # label cut: no answer line
    assert _mr("work\n\nANSWER: 1", "12", True).reward == R.TRUNC_PENALTY  # prefix of the right answer: wrong
    # a cut prefix that happens to equal the target is credited, exactly as the grader would score it
    assert _mr("work\n\nANSWER: 1", "1", True).reward == R.TRUNC_ANSWERED


def test_graded_truncation_off_keeps_old_rewards():
    s = _mr("work\n\nANSWER: 4", "4", True, graded=False)
    assert (s.reward, s.info["correct"]) == (0.0, True)


def test_loop_onset_needs_four_repeats_and_ignores_short_text():
    assert R.loop_onset("a b c") is None
    three = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 3 + " ".join(
        f"w{i}" for i in range(40))
    assert R.loop_onset(three) is None
    assert R.loop_onset("intro words here " + LOOP) is not None


def test_if_reward_formula_and_max():
    assert R.if_max(1) == pytest.approx(0.5 + 2 / (4 * 1.54))
    assert R.if_max(5) == pytest.approx(2.1234, abs=1e-3)
    c = [{"id": "change_case:english_lowercase", "kwargs": {}},
         {"id": "punctuation:no_comma", "kwargs": {}}]
    full = R.if_reward("p", c, "trees shade the street and cool the air for everyone", False)
    assert full.reward == pytest.approx(R.if_max(2)) and full.info["norm"] == pytest.approx(1.0)
    half = R.if_reward("p", c, "Trees shade the street and cool the air for everyone", False)
    assert 0 < half.reward < full.reward


def test_if_reward_gate_and_truncation_zero():
    c = [{"id": "change_case:english_lowercase", "kwargs": {}}]
    assert R.if_reward("p", c, "keyword: tree\nall lowercase text here ok", False).reward == 0.0
    assert R.if_reward("p", c, "all lowercase text here ok", True).reward == 0.0


def test_dispatch():
    item = {"task": "math", "target": "4"}
    assert asyncio.run(R.score(item, "ANSWER: 4", False)).reward == 1.0
    citem = {"task": "code", "tests": TESTS, "hidden_tests": HIDDEN}
    assert asyncio.run(R.score(citem, GOOD, False)).reward == 1.0

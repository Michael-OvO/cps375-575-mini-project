import json

import pytest

from pipeline import preflight as pf

HARNESS = ("Instructions...\n# Now, do it for the following task.\n\n## Prompt:\n```python\n{text}\n```\n\n"
           "## Test Case:\n```python\nassert add_two(1) == 3\n```\n\n## Completion:\n")


def ex(task, prompt, answer, i="x"):
    return {"id": i, "task": task, "messages": [{"role": "user", "content": prompt},
                                                {"role": "assistant", "content": answer}]}


GOOD_CODE = "```python\ndef add_two(x):\n    return x + 2\n```"


def test_code_format_rules():
    p = HARNESS.format(text="Write a function to add two.")
    assert pf.format_errors(ex("code", p, GOOD_CODE)) == []
    assert "code_not_fence_first" in pf.format_errors(ex("code", p, "Plan: add 2.\n\n" + GOOD_CODE))
    assert "code_not_single_block" in pf.format_errors(ex("code", p, GOOD_CODE + "\n\n```python\nprint(1)\n```"))


def test_math_format_rules():
    assert pf.format_errors(ex("math", "q", "Steps.\n\nANSWER: 4")) == []
    assert pf.format_errors(ex("math", "q", "Steps. The answer is 4.")) == ["math_no_answer_line"]
    assert pf.format_errors(ex("math", "q", "Steps.\n\nANSWER: \\boxed{4}")) == ["math_boxed_answer"]


def test_names_function_feature_ignores_fewshot_and_asserts():
    plain = ex("code", HARNESS.format(text="Write a function to add two."), GOOD_CODE)
    named = ex("code", HARNESS.format(text="Write a function add_two(x) that adds two."), GOOD_CODE)
    assert pf.features(plain)["prompt_names_function"] == 0.0
    assert pf.features(named)["prompt_names_function"] == 1.0


def test_drift_flags_style_shift_and_exec_check(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "RUNS", tmp_path)
    base = [ex("code", HARNESS.format(text="Write a function to add two."), GOOD_CODE, f"b{i}") for i in range(20)]
    new = base[:10] + [ex("code", HARNESS.format(text="Write a function add_two(x)."), GOOD_CODE, f"n{i}")
                       for i in range(10)]
    for name, rows in (("v1", base), ("v2", new)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = pf.run("v2", "v1", {})
    assert not r["ok"] and any(w.startswith("code.prompt_names_function") for w in r["unacknowledged"])
    assert pf.run("v2", "v1", {"code.prompt_names_function": "intended"})["ok"]
    with pytest.raises(SystemExit):
        pf.run("v2", "v1", {}) and pf.require_passed("v2")


def test_code_exec_catches_wrong_target(tmp_path, monkeypatch):
    rows = [ex("code", HARNESS.format(text="Write a function to add two."),
               "```python\ndef add_two(x):\n    return x + 3\n```")]
    assert pf.code_exec_failures(rows) == ["x"]

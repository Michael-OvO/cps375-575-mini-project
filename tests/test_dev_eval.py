import asyncio

import pytest

from pipeline import dev_eval, jsonl


def test_if_metric_matches_public_definition():
    scores = [
        {"n": 2, "inst_strict": 2, "inst_loose": 2, "prompt_strict": True, "prompt_loose": True},
        {"n": 2, "inst_strict": 1, "inst_loose": 2, "prompt_strict": False, "prompt_loose": True},
    ]
    # prompt strict 0.5, prompt loose 1.0, inst strict 0.75, inst loose 1.0
    assert abs(dev_eval.if_metric(scores) - 0.8125) < 1e-9


def test_score_items_with_fake_sampler():
    class Fake:
        async def answer(self, text, max_tokens, temperature=0.0, n=1):
            if "ANSWER" in text:
                return ["Work.\nANSWER: 5"]
            if "## Completion:" in text:
                return ["```python\ndef double(n):\n    return n * 2\n```"]
            return ["one two three"]

    math = [{"id": "m", "problem": "2+3?", "target": "5", "vault": False}]
    code = [{"id": "c", "text": "Write a function to double.", "tests": ["assert double(2) == 4"],
             "hidden_tests": ["assert double(3) == 6"], "vault": False}]
    ifd = [{"id": "i", "prompt": "Say it.", "constraints": [{"id": "x:exact_words", "kwargs": {"n": 3}}],
            "kind": "seen", "vault": False}]
    out = asyncio.run(dev_eval.score_items(Fake(), math, code, ifd))
    assert out["math"] == 1.0 and out["code"] == 1.0 and out["ifeval_seen"] == 1.0
    assert out["counts"] == {"math": 1, "code": 1, "ifeval_seen": 1, "ifeval_unseen": 0}


class FixedSampler:
    """Answers math prompts with `math_reply` and code prompts with `code_reply`."""

    def __init__(self, math_reply="ANSWER: 5", code_reply="", other="x"):
        self.math_reply, self.code_reply, self.other = math_reply, code_reply, other

    async def answer(self, text, max_tokens, temperature=0.0, n=1):
        if "## Completion:" in text:
            return [self.code_reply]
        if "ANSWER" in text:
            return [self.math_reply]
        return [self.other]


def test_wrong_math_answer_scores_zero():
    math = [{"id": "m", "problem": "2+3?", "target": "5", "vault": False}]
    out = asyncio.run(dev_eval.score_items(FixedSampler(math_reply="Work.\nANSWER: 6"), math, [], []))
    assert out["math"] == 0.0


def test_code_failing_hidden_test_scores_zero():
    code = [{"id": "c", "text": "Write a function to double.", "tests": ["assert double(2) == 4"],
             "hidden_tests": ["assert double(3) == 6"], "vault": False}]
    # Passes the visible test (2 -> 4) but not the hidden one (3 -> 6).
    reply = "```python\ndef double(n):\n    return 4\n```"
    out = asyncio.run(dev_eval.score_items(FixedSampler(code_reply=reply), [], code, []))
    assert out["code"] == 0.0


def test_empty_items_give_zero_without_dividing_by_zero():
    out = asyncio.run(dev_eval.score_items(FixedSampler(), [], [], []))
    assert out["math"] == 0.0 and out["code"] == 0.0
    assert out["ifeval_seen"] == 0.0 and out["ifeval_unseen"] == 0.0
    assert out["counts"] == {"math": 0, "code": 0, "ifeval_seen": 0, "ifeval_unseen": 0}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    jsonl.write(tmp_path / "math_dev.jsonl", [
        {"id": "m-dev", "problem": "2+3?", "target": "5", "vault": False},
        {"id": "m-vault", "problem": "2+3?", "target": "5", "vault": True},
        {"id": "m-vault2", "problem": "2+3?", "target": "5", "vault": True},
    ])
    jsonl.write(tmp_path / "code_dev.jsonl", [
        {"id": "c-dev", "text": "t", "tests": ["assert True"], "vault": False},
        {"id": "c-vault", "text": "t", "tests": ["assert True"], "vault": True},
    ])
    jsonl.write(tmp_path / "if_dev.jsonl", [
        {"id": "i-dev", "prompt": "p", "constraints": [{"id": "x:exact_words", "kwargs": {"n": 3}}], "kind": "seen", "vault": False},
        {"id": "i-vault", "prompt": "p", "constraints": [{"id": "x:exact_words", "kwargs": {"n": 3}}], "kind": "unseen", "vault": True},
        {"id": "i-vault2", "prompt": "p", "constraints": [{"id": "x:exact_words", "kwargs": {"n": 3}}], "kind": "seen", "vault": True},
    ])
    monkeypatch.setattr(dev_eval, "DATA", tmp_path)


def test_split_selects_vault_flag(data_dir):
    dev = asyncio.run(dev_eval.evaluate(FixedSampler(), split="dev"))
    vault = asyncio.run(dev_eval.evaluate(FixedSampler(), split="vault"))
    assert dev["counts"] == {"math": 1, "code": 1, "ifeval_seen": 1, "ifeval_unseen": 0}
    assert vault["counts"] == {"math": 2, "code": 1, "ifeval_seen": 1, "ifeval_unseen": 1}


def test_limit_applies_after_split_filter(data_dir):
    out = asyncio.run(dev_eval.evaluate(FixedSampler(), split="vault", limit=1))
    assert out["counts"]["math"] == 1


def test_code_timeout_matches_public_scorer(monkeypatch):
    seen = {}

    def fake_run_tests(code, tests, timeout=10.0):
        seen["timeout"] = timeout
        return True

    monkeypatch.setattr(dev_eval.vc, "run_tests", fake_run_tests)
    code = [{"id": "c", "text": "t", "tests": ["assert True"], "vault": False}]
    asyncio.run(dev_eval.score_items(FixedSampler(), [], code, []))
    assert seen["timeout"] == 30


def _build_math_rest(limit):
    from pipeline import proxy_eval
    try:
        return proxy_eval.math_rest_samples(limit)
    except Exception as exc:  # datasets unavailable offline
        pytest.skip(f"cannot download datasets: {exc}")


def test_math_rest_excludes_math500_and_is_deterministic():
    from datasets import load_dataset
    rest = _build_math_rest(200)
    again = _build_math_rest(200)
    assert [s.input for s in rest] == [s.input for s in again]
    assert len(rest) == 200
    in_500 = set(load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"])
    assert not {s.input for s in rest} & in_500

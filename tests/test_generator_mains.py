"""Generator entry points: fail fast without credentials; record teacher usage at the end of a run."""

import json
import sys

import pytest

from pipeline import teachers
from pipeline.generate import code as gc
from pipeline.generate import ifeval as gi
from pipeline.generate import math as gm


@pytest.mark.parametrize("module,argv", [
    (gm, ["solve-train", "--teachers", "claude", "gpt"]),
    (gc, ["write", "--teachers", "claude", "gpt"]),
    (gi, ["respond", "--teachers", "claude", "gpt"]),
])
def test_main_fails_fast_without_credentials(module, argv, monkeypatch, tmp_path):
    for var in ("OPENAI_API_KEY", "OPENAI_TEACHER_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(teachers.Path, "home", lambda: tmp_path)

    async def must_not_run(*a, **k):
        raise AssertionError("no work may start before the configuration check")

    monkeypatch.setattr(module, "run_many", must_not_run)
    if module is gm:
        monkeypatch.setattr(gm, "load_train", lambda: pytest.fail("loaded data before the check"))
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        module.main()


def test_code_main_writes_usage_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "x")
    monkeypatch.setattr(gc, "RAW", tmp_path / "raw.jsonl")
    seen = {}

    async def fake_run_many(items, fn, out, concurrency):
        seen["n"], seen["out"] = len(items), out

    monkeypatch.setattr(gc, "run_many", fake_run_many)
    monkeypatch.setattr(teachers, "usage_snapshot",
                        lambda: {"claude": {"calls": 1, "input_tokens": 2, "output_tokens": 3, "models": {}}})
    monkeypatch.setattr(sys, "argv", ["prog", "write", "--teachers", "tinker", "--limit", "5"])
    gc.main()
    assert seen == {"n": 5, "out": tmp_path / "raw.jsonl"}
    usage = json.loads((tmp_path / "raw.jsonl.usage.json").read_text())
    assert usage["claude"]["output_tokens"] == 3

import json
import random

import pytest

from pipeline import jsonl, leakage, select
from pipeline.dev import is_code_dev


def _sol(pid, words, level=3, source="math_train"):
    return {"id": f"{pid}-t", "pid": pid, "problem": f"Problem {pid}", "target": "1",
            "solution": " ".join(["step"] * words), "teacher": "claude", "level": level,
            "subject": "Algebra", "source": source}


def test_pick_math_one_shortest_per_problem():
    train = [_sol("mt-1", 80), _sol("mt-1", 50), _sol("mt-1", 10), _sol("mt-2", 60)]
    new = [_sol(f"mn-{i}", 45, level=1 + i % 5, source="math_new") for i in range(200)]
    picked = select.pick_math(train, new, 50, random.Random(0))
    assert len(picked) == 50
    assert len({r["pid"] for r in picked}) == 50
    mt1 = next(r for r in picked if r["pid"] == "mt-1")
    assert len(mt1["solution"].split()) == 50  # shortest with at least 40 words
    assert {"mt-1", "mt-2"} <= {r["pid"] for r in picked}
    hard = sum(1 for r in picked if r["source"] == "math_new" and r["level"] >= 4)
    easy = sum(1 for r in picked if r["source"] == "math_new" and r["level"] <= 2)
    assert hard > easy


def test_to_example_shapes():
    ex = select.math_example(_sol("mt-1", 50), random.Random(0))
    assert ex["task"] == "math" and ex["messages"][1]["content"].endswith("ANSWER: 1")
    code = {"id": "ct-0-0", "text": "Write a function to f.", "code": "def f():\n    return 1",
            "tests": ["assert f() == 1"] * 3}
    ex = select.code_example(code, random.Random(0))
    assert ex["messages"][1]["content"] == "```python\ndef f():\n    return 1\n```"


# ---- end-to-end build() on synthetic pools ----

SIZES = {"math": 20, "ifeval": 8, "code": 10, "general": 5}
LONG_TASK = next(f"ct-long-{k}" for k in range(1000) if not is_code_dev(f"ct-long-{k}"))


def _word_count(messages):
    return sum(len(m["content"].split()) for m in messages)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    monkeypatch.setattr(select, "DATA", data)
    monkeypatch.setattr(select, "RUNS", runs)
    monkeypatch.setattr(select, "_token_counter", lambda: _word_count)
    monkeypatch.setattr(select, "MAX_TRAIN_TOKENS", 2000)

    math_train = [_sol(f"mt-{i}", 45) for i in range(15)]
    math_train.append(dict(math_train[0]))  # duplicated pool row
    math_new = [_sol(f"mn-{i}", 45, level=1 + i % 5, source="math_new") for i in range(30)]
    code = [{"id": f"ct-{i}-0", "text": f"Write a function to do thing {i}.",
             "code": f"def f{i}():\n    return {i}", "tests": [f"assert f{i}() == {i}"] * 3}
            for i in range(80)]
    code.append({"id": LONG_TASK, "text": "Write a long thing.", "code": "x = 1\n" * 3000,
                 "tests": ["assert True"] * 3})
    assert any(is_code_dev(t["id"]) for t in code)
    ifs = [{"id": f"if-{i}", "prompt": f"Write about topic {i} in detail.", "constraints": [],
            "response": "an answer " * 10, "teacher": "gpt"} for i in range(20)]
    general = [{"id": f"gen-{i}", "source": "tulu-3:x",
                "messages": [{"role": "user", "content": f"question {i}"},
                             {"role": "assistant", "content": f"answer {i} " * 30}]} for i in range(10)]
    jsonl.write(data / "math_train_solutions.jsonl", math_train)
    jsonl.write(data / "math_new_solutions.jsonl", math_new)
    jsonl.write(data / "code_tasks.jsonl", code)
    jsonl.write(data / "if_responses.jsonl", ifs)
    jsonl.write(data / "general.jsonl", general)
    monkeypatch.setattr(leakage, "load_overlap_filter", lambda: None)
    return runs


def _rows(runs, name="t"):
    return jsonl.read(runs / name / "train.jsonl")


def test_build_end_to_end(env):
    manifest = select.build("t", SIZES)
    rows = _rows(env)
    assert all(set(r) == {"id", "task", "messages"} for r in rows)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert not any(r["task"] == "code" and is_code_dev(r["id"]) for r in rows)
    for task, key in (("math", "math"), ("ifeval", "ifeval"), ("code", "code"), ("general", "general")):
        assert sum(r["task"] == task for r in rows) == SIZES[key]
    assert manifest["total_examples"] == len(rows)
    assert manifest["overlap_filter_enabled"] is False
    leakage.check_sources(manifest["sources"])
    assert {s["name"] for s in manifest["sources"]} >= {"instruction_following"}
    assert manifest["teachers"] == ["claude", "gpt"]
    assert json.loads((env / "t" / "manifest.json").read_text()) == manifest


def test_build_drops_too_long_and_excludes_dev_code(env):
    # Request every available code task: no dev id may appear, the over-long one is dropped.
    manifest = select.build("t", {**SIZES, "code": 1000})
    assert manifest["dropped_too_long"] == 1
    rows = _rows(env)
    code_ids = {r["id"] for r in rows if r["task"] == "code"}
    assert LONG_TASK not in code_ids
    assert not any(is_code_dev(i) for i in code_ids)
    non_dev = [i for i in (f"ct-{k}-0" for k in range(80)) if not is_code_dev(i)]
    assert code_ids == set(non_dev)
    assert manifest["shortfall"]["code"] == 1000 - len(non_dev)


def test_build_deterministic(env):
    select.build("a", SIZES)
    select.build("b", SIZES)
    assert (env / "a" / "train.jsonl").read_bytes() == (env / "b" / "train.jsonl").read_bytes()


def test_build_drops_overlap_on_raw_text(env, monkeypatch):
    item = "the secret held out test item about purple elephants dancing"
    jsonl.write(select.DATA / "if_responses.jsonl",
                [{"id": "if-leak", "prompt": item, "constraints": [], "response": "ok " * 20,
                  "teacher": "gpt"}]
                + [{"id": f"if-{i}", "prompt": f"Write about topic {i} in detail.", "constraints": [],
                    "response": "ok " * 20, "teacher": "gpt"} for i in range(5)])
    monkeypatch.setattr(leakage, "load_overlap_filter", lambda: leakage.OverlapFilter([item]))
    manifest = select.build("t", {**SIZES, "ifeval": 100})
    assert manifest["dropped_overlap"] == 1
    assert manifest["overlap_filter_enabled"] is True
    assert "if-leak" not in {r["id"] for r in _rows(env)}
    assert sum(r["task"] == "ifeval" for r in _rows(env)) == 5


def _general(i, messages):
    return {"id": f"gen-bad-{i}", "source": "tulu-3:x", "messages": messages}


def test_build_drops_invalid_examples(env):
    u = {"role": "user", "content": "hi"}
    a = {"role": "assistant", "content": "hello"}
    bad = [
        _general(0, [u, a, u]),
        _general(1, [{"role": "system", "content": "s"}, a]),
        _general(2, [u, {"role": "assistant", "content": "  \n"}]),
        _general(3, [u, {"role": "assistant", "content": ["x"]}]),
    ]
    good = jsonl.read(select.DATA / "general.jsonl")
    jsonl.write(select.DATA / "general.jsonl", bad + good)
    manifest = select.build("t", {**SIZES, "general": 1000})
    assert manifest["dropped_invalid"] == 4
    ids = {r["id"] for r in _rows(env)}
    assert not any(i.startswith("gen-bad-") for i in ids)
    assert sum(r["task"] == "general" for r in _rows(env)) == len(good)


def test_unexpected_math_source_raises(env):
    rows = jsonl.read(select.DATA / "math_train_solutions.jsonl")
    rows[3]["source"] = "mystery"
    jsonl.write(select.DATA / "math_train_solutions.jsonl", rows)
    with pytest.raises(ValueError, match=rows[3]["id"]):
        select.build("t", SIZES)
    assert not (env / "t").exists()


def test_source_count_mismatch_raises():
    sources = [{"name": "a", "count": 2}, {"name": "b", "count": 1}]
    select._check_counts(sources, [{}] * 3)
    with pytest.raises(ValueError):
        select._check_counts(sources, [{}] * 4)


def test_backfill_replaces_dropped_candidates(env):
    u = {"role": "user", "content": "hi"}
    bad = [_general(i, [u]) for i in range(20)]
    good = jsonl.read(select.DATA / "general.jsonl")  # 10 valid rows
    jsonl.write(select.DATA / "general.jsonl", bad + good)
    manifest = select.build("t", {**SIZES, "general": 10})
    assert sum(r["task"] == "general" for r in _rows(env)) == 10
    assert manifest["dropped_invalid"] >= 1
    assert manifest["shortfall"]["general"] == 0
    assert manifest["requested"] == {**SIZES, "general": 10}


def test_shortfall_reported_when_pool_too_small(env):
    manifest = select.build("t", {**SIZES, "ifeval": 50, "general": 10})
    assert manifest["shortfall"]["ifeval"] == 30
    assert manifest["shortfall"]["general"] == 0
    assert manifest["shortfall"]["math"] == 0
    assert sum(r["task"] == "ifeval" for r in _rows(env)) == 20

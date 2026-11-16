import json
import random

import pytest

from pipeline import jsonl, leakage, select
from pipeline.config import is_dev
from pipeline.dev import is_code_dev


def _sol(pid, words, level=3, source="math_train"):
    return {"id": f"{pid}-t", "pid": pid, "problem": f"Problem {pid}", "target": "1",
            "solution": " ".join(["step"] * words), "teacher": "claude", "level": level,
            "subject": "Algebra", "source": source}


def test_pick_math_one_shortest_per_problem():
    train = [_sol("mt-1", 80), _sol("mt-1", 50), _sol("mt-1", 5), _sol("mt-2", 60)]
    new = [_sol(f"mn-{i}", 45, level=1 + i % 5, source="math_new") for i in range(200)]
    picked = select.pick_math(train, new, 50, random.Random(0))
    assert len(picked) == 50
    assert len({r["pid"] for r in picked}) == 50
    mt1 = next(r for r in picked if r["pid"] == "mt-1")
    assert len(mt1["solution"].split()) == 50  # shortest eligible; the 5-word (20 char) body is too trivial
    assert {"mt-1", "mt-2"} <= {r["pid"] for r in picked}
    hard = sum(1 for r in picked if r["source"] == "math_new" and r["level"] >= 4)
    easy = sum(1 for r in picked if r["source"] == "math_new" and r["level"] <= 2)
    assert hard > easy


def test_short_but_real_solution_is_eligible():
    row = _sol("mt-9", 1)
    row["solution"] = "Rationalise: $\\frac{3}{\\sqrt{7}}=\\frac{3\\sqrt{7}}{7}$."
    assert select._shortest_per_problem([row]) == [row]
    assert select._shortest_per_problem([dict(row, solution="  so   x  ")]) == []


def test_math_example_uses_verified_answer_else_target():
    row = dict(_sol("mt-1", 50), target="\\!\\sqrt{21}", answer="\\sqrt{21}")
    ex = select.math_example(row, random.Random(0))
    assert ex["messages"][1]["content"].endswith("ANSWER: \\sqrt{21}")
    old = dict(_sol("mt-1", 50), target="\\!\\sqrt{21}")
    assert select.math_example(old, random.Random(0))["messages"][1]["content"].endswith("ANSWER: \\!\\sqrt{21}")
    empty = dict(old, answer="")
    assert select.math_example(empty, random.Random(0))["messages"][1]["content"].endswith("ANSWER: \\!\\sqrt{21}")


def test_to_example_shapes():
    ex = select.math_example(_sol("mt-1", 50), random.Random(0))
    assert ex["task"] == "math" and ex["messages"][1]["content"].endswith("ANSWER: 1")
    code = {"id": "ct-0-0", "text": "Write a function to f.", "code": "def f():\n    return 1",
            "tests": ["assert f() == 1"] * 3}
    ex = select.code_example(code, random.Random(0))
    assert ex["messages"][1]["content"] == "```python\ndef f():\n    return 1\n```"


# ---- end-to-end build() on synthetic pools ----

SIZES = {"math": 20, "ifeval": 8, "code": 10, "general": 5}
LONG_TEXT = next(f"Write a long thing {k}." for k in range(1000) if not is_code_dev(f"Write a long thing {k}."))
LONG_TASK = "ct-long-0"
# Synthetic ids/texts outside the private dev splits, so dev-membership checks keep them.
MATH_IDS = [f"mt-{i}" for i in range(400) if not is_dev(f"mt-{i}", select.MATH_DEV_PER_MILLE)][:15]
IF_IDS = [f"if-bt-{i}" for i in range(400) if not is_dev(f"bt-{i}", select.IF_DEV_PER_MILLE)][:20]


def _code_text(i):
    return f"Write a function to do thing {i}."


def build(name, sizes, **kw):
    """select.build with the overlap decision acknowledged (the filter is off in these tests)."""
    kw.setdefault("no_overlap_check_ok", True)
    return select.build(name, sizes, **kw)


def _word_count(messages):
    return sum(len(m["content"].split()) for m in messages)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, runs = tmp_path / "data", tmp_path / "runs"
    monkeypatch.setattr(select, "DATA", data)
    monkeypatch.setattr(select, "RUNS", runs)
    monkeypatch.setattr(select, "_token_counter", lambda: _word_count)
    monkeypatch.setattr(select, "MAX_TRAIN_TOKENS", 2000)

    math_train = [_sol(pid, 45) for pid in MATH_IDS]
    math_train.append(dict(math_train[0]))  # duplicated pool row
    math_new = [_sol(f"mn-{i}", 45, level=1 + i % 5, source="math_new") for i in range(30)]
    code = [{"id": f"ct-{i}-0", "text": _code_text(i),
             "code": f"def f{i}():\n    return {i}", "tests": [f"assert f{i}() == {i}"] * 3}
            for i in range(80)]
    code.append({"id": LONG_TASK, "text": LONG_TEXT, "code": "x = 1\n" * 3000,
                 "tests": ["assert True"] * 3})
    assert any(is_code_dev(t["text"]) for t in code)
    ifs = [{"id": iid, "task": f"Write about topic {i}.", "prompt": f"Write about topic {i} in detail.",
            "constraints": [], "response": "an answer " * 10, "teacher": "gpt"} for i, iid in enumerate(IF_IDS)]
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
    manifest = build("t", SIZES)
    rows = _rows(env)
    assert all(set(r) == {"id", "task", "messages"} for r in rows)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    code_text = {f"ct-{i}-0": _code_text(i) for i in range(80)}
    assert not any(r["task"] == "code" and is_code_dev(code_text[r["id"]]) for r in rows)
    assert not any(r["task"] == "ifeval" for r in rows)
    for task, key in (("math", "math"), ("instruction_following", "ifeval"), ("code", "code"),
                      ("general", "general")):
        assert sum(r["task"] == task for r in rows) == SIZES[key]
    assert manifest["total_examples"] == len(rows)
    assert manifest["overlap_filter_enabled"] is False
    assert manifest["no_overlap_check_acknowledged"] is True
    leakage.check_sources(manifest["sources"])
    assert {s["name"] for s in manifest["sources"]} >= {"instruction_following"}
    assert manifest["teachers"] == ["claude", "gpt"]
    assert json.loads((env / "t" / "manifest.json").read_text()) == manifest


def test_build_drops_too_long_and_excludes_dev_code(env):
    # Request every available code task: no dev id may appear, the over-long one is dropped.
    manifest = build("t", {**SIZES, "code": 1000})
    assert manifest["dropped_too_long"] == 1
    rows = _rows(env)
    code_ids = {r["id"] for r in rows if r["task"] == "code"}
    assert LONG_TASK not in code_ids
    non_dev = [f"ct-{k}-0" for k in range(80) if not is_code_dev(_code_text(k))]
    assert code_ids == set(non_dev)
    assert manifest["dropped_dev_member"] == 80 - len(non_dev)
    assert manifest["dropped_by_task"]["code"]["dev_member"] == 80 - len(non_dev)
    assert manifest["shortfall"]["code"] == 1000 - len(non_dev)


def test_build_deterministic(env):
    build("a", SIZES)
    build("b", SIZES)
    assert (env / "a" / "train.jsonl").read_bytes() == (env / "b" / "train.jsonl").read_bytes()


def test_build_drops_overlap_on_raw_text(env, monkeypatch):
    item = "the secret held out test item about purple elephants dancing"
    jsonl.write(select.DATA / "if_responses.jsonl",
                [{"id": IF_IDS[0], "prompt": item, "constraints": [], "response": "ok " * 20,
                  "teacher": "gpt"}]
                + [{"id": iid, "prompt": f"Write about topic {i} in detail.", "constraints": [],
                    "response": "ok " * 20, "teacher": "gpt"} for i, iid in enumerate(IF_IDS[1:6])])
    monkeypatch.setattr(leakage, "load_overlap_filter", lambda: leakage.OverlapFilter([item]))
    manifest = build("t", {**SIZES, "ifeval": 100})
    assert manifest["dropped_overlap"] == 1
    assert manifest["overlap_filter_enabled"] is True
    assert IF_IDS[0] not in {r["id"] for r in _rows(env)}
    assert sum(r["task"] == "instruction_following" for r in _rows(env)) == 5


def test_if_overlap_checks_base_task_not_constraint_wording(env, monkeypatch):
    # The constraint text matches a test prompt verbatim, but the base tasks are our own: keep all.
    boiler = "first repeat the request word for word without change then give your answer"
    test_item = f"Is ballistics an actual science? {boiler}"
    leaky_task = "the secret held out test item about purple elephants dancing"
    rows = [{"id": iid, "task": f"Write about topic {i} in detail.",
             "prompt": f"Write about topic {i} in detail. {boiler}", "constraints": [],
             "response": "ok " * 20, "teacher": "claude"} for i, iid in enumerate(IF_IDS[1:6])]
    rows.append({"id": IF_IDS[0], "task": leaky_task, "prompt": f"{leaky_task} {boiler}",
                 "constraints": [], "response": "ok " * 20, "teacher": "claude"})
    jsonl.write(select.DATA / "if_responses.jsonl", rows)
    monkeypatch.setattr(leakage, "load_overlap_filter",
                        lambda: leakage.OverlapFilter([test_item, leaky_task]))
    manifest = build("t", {**SIZES, "ifeval": 100})
    assert manifest["dropped_overlap"] == 1
    assert IF_IDS[0] not in {r["id"] for r in _rows(env)}
    assert sum(r["task"] == "instruction_following" for r in _rows(env)) == 5


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
    manifest = build("t", {**SIZES, "general": 1000})
    assert manifest["dropped_invalid"] == 4
    ids = {r["id"] for r in _rows(env)}
    assert not any(i.startswith("gen-bad-") for i in ids)
    assert sum(r["task"] == "general" for r in _rows(env)) == len(good)


def test_unexpected_math_source_raises(env):
    rows = jsonl.read(select.DATA / "math_train_solutions.jsonl")
    rows[3]["source"] = "mystery"
    jsonl.write(select.DATA / "math_train_solutions.jsonl", rows)
    with pytest.raises(ValueError, match=rows[3]["id"]):
        build("t", SIZES)
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
    manifest = build("t", {**SIZES, "general": 10})
    assert sum(r["task"] == "general" for r in _rows(env)) == 10
    assert manifest["dropped_invalid"] >= 1
    assert manifest["shortfall"]["general"] == 0
    assert manifest["requested"] == {**SIZES, "general": 10}


def test_shortfall_reported_when_pool_too_small(env):
    manifest = build("t", {**SIZES, "ifeval": 50, "general": 10})
    assert manifest["shortfall"]["ifeval"] == 30
    assert manifest["shortfall"]["general"] == 0
    assert manifest["shortfall"]["math"] == 0
    assert sum(r["task"] == "instruction_following" for r in _rows(env)) == 20


def test_build_drops_examples_with_control_characters(env):
    good = jsonl.read(select.DATA / "general.jsonl")
    bad = [_general(0, [{"role": "user", "content": "compute \x0crac{1}{2}"},
                        {"role": "assistant", "content": "half " * 30}]),
           _general(1, [{"role": "user", "content": "hello there"},
                        {"role": "assistant", "content": "bell\x07here " * 30}])]
    tabbed = {"id": "gen-tab", "source": "tulu-3:x", "messages": [
        {"role": "user", "content": "format this table"},
        {"role": "assistant", "content": "col\tvalue " * 30}]}
    jsonl.write(select.DATA / "general.jsonl", bad + good + [tabbed])
    manifest = build("t", {**SIZES, "general": 1000})
    assert manifest["dropped_invalid"] == 2
    ids = {r["id"] for r in _rows(env)}
    assert not any(i.startswith("gen-bad-") for i in ids)
    assert "gen-tab" in ids  # tabs are legitimate outside new math problems


# ---- overlap decision gate (B1) ----

def test_teacher_math_or_code_without_overlap_filter_needs_decision(env):
    with pytest.raises(select.OverlapCheckNotDecided, match=r"(?s)ALLOW_TEST_OVERLAP_CHECK=1.*--no-overlap-check-ok"):
        select.build("t", SIZES)
    assert not (env / "t").exists()
    with pytest.raises(select.OverlapCheckNotDecided):
        select.build("t", {**SIZES, "code": 0, "math": 1000})  # math_new reached once math_train runs out
    # math_train, instruction following and general only: no teacher-written math or code, no decision needed
    manifest = select.build("t", {**SIZES, "code": 0, "math": 5})
    assert manifest["no_overlap_check_acknowledged"] is False and manifest["overlap_filter_enabled"] is False
    manifest = build("t", SIZES)
    assert manifest["no_overlap_check_acknowledged"] is True


def test_overlap_filter_on_needs_no_acknowledgement(env, monkeypatch):
    monkeypatch.setattr(leakage, "load_overlap_filter", lambda: leakage.OverlapFilter(["unrelated text here"]))
    manifest = select.build("t", SIZES)
    assert manifest["overlap_filter_enabled"] is True and manifest["no_overlap_check_acknowledged"] is False


def test_cli_stops_with_clear_error_and_accepts_flag(env, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["select", "--name", "t", "--math", "20", "--ifeval", "8",
                                      "--code", "10", "--general", "5"])
    with pytest.raises(SystemExit, match="--no-overlap-check-ok"):
        select.main()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--no-overlap-check-ok"])
    select.main()
    assert json.loads((env / "t" / "manifest.json").read_text())["no_overlap_check_acknowledged"] is True


# ---- dev membership and dev overlap (B2), code text de-duplication (B3) ----

def test_dev_members_are_dropped(env):
    dev_pid = next(f"mt-{i}" for i in range(2000) if is_dev(f"mt-{i}", select.MATH_DEV_PER_MILLE))
    dev_if = next(f"if-bt-{i}" for i in range(2000) if is_dev(f"bt-{i}", select.IF_DEV_PER_MILLE))
    rows = jsonl.read(select.DATA / "math_train_solutions.jsonl")
    jsonl.write(select.DATA / "math_train_solutions.jsonl", [_sol(dev_pid, 45)] + rows)
    ifs = jsonl.read(select.DATA / "if_responses.jsonl")
    jsonl.write(select.DATA / "if_responses.jsonl", [dict(ifs[0], id=dev_if, task="Write about dev.")] + ifs)
    manifest = build("t", {**SIZES, "math": 1000, "ifeval": 1000})
    ids = {r["id"] for r in _rows(env)}
    assert f"{dev_pid}-t" not in ids and dev_if not in ids
    assert manifest["dropped_by_task"]["math"]["dev_member"] == 1
    assert manifest["dropped_by_task"]["ifeval"]["dev_member"] == 1


def test_dev_overlap_filter_from_private_dev_files(env):
    math_text = "Problem " + MATH_IDS[2]
    jsonl.write(select.DATA / "math_dev.jsonl", [{"id": "mt-x", "problem": math_text, "target": "1"}])
    jsonl.write(select.DATA / "code_dev.jsonl", [{"id": "ct-x", "text": _code_text(3)}])
    jsonl.write(select.DATA / "if_dev.jsonl", [{"id": "ifdev-x", "task": "Write about topic 4.",
                                                "prompt": "Write about topic 4. Use exactly 3 sentences."}])
    manifest = build("t", {**SIZES, "math": 1000, "code": 1000, "ifeval": 1000})
    ids = {r["id"] for r in _rows(env)}
    assert f"{MATH_IDS[2]}-t" not in ids and IF_IDS[4] not in ids
    assert "ct-3-0" not in ids
    expected_code = 0 if is_code_dev(_code_text(3)) else 1  # a dev member is dropped before the overlap check
    assert manifest["dropped_by_task"]["code"]["dev_overlap"] == expected_code
    assert manifest["dropped_by_task"]["math"]["dev_overlap"] == 1
    assert manifest["dropped_by_task"]["ifeval"]["dev_overlap"] == 1
    assert manifest["dropped_dev_overlap"] == 2 + expected_code
    assert manifest["dev_overlap_filter_enabled"] is True and manifest["missing_dev_files"] == []


def test_code_tasks_deduplicated_by_normalised_text(env):
    code = jsonl.read(select.DATA / "code_tasks.jsonl")
    keep = next(t for t in code if not is_code_dev(t["text"]) and t["id"] != LONG_TASK)
    dup = dict(keep, id="ct-dup-0", text="  " + keep["text"].upper().replace(".", " !"))
    jsonl.write(select.DATA / "code_tasks.jsonl", code + [dup])
    manifest = build("t", {**SIZES, "code": 1000})
    ids = {r["id"] for r in _rows(env)}
    assert keep["id"] in ids and "ct-dup-0" not in ids
    assert manifest["dropped_by_task"]["code"]["duplicate"] == 1


# ---- general slice (B4), dropped log and general overlap (C1), warnings (C4) ----

def test_general_contents_are_stripped(env):
    rows = jsonl.read(select.DATA / "general.jsonl")
    rows[0]["messages"][0]["content"] = "\n  padded question  \n"
    jsonl.write(select.DATA / "general.jsonl", rows)
    build("t", {**SIZES, "general": 1000})
    general = [r for r in _rows(env) if r["task"] == "general"]
    padded = next(r for r in general if r["id"] == rows[0]["id"])
    assert padded["messages"][0]["content"] == "padded question"
    assert all(m["content"] == m["content"].strip() for r in general for m in r["messages"])


def test_coconot_is_skipped():
    from pipeline.generate import general as gg
    assert "coconot" in gg.SKIP


def test_dropped_log_and_general_overlap(env, monkeypatch):
    item = "the secret held out test item about purple elephants dancing"
    general = jsonl.read(select.DATA / "general.jsonl")
    general.append({"id": "gen-leak", "source": "tulu-3:x", "messages": [
        {"role": "user", "content": item}, {"role": "assistant", "content": "ok " * 30}]})
    general.append(dict(general[0]))  # duplicate id
    jsonl.write(select.DATA / "general.jsonl", general)
    monkeypatch.setattr(leakage, "load_overlap_filter", lambda: leakage.OverlapFilter([item]))
    manifest = select.build("t", {**SIZES, "general": 1000, "code": 1000})
    dropped = jsonl.read(env / "t" / "dropped.jsonl")
    assert {"id": "gen-leak", "task": "general", "reason": "overlap"} in dropped
    assert {"id": general[0]["id"], "task": "general", "reason": "duplicate"} in dropped
    assert {"id": LONG_TASK, "task": "code", "reason": "too_long"} in dropped
    assert {d["reason"] for d in dropped} <= set(select.DROP_REASONS)
    by_task = manifest["dropped_by_task"]
    assert by_task["general"]["overlap"] == 1 and by_task["general"]["duplicate"] == 1
    assert sum(sum(v.values()) for v in by_task.values()) == len(dropped)


def test_loud_warnings_for_shortfall_and_missing_pool(env, capsys):
    (select.DATA / "general.jsonl").unlink()
    manifest = build("t", {**SIZES, "ifeval": 50})
    out = capsys.readouterr().out
    assert "WARNING: ifeval: shortfall of 30" in out
    assert "WARNING: general: shortfall of 5" in out
    assert "WARNING: pool file missing" in out and "general.jsonl" in out
    assert manifest["missing_pools"] == ["general.jsonl"]


def test_long_if_response_dropped_but_not_for_other_tasks(env):
    assert select.MAX_IF_RESPONSE_TOKENS == 900
    ifs = jsonl.read(select.DATA / "if_responses.jsonl")
    ifs.append(dict(ifs[0], id=IF_IDS[0] + "-long", response="word " * 901))
    jsonl.write(select.DATA / "if_responses.jsonl", ifs)
    math_new = jsonl.read(select.DATA / "math_new_solutions.jsonl")
    math_new.append(dict(math_new[0], id="mn-big-t", pid="mn-big", solution="step " * 950))
    jsonl.write(select.DATA / "math_new_solutions.jsonl", math_new)
    manifest = build("t", {**SIZES, "ifeval": 1000, "math": 1000})
    dropped = jsonl.read(env / "t" / "dropped.jsonl")
    assert {"id": IF_IDS[0] + "-long", "task": "instruction_following", "reason": "too_long_response"} in dropped
    assert manifest["dropped_by_task"]["ifeval"]["too_long_response"] == 1
    assert manifest["dropped_by_task"]["math"]["too_long_response"] == 0
    assert not any(d["id"] == "mn-big-t" for d in dropped)  # 950 words is under the 2,048 rule
    assert manifest["dropped_too_long"] >= 1 and manifest["dropped_too_long_response"] == 1
    assert IF_IDS[0] + "-long" not in {r["id"] for r in _rows(env)}


def test_manifest_counts_math_verified(env):
    math_new = jsonl.read(select.DATA / "math_new_solutions.jsonl")
    for i, row in enumerate(math_new):
        if i < 3:
            row["verified"] = "single_solve"
        elif i < 5:
            row["verified"] = "tiebreak"
    jsonl.write(select.DATA / "math_new_solutions.jsonl", math_new)
    train = jsonl.read(select.DATA / "math_train_solutions.jsonl")
    train[1]["verified"] = "known_answer"
    jsonl.write(select.DATA / "math_train_solutions.jsonl", train)
    manifest = build("t", {**SIZES, "math": 1000})
    counts = manifest["math_verified"]
    assert counts == {"known_answer": len(MATH_IDS), "single_solve": 3, "tiebreak": 2, "agreement": 25}
    assert sum(counts.values()) == manifest["sources"][0]["count"] + manifest["sources"][1]["count"]


def test_v2_long_solution_replaces_short_and_drops_give_up():
    import random
    from pipeline.select import math_candidates_v2
    short = [{"id": "a-s", "pid": "a", "problem": "p", "solution": "x" * 40, "target": "3", "level": 5, "source": "math_new"},
             {"id": "b-s", "pid": "b", "problem": "q", "solution": "y" * 40, "target": "No such n", "level": 3, "source": "math_new"},
             {"id": "c-s", "pid": "c", "problem": "r", "solution": "z" * 40, "target": "7", "level": 3, "source": "math_new"}]
    long = [{"id": "a-long", "pid": "a", "problem": "p", "solution": "w" * 400, "target": "3", "answer": "3", "level": 5,
             "source": "math_new"}]
    cells = [{"id": "mc-0-1-long", "pid": "mc-0-1", "problem": "s", "solution": "v" * 400, "target": "2", "answer": "2",
              "level": 4, "source": "v2_cells"}]
    out = math_candidates_v2([], short, long, cells, random.Random(0))
    ids = [r["id"] for r in out]
    assert set(ids[:2]) == {"a-long", "mc-0-1-long"}  # v2 rows first
    assert "a-s" not in ids and "b-s" not in ids and "c-s" in ids


def test_v2_code_plan_row_keeps_full_response():
    import random
    from pipeline.select import code_example
    row = {"id": "t-plan", "text": "Write f.", "tests": ["assert f() == 1"],
           "response": "Return 1.\n\n```python\ndef f():\n    return 1\n```"}
    ex = code_example(row, random.Random(0))
    assert ex["messages"][1]["content"] == row["response"]


def test_v4_puts_if_rlvr_rows_first_and_records_source(env):
    rl_ids = [i for i in (f"if-rl-{k}" for k in range(400))
              if not is_dev(i[len("if-"):], select.IF_DEV_PER_MILLE)][:6]
    rlvr = [{"id": iid, "task": f"Describe gadget {k} for a reader.", "prompt": f"Describe gadget {k}. No commas.",
             "constraints": [{"id": "punctuation:no_comma", "kwargs": None}], "response": "fine " * 20,
             "teacher": "codex", "style": "if_rlvr"} for k, iid in enumerate(rl_ids)]
    jsonl.write(select.DATA / "if_rlvr.codex.jsonl", rlvr[:4])
    jsonl.write(select.DATA / "if_rlvr.codex.retry.jsonl", rlvr[3:])  # one id in both files: kept once
    manifest = build("t", {**SIZES, "ifeval": 8}, v4=True)
    src = {s["name"]: s for s in manifest["sources"]}
    assert src["if_rlvr"]["count"] == 6 and src["if_rlvr"]["dataset"] == "allenai/IF_multi_constraints_upto5"
    assert src["instruction_following"]["count"] == 2
    assert manifest["dropped_by_task"]["ifeval"]["duplicate"] == 1

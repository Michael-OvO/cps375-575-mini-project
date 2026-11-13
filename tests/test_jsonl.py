from pipeline import config, jsonl


def test_append_read_and_done(tmp_path):
    p = tmp_path / "a.jsonl"
    jsonl.append(p, {"id": "x", "v": 1})
    jsonl.append(p, {"id": "y", "v": 2})
    assert jsonl.read(p) == [{"id": "x", "v": 1}, {"id": "y", "v": 2}]
    assert jsonl.done_ids(p) == set()
    jsonl.mark_done(p, "x")
    assert jsonl.done_ids(p) == {"x"}
    assert jsonl.read(tmp_path / "missing.jsonl") == []


def test_append_many_single_write(tmp_path):
    p = tmp_path / "sub" / "m.jsonl"
    jsonl.append_many(p, [{"id": "x"}, {"id": "y"}])
    jsonl.append_many(p, [])
    assert jsonl.read(p) == [{"id": "x"}, {"id": "y"}]


def test_split_helpers_are_deterministic():
    ids = [f"item-{i}" for i in range(5000)]
    dev = [i for i in ids if config.is_dev(i, 100)]
    assert 350 < len(dev) < 650
    assert dev == [i for i in ids if config.is_dev(i, 100)]
    vault = [i for i in dev if config.is_vault(i)]
    assert 0.2 < len(vault) / len(dev) < 0.47


def test_read_skips_only_a_truncated_final_line(tmp_path):
    import pytest
    p = tmp_path / "t.jsonl"
    p.write_text('{"id": "a"}\n{"id": "b", "v": "café"}\n{"id": "c", "v": [1, 2', encoding="utf-8")
    assert jsonl.read(p) == [{"id": "a"}, {"id": "b", "v": "café"}]
    p.write_text('{"id": "a"}\n{"id": broken}\n{"id": "c"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        jsonl.read(p)


def test_utf8_round_trip(tmp_path):
    p = tmp_path / "u.jsonl"
    jsonl.write(p, [{"id": "x", "t": "π ≤ 4 — ok"}])
    jsonl.append(p, {"id": "y", "t": "日本"})
    assert [r["t"] for r in jsonl.read(p)] == ["π ≤ 4 — ok", "日本"]
    assert "π" in p.read_bytes().decode("utf-8")


def test_append_after_torn_write_keeps_file_readable(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"id": "a"}\n{"id": "c", "v": [1, 2', encoding="utf-8")
    jsonl.append_many(p, [{"id": "d"}])
    jsonl.append(p, {"id": "e"})
    assert [r["id"] for r in jsonl.read(p)] == ["a", "d", "e"]
    q = tmp_path / "q.jsonl"
    q.write_text('{"id": "a"}', encoding="utf-8")  # complete row, just no newline
    jsonl.append(q, {"id": "b"})
    assert [r["id"] for r in jsonl.read(q)] == ["a", "b"]

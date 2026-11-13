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

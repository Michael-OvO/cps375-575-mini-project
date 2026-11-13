from pipeline import dev
from pipeline.constraints import extra
from pipeline.generate import ifeval


def test_if_dev_items_split_by_kind():
    tasks = [{"id": f"bt-0-{i}", "task": f"Write about topic {i}."} for i in range(40)]
    items = dev.build_if_dev(tasks)
    kinds = [it["kind"] for it in items]
    assert len(items) == 40 and kinds.count("seen") == 20 and kinds.count("unseen") == 20
    for it in items:
        has_dev_only = any(c["id"] in extra.REGISTRY and extra.REGISTRY[c["id"]].dev_only
                           for c in it["constraints"])
        assert has_dev_only == (it["kind"] == "unseen")
        assert set(it) == {"id", "task", "prompt", "constraints", "kind", "vault"}


def test_code_dev_is_deterministic():
    texts = [f"Write a function to do thing number {i}." for i in range(2000)]
    picked = [t for t in texts if dev.is_code_dev(t)]
    assert 100 < len(picked) < 230
    assert picked == [t for t in texts if dev.is_code_dev(t)]


def test_code_dev_decided_by_normalised_text_not_id():
    a = "Write a function to find the  LARGEST gap, in a list."
    b = "write a function to find the largest gap in a list"
    assert dev.normalize_code_text(a) == dev.normalize_code_text(b) == b
    texts = [f"Write a function to do thing number {i}." for i in range(2000)]
    dev_text = next(t for t in texts if dev.is_code_dev(t))
    assert dev.is_code_dev(dev_text.upper() + "  ")


def test_dedupe_code_tasks_keeps_first():
    tasks = [{"id": "ct-0-0", "text": "Write a function to add."}, {"id": "ct-1-0", "text": "write a FUNCTION to add"},
             {"id": "ct-2-0", "text": "Write a function to subtract."}]
    kept, dupes = dev.dedupe_code_tasks(tasks)
    assert [t["id"] for t in kept] == ["ct-0-0", "ct-2-0"] and [t["id"] for t in dupes] == ["ct-1-0"]


def test_dev_main_splits_code_by_text(tmp_path, monkeypatch):
    from pipeline import jsonl
    texts = [f"Write a function to do thing number {i}." for i in range(400)]
    tasks = [{"id": f"ct-{i}-0", "text": t} for i, t in enumerate(texts)]
    tasks.append({"id": "ct-dup", "text": texts[0].lower()})  # same description, different id
    jsonl.write(tmp_path / "code_tasks.jsonl", tasks)
    jsonl.write(tmp_path / "if_base_tasks.jsonl", [])
    monkeypatch.setattr(dev, "DATA", tmp_path)
    dev.main()
    code_dev = jsonl.read(tmp_path / "code_dev.jsonl")
    assert code_dev and all(dev.is_code_dev(t["text"]) for t in code_dev)
    assert "ct-dup" not in {t["id"] for t in code_dev}


def test_if_dev_share_matches_ifeval_split():
    assert dev.IF_DEV_PER_MILLE == ifeval.DEV_PER_MILLE

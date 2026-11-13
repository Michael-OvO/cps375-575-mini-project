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
        assert set(it) == {"id", "prompt", "constraints", "kind", "vault"}


def test_code_dev_is_deterministic():
    ids = [f"ct-1-{i}-0" for i in range(2000)]
    picked = [i for i in ids if dev.is_code_dev(i)]
    assert 100 < len(picked) < 230


def test_if_dev_share_matches_ifeval_split():
    assert dev.IF_DEV_PER_MILLE == ifeval.DEV_PER_MILLE

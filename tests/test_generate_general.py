import random

from pipeline.generate import general as gg


def row(i, source="ai2-adapt-dev/oasst1_converted", user="Tell me about tea please", words=30):
    return {"source": source, "messages": [
        {"role": "user", "content": f"{user} {i}"},
        {"role": "assistant", "content": " ".join(["word"] * words)}]}


def test_uniform_over_qualifying_rows():
    # Only every 10th row qualifies; late qualifiers must appear in proportion.
    rows = [row(i) if i % 10 == 0 else {"source": "x", "messages": []} for i in range(40000)]
    kept, stats = gg.select(rows, 400, random.Random(1), max_share=1.0)
    assert stats["qualified"] == 4000 and len(kept) == 400
    idx = [int(k["messages"][0]["content"].split()[-1]) for k in kept]
    late = sum(i >= 20000 for i in idx)
    assert 160 <= late <= 240


def test_per_source_cap_and_ids():
    rows = [row(i, "big") for i in range(3000)] + [row(i, "a") for i in range(300)] + [row(i, "b") for i in range(300)]
    random.Random(0).shuffle(rows)
    kept, stats = gg.select(rows, 1000, random.Random(2))
    assert len(kept) == 1000
    assert stats["per_source"]["tulu-3:big"] == 400
    assert stats["per_source"]["tulu-3:a"] == 300 and stats["per_source"]["tulu-3:b"] == 300
    assert [k["id"] for k in kept] == [f"gen-{i}" for i in range(1000)]


def test_skips_sources_and_non_english():
    rows = [row(1, "ai2-adapt-dev/flan_v2_converted"), row(2, "x/math_code_stuff"), row(3, "x/cot_thing"),
            row(4, "good"), row(5, "good", user="Dites-moi éééééééé 中文中文中文中文" * 10, words=20)]
    rows[4]["messages"][1]["content"] = "中文 " * 40
    kept, _ = gg.select(rows, 10, random.Random(0), max_share=1.0)
    assert [k["source"] for k in kept] == ["tulu-3:good"]

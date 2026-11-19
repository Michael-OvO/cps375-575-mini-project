from pipeline import jsonl
from pipeline.rl import demos as D


def _row(i, task, user, answer):
    return {"id": i, "task": task, "messages": [{"role": "user", "content": user},
                                                {"role": "assistant", "content": answer}]}


def test_id_prefixes():
    assert D.id_prefixes("mn-642-3-long") == ["mn-642", "mn-642-3", "mn-642-3-long"]


def test_pick_needs_prefix_and_text_and_prefers_shortest(tmp_path):
    p = tmp_path / "train.jsonl"
    jsonl.write(p, [_row("mn-642-3-long", "math", "Solve.\nx  + 1 = 2", "long answer, x = 1"),
                    _row("mn-642-3-short", "math", "Solve.\nx + 1 = 2", "x = 1"),
                    _row("mn-642-30-long", "math", "Other problem", "y"),
                    _row("if-1", "instruction_following", "x + 1 = 2", "z")])
    idx = D.load(p)
    assert D.pick({"id": "mn-642-3", "text": "x + 1 = 2"}, idx)[1]["content"] == "x = 1"
    assert D.pick({"id": "mn-642-3", "text": "a different problem"}, idx) is None
    assert D.pick({"id": "if-1", "text": "x + 1 = 2"}, idx) is None  # IF rows are never demonstrations
    assert D.pick({"id": "zz-1", "text": "x"}, idx) is None

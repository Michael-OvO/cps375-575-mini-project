import numpy as np
import pytest

from pipeline import train_sft


class FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


def test_lr_schedule():
    assert train_sft.lr_at(0, 100, 1e-3) == 1e-3
    assert abs(train_sft.lr_at(50, 100, 1e-3) - 5e-4) < 1e-12
    assert train_sft.lr_at(99, 100, 1e-3) > 0


def test_batches_cover_everything_once():
    bs = train_sft.batches(70, 32)
    assert [len(b) for b in bs] == [32, 32, 6]
    assert sorted(i for b in bs for i in b) == list(range(70))


def test_parse_args_defaults_and_flags():
    a = train_sft.parse_args(["--name", "smoke"])
    assert (a.name, a.rank, a.lr, a.batch_size, a.no_publish) == ("smoke", 32, None, 32, False)
    a = train_sft.parse_args(["--name", "v1", "--rank", "8", "--lr", "1e-4",
                              "--batch_size", "4", "--no_publish"])
    assert (a.rank, a.lr, a.batch_size, a.no_publish) == (8, 1e-4, 4, True)
    with pytest.raises(SystemExit):
        train_sft.parse_args([])


def test_batch_loss_is_weighted_mean_nll():
    outputs = [{"logprobs": FakeTensor([-1.0, -2.0, -4.0])},
               {"logprobs": FakeTensor([-3.0, -5.0])}]
    inputs = [{"weights": FakeTensor([0.0, 1.0, 1.0])},
              {"weights": FakeTensor([0.0, 1.0])}]
    # supervised tokens: -2, -4, -5 -> mean nll 11/3
    assert abs(train_sft.batch_loss(outputs, inputs) - 11 / 3) < 1e-9


def test_batch_loss_all_zero_weights_is_zero():
    outputs = [{"logprobs": FakeTensor([-1.0, -2.0])}]
    inputs = [{"weights": FakeTensor([0.0, 0.0])}]
    assert train_sft.batch_loss(outputs, inputs) == 0.0


def test_count_tokens_and_render_with_fake_renderer(monkeypatch):
    class FakeInput:
        def __init__(self, n):
            self.n = n

        def to_ints(self):
            return list(range(self.n))

    class FakeDatum:
        def __init__(self, n):
            self.model_input = FakeInput(n)

    assert train_sft.count_tokens([FakeDatum(3), FakeDatum(4)]) == 7

    calls = []

    def fake_c2d(messages, renderer, max_length, train_on_what):
        calls.append((messages, renderer, max_length, train_on_what))
        return FakeDatum(len(messages))

    monkeypatch.setattr(train_sft, "conversation_to_datum", fake_c2d)
    rows = [{"messages": [1, 2]}, {"messages": [1, 2, 3, 4]}]
    data = train_sft.render_examples(rows, "R")
    assert train_sft.count_tokens(data) == 6
    assert all(c[1] == "R" and c[2] is None for c in calls)
    assert all(c[3] == train_sft.renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES for c in calls)


@pytest.fixture(scope="module")
def real_renderer():
    try:
        return train_sft.build_renderer()
    except Exception as e:  # no network or tokenizer unavailable
        pytest.skip(f"tokenizer unavailable: {e}")


def test_real_rendering_weights_only_on_assistant_with_stop(real_renderer):
    rows = [
        {"messages": [{"role": "user", "content": "What is 2+2?"},
                      {"role": "assistant", "content": "The answer is 4."}]},
        {"messages": [{"role": "user", "content": "Say hi."},
                      {"role": "assistant", "content": "Hi there, friend."}]},
    ]
    data = train_sft.render_examples(rows, real_renderer)
    assert train_sft.count_tokens(data) > 0
    tok = real_renderer.tokenizer
    for d in data:
        weights = np.array(d.loss_fn_inputs["weights"].tolist())
        targets = d.loss_fn_inputs["target_tokens"].tolist()
        assert (weights > 0).any() and (weights == 0).any()
        supervised = tok.decode([t for t, w in zip(targets, weights) if w > 0])
        unsupervised = tok.decode([t for t, w in zip(targets, weights) if w == 0])
        assert supervised.endswith("\n\nUser:")
        assert "User:" not in supervised[:-len("\n\nUser:")]
        # targets are shifted by one, so the leading "User" token is input only
        assert "Assistant:" in unsupervised
        assert "Assistant:" not in supervised


def test_overlong_example_is_rejected_not_truncated(monkeypatch):
    class FakeInput:
        def __init__(self, n):
            self.n = n

        def to_ints(self):
            return list(range(self.n))

    class FakeDatum:
        def __init__(self, n):
            self.model_input = FakeInput(n)

    monkeypatch.setattr(train_sft, "conversation_to_datum",
                        lambda m, r, max_length, train_on_what: FakeDatum(m[0]))
    rows = [{"id": "ok", "messages": [10]},
            {"id": "too-long-1", "messages": [train_sft.MAX_TRAIN_TOKENS + 1]}]
    with pytest.raises(ValueError, match=r"1 of 2.*too-long-1.*pipeline\.select"):
        train_sft.render_examples(rows, "R")
    assert len(train_sft.render_examples(rows[:1], "R")) == 1


def test_real_overlong_example_rejected(real_renderer):
    rows = [{"id": "big", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "word " * (train_sft.MAX_TRAIN_TOKENS + 50)}]}]
    with pytest.raises(ValueError, match="big"):
        train_sft.render_examples(rows, real_renderer)


class _Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _FakeClient:
    def save_weights_for_sampler(self, name):
        return _Done(type("C", (), {"path": f"tinker://run/{name}"})())


_INFO = {"base_model": "m", "rank": 32}


def _read(run_dir):
    import json
    return json.loads((run_dir / "checkpoint.json").read_text())


def test_publish_failure_keeps_checkpoint_path(tmp_path):
    def boom(path):
        raise RuntimeError("publish down")

    with pytest.raises(RuntimeError):
        train_sft.save_and_publish(_FakeClient(), boom, tmp_path, "r1", _INFO, False)
    info = _read(tmp_path)
    assert info["checkpoint_path"] == "tinker://run/r1" and info["published"] is False


def test_publish_success_marks_published(tmp_path):
    seen = []
    out = train_sft.save_and_publish(_FakeClient(), seen.append, tmp_path, "r1", _INFO, False)
    assert seen == ["tinker://run/r1"] and out["published"] is True
    assert _read(tmp_path)["published"] is True


def test_no_publish_writes_false_and_skips_publish(tmp_path):
    def boom(path):
        raise AssertionError("must not publish")

    out = train_sft.save_and_publish(_FakeClient(), boom, tmp_path, "r1", _INFO, True)
    assert out["published"] is False and _read(tmp_path)["published"] is False


def test_missing_or_empty_train_file_raises_before_any_client(tmp_path, monkeypatch):
    def no_client(*a, **k):
        raise AssertionError("no client may be created")

    monkeypatch.setattr(train_sft.tinker, "ServiceClient", no_client)
    monkeypatch.setattr(train_sft, "RUNS", tmp_path)
    monkeypatch.setattr(train_sft, "build_renderer", lambda: "R")
    monkeypatch.setattr(train_sft, "parse_args", lambda argv=None: train_sft.argparse.Namespace(
        name="r1", rank=32, lr=1e-4, batch_size=32, no_publish=True))
    (tmp_path / "r1").mkdir()
    with pytest.raises(FileNotFoundError, match="pipeline.select"):
        train_sft.main()
    (tmp_path / "r1" / "train.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="no training examples"):
        train_sft.main()


def test_save_state_logs_and_prints_path(tmp_path, capsys):
    import json

    class Client:
        def save_state(self, name, ttl_seconds):
            assert ttl_seconds == train_sft.STATE_TTL_SECONDS
            return _Done(type("S", (), {"path": f"tinker://run/state/{name}"})())

    log = tmp_path / "train_log.jsonl"
    path = train_sft.save_state(Client(), log, "r1-step250", 250)
    assert path == "tinker://run/state/r1-step250"
    assert json.loads(log.read_text()) == {"step": 250, "state_path": path, "state_name": "r1-step250"}
    assert path in capsys.readouterr().out


def test_stale_checkpoint_kept_when_client_creation_fails(tmp_path, monkeypatch):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "train.jsonl").write_text('{"id": "a", "messages": []}\n')
    (run / "checkpoint.json").write_text('{"checkpoint_path": "tinker://old"}')

    def down():
        raise RuntimeError("service down")

    monkeypatch.setattr(train_sft, "RUNS", tmp_path)
    monkeypatch.setattr(train_sft, "build_renderer", lambda: "R")
    monkeypatch.setattr(train_sft, "render_examples", lambda rows, r: ["d"])
    monkeypatch.setattr(train_sft, "count_tokens", lambda data: 1)
    monkeypatch.setattr(train_sft.tinker, "ServiceClient", down)
    monkeypatch.setattr(train_sft, "parse_args", lambda argv=None: train_sft.argparse.Namespace(
        name="r1", rank=32, lr=1e-4, batch_size=32, no_publish=True))
    with pytest.raises(RuntimeError, match="service down"):
        train_sft.main()
    assert (run / "checkpoint.json").exists()

    class Service:
        def create_lora_training_client(self, base_model, rank):
            return _Done(object())  # a client with no methods: the loop fails on first use

    monkeypatch.setattr(train_sft.tinker, "ServiceClient", Service)
    with pytest.raises(AttributeError):
        train_sft.main()
    assert not (run / "checkpoint.json").exists()  # removed once the client existed, before training

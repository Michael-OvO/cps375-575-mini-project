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
    assert all(c[1] == "R" and c[2] == train_sft.MAX_TRAIN_TOKENS for c in calls)
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

from pipeline import train_sft as T


def test_replay_flags_parse():
    a = T.parse_args(["--name", "v4r", "--rows_from", "v4", "--resume_state", "tinker://x/weights/v4-step1500",
                      "--start_step", "1501", "--lr", "4.7297908091376354e-4", "--no_publish"])
    assert a.rows_from == "v4" and a.start_step == 1501 and a.resume_state.endswith("v4-step1500")
    assert a.final_state_ttl == "none"


def test_step_range_and_lr_match_v4_schedule():
    assert T.step_range(1501, 1688) == range(1501, 1688)
    assert len(T.step_range(1501, 1688)) == 187
    assert T.lr_at(1501, 1688, 4.7297908091376354e-4) == 4.7297908091376354e-4 * (1 - 1501 / 1688)


def test_final_state_ttl_validated_early():
    import pytest
    for bad in ("1h", "0", "-5", "1.5", ""):
        with pytest.raises(SystemExit):
            T.parse_args(["--name", "x", "--final_state_ttl", bad])
    assert T.parse_args(["--name", "x", "--final_state_ttl", "3600"]).final_state_ttl == "3600"


def test_final_state_failure_still_saves_and_publishes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace as NS

    class Done:
        def __init__(self, v):
            self.v = v

        def result(self):
            return self.v

    class Client:
        def forward_backward(self, batch, loss_fn):
            return Done(NS(loss_fn_outputs=[]))

        def optim_step(self, adam):
            return Done(None)

        def save_state(self, name, ttl_seconds):
            raise RuntimeError("state save down")

        def save_weights_for_sampler(self, name):
            return Done(NS(path=f"tinker://run/{name}"))

    run = tmp_path / "r1"
    run.mkdir()
    (run / "train.jsonl").write_text('{"id": "a", "messages": []}\n')
    monkeypatch.setattr(T, "RUNS", tmp_path)
    monkeypatch.setattr(T, "build_renderer", lambda: "R")
    monkeypatch.setattr(T, "render_examples", lambda rows, r: [NS(loss_fn_inputs={})])
    monkeypatch.setattr(T, "count_tokens", lambda data: 1)
    monkeypatch.setattr(T, "batch_loss", lambda o, b: 0.0)
    monkeypatch.setattr(T.tinker, "ServiceClient", lambda: NS(
        create_lora_training_client=lambda base_model, rank: Done(Client())))
    monkeypatch.setattr(T, "parse_args", lambda argv=None: T.argparse.Namespace(
        name="r1", rank=32, lr=1e-4, batch_size=32, no_publish=True, skip_preflight=True,
        rows_from=None, resume_state=None, start_step=0, ledger=None, final_state_ttl="none"))
    T.main()
    log = [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
    assert {"final_state_error": "state save down"} in log
    assert json.loads((run / "checkpoint.json").read_text())["checkpoint_path"] == "tinker://run/r1"

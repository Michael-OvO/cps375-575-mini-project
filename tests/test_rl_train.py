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


import pytest
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from pipeline.rl import train_rl as TR


def test_build_datum_shapes_and_masking():
    prompt = types.ModelInput.from_ints([1, 2, 3, 4])
    d = TR.build_datum(prompt, [10, 11, 12], [-0.1, -0.2, -0.3], [0.5, 0.5, 0.5])
    tgt = d.loss_fn_inputs["target_tokens"].to_torch().tolist()
    adv = d.loss_fn_inputs["advantages"].to_torch().tolist()
    lp = d.loss_fn_inputs["logprobs"].to_torch().tolist()
    assert d.model_input.length == len(tgt) == len(adv) == len(lp) == 3 + 3
    assert tgt == [0, 0, 0, 10, 11, 12] and adv == [0, 0, 0, 0.5, 0.5, 0.5]
    assert lp[3:] == pytest.approx([-0.1, -0.2, -0.3])


def test_ratio_kl():
    assert TR.ratio_kl([-1.0, -2.0], [-1.1, -1.9]) == pytest.approx(0.0)
    assert TR.ratio_kl([-1.0, -1.0], [-1.5, -1.5]) == pytest.approx(0.5)


def test_step_metrics_per_domain():
    groups = [{"task": "math", "rewards": [1, 0], "lens": [100, 300], "trunc": [False, True],
               "reasons": ["pass", "truncated"], "vis": [None, None], "kept": True, "adv_tokens": 200.0},
              {"task": "math", "rewards": [0, 0], "lens": [80, 80], "trunc": [False, False],
               "reasons": ["fail", "fail"], "vis": [None, None], "kept": False, "adv_tokens": 0.0,
               "sft_tokens": 40},
              {"task": "if", "rewards": [0.5, 0.5], "lens": [50, 50], "trunc": [False, False],
               "reasons": ["gate", "scored"], "vis": [None, None], "kept": False, "adv_tokens": 0.0}]
    m = TR.step_metrics(groups)
    assert m["math/reward"] == 0.25 and m["math/len"] == 140 and m["math/trunc"] == 0.25
    assert m["if/informative"] == 0.0 and m["if/gate"] == 0.5
    assert m["math/adv_share"] == 1.0
    assert m["math/sft_groups"] == 1 and m["math/sft_tokens"] == 40 and m["if/sft_groups"] == 0


def test_demo_datum_is_ratio_one_weighted_nll():
    sft = types.Datum(model_input=types.ModelInput.from_ints([1, 2, 3, 4]), loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4, 5])),
        "weights": TensorData.from_torch(torch.tensor([0.0, 0.0, 1.0, 1.0]))})
    full_lp = [None, -0.5, -0.6, -0.7, -0.8]  # logprob of each token of [1, 2, 3, 4, 5] given its prefix
    d = TR.demo_datum(sft, full_lp, 0.25)
    assert d.loss_fn_inputs["target_tokens"].to_torch().tolist() == [2, 3, 4, 5]
    assert d.loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, 0.0, 0.25, 0.25]
    assert d.loss_fn_inputs["logprobs"].to_torch().tolist() == pytest.approx([0.0, 0.0, -0.7, -0.8])


def test_step_metrics_code_vphf_matches_calibration_definition():
    from pipeline.rl.calibrate import domain_baseline

    g = {"task": "code", "rewards": [1.0, 0.0, 0.0, 0.0], "lens": [10, 20, 30, 40], "trunc": [False] * 3 + [True],
         "reasons": ["pass", "fail", "guard", "truncated"], "vis": [None, True, None, None], "kept": True,
         "adv_tokens": 1.0}
    m = TR.step_metrics([g])
    b = domain_baseline([g])["code"]
    assert (m["code/vphf"], m["code/gate"], m["code/len"], m["code/trunc"]) == \
        (b["vphf"], b["gate"], b["mean_len"], b["trunc"]) == (0.25, 0.25, 25.0, 0.25)


def test_worst_case_usd_counts_every_sample_at_cap():
    from pipeline.rl.ledger import cost

    usd = TR.worst_case_usd([("math", 100), ("if", 50)], {"math": 4096, "if": 1024}, G=8, demos=1)
    exp = (cost("prefill", 100) + cost("sample", 8 * 4096) + cost("train", 8 * (100 + 4096))
           + cost("prefill", 50) + cost("sample", 8 * 1024) + cost("train", 8 * (50 + 1024))
           + cost("prefill", 100 + 4096) + cost("train", 100 + 4096))  # one demo bounded by the longest group
    assert usd == pytest.approx(exp)


def test_log_rows_keeps_last_row_per_step_before_current(tmp_path):
    from pipeline import jsonl

    p = tmp_path / "train_log.jsonl"
    for r in ({"step": 0, "x": 1}, {"step": 1, "x": 1}, {"step": 2, "x": 1}, {"step": 1, "x": 2}):
        jsonl.append(p, r)
    assert TR.log_rows(p, 2) == [{"step": 0, "x": 1}, {"step": 1, "x": 2}]

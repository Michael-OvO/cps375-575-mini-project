from types import SimpleNamespace as _NS

import pytest
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from pipeline import train_sft as T
from pipeline.rl import train_rl as TR


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


# ---- fake-client tests of the loop: no Tinker calls, no network ----

_RUN = TR.Run  # the real class: _main patches TR.Run


class _Done:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


class _Seq:
    def __init__(self, n, stop="stop"):
        self.tokens, self.logprobs, self.stop_reason = list(range(5, 5 + n)), [-0.5] * n, stop


class _Sampling:
    def __init__(self, lens):
        self.lens = lens

    async def sample_async(self, prompt, num_samples, sampling_params):
        return _NS(sequences=[_Seq(self.lens[i % len(self.lens)], "length" if i == 0 else "stop")
                              for i in range(num_samples)])

    async def compute_logprobs_async(self, mi):
        return [None] + [-0.5] * (mi.length - 1)


class _Training:
    def __init__(self, lens=(1, 2), save_failures=0):
        self.lens, self.save_failures, self.names, self.calls = lens, save_failures, [], []

    def save_weights_and_get_sampling_client(self):
        return _Done(_Sampling(self.lens))

    def forward_backward(self, datums, loss_fn):
        assert loss_fn == "importance_sampling"
        self.calls.append(len(datums))
        return _Done(_NS(loss_fn_outputs=[{"logprobs": TensorData.from_torch(d.loss_fn_inputs["logprobs"].to_torch())}
                                          for d in datums]))

    def optim_step(self, p):
        return _Done(None)

    def save_state(self, name, ttl_seconds):
        self.names.append(name)
        if self.save_failures:
            self.save_failures -= 1
            raise RuntimeError("save down")
        return _Done(_NS(path=f"tinker://s/{name}"))

    def save_weights_for_sampler(self, name, ttl_seconds):
        return _Done(_NS(path=f"tinker://w/{name}"))


class _Renderer:
    def build_generation_prompt(self, msgs):
        return types.ModelInput.from_ints([1, 2, 3])

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, toks):
        return {"content": "x" * len(toks)}, True


async def _score(item, text, truncated, check_visible=False, graded_trunc=False):
    from pipeline.rl.rewards import Scored

    if item["id"] == "bad":
        raise RuntimeError("scorer down")
    if item["task"] == "math":
        return Scored(0.0, {"reason": "fail"})  # all-fail: gated SFT
    return Scored(float(len(text) % 2), {"reason": "fail", "visible_pass": True} if item["task"] == "code"
                  else {"reason": "scored"})


def _fake_run(tmp_path, monkeypatch, cap=100.0, lens=(1, 2), steps=2, **kw):
    from pipeline.rl import advantages as A
    from pipeline.rl.ledger import Ledger

    monkeypatch.setattr(TR.rewards, "score", _score)
    monkeypatch.setattr(TR.demos, "pick", lambda item, idx: [{"role": "user", "content": "q"},
                                                             {"role": "assistant", "content": "a"}])
    import tinker_cookbook.supervised.data as D

    monkeypatch.setattr(D, "conversation_to_datum", lambda m, r, max_length, train_on_what, reduction: types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={"target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4, 5])),
                        "weights": TensorData.from_torch(torch.tensor([0.0, 0.0, 1.0, 1.0]))}))
    monkeypatch.setattr(TR.time, "sleep", lambda s: None)
    items = ([{"id": "m0", "task": "math", "prompt": "p", "text": "t"}]
             + [{"id": f"c{i}", "task": "code", "prompt": "p"} for i in range(2)]
             + [{"id": f"i{i}", "task": "if", "prompt": "p"} for i in range(2)] + [{"id": "bad", "task": "if", "prompt": "p"}])
    run = object.__new__(_RUN)
    run.dir, run.smoke, run.phase = tmp_path, False, "4"
    run.log, run.resume_path = tmp_path / "train_log.jsonl", tmp_path / "resume.json"
    run.cfg = {"G": 8, "caps": {"math": 4096, "if": 1024, "code": 4096}, "lr": 2e-5, "ttl_seconds": 10,
               "save_every": 10, "baseline": {d: {"L": 100, "mean_len": 2, "trunc": 0.5, "gate": 0.0, "vphf": 0.0}
                                              for d in TR.DOMAINS},
               "gated_sft": {"on": True, "mu": 0.5, "max_per_step": 8}, "teacher": {"on": False, "lam": 0.0}} | kw
    run.pools = {i["id"]: i for i in items}
    run.sched = [[i["id"] for i in items]] * steps
    run.led = Ledger(tmp_path / "ledger.jsonl", cap_usd=cap)
    run.teacher, run.saved_step, run.step, run.demos = None, None, 0, {}
    run.graded_trunc, run.trunc_since, run.trunc_ceiling = TR.trunc_rules(run.cfg)
    run.renderer, run.client = _Renderer(), _Training(lens)
    run.scaler = A.DomainScaler({"math": (0.3, 100), "if": (0.2, 50), "code": (0.4, 200)})
    run.start = lambda service: None
    return run


def test_one_step_datums_demos_ledger_order_and_coef(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, monkeypatch)
    coef_before = {d: run.scaler.coef(d) for d in TR.DOMAINS}
    m = run.one_step()
    assert m["score_errors"] == 1 and m["math/sft_groups"] == 1 and m["math/sft_tokens"] == 2
    assert m["kl_sample_train"] == pytest.approx(0.0) and m["kl_abs"] == pytest.approx(0.0)
    assert m["coef"] == coef_before and {d: run.scaler.coef(d) for d in TR.DOMAINS} != coef_before
    assert m["datums"] == 4 * 8 + 1 and run.client.calls == [33]  # 2 code + 2 if groups, one demo, one call
    rows = run.led.rows()
    assert [(r["kind"], r["note"]) for r in rows] == [("sample", "step 0"), ("prefill", "step 0"),
                                                      ("prefill", "step 0 demos"), ("train", "step 0")]
    assert rows[1]["tokens"] == 6 * 3 and rows[2]["tokens"] == 5


def test_no_rl_datums_logs_none_and_monitor_skips_kl(tmp_path, monkeypatch):
    from pipeline.rl import monitor

    run = _fake_run(tmp_path, monkeypatch, lens=(2,))  # every sample the same length: all groups constant
    m = run.one_step()
    assert m["kl_sample_train"] is None and m["kl_abs"] is None and m["datums"] == 3  # demos only: 1 math + 2 all-fail code
    assert monitor.train_checks([m], run.cfg["baseline"]) == ([], [])


def test_constant_math_group_of_loops_trains_only_loop_tokens_and_keeps_demo(tmp_path, monkeypatch):
    from pipeline.rl.rewards import LOOP_TOKEN_PENALTY, TRUNC_PENALTY, Scored

    seen = []

    async def score(item, text, truncated, check_visible=False, graded_trunc=False):
        seen.append(graded_trunc)
        if item["task"] == "math":  # every sample cut off in a loop halfway: a constant group of -0.25
            return Scored(TRUNC_PENALTY, {"reason": "truncated_loop", "correct": False, "loop_onset": 0.5})
        return Scored(0.0, {"reason": "fail"})

    run = _fake_run(tmp_path, monkeypatch, lens=(4,), graded_trunc={"on": True})
    monkeypatch.setattr(TR.rewards, "score", score)
    c = run.scaler.coef("math")
    got = []
    fb = run.client.forward_backward
    run.client.forward_backward = lambda datums, loss_fn: (got.extend(datums), fb(datums, loss_fn))[1]
    m = run.one_step()
    assert all(seen) and m["math/loops"] == 8 and m["math/acc"] == 0.0
    adv = got[0].loss_fn_inputs["advantages"].to_torch().tolist()[-4:]  # 4 completion tokens, loop from token 2
    assert adv == pytest.approx([0.0, 0.0, -c * LOOP_TOKEN_PENALTY, -c * LOOP_TOKEN_PENALTY])
    # the 8 loop samples (constant group: zero outcome advantage) plus the math demo; code/if are constant 0
    assert m["datums"] == 8 + 1 + 2 and m["math/sft_groups"] == 1
    assert m["kl_sample_train"] == pytest.approx(0.0)


def test_loop_tail_and_all_fail():
    from pipeline.rl import advantages as A

    assert A.loop_tail([0.1, 0.1, 0.1, 0.1], 2, 0.25) == pytest.approx([0.1, 0.1, -0.15, -0.15])
    assert A.loop_tail([0.0, 0.0], 5, 1.0) == [0.0, 0.0]
    assert TR.all_fail({"task": "math", "rewards": [-0.25, 0.0], "correct": [False, False]})
    assert not TR.all_fail({"task": "math", "rewards": [0.75, -0.25], "correct": [True, False]})
    assert TR.all_fail({"task": "code", "rewards": [0.0, 0.0]})


def test_step_metrics_math_accuracy_is_unshaped():
    g = {"task": "math", "rewards": [1.0, -0.25, 0.75, 0.0], "lens": [10] * 4, "trunc": [False, True, True, False],
         "reasons": ["pass", "truncated_loop", "truncated_answered", "fail"], "vis": [None] * 4, "kept": True,
         "adv_tokens": 1.0, "correct": [True, False, True, False], "loop": [None, 0.3, None, None]}
    m = TR.step_metrics([g])
    assert m["math/acc"] == 0.5 and m["math/loops"] == 1 and m["math/reward"] == pytest.approx(0.375)


def test_save_retries_with_fresh_names_then_exits(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, monkeypatch)
    run.client.save_failures = 2
    st = run.save("step0010")
    assert run.client.names == ["rl1-step0010", "rl1-step0010-r1", "rl1-step0010-r2"]
    assert st["state_path"] == "tinker://s/rl1-step0010-r2" and run.saved_step == 0
    run.client.save_failures = 9
    with pytest.raises(SystemExit, match="failed 4 times"):
        run.save("step0020")


def _main(monkeypatch, run, *argv):
    monkeypatch.setattr(TR, "Run", lambda smoke: run)
    monkeypatch.setattr(TR.data_gates, "require_passed", lambda: None)
    monkeypatch.setattr(TR.tinker, "ServiceClient", lambda: None)
    monkeypatch.setattr(TR, "sync_billing", lambda led: 0.0)
    monkeypatch.setattr(TR.monitor, "warn", lambda msgs, step: None)
    monkeypatch.setattr(TR.sys, "argv", ["train_rl", *argv])
    with pytest.raises(SystemExit) as e:
        TR.main()
        raise SystemExit(0)
    return e.value.code


def test_main_done_paused_and_cap_exit_codes(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    run = _fake_run(tmp_path / "a", monkeypatch)
    assert _main(monkeypatch, run) == 0 and run.step == 2
    assert [r["step"] for r in TR.log_rows(run.log, 9)] == [0, 1] and run.saved_step == 2
    (tmp_path / "b").mkdir()
    run = _fake_run(tmp_path / "b", monkeypatch)
    (run.dir / "PAUSED").write_text("eval pause\n")
    assert _main(monkeypatch, run) == 3 and run.step == 0
    (tmp_path / "c").mkdir()
    run = _fake_run(tmp_path / "c", monkeypatch, cap=0.001)
    assert _main(monkeypatch, run) == 4 and run.saved_step == 0 and not run.led.rows()


def test_main_over_budget_and_monitor_error_pause(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    run = _fake_run(tmp_path / "a", monkeypatch)
    monkeypatch.setattr(run, "step_estimate", lambda step: 0.0)
    assert _main(monkeypatch, run) == 3 and run.step == 1 and run.saved_step == 1
    assert "over budget estimate" in (run.dir / "PAUSED").read_text()
    (tmp_path / "b").mkdir()
    run = _fake_run(tmp_path / "b", monkeypatch)
    monkeypatch.setattr(TR.monitor, "train_checks", lambda rows, base: 1 / 0)
    assert _main(monkeypatch, run) == 3 and "monitor error" in (run.dir / "PAUSED").read_text()


def test_main_kl_monitor_failure_keeps_training(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, monkeypatch, save_every=1)
    monkeypatch.setattr(TR, "KL_EVERY", 1)
    monkeypatch.setattr(run, "kl_to_v4", lambda path: (_ for _ in ()).throw(RuntimeError("v4 down")))
    assert _main(monkeypatch, run) == 0 and run.step == 2


def test_main_reraises_bugs_and_retries_transient(tmp_path, monkeypatch):
    run = _fake_run(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(run, "one_step", lambda: calls.append(1) or {}["missing"])
    with pytest.raises(KeyError):
        _main(monkeypatch, run)
    assert len(calls) == 1
    calls.clear()
    monkeypatch.setattr(run, "one_step", lambda: calls.append(1) or (_ for _ in ()).throw(RuntimeError("503")))
    with pytest.raises(RuntimeError):
        _main(monkeypatch, run)
    assert len(calls) == 3

"""rl1 training loop: synchronous on-policy RLVR on Tinker (spec §4, §6, §8). One optimizer step per batch of
64 groups x G=8; clip-free importance_sampling loss pinned; centered group advantages with constant groups dropped;
per-domain EMA scaling from previous steps; gated SFT on all-fail math/code groups as ratio-1 datums in the same
call; checkpoints and a resume JSON every 10 steps; pause rules; spend cap.

Exit codes: 0 done, 3 paused (PAUSED in the run dir holds the reasons), 4 spend cap."""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from pipeline import jsonl
from pipeline.config import RUNS, SEED, STUDENT
from pipeline.rl import advantages as A
from pipeline.rl import data_gates, demos, monitor, rewards
from pipeline.rl.ledger import CapExceeded, Ledger, cost, sync_billing
from pipeline.student import _text

TRUNC = "length"  # tinker.types.StopReason = Literal['length', 'stop']
DOMAINS = ("math", "if", "code")
DEMO_TASKS = ("math", "code")
KL_EVERY = 20  # spec §6.5: KL monitor every 20 steps, 64 rollouts scored under v4
KL_ROLLOUTS = 64
SMOKE_GROUPS = 8  # spec §7: smoke = 3 steps x 8 groups
BACKOFF = (10, 30, 60)  # seconds before each retry of a crashed step or a failed save
BUGS = (AssertionError, KeyError, TypeError, ValueError)  # re-raised at once: retrying cannot fix them
OVER_BUDGET = 1.3  # soft stop: this run's spend > 1.3 x the schedule estimate of the steps it has done


def build_datum(prompt: types.ModelInput, tokens: list[int], logprobs: list[float],
                token_adv: list[float]) -> types.Datum:
    ob = prompt.length - 1
    model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
    tgt = [0] * ob + tokens
    lp = [0.0] * ob + logprobs
    adv = [0.0] * ob + token_adv
    assert model_input.length == len(tgt) == len(lp) == len(adv)
    return types.Datum(model_input=model_input, loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(tgt)),
        "logprobs": TensorData.from_torch(torch.tensor(lp, dtype=torch.float32)),
        "advantages": TensorData.from_torch(torch.tensor(adv, dtype=torch.float32))})


def demo_datum(sft: types.Datum, full_lp: list[float | None], weight: float) -> types.Datum:
    """Ratio-1 importance_sampling datum from a cross_entropy datum: position j predicts token j+1 of the full
    sequence, so its sampler logprob is full_lp[j + 1] under the current policy; advantage = weight on trained
    tokens. The loss -ratio*A then has gradient -weight * grad log p, a weighted NLL."""
    tgt = sft.loss_fn_inputs["target_tokens"].to_torch().tolist()
    w = sft.loss_fn_inputs["weights"].to_torch().tolist()
    assert len(full_lp) == len(tgt) + 1 == sft.model_input.length + 1
    lp = [float(full_lp[j + 1]) if w[j] > 0 else 0.0 for j in range(len(tgt))]
    return types.Datum(model_input=sft.model_input, loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(tgt)),
        "logprobs": TensorData.from_torch(torch.tensor(lp, dtype=torch.float32)),
        "advantages": TensorData.from_torch(torch.tensor([weight * x for x in w], dtype=torch.float32))})


def ratio_kl(sampler_lp: list[float], train_lp: list[float]) -> float:
    return sum(s - t for s, t in zip(sampler_lp, train_lp)) / max(1, len(sampler_lp))


def abs_kl(sampler_lp: list[float], train_lp: list[float]) -> float:
    """Mean |sampler - train| logprob gap: visibility only, since signed errors can cancel in ratio_kl."""
    return sum(abs(s - t) for s, t in zip(sampler_lp, train_lp)) / max(1, len(sampler_lp))


def step_metrics(groups: list[dict]) -> dict:
    """Per-domain step metrics. len, trunc, gate and code/vphf use calibrate.domain_baseline's definitions
    (per-sample means), so monitor.train_checks compares like with like."""
    m = {}
    total_adv = sum(g["adv_tokens"] for g in groups) or 1.0
    for d in DOMAINS:
        gs = [g for g in groups if g["task"] == d]
        if not gs:
            continue
        n = sum(len(g["rewards"]) for g in gs)
        m[f"{d}/reward"] = sum(sum(g["rewards"]) for g in gs) / n
        m[f"{d}/len"] = sum(sum(g["lens"]) for g in gs) / n
        m[f"{d}/trunc"] = sum(sum(g["trunc"]) for g in gs) / n
        m[f"{d}/gate"] = sum(r in ("gate", "guard") for g in gs for r in g["reasons"]) / n
        m[f"{d}/informative"] = sum(g["kept"] for g in gs) / len(gs)
        m[f"{d}/adv_share"] = sum(g["adv_tokens"] for g in gs) / total_adv
        m[f"{d}/sft_groups"] = sum("sft_tokens" in g for g in gs)
        m[f"{d}/sft_tokens"] = sum(g.get("sft_tokens", 0) for g in gs)
        if d == "code":
            m["code/vphf"] = sum(v is True for g in gs for v in g["vis"]) / n
        if d == "math" and all(c is not None for g in gs for c in g.get("correct", [None])):
            m["math/acc"] = sum(sum(g["correct"]) for g in gs) / n  # grader accuracy, unshaped
            m["math/loops"] = sum(o is not None for g in gs for o in g.get("loop", []))
    return m


def all_fail(g: dict) -> bool:
    """No sample solved the task. Math uses the grader's verdict, since graded truncation makes rewards negative
    or fractional; code and IF use the reward."""
    if g["task"] == "math" and all(c is not None for c in g.get("correct", [None])):
        return not any(g["correct"])
    return max(g["rewards"]) == 0


def worst_case_usd(groups: list[tuple[str, int]], caps: dict[str, int], G: int, demos: int) -> float:
    """Upper bound on one step's cost, reserved before sampling: every completion runs to its cap (one prefill
    per group, training on G x (prompt + cap)), plus up to `demos` gated-SFT datums, each bounded by the longest
    prompt + cap of the step (one prefill for its sampler logprobs, then training)."""
    usd = sum(cost("prefill", p) + cost("sample", G * caps[t]) + cost("train", G * (p + caps[t])) for t, p in groups)
    longest = max((p + caps[t] for t, p in groups), default=0)
    return usd + demos * (cost("prefill", longest) + cost("train", longest))


def log_rows(path: Path, before: int) -> list[dict]:
    """train_log rows for steps < before, the last row per step: after a resume, rows of steps that were rolled
    back (crash between checkpoints) are superseded by the retrained ones."""
    by = {}
    for r in jsonl.read(path) if path.exists() else []:
        if "step" in r and r["step"] < before:
            by[r["step"]] = r
    return [by[s] for s in sorted(by)]


class Run:
    def __init__(self, smoke: bool):
        self.dir = RUNS / ("rl1_smoke" if smoke else "rl1")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.smoke = smoke
        self.cfg = json.loads((RUNS / "rl1" / "config.json").read_text())
        sched = json.loads((RUNS / "rl1" / "schedule.json").read_text())
        pools = {}
        for t in DOMAINS:
            for r in jsonl.read(RUNS / "rl1" / "pools" / f"{t}.jsonl"):
                pools[r["id"]] = r
        self.pools = pools
        main_steps = self.cfg["steps"]  # schedule.json holds main_steps + 3 smoke steps at the end
        assert len(sched) == main_steps + 3, "schedule.json must hold the main steps plus 3 smoke steps"
        if smoke:  # 8 groups from each of the 3 reserved tail steps: disjoint, and never trained by the main run.
            # A seeded sample, not the first 8: schedule.build lists each step's groups task by task.
            rng = random.Random(SEED)
            self.sched = [rng.sample(step, SMOKE_GROUPS) for step in sched[main_steps:]]
        else:
            self.sched = sched[:main_steps]
        self.led = Ledger()
        self.log = self.dir / "train_log.jsonl"
        self.resume_path = self.dir / "resume.json"
        self.demos = demos.load() if self.cfg["gated_sft"]["on"] else {}
        self.graded_trunc = bool(self.cfg.get("graded_trunc", {}).get("on", False))
        self.teacher = None
        if self.cfg["teacher"]["on"] and not smoke:
            from pipeline.rl.teacher import TeacherTerm
            self.teacher = TeacherTerm(self.led)
        self.saved_step = None
        self.phase = "3" if smoke else "4"

    def start(self, service: tinker.ServiceClient):
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.service = service
        self.renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(STUDENT),
                                               get_tokenizer(STUDENT))
        if self.resume_path.exists():
            st = json.loads(self.resume_path.read_text())
            client = service.create_training_client_from_state_with_optimizer(st["state_path"]).result()
            self.step = st["step"]
            self.saved_step = st["step"]
            self.scaler = A.DomainScaler.from_state(st["ema"])
            random.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
        else:
            client = service.create_training_client_from_state(self.cfg["start_state"]).result()  # fresh Adam
            self.step = 0
            self.scaler = A.DomainScaler({t: tuple(v) for t, v in self.cfg["ema_init"].items()})
            random.seed(SEED)
        self.client = client
        self.v4 = service.create_sampling_client(model_path=self.cfg["v4_sampler"]).result()

    def prompt(self, item: dict) -> types.ModelInput:
        # Same prompt stripping as student.Sampler.answer / calibrate, which the official eval mirrors.
        return self.renderer.build_generation_prompt([{"role": "user", "content": item["prompt"].strip()}])

    def params(self, task: str) -> tinker.SamplingParams:
        return tinker.SamplingParams(temperature=1.0, top_p=1.0, stop=self.renderer.get_stop_sequences(),
                                     max_tokens=self.cfg["caps"][task])

    def text(self, seq) -> str:
        # Same completion text as student.Sampler.answer, so rewards score the string the official eval scores.
        return _text(self.renderer.parse_response(seq.tokens)[0]["content"])

    async def sample(self, sampling_client, items: list[dict], prompts: list[types.ModelInput]) -> list[dict]:
        async def one(item, prompt):
            res = await sampling_client.sample_async(prompt=prompt, num_samples=self.cfg["G"],
                                                     sampling_params=self.params(item["task"]))
            return {"item": item, "task": item["task"], "prompt": prompt, "seqs": list(res.sequences)}

        return list(await asyncio.gather(*(one(i, p) for i, p in zip(items, prompts))))

    async def score(self, groups: list[dict]) -> None:
        async def one(g):
            seqs = g["seqs"]
            g["texts"] = [self.text(s) for s in seqs]
            g["trunc"] = [s.stop_reason == TRUNC for s in seqs]
            g["lens"] = [len(s.tokens) for s in seqs]
            try:
                scored = await asyncio.gather(*(rewards.score(g["item"], t, tr, check_visible=g["task"] == "code",
                                                              graded_trunc=self.graded_trunc)
                                                for t, tr in zip(g["texts"], g["trunc"])))
            except Exception as e:  # the samples are paid for: drop this group, not the step
                print(f"score error {g['item']['id']}: {e!r}", file=sys.stderr)
                g["score_error"] = repr(e)
                return
            g["rewards"] = [s.reward for s in scored]
            g["reasons"] = [s.info.get("reason") for s in scored]
            g["vis"] = [s.info.get("visible_pass") for s in scored]
            g["correct"] = [s.info.get("correct") for s in scored]  # math only: the grader's verdict
            g["loop"] = [s.info.get("loop_onset") for s in scored]  # math with graded_trunc: loop start (word share)

        await asyncio.gather(*(one(g) for g in groups))

    def save(self, tag: str) -> dict:
        """Save state + sampler weights and the resume JSON; up to 3 retries with backoff, then exit 1. A retry
        uses a fresh name, since a half-finished attempt may have taken the first (save_state never overwrites)."""
        ttl = self.cfg["ttl_seconds"]
        base = f"{'rl1smoke' if self.smoke else 'rl1'}-{tag}"
        for attempt in range(len(BACKOFF) + 1):
            name = base if attempt == 0 else f"{base}-r{attempt}"
            try:
                state = self.client.save_state(name, ttl_seconds=ttl).result().path
                sampler = self.client.save_weights_for_sampler(name, ttl_seconds=ttl).result().path
                break
            except Exception as e:
                if attempt == len(BACKOFF):
                    raise SystemExit(f"save {base} failed {attempt + 1} times, last: {e!r}; the last good state is "
                                     f"in {self.resume_path}")
                print(f"save {name} failed ({e!r}); retrying in {BACKOFF[attempt]}s", file=sys.stderr)
                time.sleep(BACKOFF[attempt])
        version, internal, gauss = random.getstate()
        st = {"step": self.step, "state_path": state, "sampler_path": sampler, "ema": self.scaler.state(),
              "rng": [version, list(internal), gauss], "ledger_total": self.led.total()}
        self.resume_path.write_text(json.dumps(st))
        jsonl.append(self.dir / "checkpoints.jsonl", {"step": self.step, "state_path": state, "sampler_path": sampler,
                                                      "ts": time.time()})
        self.saved_step = self.step
        return st

    def save_once(self, tag: str) -> None:
        """Save for a pause or cap stop unless the current step's state is already saved."""
        if self.saved_step != self.step:
            self.save(f"{tag}-{self.step:04d}-{int(time.time())}")

    def one_step(self) -> dict:
        items = [self.pools[i] for i in self.sched[self.step]]
        prompts = [self.prompt(i) for i in items]
        gs_cfg = self.cfg["gated_sft"]
        n_demos = gs_cfg["max_per_step"] if gs_cfg["on"] else 0
        self.led.reserve(worst_case_usd([(i["task"], p.length) for i, p in zip(items, prompts)], self.cfg["caps"],
                                        self.cfg["G"], n_demos), f"rl1 step {self.step}")
        phase, note = self.phase, f"step {self.step}"
        sampling_client = self.client.save_weights_and_get_sampling_client().result()
        groups = asyncio.run(self.sample(sampling_client, items, prompts))
        self.led.record(phase, "sample", sum(len(s.tokens) for g in groups for s in g["seqs"]), note=note)
        self.led.record(phase, "prefill", sum(p.length for p in prompts), note=note)
        asyncio.run(self.score(groups))
        errors = [g for g in groups if "score_error" in g]
        groups = [g for g in groups if "score_error" not in g]

        datums, sampler_lps = [], []
        stats: dict[str, list] = {t: [] for t in DOMAINS}
        lens_by: dict[str, list] = {t: [] for t in DOMAINS}
        teacher_ratios = self.teacher.ratios(groups) if self.teacher else {}
        demo_slots = []  # (group, demo messages, weight) for all-fail math/code groups, spec §6.4
        for gi, g in enumerate(groups):
            d = g["task"]
            lens_by[d] += g["lens"]
            g["adv_tokens"] = 0.0
            constant = A.is_constant(g["rewards"])
            keep_for_teacher = self.teacher is not None and d == "if"
            g["kept"] = not constant
            if (gs_cfg["on"] and d in DEMO_TASKS and all_fail(g)
                    and len(demo_slots) < gs_cfg["max_per_step"]):
                msgs = demos.pick(g["item"], self.demos)
                if msgs is not None:
                    demo_slots.append((g, msgs, gs_cfg["mu"] * self.scaler.coef(d)))
            # loop starts as token indices (word share x tokens: approximate by a few tokens, which is harmless)
            loops = {si: int(o * len(seq.tokens)) for si, (seq, o) in enumerate(zip(g["seqs"], g["loop"]))
                     if o is not None and seq.tokens}
            skip = constant and not keep_for_teacher
            if skip and not loops:
                continue
            adv = A.centered(g["rewards"])
            if not constant:
                stats[d] += adv
            c = self.scaler.coef(d)
            for si, (seq, a) in enumerate(zip(g["seqs"], adv)):
                if not seq.tokens or (skip and si not in loops):
                    continue  # in a constant group only loop samples carry signal (their loop-token penalty)
                tr = teacher_ratios.get((gi, si))
                tok_adv = A.token_advantages(c * a, len(seq.tokens),
                                             teacher_logratio=tr["ratio"] if tr else None,
                                             lam=self.cfg["teacher"]["lam"], mask=tr["mask"] if tr else None)
                g["adv_tokens"] += abs(c * a) * len(seq.tokens)
                if si in loops:
                    pen = c * rewards.LOOP_TOKEN_PENALTY
                    tok_adv = A.loop_tail(tok_adv, loops[si], pen)
                    g["adv_tokens"] += pen * (len(seq.tokens) - loops[si])
                datums.append(build_datum(g["prompt"], list(seq.tokens), list(seq.logprobs), tok_adv))
                sampler_lps.append(list(seq.logprobs))
        if demo_slots:  # appended after the RL datums, so the KL zip below covers RL datums only
            from tinker_cookbook.renderers import TrainOnWhat
            from tinker_cookbook.supervised.data import conversation_to_datum

            sfts = [conversation_to_datum(m, self.renderer, max_length=None,
                                          train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, reduction="none")
                    for _, m, _ in demo_slots]

            async def full_lp(sd):
                last = sd.loss_fn_inputs["target_tokens"].to_torch().tolist()[-1]
                return await sampling_client.compute_logprobs_async(
                    sd.model_input.append(types.EncodedTextChunk(tokens=[last])))

            lps = asyncio.run(_gather_async(full_lp, sfts))
            self.led.record(phase, "prefill", sum(sd.model_input.length + 1 for sd in sfts), note=f"{note} demos")
            for (g, _, wgt), sd, lp in zip(demo_slots, sfts, lps):
                datums.append(demo_datum(sd, lp, wgt))
                g["sft_tokens"] = int(sum(x > 0 for x in sd.loss_fn_inputs["weights"].to_torch().tolist()))
        metrics = {"step": self.step, **step_metrics(groups), "datums": len(datums), "score_errors": len(errors)}
        if datums:
            fwd = self.client.forward_backward(datums, loss_fn="importance_sampling")
            opt = self.client.optim_step(types.AdamParams(learning_rate=self.cfg["lr"], beta1=0.9, beta2=0.95,
                                                          eps=1e-8))
            out = fwd.result()
            opt.result()
            self.led.record(phase, "train", sum(dat.model_input.length for dat in datums), note=note)
            kls, kabs = [], []
            for slp, o in zip(sampler_lps, out.loss_fn_outputs):
                tlp = o["logprobs"].to_torch().tolist()[-len(slp):]
                kls.append(ratio_kl(slp, tlp))
                kabs.append(abs_kl(slp, tlp))
        else:
            kls, kabs = [], []
        # None: no RL datums this step (every group constant; at most gated-SFT datums); monitor skips the KL pause
        metrics["kl_sample_train"] = sum(kls) / len(kls) if kls else None
        metrics["kl_abs"] = sum(kabs) / len(kabs) if kabs else None
        metrics["coef"] = {d: self.scaler.coef(d) for d in DOMAINS}  # the coefficients used this step
        for d in DOMAINS:
            self.scaler.update(d, stats[d], sum(lens_by[d]) / max(1, len(lens_by[d])))
        metrics["ledger_total"] = self.led.total()
        jsonl.append(self.log, metrics)
        jsonl.append(self.dir / "samples.jsonl", {"step": self.step, "samples": [
            {"id": g["item"]["id"], "task": g["task"], "reward": g["rewards"][0], "text": g["texts"][0][:4000]}
            for g in groups[:6]]})
        return metrics

    def kl_to_v4(self, sampler_path: str) -> float:
        """k1 estimate of KL(current || v4) per token on 64 fresh rollouts from the next schedule items."""
        cur = self.service.create_sampling_client(model_path=sampler_path).result()
        items = [self.pools[i] for i in self.sched[min(self.step, len(self.sched) - 1)]][:KL_ROLLOUTS]
        prompts = [self.prompt(i) for i in items]
        caps = self.cfg["caps"]
        self.led.reserve(sum(cost("sample", caps[i["task"]]) + cost("prefill", 2 * p.length + caps[i["task"]])
                             for i, p in zip(items, prompts)), f"kl monitor step {self.step}")

        async def one(item, prompt):
            r = await cur.sample_async(prompt=prompt, num_samples=1, sampling_params=self.params(item["task"]))
            s = r.sequences[0]
            if not s.tokens:
                return 0.0, 0, prompt.length
            full = prompt.append(types.EncodedTextChunk(tokens=list(s.tokens)))
            ref = await self.v4.compute_logprobs_async(full)
            ref_c = ref[-len(s.tokens):]
            return sum(a - b for a, b in zip(s.logprobs, ref_c)), len(s.tokens), prompt.length + full.length

        res = asyncio.run(_gather_async(lambda ip: one(*ip), list(zip(items, prompts))))
        self.led.record("4", "sample", sum(n for _, n, _ in res), note=f"kl monitor step {self.step}")
        self.led.record("4", "prefill", sum(f for _, _, f in res), note=f"kl monitor step {self.step}")
        return sum(k for k, _, _ in res) / max(1, sum(n for _, n, _ in res))

    def step_estimate(self, step: int) -> float:
        """schedule.step_usd for this step's groups: the per-step estimate the schedule's budget was built on."""
        from pipeline.rl.schedule import step_usd

        counts: dict[str, int] = {}
        for i in self.sched[step]:
            counts[self.pools[i]["task"]] = counts.get(self.pools[i]["task"], 0) + 1
        return step_usd(counts, self.cfg["baseline"], self.cfg["G"])

    def spent_since(self, ts: float) -> float:
        """This run's own spend since ts (its phase's local ledger rows), excluding checkpoint evals that run
        alongside it and write the same ledger."""
        return sum(r["usd"] for r in self.led.rows()
                   if r["ts"] >= ts and r["phase"] == self.phase and r["kind"] != "billing_sync")


async def _gather_async(fn, items):
    return await asyncio.gather(*(fn(i) for i in items))


def pause(run: Run, reasons: list[str]) -> None:
    run.save_once("pause")
    (run.dir / "PAUSED").write_text("\n".join(reasons) + "\n")
    print("PAUSED:", reasons)
    sys.exit(3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--until_step", type=int, default=None, help="stop after this many total steps (pilot: 40)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    data_gates.require_passed()
    run = Run(a.smoke)
    if (run.dir / "PAUSED").exists():
        print(f"{run.dir / 'PAUSED'} exists: review it, then delete it to continue")
        sys.exit(3)
    run.start(tinker.ServiceClient())
    end = len(run.sched) if a.until_step is None else min(a.until_step, len(run.sched))
    start_ts, expected, first = time.time(), 0.0, run.step
    crashes = 0
    while run.step < end:
        if (run.dir / "PAUSED").exists():  # written by ckpt_eval's eval pause rules while training runs
            run.save_once("pause")
            print("PAUSED:", (run.dir / "PAUSED").read_text().strip())
            sys.exit(3)
        try:
            m = run.one_step()
            crashes = 0
        except CapExceeded as e:
            run.save_once("cap")
            print("CAP:", e)
            sys.exit(4)
        except BUGS:
            raise
        except Exception as e:  # a crashed step is retried from the same weights; 3 in a row stop the run
            crashes += 1
            print(f"step {run.step} crashed ({crashes}/3): {e!r}")
            if crashes >= 3:
                raise
            time.sleep(BACKOFF[crashes - 1])
            continue
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                          if not isinstance(v, dict)}))
        expected += run.step_estimate(run.step)
        run.step += 1
        saved = None
        if run.step % run.cfg["save_every"] == 0 or run.step == end:
            saved = run.save(f"step{run.step:04d}")
            try:
                sync_billing(run.led)
            except Exception as e:  # the local ledger still enforces the cap
                print(f"billing sync failed: {e!r}", file=sys.stderr)
        # Pause rules every step (the sample-train KL pause applies at any step, spec §8); warnings are written
        # at checkpoints only, so window warnings are not repeated every step. A monitor error pauses the run:
        # it must never keep training unwatched, nor crash it unsaved.
        try:
            gt = run.cfg.get("graded_trunc", {})
            pauses, warns = monitor.train_checks(log_rows(run.log, run.step), run.cfg["baseline"],
                                                 trunc_since={"math": gt["from_step"]} if gt.get("on") else None)
            if saved is not None and not a.smoke:
                if run.step % KL_EVERY == 0:
                    try:
                        kl_v4 = run.kl_to_v4(saved["sampler_path"])
                        with open(run.dir / "kl_v4.jsonl", "a") as f:
                            f.write(json.dumps({"step": run.step, "kl_v4": kl_v4}) + "\n")
                        warns += monitor.kl_warning(kl_v4)
                    except CapExceeded as e:  # state of this step is already saved
                        print("CAP:", e)
                        sys.exit(4)
                    except Exception as e:  # a monitor, not a gate: note it and keep training
                        print(f"KL-to-v4 monitor failed: {e!r}", file=sys.stderr)
                        warns.append(f"KL-to-v4 monitor failed at step {run.step}: {e!r}")
                monitor.warn(warns, run.step)
        except Exception as e:
            pauses = [f"monitor error at step {run.step}: {e!r}"]
        spent = run.spent_since(start_ts)
        if spent > OVER_BUDGET * expected:
            pauses.append(f"over budget estimate: ${spent:.2f} spent by this run over {run.step - first} "
                          f"steps > {OVER_BUDGET} x ${expected:.2f} estimated")
        if pauses:
            pause(run, pauses)


if __name__ == "__main__":
    main()

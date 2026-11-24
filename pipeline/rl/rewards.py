"""Rewards for rl1. Each is the graded scorer plus only the guards the spec requires. Truncation is 0 for every
task, except math with graded_trunc on (see TRUNC_ANSWERED). Pure functions apart from the code sandbox."""

import asyncio
import signal
from collections import Counter
from dataclasses import dataclass, field

from pipeline.preflight import top_level_unsafe
from pipeline.rl import code_guard, sandbox
from pipeline.rl.gates import gate_reasons
from pipeline.rl.tests_split import Check, check, parse_value, split_tests
from pipeline.verify import ifeval as vi
from pipeline.verify import math as vm
from pipeline.verify.code import extract_code


@dataclass
class Scored:
    reward: float
    info: dict = field(default_factory=dict)


def _outcomes(values: list[str] | None, checks: list[Check]) -> list[bool]:
    """Per check: did the child report a literal that passes it. The expected values never left the parent."""
    out = []
    for i, c in enumerate(checks):
        ok = False
        if values is not None and values[i] != "ERR":
            try:
                ok = check(c, parse_value(values[i]))
            except ValueError:
                pass
        out.append(ok)
    return out


def _verify(code: str, checks: list[Check]) -> tuple[sandbox.Result, list[bool]]:
    res, values = sandbox.run_checks(code, [c.expr for c in checks])
    return res, _outcomes(values, checks)


def passes(code: str, tests: list[str]) -> bool:
    """Parent-side verification alone (no harness or guard checks): True only if every assert is supported and
    the child reported a passing value for each."""
    checks = split_tests(tests)
    return bool(checks) and all(_verify(code, checks)[1])


def code_reward(text: str, tests: list[str], hidden: list[str], truncated: bool,
                check_visible: bool = False) -> Scored:
    if truncated:
        return Scored(0.0, {"reason": "truncated"})
    code = extract_code(text)
    if not code.strip():
        return Scored(0.0, {"reason": "no_code"})
    if top_level_unsafe(code, tests):
        return Scored(0.0, {"reason": "harness_unsafe"})
    bad = code_guard.violations(code)
    if bad:
        return Scored(0.0, {"reason": "guard", "guard": bad})
    checks = split_tests(tests + hidden)
    if not checks:  # None (an unsupported form) or no tests at all
        return Scored(0.0, {"reason": "unsupported_tests"})
    res, ok = _verify(code, checks)
    if all(ok):
        return Scored(1.0, {"reason": "pass"})
    cpu_killed = res.returncode == -signal.SIGXCPU
    reason = "timeout" if res.timed_out or cpu_killed else "oom" if res.over_memory else "fail"
    info = {"reason": reason, "first_fail": ok.index(False)}
    if check_visible:
        info["visible_pass"] = all(ok[: len(tests)])  # visible checks are evaluated first
    return Scored(0.0, info)


K_BAR = 1.54  # mean instructions per IFEval prompt (834 / 541)


def if_max(k: int) -> float:
    return 0.5 + 2 * k / (4 * K_BAR)


# Graded truncation for math (rl1 from step 161; EXPERIMENTS.md "rl1 → Graded math truncation"). The grader reads an
# ANSWER line only if it ends the text, cut off or not (vm.answer_of uses the same pattern); otherwise wrong and cut
# off both score 0, so these values only break that tie and keep the order correct > wrong > cut off. Correct
# answers are never penalized for length.
TRUNC_ANSWERED = 0.75  # cut off right at a correct final ANSWER line: credited, but it ran to the cap
TRUNC_PENALTY = -0.25  # cut off without a correct answer (reasoning or loop)
LOOP_TOKEN_PENALTY = 0.25  # extra, on the tokens of a repetition loop only (applied in train_rl)
LOOP_NGRAM, LOOP_REPEATS = 12, 4


def loop_onset(text: str, n: int = LOOP_NGRAM, k: int = LOOP_REPEATS) -> float | None:
    """Where a repetition loop starts, as a fraction of the words: the first occurrence of the most common n-word
    chunk if it appears >= k times; None if there is none. Run only on cut-off text (finished answers can repeat
    working legitimately)."""
    w = text.split()
    if len(w) < n * k:
        return None
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    top, count = Counter(grams).most_common(1)[0]  # ties: the first-seen chunk
    return grams.index(top) / len(w) if count >= k else None


async def math_reward(text: str, target: str, truncated: bool, graded_trunc: bool = False) -> Scored:
    """info["correct"] is the grader's verdict (last ANSWER line, cut off or not), for accuracy logging and the
    all-fail test; the reward differs from it only for cut-off text when graded_trunc is on."""
    ok = await vm.equivalent(vm.answer_of(text), target)
    if not truncated:
        return Scored(1.0 if ok else 0.0, {"reason": "pass" if ok else "fail", "correct": ok})
    if not graded_trunc:
        return Scored(0.0, {"reason": "truncated", "correct": ok})
    if ok:
        return Scored(TRUNC_ANSWERED, {"reason": "truncated_answered", "correct": True})
    onset = loop_onset(text)
    return Scored(TRUNC_PENALTY, {"reason": "truncated" if onset is None else "truncated_loop", "correct": False,
                                  "loop_onset": onset})


def if_reward(prompt: str, constraints: list[dict], text: str, truncated: bool) -> Scored:
    if truncated:
        return Scored(0.0, {"reason": "truncated", "norm": 0.0})
    gate = gate_reasons(text, constraints)
    if gate:
        return Scored(0.0, {"reason": "gate", "gate": gate, "norm": 0.0})
    s = vi.score(prompt, constraints, text)
    r = 0.25 * (s["prompt_strict"] + s["prompt_loose"]) + (s["inst_strict"] + s["inst_loose"]) / (4 * K_BAR)
    return Scored(r, {"reason": "scored", "score": s, "norm": r / if_max(s["n"])})


async def score(item: dict, text: str, truncated: bool, check_visible: bool = False,
                graded_trunc: bool = False) -> Scored:
    task = item["task"]
    if task == "math":
        return await math_reward(text, item["target"], truncated, graded_trunc)
    if task == "if":
        return await asyncio.to_thread(if_reward, item["prompt"], item["constraints"], text, truncated)
    if task == "code":
        return await asyncio.to_thread(code_reward, text, item["tests"], item.get("hidden_tests", []),
                                       truncated, check_visible)
    raise ValueError(f"unknown task {task!r}")

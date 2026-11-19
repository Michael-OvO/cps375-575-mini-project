"""Rewards for rl1. Each is the graded scorer plus only the guards the spec requires. Truncation is 0 for every
task. Pure functions apart from the code sandbox."""

from dataclasses import dataclass, field

from pipeline.preflight import top_level_unsafe
from pipeline.rl import code_guard, sandbox
from pipeline.verify.code import extract_code


@dataclass
class Scored:
    reward: float
    info: dict = field(default_factory=dict)


def passes(code: str, tests: list[str]) -> bool:
    return sandbox.run_tests(code, tests)[1]


def code_reward(text: str, tests: list[str], hidden: list[str], truncated: bool,
                check_visible: bool = False) -> Scored:
    if truncated:
        return Scored(0.0, {"reason": "truncated"})
    code = extract_code(text)
    if top_level_unsafe(code, tests):
        return Scored(0.0, {"reason": "harness_unsafe"})
    bad = code_guard.violations(code)
    if bad:
        return Scored(0.0, {"reason": "guard", "guard": bad})
    if passes(code, tests + hidden):
        return Scored(1.0, {"reason": "pass"})
    info = {"reason": "fail"}
    if check_visible:
        info["visible_pass"] = passes(code, tests)
    return Scored(0.0, info)

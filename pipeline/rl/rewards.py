"""Rewards for rl1. Each is the graded scorer plus only the guards the spec requires. Truncation is 0 for every
task. Pure functions apart from the code sandbox."""

import secrets
from dataclasses import dataclass, field

from pipeline.preflight import top_level_unsafe
from pipeline.rl import code_guard, sandbox
from pipeline.verify.code import extract_code


@dataclass
class Scored:
    reward: float
    info: dict = field(default_factory=dict)


def harness(code: str, tests: list[str], sentinel: str) -> str:
    return (f"{code}\n\n" + "\n".join(tests) + "\n"
            "import sys as _rl_sys\n"
            "if 'unittest.mock' in _rl_sys.modules:\n    raise SystemExit(3)\n"
            f"print({sentinel!r})\n")


def passes(code: str, tests: list[str]) -> bool:
    sentinel = secrets.token_hex(16)
    res = sandbox.run_program(harness(code, tests, sentinel))
    return (res.returncode == 0 and not res.timed_out and not res.over_memory
            and res.stdout.rstrip().endswith(sentinel))


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

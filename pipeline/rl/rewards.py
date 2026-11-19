"""Rewards for rl1. Each is the graded scorer plus only the guards the spec requires. Truncation is 0 for every
task. Pure functions apart from the code sandbox."""

import signal
from dataclasses import dataclass, field

from pipeline.preflight import top_level_unsafe
from pipeline.rl import code_guard, sandbox
from pipeline.rl.tests_split import Check, check, parse_value, split_tests
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

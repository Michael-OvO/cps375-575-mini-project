"""Code extraction and assert-based execution, matching the MBPP eval scorer."""

import re
import subprocess
import sys
import tempfile

_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEF = re.compile(r"^def .*:\s*$", re.MULTILINE)


def extract_code(completion: str) -> str:
    matches = _FENCE.findall(completion)
    return matches[0] if matches else completion


def run_tests(code: str, tests: list[str], timeout: float = 10.0) -> bool:
    program = code + "\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True,
                timeout=timeout,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def signature_of(code: str) -> str | None:
    match = _DEF.search(code)
    return match.group(0).strip() if match else None

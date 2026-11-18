"""Run one untrusted Python program under macOS sandbox-exec, with no network, no fork and writes only in a
fresh temp dir, plus a CPU rlimit (set inside the child, hard = soft + 1 so it cannot be raised), a wall timeout,
a psutil RSS watchdog and a capped stdout. RLIMIT_AS is not used: it raises ValueError on macOS."""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

WORKERS = 12
_SLOTS = threading.BoundedSemaphore(WORKERS)
_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))
"""


@dataclass
class Result:
    returncode: int | None
    stdout: str
    timed_out: bool = False
    over_memory: bool = False


def _prefix(cpu_s: int) -> str:
    return ("import resource as _rl_r\n"
            f"_rl_r.setrlimit(_rl_r.RLIMIT_CPU, ({cpu_s}, {cpu_s + 1}))\n"
            "_rl_r.setrlimit(_rl_r.RLIMIT_FSIZE, (1 << 24, 1 << 24))\n"
            "del _rl_r\n")


def run_program(program: str, cpu_s: int = 10, wall_s: float = 30, rss_mb: int = 512,
                stdout_cap: int = 65536) -> Result:
    with _SLOTS, tempfile.TemporaryDirectory() as raw_tmp:
        tmp = os.path.realpath(raw_tmp)  # the sandbox matches real paths (/private/var/...)
        profile = Path(tmp) / "profile.sb"
        profile.write_text(_PROFILE.format(tmp=tmp))
        proc = subprocess.Popen(
            ["sandbox-exec", "-f", str(profile), sys.executable, "-I", "-B", "-c", _prefix(cpu_s) + program],
            cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        chunks: list[bytes] = []
        size = 0

        def read() -> None:
            nonlocal size
            while True:
                block = proc.stdout.read(65536)
                if not block:
                    return
                if size < stdout_cap:
                    keep = block[: stdout_cap - size]
                    chunks.append(keep)
                    size += len(keep)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        watched = psutil.Process(proc.pid)
        deadline = time.monotonic() + wall_s
        timed_out = over_memory = False
        while proc.poll() is None:
            if time.monotonic() > deadline:
                timed_out = True
                break
            try:
                if watched.memory_info().rss > rss_mb << 20:
                    over_memory = True
                    break
            except psutil.Error:
                pass
            time.sleep(0.02)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()
        reader.join(timeout=5)
        return Result(proc.returncode, b"".join(chunks).decode("utf-8", "replace"), timed_out, over_memory)

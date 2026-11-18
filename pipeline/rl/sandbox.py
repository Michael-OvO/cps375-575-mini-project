"""Run one untrusted Python program under macOS sandbox-exec (deny-default profile): no network, no fork, no
exec beyond the interpreter, no Mach/XPC lookups, reads only of the Python install and system paths, writes only
in a fresh temp dir. Also a minimal env, a CPU rlimit (set inside the child, hard = soft + 1 so it cannot be
raised), RLIMIT_FSIZE/NOFILE, a wall timeout, a psutil RSS watchdog, a temp-dir disk cap and a capped stdout.
RLIMIT_AS is not used: it raises ValueError on macOS. The program is delivered as a file, not on the command line."""

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
DISK_CAP = 64 << 20
_SLOTS = threading.BoundedSemaphore(WORKERS)
_READ_PATHS = ("/usr", "/System", "/Library", "/private/etc", "/private/var/db/timezone", "/dev", "/etc",
               "/var/db/timezone", "/private/var/db/dyld", "/private/var/db/datadetectors", "/Applications")
_PROFILE = """(version 1)
(deny default)
(allow process-exec (literal "{exe}"))
(allow sysctl-read)
(allow signal (target self))
(allow file-read-metadata)
(allow file-read* {reads})
(allow file-read* {ancestors})
(allow file-write* (subpath "{tmp}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))
"""
_LAUNCHER = """import resource as _r
_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu1}))
_r.setrlimit(_r.RLIMIT_FSIZE, (1 << 24, 1 << 24))
_r.setrlimit(_r.RLIMIT_NOFILE, (64, 64))
del _r
_f = open('solution.py', 'rb')
_src = _f.read().decode('utf-8', 'replace')
_f.close()
_g = {{'__name__': '__main__', '__file__': 'solution.py', '__builtins__': __builtins__}}
exec(compile(_src, 'solution.py', 'exec'), _g)
"""


@dataclass
class Result:
    returncode: int | None
    stdout: str
    timed_out: bool = False
    over_memory: bool = False


def _subpaths(paths) -> str:
    return " ".join(f'(subpath "{p}")' for p in dict.fromkeys(paths))


def _read_paths(tmp: str) -> list[str]:
    paths = [tmp, sys.prefix, sys.base_prefix, os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix),
             os.path.dirname(os.path.realpath(sys.executable))]
    paths += [os.path.realpath(p) for p in _READ_PATHS if os.path.exists(p)] + list(_READ_PATHS)
    return paths


def _ancestors(paths) -> str:
    """Literal (non-recursive) read of each parent directory: Python needs it to resolve its own install path."""
    out = {"/"}
    for p in paths:
        out.update(str(a) for a in Path(p).parents)
    return " ".join(f'(literal "{a}")' for a in sorted(out))


def _dir_size(path: str) -> int:
    total = 0
    for entry in os.scandir(path):
        try:
            total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            pass
    return total


def run_program(program: str, cpu_s: int = 10, wall_s: float = 30, rss_mb: int = 512,
                stdout_cap: int = 65536) -> Result:
    with _SLOTS, tempfile.TemporaryDirectory() as raw_tmp:
        tmp = os.path.realpath(raw_tmp)  # the sandbox matches real paths (/private/var/...)
        exe = os.path.realpath(sys.executable)
        reads = _read_paths(tmp)
        profile = Path(tmp) / "profile.sb"
        profile.write_text(_PROFILE.format(tmp=tmp, exe=exe, reads=_subpaths(reads), ancestors=_ancestors(reads)))
        try:
            (Path(tmp) / "solution.py").write_bytes(program.encode("utf-8", "surrogatepass"))
        except (UnicodeEncodeError, OSError):
            return Result(1, "")
        proc = subprocess.Popen(
            ["sandbox-exec", "-f", str(profile), sys.executable, "-I", "-B", "-c",
             _LAUNCHER.format(cpu=cpu_s, cpu1=cpu_s + 1)],
            cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8"}, start_new_session=True,
        )
        try:
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
                    if watched.memory_info().rss > rss_mb << 20 or _dir_size(tmp) > DISK_CAP:
                        over_memory = True
                        break
                except (psutil.Error, OSError):
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
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

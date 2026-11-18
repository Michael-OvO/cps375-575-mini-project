import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from pipeline.rl.sandbox import run_program


def test_ok_program_prints():
    r = run_program("print('hi')")
    assert r.returncode == 0 and r.stdout.strip() == "hi" and not r.timed_out


def test_failing_assert_nonzero():
    assert run_program("assert 1 == 2").returncode != 0


def test_cpu_limit_kills_busy_loop():
    t = time.monotonic()
    r = run_program("while True: pass", cpu_s=1, wall_s=10)
    assert r.returncode != 0 and not r.timed_out and time.monotonic() - t < 6


def test_wall_timeout_kills_sleeper():
    r = run_program("import time; time.sleep(100)", wall_s=2)
    assert r.timed_out and r.returncode != 0


def test_memory_bomb_killed():
    r = run_program("x = b'x' * (2 * 1024 ** 3)\nimport time; time.sleep(5)", rss_mb=512)
    assert r.over_memory and r.returncode != 0


def test_fork_denied():
    r = run_program("import os\nos.fork()\nprint('forked')")
    assert "forked" not in r.stdout and r.returncode != 0


def test_network_denied():
    r = run_program("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)\nprint('connected')")
    assert "connected" not in r.stdout and r.returncode != 0


def test_write_outside_tmp_denied():
    target = Path.home() / f".rl_sandbox_probe_{uuid.uuid4().hex}"
    r = run_program(f"open({str(target)!r}, 'w').write('x')\nprint('wrote')")
    assert "wrote" not in r.stdout and not target.exists()


def test_write_inside_tmp_allowed():
    assert run_program("open('scratch.txt', 'w').write('x')\nprint('ok')").stdout.strip() == "ok"


def test_stdout_capped():
    r = run_program("print('x' * 200000)")
    assert len(r.stdout) <= 65536


def test_parallel_runs_are_isolated():
    outs = []

    def go(i):
        outs.append(run_program(f"print({i})").stdout.strip())

    ts = [threading.Thread(target=go, args=(i,)) for i in range(30)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(outs, key=int) == [str(i) for i in range(30)]


def test_exec_of_other_binaries_denied_defaults_write():
    domain = f"com.rlprobe.{uuid.uuid4().hex}"
    try:
        r = run_program(f"import os\nos.execv('/usr/bin/defaults', ['defaults', 'write', {domain!r}, 'x', '1'])")
        assert r.returncode != 0
        read = subprocess.run(["defaults", "read", domain, "x"], capture_output=True)
        assert read.returncode != 0
    finally:
        subprocess.run(["defaults", "delete", domain], capture_output=True)


def test_clipboard_not_readable():
    old = subprocess.run(["pbpaste"], capture_output=True).stdout
    secret = f"rl-secret-{uuid.uuid4().hex}"
    subprocess.run(["pbcopy"], input=secret.encode(), check=True)
    try:
        r = run_program("import os\nos.execv('/usr/bin/pbpaste', ['pbpaste'])")
        assert secret not in r.stdout and r.returncode != 0
    finally:
        subprocess.run(["pbcopy"], input=old, check=True)


def test_home_and_repo_unreadable():
    repo_file = Path(__file__).resolve()  # a repo file outside the Python install
    r = run_program(f"print(open({str(repo_file)!r}).read(20))")
    assert r.returncode != 0 and r.stdout == ""


def test_parent_env_not_visible():
    os.environ["RL_SANDBOX_SECRET"] = "leak"
    try:
        r = run_program("import os\nprint(sorted(os.environ))")
    finally:
        del os.environ["RL_SANDBOX_SECRET"]
    assert "RL_SANDBOX_SECRET" not in r.stdout and r.returncode == 0


def test_future_import_works():
    r = run_program("from __future__ import annotations\nprint('ok')")
    assert r.returncode == 0 and r.stdout.strip() == "ok"


def test_main_guard_and_name():
    r = run_program("print(__name__)")
    assert r.stdout.strip() == "__main__"


def test_nul_byte_program_fails_cleanly():
    r = run_program("print('a')\x00print('b')")
    assert r.returncode != 0


def test_two_mb_program_returns():
    r = run_program("x = 1\n" * 400000)
    assert isinstance(r.returncode, int) or r.returncode is None


def test_disk_fill_killed():
    prog = ("import time\nfor i in range(40):\n    open(f'f{i}', 'wb').write(b'x' * (4 << 20))\n"
            "    time.sleep(0.05)\nprint('done')")
    r = run_program(prog)
    assert "done" not in r.stdout and r.returncode != 0 and r.over_memory


def test_disk_fill_in_subdirectory_killed():
    prog = ("import os, time\nos.mkdir('d')\nfor i in range(40):\n"
            "    open(f'd/f{i}', 'wb').write(b'x' * (4 << 20))\n    time.sleep(0.05)\nprint('done')")
    r = run_program(prog)
    assert "done" not in r.stdout and r.returncode != 0 and r.over_memory

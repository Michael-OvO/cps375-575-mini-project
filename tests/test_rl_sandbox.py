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
    assert r.returncode != 0 and time.monotonic() - t < 6


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

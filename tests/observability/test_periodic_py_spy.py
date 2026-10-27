"""Periodic py-spy dump: opt-in, interval-triggered, failure-contained (CPU-only, no py-spy binary needed)."""

import threading
import time

import pytest

from vime.observability import periodic_py_spy as ps


@pytest.fixture(autouse=True)
def _reset_started(monkeypatch):
    monkeypatch.setattr(ps, "_started", False)
    monkeypatch.delenv(ps.DUMP_INTERVAL_ENV, raising=False)


@pytest.fixture
def park_loops():
    """Clamp sleeps and neutralize leaked daemon dump-loop threads after the test.

    Tests using this fixture must assign ``ps._dump_all_processes`` directly (NOT via
    monkeypatch), so no later teardown can restore the real dumper; this fixture's final
    assignment of a SystemExit-raising sentinel then permanently terminates leaked threads
    (the loop's ``except Exception`` does not catch SystemExit).
    """
    stop = threading.Event()
    real_sleep = ps.time.sleep

    def fake_sleep(seconds):
        if stop.is_set() and threading.current_thread().daemon:
            raise SystemExit
        real_sleep(min(seconds, 0.05))

    ps.time.sleep = fake_sleep
    yield
    stop.set()

    def _kill():
        raise SystemExit

    ps._dump_all_processes = _kill
    ps.time.sleep = real_sleep


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(threading.Thread, "__init__", lambda *a, **k: pytest.fail("must not start a thread"))
    ps.maybe_start_periodic_pyspy_dump()
    monkeypatch.setenv(ps.DUMP_INTERVAL_ENV, "0")
    ps.maybe_start_periodic_pyspy_dump()


def test_missing_pyspy_binary(monkeypatch, caplog):
    monkeypatch.setenv(ps.DUMP_INTERVAL_ENV, "10")
    monkeypatch.setattr(ps.shutil, "which", lambda name: None)
    monkeypatch.setattr(threading.Thread, "__init__", lambda *a, **k: pytest.fail("must not start a thread"))
    ps.maybe_start_periodic_pyspy_dump()
    assert "py-spy not found" in caplog.text


def test_interval_triggering_fires(park_loops, monkeypatch):
    fires = []
    monkeypatch.setenv(ps.DUMP_INTERVAL_ENV, "0.05")
    monkeypatch.setattr(ps.shutil, "which", lambda name: "/usr/bin/py-spy")
    ps._dump_all_processes = lambda: fires.append(time.time())  # direct: park_loops owns teardown
    ps.maybe_start_periodic_pyspy_dump()
    deadline = time.time() + 2.0
    while len(fires) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert len(fires) >= 2


def test_starts_only_once(park_loops, monkeypatch):
    started = []
    real_thread = threading.Thread
    monkeypatch.setenv(ps.DUMP_INTERVAL_ENV, "100")
    monkeypatch.setattr(ps.shutil, "which", lambda name: "/usr/bin/py-spy")
    ps._dump_all_processes = lambda: None  # direct: park_loops owns teardown

    class _CountingThread(real_thread):
        def __init__(self, *a, **k):
            started.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(ps.threading, "Thread", _CountingThread)
    ps.maybe_start_periodic_pyspy_dump()
    ps.maybe_start_periodic_pyspy_dump()
    assert len(started) == 1


def test_process_keyword_filter(monkeypatch):
    class _Proc:
        def __init__(self, pid, cmdline):
            self.info = {"pid": pid, "cmdline": cmdline}

    procs = [
        _Proc(1, ["python", "train.py"]),
        _Proc(2, ["ray::RolloutManager"]),
        _Proc(3, ["vllm::EngineCore"]),
        _Proc(4, ["bash", "-c", "sleep 1"]),
    ]
    monkeypatch.setattr(ps.psutil, "process_iter", lambda *a, **k: procs)
    targets = ps._collect_target_processes()
    assert [pid for pid, _ in targets] == [1, 2, 3]


def test_dump_iteration_failure_is_contained(park_loops, monkeypatch):
    calls = []

    def boom():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first tick fails")

    ps._dump_all_processes = boom  # direct: park_loops owns teardown
    thread = threading.Thread(target=ps._dump_loop, args=(0.02,), daemon=True)
    thread.start()
    deadline = time.time() + 2.0
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert len(calls) >= 3

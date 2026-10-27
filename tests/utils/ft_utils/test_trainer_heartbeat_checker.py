"""TrainerHeartbeatChecker: driver-side staleness detection over actor get_heartbeat_status."""

import logging
import time
from types import SimpleNamespace

import pytest

from vime.utils.ft_utils import trainer_heartbeat_checker as thc
from vime.utils.ft_utils.heartbeat_utils import HeartbeatStatus
from vime.utils.ft_utils.trainer_heartbeat_checker import TrainerHeartbeatChecker


class _FakeActor:
    """Mimics a ray actor handle: get_heartbeat_status.remote() returns a 'handle' that the
    (monkeypatched) ray.get resolves to a HeartbeatStatus."""

    def __init__(self):
        self.status = HeartbeatStatus(last_active_timestamp=time.time(), bump_count=0)
        self.fail = False

    @property
    def get_heartbeat_status(self):
        actor = self

        class _Call:
            def remote(self):
                return actor

        return _Call()


def _fake_ray_get(monkeypatch):
    def fake_get(handle, timeout=None):
        if handle.fail:
            raise RuntimeError("actor unreachable")
        return handle.status

    monkeypatch.setattr(thc.ray, "get", fake_get)


def _args(interval=None, timeout=600.0, ci_test=False):
    return SimpleNamespace(
        trainer_heartbeat_checker_interval=interval,
        trainer_heartbeat_checker_timeout=timeout,
        ci_test=ci_test,
    )


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(
        thc.threading.Thread, "start", lambda *a, **k: pytest.fail("must not start a thread")
    )
    checker = TrainerHeartbeatChecker(_args(interval=None), lambda: [_FakeActor()])
    assert checker.start() is False
    assert thc.maybe_start_trainer_heartbeat_checker(_args(interval=None), SimpleNamespace(_actor_handlers=[])) is None


def test_fresh_heartbeat_is_healthy(monkeypatch, caplog):
    _fake_ray_get(monkeypatch)
    actor = _FakeActor()
    checker = TrainerHeartbeatChecker(_args(), lambda: [actor])
    with caplog.at_level(logging.ERROR):
        checker._check_once()
        checker._check_once()
    assert not caplog.records


def test_stale_actor_logs_error(monkeypatch, caplog):
    _fake_ray_get(monkeypatch)
    actor = _FakeActor()
    checker = TrainerHeartbeatChecker(_args(timeout=10.0), lambda: [actor])
    checker._check_once()  # establishes baseline bump_count
    actor.status = HeartbeatStatus(last_active_timestamp=time.time() - 30.0, bump_count=0)
    with caplog.at_level(logging.ERROR):
        checker._check_once()
    assert any("heartbeat stale" in r.message and "no training progress" in r.message for r in caplog.records)


def test_progress_resets_staleness(monkeypatch, caplog):
    _fake_ray_get(monkeypatch)
    actor = _FakeActor()
    checker = TrainerHeartbeatChecker(_args(timeout=10.0), lambda: [actor])
    checker._check_once()
    actor.status = HeartbeatStatus(last_active_timestamp=time.time() - 30.0, bump_count=1)  # bumped
    with caplog.at_level(logging.ERROR):
        checker._check_once()
    assert not caplog.records


def test_rpc_failure_is_stale(monkeypatch, caplog):
    _fake_ray_get(monkeypatch)
    actor = _FakeActor()
    actor.fail = True
    checker = TrainerHeartbeatChecker(_args(), lambda: [actor])
    with caplog.at_level(logging.ERROR):
        checker._check_once()
    assert any("heartbeat RPC failed" in r.message for r in caplog.records)


def test_ci_test_raises_on_stale(monkeypatch):
    _fake_ray_get(monkeypatch)
    actor = _FakeActor()
    checker = TrainerHeartbeatChecker(_args(timeout=10.0, ci_test=True), lambda: [actor])
    checker._check_once()
    actor.status = HeartbeatStatus(last_active_timestamp=time.time() - 30.0, bump_count=0)
    with pytest.raises(RuntimeError, match="heartbeat stale"):
        checker._check_once()


def test_argparse_flags_registered():
    pytest.importorskip("vllm")
    import argparse

    from vime.utils.arguments import get_vime_extra_args_provider

    parser = argparse.ArgumentParser()
    get_vime_extra_args_provider()(parser)
    args, _ = parser.parse_known_args([])
    assert args.trainer_heartbeat_checker_interval is None
    assert args.trainer_heartbeat_checker_timeout == 600.0
    assert args.rollout_health_check_failure_threshold == 1  # same fault-tolerance group

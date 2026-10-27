"""RolloutHealthMonitor consecutive-failure debounce: a single failed probe must not retire a
live engine; only N consecutive failures (threshold) trigger retire_engine."""

from types import SimpleNamespace

import pytest

from vime.utils import health_monitor
from vime.utils.health_monitor import RolloutHealthMonitor


class _FakeEngine:
    def __init__(self, healthy: bool):
        self.healthy = healthy

    @property
    def health_generate(self):
        engine = self

        class _Call:
            def remote(self, timeout=None):
                return ("ok" if engine.healthy else "stuck", id(engine))

        return _Call()


class _FakeServerGroup:
    def __init__(self, engines):
        self.engines = engines
        self.all_engines = [e for e in engines if e is not None]
        self.retired = []

    def retire_engine(self, rollout_engine_id, timeout=None):
        self.retired.append((rollout_engine_id, timeout))


def _make_monitor(outcomes, threshold=3, timeout=7.0):
    """outcomes: list of per-check health booleans for a single engine."""
    engine = _FakeEngine(healthy=True)
    group = _FakeServerGroup([engine])
    args = SimpleNamespace(
        rollout_health_check_interval=0.01,
        rollout_health_check_timeout=timeout,
        rollout_health_check_first_wait=0.0,
        rollout_health_check_failure_threshold=threshold,
    )
    monitor = RolloutHealthMonitor(group, args)

    state = {"step": 0}
    monkeypatchable = {"outcomes": outcomes}

    def fake_wait(handles, num_returns, timeout):
        return handles, []

    def fake_get(handle, timeout=None):
        ok, _ = handle
        step = min(state["step"], len(monkeypatchable["outcomes"]) - 1)
        if not monkeypatchable["outcomes"][step]:
            raise RuntimeError("engine wedged")
        return None

    return monitor, group, state, monkeypatchable, fake_wait, fake_get


@pytest.fixture
def fake_ray(monkeypatch):
    monkeypatch.setattr(health_monitor.ray, "wait", lambda handles, num_returns, timeout: (handles, []))
    return monkeypatch


def _run_checks(monitor, state, monkeypatch, outcomes, fake_get, n):
    monkeypatch.setattr(health_monitor.ray, "get", fake_get)
    for i in range(n):
        state["step"] = i
        monitor._run_health_checks(force=True)


def test_two_failures_below_threshold_do_not_retire(fake_ray, monkeypatch):
    monitor, group, state, mp, _, fake_get = _make_monitor([False, False, False], threshold=3)
    _run_checks(monitor, state, monkeypatch, None, fake_get, 2)
    assert group.retired == []


def test_threshold_consecutive_failures_retire_once(fake_ray, monkeypatch):
    monitor, group, state, mp, _, fake_get = _make_monitor([False, False, False, False], threshold=3)
    _run_checks(monitor, state, monkeypatch, None, fake_get, 4)
    assert group.retired == [(0, 7.0)]


def test_success_resets_the_counter(fake_ray, monkeypatch):
    monitor, group, state, mp, _, fake_get = _make_monitor([False, False, True, False, False], threshold=3)
    _run_checks(monitor, state, monkeypatch, None, fake_get, 5)
    assert group.retired == []


def test_resume_clears_failures(fake_ray, monkeypatch):
    monitor, group, state, mp, _, fake_get = _make_monitor([False, False, False, False], threshold=3)
    _run_checks(monitor, state, monkeypatch, None, fake_get, 2)
    monitor.resume()
    _run_checks(monitor, state, monkeypatch, None, fake_get, 2)
    assert group.retired == []


def test_check_once_counts_toward_same_counter(fake_ray, monkeypatch):
    monitor, group, state, mp, _, fake_get = _make_monitor([False, False, False], threshold=3)
    monkeypatch.setattr(health_monitor.ray, "get", fake_get)
    monitor.check_once()
    monitor.check_once()
    assert group.retired == []
    monitor.check_once()
    assert group.retired == [(0, 7.0)]


def _parse_vime_args(argv):
    pytest.importorskip("vllm")  # the vime args provider imports vllm engine arg utils
    import argparse

    from vime.utils.arguments import get_vime_extra_args_provider

    parser = argparse.ArgumentParser()
    get_vime_extra_args_provider()(parser)
    args, _ = parser.parse_known_args(argv)
    return args


def test_argparse_default_is_one():
    assert _parse_vime_args([]).rollout_health_check_failure_threshold == 1


def test_argparse_explicit_value():
    assert _parse_vime_args(["--rollout-health-check-failure-threshold", "5"]).rollout_health_check_failure_threshold == 5

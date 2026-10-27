"""Eval-fleet pin/version contract tests (CPU; fake engines, no Ray)."""

from pathlib import Path

import pytest

from vime.ray.eval_fleet import (
    HF_EXPORT_MARKER,
    EvalSkip,
    check_eval_fleet_version,
    get_eval_fleet_info,
    pin_eval_fleet,
)


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return lambda: self.fn(*args, **kwargs)


def _gather(thunks):
    return [t() for t in thunks]


class _FakeEngine:
    def __init__(self, name, fail_load=False, fail_health=False):
        self.name = name
        self.version = None
        self.loads = []
        self.fail_load = fail_load
        self.fail_health = fail_health
        self.update_weights_from_disk = _RemoteMethod(self._load)
        self.get_weight_version = _RemoteMethod(lambda: self.version)
        self.health_generate = _RemoteMethod(self._health)

    def _load(self, model_path, weight_version=None, flush_cache=True):
        if self.fail_load:
            raise RuntimeError(f"{self.name} load boom")
        self.loads.append(model_path)
        self.version = str(weight_version)

    def _health(self):
        if self.fail_health:
            raise RuntimeError(f"{self.name} health boom")
        return True


def _write_marker(d: Path):
    (d / HF_EXPORT_MARKER).write_text("rollout_id=3\n")


def test_pin_success(tmp_path):
    _write_marker(tmp_path)
    engines = [_FakeEngine("e0"), _FakeEngine("e1")]
    pin = pin_eval_fleet(engines, str(tmp_path), "3", require_marker=True, gather=_gather)
    assert pin.skip_reason is None
    assert [e.loads for e in engines] == [[str(tmp_path)], [str(tmp_path)]]
    assert [e.version for e in engines] == ["3", "3"]


def test_pin_requires_marker(tmp_path):
    engines = [_FakeEngine("e0")]
    pin = pin_eval_fleet(engines, str(tmp_path), "3", require_marker=True, gather=_gather)
    assert pin.skip_reason == "export_incomplete"
    assert engines[0].loads == []


def test_pin_without_marker_requirement_loads_anyway(tmp_path):
    engines = [_FakeEngine("e0")]
    pin = pin_eval_fleet(engines, str(tmp_path), "3", require_marker=False, gather=_gather)
    assert pin.skip_reason is None


def test_pin_load_failure_is_unhealthy(tmp_path):
    _write_marker(tmp_path)
    engines = [_FakeEngine("e0", fail_load=True), _FakeEngine("e1")]
    pin = pin_eval_fleet(engines, str(tmp_path), "3", require_marker=True, gather=_gather)
    assert pin.skip_reason == "unhealthy"


def test_pin_detects_stale_engine(tmp_path):
    _write_marker(tmp_path)
    engines = [_FakeEngine("e0"), _FakeEngine("e1")]
    # e1 lies about its version after load.
    engines[1].get_weight_version = _RemoteMethod(lambda: "2")
    pin = pin_eval_fleet(engines, str(tmp_path), "3", require_marker=True, gather=_gather)
    assert pin.skip_reason == "pin_violation"


def test_pin_health_probe_failure(tmp_path):
    _write_marker(tmp_path)
    engines = [_FakeEngine("e0", fail_health=True)]
    pin = pin_eval_fleet(engines, str(tmp_path), "3", require_marker=True, gather=_gather)
    assert pin.skip_reason == "unhealthy"


def test_check_version_without_reload(tmp_path):
    engines = [_FakeEngine("e0"), _FakeEngine("e1")]
    for e in engines:
        e.version = "7"
    assert check_eval_fleet_version(engines, "7", gather=_gather).skip_reason is None
    engines[1].version = "6"
    assert check_eval_fleet_version(engines, "7", gather=_gather).skip_reason == "pin_violation"
    assert all(e.loads == [] for e in engines)  # no reload happened


def test_get_eval_fleet_info():
    class _Server:
        router_ip, router_port = "10.0.0.1", 3100
        engine_gpu_counts = [1, 1]

    info = get_eval_fleet_info({"default": object(), "eval": _Server()})
    assert info.router == ("10.0.0.1", 3100)
    assert info.engine_gpu_counts == [1, 1]
    assert get_eval_fleet_info({"default": object()}) is None
    assert get_eval_fleet_info(None) is None


def test_eval_skip_carries_reason():
    assert EvalSkip("busy").reason == "busy"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

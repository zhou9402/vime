"""CPU unit tests for the weight-version staleness guard (F02-5)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import pytest

from vime.utils.weight_version import (
    assert_engine_weight_versions_sane,
    assert_samples_weight_version_sane,
    assert_weight_version_is_published,
    max_rollouts_without_published_weight_version,
)

NUM_GPUS = 0


def _args(**overrides) -> SimpleNamespace:
    base = dict(debug_rollout_only=False, debug_train_only=False, debug_skip_weight_update=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _sample(index: int, versions: list[str]) -> SimpleNamespace:
    return SimpleNamespace(index=index, weight_versions=versions)


def test_publish_guard_raises_after_threshold_without_update():
    args = _args()
    threshold = max_rollouts_without_published_weight_version(args)
    assert threshold >= 3
    assert_weight_version_is_published(args, rollouts_since_publish=threshold)  # boundary ok
    with pytest.raises(AssertionError, match="without anyone publishing"):
        assert_weight_version_is_published(args, rollouts_since_publish=threshold + 1)


def test_publish_guard_threshold_honors_update_weights_interval():
    assert max_rollouts_without_published_weight_version(_args(update_weights_interval=10)) == 11
    assert max_rollouts_without_published_weight_version(_args()) == 3


def test_guards_skip_debug_modes():
    args = _args(debug_rollout_only=True)
    assert_weight_version_is_published(args, rollouts_since_publish=10**9)
    assert_engine_weight_versions_sane(args, ["default"])
    assert_samples_weight_version_sane(args, [_sample(0, ["default"])])
    args = _args(debug_skip_weight_update=True)
    assert_samples_weight_version_sane(args, [_sample(0, ["not-a-version"])])


def test_engine_versions_reject_placeholder_and_non_numeric():
    with pytest.raises(AssertionError, match="never updated"):
        assert_engine_weight_versions_sane(_args(), ["3", "default"])
    with pytest.raises(AssertionError, match="not the numeric version"):
        assert_engine_weight_versions_sane(_args(), ["abc"])
    assert_engine_weight_versions_sane(_args(), ["0", "12"])  # no raise


def test_samples_reject_placeholder_and_non_numeric_versions():
    with pytest.raises(AssertionError, match="never updated"):
        assert_samples_weight_version_sane(_args(), [_sample(7, ["1", "default"])])
    with pytest.raises(AssertionError, match="somewhere vime does not"):
        assert_samples_weight_version_sane(_args(), [_sample(3, ["v2"])])


def test_samples_pass_numeric_versions_and_nested_groups():
    samples = [[_sample(0, ["1", "2"]), _sample(1, ["2"])], [_sample(2, [])]]
    assert_samples_weight_version_sane(_args(), samples)  # no raise
    assert_samples_weight_version_sane(_args(), [])  # no raise

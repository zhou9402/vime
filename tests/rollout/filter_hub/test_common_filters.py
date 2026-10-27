"""Group-level built-in filters: aborted, missing-reward, and preput composition."""

from argparse import Namespace

import pytest

from vime.rollout.filter_hub.base_types import DynamicFilterOutput
from vime.rollout.filter_hub.common_filters import (
    apply_aborted_filter,
    apply_missing_reward_filter,
    apply_preput_filters,
)
from vime.utils.types import Sample


def _args(reward_key=None):
    return Namespace(reward_key=reward_key)


def _sample(status=Sample.Status.COMPLETED, reward=1.0):
    return Sample(status=status, reward=reward)


def test_aborted_filter_rejects_group_with_one_aborted():
    group = [_sample(), _sample(status=Sample.Status.ABORTED)]
    out = apply_aborted_filter(_args(), group)
    assert out.keep is False and out.reason == "group_has_aborted"


def test_aborted_filter_keeps_all_completed():
    out = apply_aborted_filter(_args(), [_sample(), _sample()])
    assert out.keep is True


def test_filters_visit_nested_groups():
    group = [[_sample()], [_sample(status=Sample.Status.ABORTED)]]
    assert apply_aborted_filter(_args(), group).keep is False


@pytest.mark.parametrize("reward", [None, {"score": None}])
def test_missing_reward_filter(reward):
    good = _sample(reward={"score": 1.0}) if isinstance(reward, dict) else _sample()
    group = [good, _sample(reward=reward)]
    args = _args(reward_key="score") if isinstance(reward, dict) else _args()
    out = apply_missing_reward_filter(args, group)
    assert out.keep is False and out.reason == "group_has_missing_reward"


def test_preput_ordering_aborted_before_missing_reward():
    group = [_sample(status=Sample.Status.ABORTED, reward=None)]
    out = apply_preput_filters(_args(), None, group)
    assert out.reason == "group_has_aborted"


def test_preput_missing_reward_before_dynamic():
    called = []

    def dynamic(args, samples, **kwargs):
        called.append(1)
        return DynamicFilterOutput(keep=True)

    out = apply_preput_filters(_args(), dynamic, [_sample(reward=None)])
    assert out.reason == "group_has_missing_reward"
    assert called == []


def test_preput_calls_dynamic_when_clean():
    def dynamic(args, samples, **kwargs):
        return DynamicFilterOutput(keep=False, reason="custom")

    out = apply_preput_filters(_args(), dynamic, [_sample()])
    assert out.keep is False and out.reason == "custom"


def test_preput_legacy_bool_normalized():
    out = apply_preput_filters(_args(), lambda *a, **k: False, [_sample()])
    assert isinstance(out, DynamicFilterOutput)
    assert out.keep is False


def test_preput_kwargs_forwarded():
    seen = {}

    def dynamic(args, samples, **kwargs):
        seen.update(kwargs)
        return True

    apply_preput_filters(_args(), dynamic, [_sample()], extra_key="extra_value")
    assert seen == {"extra_key": "extra_value"}


def test_preput_dynamic_exception_propagates():
    def dynamic(args, samples, **kwargs):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        apply_preput_filters(_args(), dynamic, [_sample()])


def test_preput_wrapper_end_to_end_via_path_loader():
    """--dynamic-sampling-filter-path vime.rollout.filter_hub.common_filters.preput_then_reward_nonzero_std"""
    from vime.utils.misc import load_function

    fn = load_function("vime.rollout.filter_hub.common_filters.preput_then_reward_nonzero_std")

    # aborted group rejected before the dynamic filter
    aborted = fn(_args(), [_sample(), _sample(status=Sample.Status.ABORTED)])
    assert aborted.keep is False and aborted.reason == "group_has_aborted"

    # missing reward rejected
    missing = fn(_args(), [_sample(reward=None)])
    assert missing.keep is False and missing.reason == "group_has_missing_reward"

    # clean group with zero reward std rejected by the dynamic filter
    zero_std = fn(_args(), [_sample(reward=1.0), _sample(reward=1.0)])
    assert bool(zero_std.keep) is False and zero_std.reason.startswith("zero_std_")

    # clean group with reward variance kept
    varied = fn(_args(), [_sample(reward=0.0), _sample(reward=1.0)])
    assert bool(varied.keep) is True

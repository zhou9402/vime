"""Group-level built-in filters composed ahead of the dynamic filter.

Ported from miles.rollout.filter_hub.common_filters (subset: aborted / missing-reward /
preput composition). The reward-nonzero-std filter is not ported: vime already ships it as
``vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std``.
"""

from argparse import Namespace

from vime.rollout.base_types import iter_samples
from vime.rollout.filter_hub.base_types import DynamicFilterOutput, call_dynamic_filter
from vime.utils.types import Sample

Group = list[Sample | list[Sample]]


def apply_preput_filters(args: Namespace, dynamic_filter, samples: Group, **kwargs) -> DynamicFilterOutput:
    output = apply_aborted_filter(args, samples, **kwargs)
    if not output.keep:
        return output

    output = apply_missing_reward_filter(args, samples, **kwargs)
    if not output.keep:
        return output

    return call_dynamic_filter(dynamic_filter, args, samples, **kwargs)


def apply_aborted_filter(args: Namespace, samples: Group, **kwargs) -> DynamicFilterOutput:
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(sample.status == Sample.Status.ABORTED for sample in iter_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def apply_missing_reward_filter(args: Namespace, samples: Group, **kwargs) -> DynamicFilterOutput:
    if any(sample.reward is None or sample.get_reward_value(args) is None for sample in iter_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_missing_reward")
    return DynamicFilterOutput(keep=True)


def preput_then_reward_nonzero_std(args: Namespace, samples: Group, **kwargs) -> DynamicFilterOutput:
    """Ready-to-use ``--dynamic-sampling-filter-path`` entry: rejected-aborted and
    missing-reward groups are dropped before the reward-nonzero-std dynamic filter runs.

    Use as ``--dynamic-sampling-filter-path vime.rollout.filter_hub.common_filters.preput_then_reward_nonzero_std``.
    """
    from vime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std

    return apply_preput_filters(args, check_reward_nonzero_std, samples, **kwargs)

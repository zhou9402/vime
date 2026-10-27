"""mask_truncated_zero_reward (DeepSWE compact filtering): cut-off zero-reward samples leave the loss."""

from argparse import Namespace

import pytest

from vime.rollout.filter_hub import sample_filters
from vime.rollout.filter_hub.sample_filters import mask_truncated_zero_reward
from vime.utils.misc import load_function
from vime.utils.types import Sample


@pytest.fixture(autouse=True)
def _reset_warn():
    sample_filters._warn_dict_reward_once.cache_clear()


def _args(reward_key=None):
    return Namespace(reward_key=reward_key)


def _sample(status=Sample.Status.COMPLETED, reward=0.0, exit_status=None):
    s = Sample(status=status, reward=reward)
    if exit_status is not None:
        s.metadata = {"exit_status": exit_status}
    return s


def test_truncated_zero_reward_masked():
    s = _sample(status=Sample.Status.TRUNCATED, reward=0.0)
    mask_truncated_zero_reward(_args(), [[s]])
    assert s.remove_sample is True


@pytest.mark.parametrize("exit_status", ["TimeLimitExceeded", "SequenceLengthLimitExceeded"])
@pytest.mark.parametrize("reward", [0.0, None])
def test_cut_off_exit_status_masked(exit_status, reward):
    s = _sample(reward=reward, exit_status=exit_status)
    mask_truncated_zero_reward(_args(), [[s]])
    assert s.remove_sample is True


def test_truncated_positive_reward_kept():
    s = _sample(status=Sample.Status.TRUNCATED, reward=1.0)
    mask_truncated_zero_reward(_args(), [[s]])
    assert s.remove_sample is False


def test_completed_zero_reward_kept():
    s = _sample(status=Sample.Status.COMPLETED, reward=0.0)
    mask_truncated_zero_reward(_args(), [[s]])
    assert s.remove_sample is False


def test_agent_error_not_masked():
    s = _sample(reward=0.0, exit_status="AgentError")
    mask_truncated_zero_reward(_args(), [[s]])
    assert s.remove_sample is False


def test_dict_reward_without_key_never_masked_and_warns_once(caplog):
    s1 = _sample(status=Sample.Status.TRUNCATED, reward={"score": 0.0})
    s2 = _sample(status=Sample.Status.TRUNCATED, reward={"score": 0.0})
    import logging

    with caplog.at_level(logging.WARNING):
        mask_truncated_zero_reward(_args(), [[s1]])
        mask_truncated_zero_reward(_args(), [[s2]])
    assert s1.remove_sample is False and s2.remove_sample is False
    assert sum("never masked" in r.message for r in caplog.records) == 1


def test_dict_reward_with_reward_key():
    masked = _sample(status=Sample.Status.TRUNCATED, reward={"score": 0.0})
    kept = _sample(status=Sample.Status.TRUNCATED, reward={"score": 1.0})
    mask_truncated_zero_reward(_args(reward_key="score"), [[masked], [kept]])
    assert masked.remove_sample is True
    assert kept.remove_sample is False


def test_loadable_via_load_function():
    assert (
        load_function("vime.rollout.filter_hub.sample_filters.mask_truncated_zero_reward") is mask_truncated_zero_reward
    )

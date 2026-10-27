"""Tests for --rollout-submission-granularity / sample-level backfill (B5c).

Port of miles's submission scheduler semantics (miles/tests/fast/rollout/
test_sample_completion_backfill.py): a finished sample frees submission
capacity, so a replacement group is admitted once ``n_samples_per_prompt``
samples complete, without waiting for a whole group to return.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from collections import deque
from types import SimpleNamespace

try:
    import vllm_router  # noqa: F401
except ImportError:
    _router_stub = types.ModuleType("vllm_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["vllm_router"] = _router_stub
try:
    import transformers  # noqa: F401
except ImportError:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import vime.rollout.fully_async_rollout as fa
from vime.rollout.submission_scheduler import (
    GroupLevelSubmission,
    SampleBackfillSubmission,
    make_submission_scheduler,
)
from vime.utils.types import Sample

NUM_GPUS = 0


def _group(index: int, size: int) -> list[Sample]:
    return [Sample(index=index, prompt=f"p{index}") for _ in range(size)]


@pytest.mark.unit
def test_group_level_capacity_frees_per_group():
    scheduler = GroupLevelSubmission()
    assert scheduler.has_capacity(pending_groups=1, group_budget=2)
    assert not scheduler.has_capacity(pending_groups=2, group_budget=2)
    assert scheduler.sample_done_callback is None  # no per-sample accounting


@pytest.mark.unit
def test_sample_level_capacity_frees_per_sample():
    scheduler = SampleBackfillSubmission(group_size=4)
    scheduler.on_submit([_group(0, 4)])
    assert scheduler.samples_in_flight == 4
    # Budget of 2 groups = 8 sample slots; one more whole group fits.
    assert scheduler.has_capacity(pending_groups=1, group_budget=2)
    scheduler.on_submit([_group(1, 4)])
    assert not scheduler.has_capacity(pending_groups=2, group_budget=2)
    # A single finished sample is not enough for a whole new group…
    scheduler.sample_done_callback()
    assert not scheduler.has_capacity(pending_groups=2, group_budget=2)
    # …but once a group-size worth of samples finished, backfill admits one.
    for _ in range(3):
        scheduler.sample_done_callback()
    assert scheduler.has_capacity(pending_groups=2, group_budget=2)


@pytest.mark.unit
def test_sample_level_self_heals_leaked_counter(caplog):
    scheduler = SampleBackfillSubmission(group_size=2)
    scheduler.on_submit([_group(0, 2)])
    # The group task returned without its sample callbacks (e.g. early abort).
    with caplog.at_level("WARNING"):
        assert scheduler.has_capacity(pending_groups=0, group_budget=1)
    assert scheduler.samples_in_flight == 0
    assert "leaked" in caplog.text


@pytest.mark.unit
def test_factory_defaults_and_validation():
    assert isinstance(
        make_submission_scheduler(SimpleNamespace(n_samples_per_prompt=2), default="sample"),
        SampleBackfillSubmission,
    )
    assert isinstance(make_submission_scheduler(SimpleNamespace(), default="group"), GroupLevelSubmission)
    assert isinstance(
        make_submission_scheduler(SimpleNamespace(rollout_submission_granularity="group"), default="sample"),
        GroupLevelSubmission,
    )
    with pytest.raises(AssertionError, match="granularity"):
        make_submission_scheduler(SimpleNamespace(rollout_submission_granularity="bogus"), default="sample")


class _RecordingBuffer:
    def __init__(self, groups):
        self._groups = deque(groups)
        self.pulled = 0

    def get_samples(self, n):
        assert n == 1
        if not self._groups:
            return []
        self.pulled += 1
        return [self._groups.popleft()]

    def add_samples(self, groups):
        pass


@pytest.mark.unit
def test_worker_backfills_while_group_samples_still_in_flight(monkeypatch):
    """With 2 fast + 2 blocked samples per group and a 2-group budget, the
    third group must be admitted after the fast samples finish — group-level
    pacing would wait for a whole group to return."""
    import threading

    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    release = threading.Event()

    async def _staggered(args, group, sampling_params, evaluation, sample_done_callback=None):
        for _ in group[:2]:
            await asyncio.sleep(0.01)
            if sample_done_callback is not None:
                sample_done_callback()
        while not release.is_set():
            await asyncio.sleep(0.005)
        for _ in group[2:]:
            if sample_done_callback is not None:
                sample_done_callback()
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _staggered)
    buffer = _RecordingBuffer([_group(i, 4) for i in range(3)])
    args = SimpleNamespace(
        n_samples_per_prompt=4, rollout_batch_size=4, rollout_submission_granularity="sample"
    )
    worker = fa.AsyncRolloutWorker(args, buffer, concurrency=8)
    assert isinstance(worker.submission, SampleBackfillSubmission)
    worker.poll_interval = 0.01

    worker.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if buffer.pulled >= 3:
                break
            time.sleep(0.01)
        assert buffer.pulled == 3, "no backfill: third group was not admitted on sample completions"
        assert worker.queue_size() == 0  # A and B still blocked in slow samples
    finally:
        release.set()
        worker.close()


@pytest.mark.unit
def test_group_granularity_waits_for_whole_group(monkeypatch):
    import threading

    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    release = threading.Event()

    async def _staggered(args, group, sampling_params, evaluation, sample_done_callback=None):
        await asyncio.sleep(0.01)
        while not release.is_set():
            await asyncio.sleep(0.005)
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _staggered)
    buffer = _RecordingBuffer([_group(i, 4) for i in range(3)])
    args = SimpleNamespace(
        n_samples_per_prompt=4, rollout_batch_size=4, rollout_submission_granularity="group"
    )
    worker = fa.AsyncRolloutWorker(args, buffer, concurrency=8)
    worker.poll_interval = 0.01

    worker.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            time.sleep(0.01)
        assert buffer.pulled == 2, "group granularity must wait for a whole group slot"
    finally:
        release.set()
        worker.close()


@pytest.mark.unit
def test_worker_defaults_to_group_granularity(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    args = SimpleNamespace(n_samples_per_prompt=4, rollout_batch_size=4)
    worker = fa.AsyncRolloutWorker(args, _RecordingBuffer([]), concurrency=8)
    assert isinstance(worker.submission, GroupLevelSubmission)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

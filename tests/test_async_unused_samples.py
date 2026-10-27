"""Tests for --async-unused-samples-handler and --async-capacity-factor (B5b).

Port of miles's unused-samples handler semantics (retry/drop) onto vime's
queue-based requeue, plus the generation look-ahead cap. The
lease-after-resume test lives in tests/test_distributed_rollout.py next to
the queue fixture it uses.
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
from vime.rollout.async_policy import (
    UnusedReason,
    async_capacity_limit,
    handle_unused_group,
    prompt_retry_stubs,
    reset_group_for_retry,
    unused_samples_handler,
)
from vime.utils.types import Sample

NUM_GPUS = 0


def _sample(index: int, versions=("-1",), response_length: int = 2) -> Sample:
    sample = Sample(index=index, prompt=f"p{index}", tokens=[10, 11, 12, 13], response="ok")
    sample.status = Sample.Status.COMPLETED
    sample.reward = 1.0
    sample.response_length = response_length
    sample.loss_mask = [1, 1]
    sample.weight_versions = list(versions)
    sample.rollout_log_probs = [-0.1, -0.2]
    return sample


class _FakeBuffer:
    def __init__(self, groups=()):
        self._groups = deque(groups)
        self.requeued = []

    def get_samples(self, n):
        return [self._groups.popleft()] if self._groups else []

    def add_samples(self, groups):
        self.requeued.extend(groups)


# -- handler resolution -------------------------------------------------------


@pytest.mark.unit
def test_handler_defaults_are_reason_dependent():
    args = SimpleNamespace()
    assert unused_samples_handler(args, UnusedReason.ABORTED) == "retry"  # vime legacy
    assert unused_samples_handler(args, UnusedReason.STALE) == "drop"  # miles parity


@pytest.mark.unit
@pytest.mark.parametrize("handler", ["retry", "drop"])
def test_explicit_handler_overrides_both_reasons(handler):
    args = SimpleNamespace(async_unused_samples_handler=handler)
    for reason in UnusedReason:
        assert unused_samples_handler(args, reason) == handler


# -- reset for retry ----------------------------------------------------------


@pytest.mark.unit
def test_reset_group_for_retry_strips_generation_but_keeps_prompt():
    group = [_sample(3, versions=["7"])]
    reset_group_for_retry(group)
    sample = group[0]
    assert sample.index == 3 and sample.prompt == "p3"
    assert sample.tokens == [10, 11]  # response portion trimmed
    assert sample.response == "" and sample.response_length == 0
    assert sample.reward is None and sample.loss_mask is None
    assert sample.weight_versions == [] and sample.rollout_log_probs is None
    assert sample.status == Sample.Status.PENDING


# -- handler behavior ----------------------------------------------------------


@pytest.mark.unit
def test_stale_retry_requeues_reset_group(monkeypatch):
    buffer = _FakeBuffer()
    monkeypatch.setattr(
        "vime.rollout.async_policy.discard_rollout_group",
        lambda *a, **k: pytest.fail("retry must not discard"),
    )
    group = [_sample(1, versions=["3"])]
    args = SimpleNamespace(async_unused_samples_handler="retry")
    assert handle_unused_group(args, group, UnusedReason.STALE, data_source=buffer) == "retry"
    assert len(buffer.requeued) == 1
    assert buffer.requeued[0][0].status == Sample.Status.PENDING


@pytest.mark.unit
def test_aborted_retry_requeues_without_reset():
    buffer = _FakeBuffer()
    group = [_sample(1)]
    group[0].status = Sample.Status.ABORTED  # partial continuation is preserved
    args = SimpleNamespace()
    assert handle_unused_group(args, group, UnusedReason.ABORTED, data_source=buffer) == "retry"
    assert buffer.requeued[0][0].response_length == 2  # not reset
    assert buffer.requeued[0][0].status == Sample.Status.ABORTED


@pytest.mark.unit
def test_drop_discards_via_transport(monkeypatch):
    discarded = []
    monkeypatch.setattr(
        "vime.rollout.async_policy.discard_rollout_group",
        lambda group, args, reason, *, controller=None: discarded.append((group, reason)),
    )
    group = [_sample(2)]
    args = SimpleNamespace(async_unused_samples_handler="drop")
    assert handle_unused_group(args, group, UnusedReason.ABORTED, data_source=_FakeBuffer()) == "drop"
    assert discarded == [(group, "aborted")]


# -- worker done-callback honors the handler -----------------------------------


@pytest.mark.unit
def test_done_callback_drop_handler_discards_aborted(monkeypatch):
    discarded = []
    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    monkeypatch.setattr(
        "vime.rollout.async_policy.discard_rollout_group",
        lambda group, args, reason, *, controller=None: discarded.append(reason),
    )
    buffer = _FakeBuffer()
    args = SimpleNamespace(
        n_samples_per_prompt=1, rollout_batch_size=2, async_unused_samples_handler="drop"
    )
    worker = fa.AsyncRolloutWorker(args, buffer, concurrency=2)

    class _DoneTask:
        def result(self):
            group = [_sample(0)]
            group[0].status = Sample.Status.ABORTED
            return group

    worker._make_done_cb(0)(_DoneTask())
    assert discarded == ["aborted"]
    assert buffer.requeued == []  # not requeued
    assert worker.queue_size() == 0  # not shipped to training


@pytest.mark.unit
def test_done_callback_default_requeues_aborted(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    buffer = _FakeBuffer()
    args = SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=2)
    worker = fa.AsyncRolloutWorker(args, buffer, concurrency=2)

    class _DoneTask:
        def result(self):
            group = [_sample(0)]
            group[0].status = Sample.Status.ABORTED
            return group

    worker._make_done_cb(0)(_DoneTask())
    assert len(buffer.requeued) == 1


# -- stale drain with retry handler --------------------------------------------


class _FakeWorker:
    def __init__(self, groups):
        self._groups = list(groups)
        self.error = None

    def resume(self):
        pass

    def queue_size(self):
        return 0

    def get_completed_groups(self, limit=None):
        out, self._groups = self._groups[:limit], self._groups[limit:]
        return [(i, g) for i, g in enumerate(out)]


@pytest.mark.unit
def test_stale_retry_recycles_prompt_instead_of_training(monkeypatch):
    fresh = [_sample(0, versions=["10"])]
    stale = [_sample(1, versions=["5"])]
    spare = [_sample(2, versions=["10"])]
    buffer = _FakeBuffer()
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: _FakeWorker([fresh, stale, spare]))
    monkeypatch.setattr(
        "vime.rollout.async_policy.discard_rollout_group",
        lambda *a, **k: pytest.fail("retry must not discard"),
    )
    args = SimpleNamespace(
        rollout_batch_size=2,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=2,
        async_unused_samples_handler="retry",
    )
    out = asyncio.run(
        fa._generate_rollout_async(args, rollout_id=0, data_buffer=buffer, current_weight_version=10)
    )
    assert [g[0].index for g in out.samples] == [0, 2]
    # The stale group's prompt was recycled as a fresh sample, not trained.
    assert len(buffer.requeued) == 1
    recycled = buffer.requeued[0][0]
    assert recycled.index == 1 and recycled.status == Sample.Status.PENDING
    assert recycled.weight_versions == [] and recycled.reward is None
    assert out.metrics["rollout/staleness/retried_groups"] == 1
    assert out.metrics["rollout/staleness/dropped_groups"] == 0


@pytest.mark.unit
def test_prompt_retry_stubs_keep_only_prompt_fields():
    group = [_sample(5, versions=["3"])]
    group[0].label = "gold"
    group[0].metadata = {"k": "v"}
    group[0].group_index = 5
    group[0]._queue_source_positions = (3,)
    stubs = prompt_retry_stubs(group)
    (sample,) = stubs
    assert sample.index == 5 and sample.group_index == 5
    assert sample.prompt == "p5" and sample.label == "gold" and sample.metadata == {"k": "v"}
    assert sample.status == Sample.Status.PENDING
    assert sample.response_length == 0 and sample.weight_versions == [] and sample.reward is None
    assert sample._queue_source_positions == (3,)
    assert not hasattr(sample, "_queue_lease")  # old lease stays with the discarded payload


# -- capacity factor ------------------------------------------------------------


@pytest.mark.unit
def test_capacity_limit_is_off_by_default():
    assert async_capacity_limit(SimpleNamespace()) is None


@pytest.mark.unit
def test_capacity_limit_scales_with_batch():
    assert async_capacity_limit(SimpleNamespace(async_capacity_factor=2.0, rollout_batch_size=64)) == 128


@pytest.mark.unit
def test_worker_rejects_cap_below_warmup_floor(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    # cap = int(1.0 * 4) = 4 < concurrency 8 -> must raise
    args = SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=4, async_capacity_factor=1.0)
    with pytest.raises(ValueError, match="async-capacity-factor"):
        fa.AsyncRolloutWorker(args, _FakeBuffer(), concurrency=8)
    # cap covers both batch and concurrency -> ok
    args = SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=8, async_capacity_factor=1.0)
    worker = fa.AsyncRolloutWorker(args, _FakeBuffer(), concurrency=4)
    assert worker.capacity_limit == 8


@pytest.mark.unit
def test_capacity_gate_bounds_queue_growth(monkeypatch):
    """With the cap set, the warm queue plateaus at the cap, not concurrency."""
    monkeypatch.setattr(fa, "GenerateState", lambda args: SimpleNamespace(sampling_params={}))
    fuel = 60
    buffer = _FakeBuffer([[_sample(i)] for i in range(fuel)])

    async def _instant_generate(args, group, sampling_params, evaluation):
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _instant_generate)
    # concurrency 6 groups, batch 6, factor 1.0 -> cap 6
    args = SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=6, async_capacity_factor=1.0)
    worker = fa.AsyncRolloutWorker(args, buffer, concurrency=6)
    assert worker.capacity_limit == 6
    worker.poll_interval = 0.01
    worker.start()
    try:
        deadline = time.time() + 3.0
        max_seen = 0
        while time.time() < deadline:
            max_seen = max(max_seen, worker.queue_size())
            time.sleep(0.02)
    finally:
        worker.close()
    assert 0 < max_seen <= 6, f"queue grew to {max_seen} with cap=6"


# -- argument validation -------------------------------------------------------


@pytest.mark.unit
def test_capacity_factor_arg_validation(monkeypatch):
    from test_megatron_argument_validation import load_vime_arguments_module, make_vime_validate_args

    module = load_vime_arguments_module(monkeypatch)
    base = dict(rollout_function_path="vime.rollout.fully_async_rollout.generate_rollout_fully_async")
    args, _ = module.vime_validate_args(make_vime_validate_args(**base, async_capacity_factor=2.0))
    assert args.async_capacity_factor == 2.0
    with pytest.raises(ValueError, match=">= 1.0"):
        module.vime_validate_args(make_vime_validate_args(**base, async_capacity_factor=0.5))
    with pytest.raises(ValueError, match="fully-async"):
        module.vime_validate_args(make_vime_validate_args(async_capacity_factor=2.0))


@pytest.mark.unit
def test_handler_arg_validation(monkeypatch):
    from test_megatron_argument_validation import load_vime_arguments_module, make_vime_validate_args

    module = load_vime_arguments_module(monkeypatch)
    base = dict(rollout_function_path="vime.rollout.fully_async_rollout.generate_rollout_fully_async")
    args, _ = module.vime_validate_args(make_vime_validate_args(**base, async_unused_samples_handler="retry"))
    assert args.async_unused_samples_handler == "retry"
    with pytest.raises(ValueError, match="fully-async"):
        module.vime_validate_args(make_vime_validate_args(async_unused_samples_handler="drop"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

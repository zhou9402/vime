"""Tests for --max-weight-staleness (B5a) built on the V4 simulation harness.

Covers the group-level staleness helpers in ``vime.utils.staleness``, the
drain-time drop in the local fully-async worker path, and a run of the V4
simulation with the *real* staleness functions plugged in as the policy.
"""

from __future__ import annotations

import asyncio
import sys
import types
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
from vime.data.transport import RolloutGroupRef
from vime.utils.staleness import group_is_stale, group_staleness, sample_staleness
from vime.utils.types import Sample

from test_async_staleness_simulation import STALE_SCHEDULE, SimGroup, SimSample, run_simulation

NUM_GPUS = 0


def _sample(index: int, versions: list[str]) -> Sample:
    sample = Sample(index=index, prompt=f"p{index}")
    sample.status = Sample.Status.COMPLETED
    sample.reward = 1.0
    sample.weight_versions = versions
    return sample


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions_by_sample,current,expected",
    [
        ([["10"], ["9"]], 10, 1),  # oldest sample governs
        ([["8", "10"], ["10"]], 10, 2),
        ([[]], 10, None),  # unversioned -> unknown -> kept
        ([["bad"]], 10, None),
        ([["11"]], 10, None),  # future version -> unknown
        ([], 10, None),
    ],
)
def test_group_staleness_uses_oldest_sample(versions_by_sample, current, expected):
    group = [_sample(i, versions) for i, versions in enumerate(versions_by_sample)]
    assert group_staleness(group, current) == expected


@pytest.mark.unit
def test_group_staleness_handles_nested_groups():
    group = [[_sample(0, ["7"])], [_sample(1, ["9"])]]
    assert group_staleness(group, 10) == 3


@pytest.mark.unit
def test_group_staleness_reads_straw_ref_summary():
    ref = RolloutGroupRef(manifest=None, root="/tmp", index=0, min_weight_version=7)
    assert group_staleness(ref, 10) == 3
    assert group_staleness(RolloutGroupRef(manifest=None, root="/tmp", index=0), 10) is None
    assert group_staleness(RolloutGroupRef(manifest=None, root="/tmp", index=0, min_weight_version=11), 10) is None


@pytest.mark.unit
def test_staleness_limit_is_inclusive():
    group = [_sample(0, ["8"])]
    assert not group_is_stale(group, 10, 2)  # age == limit: kept
    assert group_is_stale(group, 10, 1)  # age > limit: dropped
    assert not group_is_stale(group, 10, None)  # disabled
    assert not group_is_stale([_sample(0, [])], 10, 0)  # unversioned always kept
    assert not group_is_stale(group, None, 1)  # no current version -> keep


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
def test_drain_drops_stale_groups_and_refills(monkeypatch):
    groups = [
        [_sample(0, ["10"])],  # fresh
        [_sample(1, ["5"])],  # stale (age 5 > 2)
        [_sample(2, ["9"])],  # fresh
        [_sample(3, [])],  # unversioned: kept
        [_sample(4, ["8"])],  # boundary (age 2 == limit): kept
    ]
    worker = _FakeWorker(groups)
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)
    discarded = []
    monkeypatch.setattr(
        fa, "discard_rollout_group", lambda group, args, reason, *, controller=None: discarded.append((group, reason))
    )
    args = SimpleNamespace(
        rollout_batch_size=4,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=2,
    )
    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None, current_weight_version=10))

    assert [g[0].index for g in out.samples] == [0, 2, 3, 4]
    assert [(g[0].index, reason) for g, reason in discarded] == [(1, "stale")]
    assert out.metrics["rollout/staleness/dropped_groups"] == 1
    assert out.metrics["rollout/staleness/max_weight_staleness"] == 2
    assert "rollout/dynamic_filter/drop_stale" not in out.metrics
    assert out.metrics["rollout/dynamic_filter/dropped_groups"] == 0


@pytest.mark.unit
def test_drain_without_staleness_flag_keeps_everything(monkeypatch):
    groups = [[_sample(0, ["1"])], [_sample(1, ["10"])]]
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: _FakeWorker(groups))
    monkeypatch.setattr(
        fa,
        "discard_rollout_group",
        lambda *a, **k: pytest.fail("no group may be discarded without --max-weight-staleness"),
    )
    args = SimpleNamespace(
        rollout_batch_size=2,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path=None,
    )
    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None, current_weight_version=10))
    # No metrics -> legacy return shape (plain sample list) is preserved.
    assert [g[0].index for g in out] == [0, 1]


class _RealPolicy:
    """Adapter: judge SimGroups with the real vime staleness functions."""
    def __init__(self, max_staleness):
        self.max_staleness = max_staleness

    def judge(self, group: SimGroup, current_weight_version: int) -> str:
        samples = [SimpleNamespace(weight_versions=[str(s.weight_version)]) for s in group.samples]
        ages = [sample_staleness(s, current_weight_version) for s in samples]
        if any(age is None for age in ages):
            return "keep"
        return "drop" if max(ages) > self.max_staleness else "keep"


@pytest.mark.unit
def test_v4_simulation_with_real_staleness_policy():
    result = run_simulation(policy=_RealPolicy(2), **STALE_SCHEDULE)
    assert result.max_seen_staleness <= 2
    assert len(result.dropped) > 0  # non-vacuous
    assert run_simulation(policy=None, **STALE_SCHEDULE).max_seen_staleness > 2


# -- current_weight_version via rollout_context ---------------------------------


@pytest.mark.unit
def test_partial_wrapped_custom_fn_sees_weight_version_via_context():
    """The version travels through rollout_context, so a functools.partial-
    wrapped custom rollout fn (no current_weight_version in its signature)
    still observes it."""
    import functools

    from vime.rollout.base_types import call_rollout_fn
    from vime.rollout.sample_hooks import current_rollout_weight_version, rollout_context

    seen = {}

    def custom(args, rollout_id, data_buffer, evaluation=False):
        seen["version"] = current_rollout_weight_version()
        return []

    wrapped = functools.partial(custom)  # signature no longer introspectable
    args = SimpleNamespace(rollout_data_transport="object-store")
    with rollout_context(3, weight_version=7):
        call_rollout_fn(wrapped, args, 3, None, evaluation=False)
    assert seen["version"] == 7
    # Outside the context the default is restored.
    assert current_rollout_weight_version() is None


@pytest.mark.unit
def test_entrypoint_reads_context_and_kwarg_overrides(monkeypatch):
    from vime.rollout.sample_hooks import rollout_context

    captured = []

    async def fake_generate(args, rollout_id, data_buffer, current_weight_version=None):
        captured.append(current_weight_version)
        return []

    monkeypatch.setattr(fa, "_generate_rollout_async", fake_generate)
    monkeypatch.setattr(fa, "run", asyncio.run)
    source = SimpleNamespace(consumers={"fully_async": object.__new__(fa.AsyncRolloutWorker)})

    with rollout_context(0, weight_version=11):
        fa.generate_rollout_fully_async(SimpleNamespace(), 0, source)
    assert captured == [11]

    with rollout_context(0, weight_version=11):
        fa.generate_rollout_fully_async(SimpleNamespace(), 0, source, current_weight_version=4)
    assert captured[-1] == 4  # explicit kwarg wins


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

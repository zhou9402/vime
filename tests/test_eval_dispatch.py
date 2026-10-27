"""EvalDispatcher contract tests (port of miles eval_dispatch).

Pure-async fakes stand in for the Ray bridges; no Ray or GPU required.
"""

import asyncio
import os
from argparse import Namespace

import pytest

from vime.ray.eval_dispatch import EvalDispatcher


def _args(**overrides):
    base = dict(
        eval_uses_snapshots=True,
        eval_max_in_flight=2,
        eval_overflow_policy="backpressure",
        eval_keep_snapshots=2,
        eval_hf_dir=None,
        offload_train=False,
    )
    base.update(overrides)
    return Namespace(**base)


class _FakeActorModel:
    def __init__(self, fail_export=False):
        self.exports = []
        self.fail_export = fail_export

    async def export_hf(self, rollout_id, hf_dir):
        if self.fail_export:
            raise RuntimeError("export boom")
        os.makedirs(hf_dir, exist_ok=True)
        self.exports.append((rollout_id, hf_dir))

    async def onload(self):
        pass

    async def offload(self):
        pass


class _FakeRolloutManager:
    def __init__(self, crash_rollout_ids=()):
        self.evals = []
        self.skips = []
        self.crash_rollout_ids = set(crash_rollout_ids)
        self.concurrent = 0
        self.max_concurrent = 0
        self._gate = asyncio.Event()
        self._gate.set()

    async def eval(self, rollout_id, **kwargs):
        self.evals.append((rollout_id, kwargs))
        if rollout_id in self.crash_rollout_ids:
            raise RuntimeError("eval boom")
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        await asyncio.sleep(0.01)
        self.concurrent -= 1

    async def report_eval_skip(self, rollout_id, reason):
        self.skips.append((rollout_id, reason))


def _run(coro):
    return asyncio.run(coro)


def test_blocking_path_when_snapshots_disabled(tmp_path):
    rm, actor = _FakeRolloutManager(), _FakeActorModel()
    d = EvalDispatcher(_args(eval_uses_snapshots=False), actor, rm)
    _run(d.dispatch(3))
    assert rm.evals == [(3, {})]
    assert actor.exports == []


def test_dispatch_exports_and_enqueues(tmp_path):
    actor, rm = _FakeActorModel(), _FakeRolloutManager()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path)), actor, rm)

    async def main():
        await d.dispatch(1)
        await d.drain()

    _run(main())
    assert actor.exports == [(1, str(tmp_path / "step_1"))]
    rollout_id, kwargs = rm.evals[0]
    assert rollout_id == 1
    assert kwargs["hf_dir"] == str(tmp_path / "step_1")
    assert kwargs["require_marker"] is True
    assert kwargs["export_time_seconds"] is not None


def test_in_flight_cap_and_backpressure(tmp_path):
    rm, actor = _FakeRolloutManager(), _FakeActorModel()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path), eval_max_in_flight=2), actor, rm)

    async def main():
        # Occupy both in-flight slots with evals that never finish until released.
        gate = asyncio.Event()

        async def gated_eval(rollout_id, **kwargs):
            rm.evals.append((rollout_id, kwargs))
            await gate.wait()

        rm.eval = gated_eval
        await d.dispatch(1)
        await d.dispatch(2)
        assert len(d.pending) == 2
        third = asyncio.ensure_future(d.dispatch(3))
        await asyncio.sleep(0.05)
        assert not third.done()  # backpressure: settling oldest blocks
        gate.set()
        await third
        assert len(d.pending) == 2  # settled 1, enqueued 3
        await d.drain()

    _run(main())
    assert rm.skips == []


def test_overflow_skip_policy(tmp_path):
    rm, actor = _FakeRolloutManager(), _FakeActorModel()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path), eval_overflow_policy="skip"), actor, rm)

    async def main():
        gate = asyncio.Event()

        async def gated_eval(rollout_id, **kwargs):
            await gate.wait()

        rm.eval = gated_eval
        await d.dispatch(1)
        await d.dispatch(2)
        await d.dispatch(3)  # dropped
        assert rm.skips == [(3, "busy")]
        # force=True overrides the skip policy with backpressure.
        forced = asyncio.ensure_future(d.dispatch(4, force=True))
        await asyncio.sleep(0.05)
        assert not forced.done()
        gate.set()
        await forced
        await d.drain()

    _run(main())


def test_export_failure_reports_skip(tmp_path):
    rm = _FakeRolloutManager()
    actor = _FakeActorModel(fail_export=True)
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path)), actor, rm)
    _run(d.dispatch(1))
    assert rm.skips == [(1, "export_failed")]
    assert rm.evals == []


def test_crashed_eval_reports_skip_and_retires_snapshot(tmp_path):
    rm = _FakeRolloutManager(crash_rollout_ids={1})
    actor = _FakeActorModel()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path)), actor, rm)

    async def main():
        await d.dispatch(1)
        await d.drain()

    _run(main())
    assert (1, "crashed") in rm.skips
    assert d._exported == [str(tmp_path / "step_1")]


def test_snapshot_gc_keeps_eval_keep_snapshots(tmp_path):
    rm, actor = _FakeRolloutManager(), _FakeActorModel()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path), eval_keep_snapshots=2), actor, rm)

    async def main():
        for i in range(4):
            await d.dispatch(i)
            await d.drain()

    _run(main())
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["step_2", "step_3"]


def test_caller_supplied_hf_dir_never_gc_and_no_marker(tmp_path):
    rm, actor = _FakeRolloutManager(), _FakeActorModel()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path)), actor, rm)
    external = tmp_path / "external_ckpt"
    external.mkdir()

    async def main():
        await d.dispatch(1, hf_dir=str(external))
        await d.drain()

    _run(main())
    assert external.exists()
    assert actor.exports == []
    _, kwargs = rm.evals[0]
    assert kwargs["require_marker"] is False
    assert kwargs["export_time_seconds"] is None


def test_drain_settles_all_pending(tmp_path):
    rm, actor = _FakeRolloutManager(), _FakeActorModel()
    d = EvalDispatcher(_args(eval_hf_dir=str(tmp_path), eval_max_in_flight=5), actor, rm)

    async def main():
        for i in range(4):
            await d.dispatch(i)
        assert len(d.pending) == 4
        await d.drain()
        assert len(d.pending) == 0

    _run(main())
    assert len(rm.evals) == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

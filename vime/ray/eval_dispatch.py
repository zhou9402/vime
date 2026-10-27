"""Non-blocking eval dispatch for the sync vime driver loop.

Port of miles@ba28991834 miles/ray/rollout/eval_dispatch.py.

The dispatcher itself is plain asyncio and engine-agnostic: ``rollout_manager``
and ``actor_model`` are duck-typed awaitable proxies (in production, the Ray
bridges below; in tests, plain async fakes). Because vime's ``train.py`` driver
is synchronous, ``AsyncEvalThread`` hosts the dispatcher's event loop on a
daemon thread and exposes blocking ``dispatch``/``drain`` calls.
"""

import asyncio
import logging
import os
import shutil
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class EvalDispatcher:
    """Dispatches eval points without holding up the training loop.

    Exports a snapshot per point, fires the eval, and reclaims the snapshot once the
    point settles. Failures degrade to a skipped point outside CI; under ``--ci-test``,
    every skipped point fails the run. Shared-engine eval takes the plain blocking call
    instead.
    """

    def __init__(self, args, actor_model, rollout_manager):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, asyncio.Task, str | None]] = deque()
        self._exported: list[str] = []

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        """A caller-supplied ``hf_dir`` is an existing checkpoint, not one of our exports."""
        if not self.args.eval_uses_snapshots:
            await self.rollout_manager.eval(rollout_id)
            return

        await self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip" and not force:
                await self.rollout_manager.report_eval_skip(rollout_id, "busy")
                return
            await self._settle(*self.pending.popleft())

        export_time = None
        exported_dir = None
        require_marker = hf_dir is None
        if hf_dir is None:
            # Snapshot eval always exports its own snapshot; --save-hf reuse is
            # deliberately unsupported (save points don't cover every eval point).
            hf_dir = os.path.join(self.args.eval_hf_dir, f"step_{rollout_id}")
            try:
                export_time = await self._export(rollout_id, hf_dir)
            except Exception as e:
                logger.error(f"HF snapshot export for eval {rollout_id} failed: {e}")
                await self.rollout_manager.report_eval_skip(rollout_id, "export_failed")
                return
            exported_dir = hf_dir

        task = asyncio.ensure_future(
            self.rollout_manager.eval(
                rollout_id, hf_dir=hf_dir, export_time_seconds=export_time, require_marker=require_marker
            )
        )
        self.pending.append((rollout_id, task, exported_dir))

    async def drain(self) -> None:
        while self.pending:
            await self._settle(*self.pending.popleft())

    async def _export(self, rollout_id: int, hf_dir: str) -> float:
        start = time.time()
        if self.args.offload_train:
            await self.actor_model.onload()
        try:
            await self.actor_model.export_hf(rollout_id, hf_dir)
        finally:
            if self.args.offload_train:
                await self.actor_model.offload()
        return time.time() - start

    async def _reap_finished(self) -> None:
        while self.pending:
            if not self.pending[0][1].done():
                break
            await self._settle(*self.pending.popleft())

    async def _settle(self, rollout_id: int, task: asyncio.Task, exported_dir: str | None) -> None:
        try:
            await task
        except Exception:
            logger.exception(f"Async eval for rollout {rollout_id} raised")
            await self.rollout_manager.report_eval_skip(rollout_id, "crashed")
        finally:
            self._retire(exported_dir)

    def _retire(self, exported_dir: str | None) -> None:
        """Every settled point retires exactly one snapshot, whatever its outcome."""
        if exported_dir is None:
            return
        self._exported.append(exported_dir)
        while len(self._exported) > self.args.eval_keep_snapshots:
            victim = self._exported.pop(0)
            shutil.rmtree(victim, ignore_errors=True)
            logger.info(f"GC'd consumed eval snapshot {victim}")


class _RayRolloutManagerEvalBridge:
    """Awaitable facade over the RolloutManager Ray actor (and the EvalRunner
    actor when a dedicated eval fleet is configured)."""

    def __init__(self, rollout_manager, eval_runner=None):
        self._actor = rollout_manager
        self._eval_runner = eval_runner

    async def eval(self, rollout_id, hf_dir=None, export_time_seconds=None, require_marker=False):
        import ray

        if hf_dir is None:
            return await asyncio.to_thread(ray.get, self._actor.eval.remote(rollout_id))
        if self._eval_runner is None:
            raise RuntimeError(
                "Snapshot eval was dispatched but no dedicated eval fleet is running; "
                "set --eval-num-gpus or disable snapshot eval."
            )
        from vime.ray.eval_fleet import EvalSkip

        try:
            data, metrics = await asyncio.to_thread(
                ray.get,
                self._eval_runner.run.remote(rollout_id, hf_dir, str(rollout_id), require_marker),
            )
        except EvalSkip as e:
            await self.report_eval_skip(rollout_id, e.reason)
            return
        await asyncio.to_thread(ray.get, self._actor.report_eval_result.remote(rollout_id, data, metrics))

    async def report_eval_skip(self, rollout_id, reason):
        import ray

        return await asyncio.to_thread(ray.get, self._actor.report_eval_skip.remote(rollout_id, reason))

    def dispose(self):
        if self._eval_runner is not None:
            import ray

            ray.get(self._eval_runner.dispose.remote())


class _RayTrainGroupBridge:
    """Awaitable facade over RayTrainGroup onload/offload/export_hf."""

    def __init__(self, actor_model):
        self._group = actor_model

    async def onload(self):
        await asyncio.to_thread(self._group.onload)

    async def offload(self):
        await asyncio.to_thread(self._group.offload)

    async def export_hf(self, rollout_id, hf_dir):
        await asyncio.to_thread(self._group.export_hf, rollout_id, hf_dir)


class AsyncEvalThread:
    """Hosts an EvalDispatcher event loop on a daemon thread for the sync driver."""

    def __init__(self, dispatcher: EvalDispatcher, on_close=None):
        self._dispatcher = dispatcher
        self._on_close = on_close
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="eval-dispatch", daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._dispatcher.dispatch(rollout_id, hf_dir=hf_dir, force=force), self._loop
        )
        future.result()  # dispatch itself only enqueues; eval keeps running on the loop

    def drain(self) -> None:
        future = asyncio.run_coroutine_threadsafe(self._dispatcher.drain(), self._loop)
        future.result()

    def close(self) -> bool:
        """Drain and stop the loop. Returns False when the thread is stuck; in
        that case the eval runner is left alone because it may be mid-RPC."""
        self.drain()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            logger.error("Eval dispatch thread did not stop within 30s; leaving the eval runner undisposed.")
            return False
        if self._on_close is not None:
            self._on_close()
        return True


def create_eval_dispatcher(args, actor_model, rollout_manager) -> AsyncEvalThread:
    eval_runner = None
    if getattr(args, "eval_uses_snapshots", False):
        import ray

        from vime.ray.eval_fleet import EvalRunner
        from vime.rollout.checkpoint_eval import is_checkpoint_eval_fn

        fleet = ray.get(rollout_manager.get_eval_fleet.remote())
        if fleet is not None:
            # Driver-owned (not detached): eval engines are not part of the
            # recoverable serving topology in v1 and die with the driver.
            eval_runner = EvalRunner.remote(args, fleet["engines"], fleet["router"])
        elif is_checkpoint_eval_fn(getattr(args, "eval_function_path", None)):
            # A CheckpointEvalFn owns its backend; the runner just hosts it.
            eval_runner = EvalRunner.remote(args)
        else:
            logger.warning("Snapshot eval requested but no eval fleet is configured; evals will be skipped.")
    bridge = _RayRolloutManagerEvalBridge(rollout_manager, eval_runner)
    dispatcher = EvalDispatcher(
        args,
        _RayTrainGroupBridge(actor_model),
        bridge,
    )
    return AsyncEvalThread(dispatcher, on_close=bridge.dispose)

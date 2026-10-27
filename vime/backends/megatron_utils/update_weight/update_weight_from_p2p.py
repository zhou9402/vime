"""P2P (Mooncake RDMA) weight updater for disaggregated deployments (F01-P3).

Replica-free ("Design A", docs/en/advanced/p2p-weight-transfer.md): the trainer
packs each HF bucket into a pinned transfer buffer, RDMA-writes it into one
staging buffer each target engine worker publishes, and the worker reshards the
staged bytes with its own vLLM `load_weights` (`load_weights_from_staged`).

Only disaggregated (non-colocate) full-mode sync; argument validation rejects
colocate / LoRA, and dspark speculative decoding (mtp is supported). Writes are
never retried. An engine that errors is shut down and skipped; the health
monitor then retires it, `recover()` replaces it, the next
`connect_rollout_engines` clears the quarantine, and the next update fully
syncs it. --p2p-on-engine-error=fail raises at the end of the update; =continue
keeps training on the healthy fleet down to --p2p-min-healthy-engines.
"""

import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from vime.utils.distributed_utils import get_gloo_group
from vime.utils.weight_sync import should_flush_cache

from .checksum_utils import hash_tensor_sha256
from .p2p import (
    MooncakeTransport,
    RemoteShard,
    TransferBuffers,
    assign_rollout_engine_ranks_for_data_replica,
    pack_bucket_into_transfer_buffer,
)

logger = logging.getLogger(__name__)


class _EngineTarget:
    """Per-engine error-isolation handle (TransferBuffers' WriteTarget).

    F01-P4 replaces this with the shared rollout-cell quarantine (with F02).
    """

    def __init__(self, engine_index: int) -> None:
        self.engine_index = engine_index
        self.errored: BaseException | None = None
        # weight_version the error happened in; quarantine is permanent, but an
        # update only fails for engines that errored during THAT update.
        self.errored_at_version: int | None = None

    @property
    def is_errored(self) -> bool:
        return self.errored is not None

    def mark_errored(self, error: BaseException, weight_version: int | None = None) -> None:
        if self.errored is None:
            self.errored = error
            self.errored_at_version = weight_version
            logger.error("P2P weight update: engine %d marked errored: %r", self.engine_index, error)


def _engine_call(engine: Any, method: str, *args, **kwargs) -> Any:
    """Call a weight-sync method on a Ray actor handle or a plain (test) object."""
    fn = getattr(engine, method)
    if hasattr(fn, "remote"):
        import ray

        return ray.get(fn.remote(*args, **kwargs))
    return fn(*args, **kwargs)


class UpdateWeightFromP2P:
    """Update disaggregated vLLM engines by RDMA-writing into their staging buffers."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, Any] | None,
        # dependency injection for CPU tests (no Megatron / Mooncake / Ray)
        iterator: Any | None = None,
        transport_factory: Callable[[], MooncakeTransport] = MooncakeTransport,
        data_replica_rank_and_size: tuple[int, int] | None = None,
    ) -> None:
        self.args = args
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        if iterator is None:
            from .hf_weight_iterator_direct import HfWeightIteratorDirect

            iterator = HfWeightIteratorDirect(
                args=args, model=model, model_name=model_name, quantization_config=quantization_config
            )
        self._iterator = iterator
        self._weights_getter = weights_getter
        self._transport_factory = transport_factory
        self._transport: MooncakeTransport | None = None  # created on connect
        self._data_replica_rank_and_size = data_replica_rank_and_size
        self._buffers: TransferBuffers | None = None
        self._assignments = []
        self._shards_by_rank: dict[int, list[RemoteShard]] = {}
        self._targets: list[_EngineTarget] = []
        self.rollout_engines: list[Any] = []

    # ---- wiring -----------------------------------------------------------------

    def _rank_and_size(self) -> tuple[int, int]:
        if self._data_replica_rank_and_size is not None:
            return self._data_replica_rank_and_size
        try:
            from megatron.core import mpu

            return mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        except ImportError:
            if dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
            return 0, 1

    def _driver_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[Any],
        rollout_engine_lock: Any,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        del rollout_engine_lock, engine_gpu_offsets
        # Reconnect (an engine recovered after being retired) starts from a clean
        # slate — same as the distributed path.
        self.disconnect_rollout_engines()
        self.rollout_engines = list(rollout_engines)
        engine_gpu_counts = list(
            engine_gpu_counts or [self.args.rollout_num_gpus_per_engine] * len(self.rollout_engines)
        )
        self._targets = [_EngineTarget(i) for i in range(len(self.rollout_engines))]

        # The engines resharding with their own loaders absorb any per-engine
        # layout differences, so one HF bucket serves every layout.
        if engine_parallel_configs is not None:
            logger.info("p2p weight update: engines reshard staged buckets with their own loaders")

        # One staging slot per worker: the per-engine write thread serializes
        # write -> load, so a second slot is never used.
        staging_nbytes = self.args.update_weight_buffer_size
        if self._driver_rank() == 0:
            for engine in self.rollout_engines:
                _engine_call(
                    engine,
                    "init_weight_transfer_engine",
                    {"init_info": {"staging_buffer_size_bytes": staging_nbytes}},
                )
        _gloo_barrier()

        rank, size = self._rank_and_size()
        self._assignments = assign_rollout_engine_ranks_for_data_replica(rank, size, engine_gpu_counts)
        if not self._assignments:
            return  # this trainer rank sends nothing

        if self._transport is None:
            self._transport = self._transport_factory()

        def query_transfer_info(engine_ind: int, engine_rank: int) -> dict | None:
            target = self._targets[engine_ind]
            if target.is_errored:
                return None

            # Control-plane calls (not data writes) may be retried on transient
            # transport errors (miles #4402 lineage); data writes never are.
            def query() -> dict | None:
                # collective_rpc fans out to every worker of the engine; index by
                # rank. Non-node-0 actors answer [] (they serve no HTTP): skip.
                infos = _engine_call(self.rollout_engines[engine_ind], "get_p2p_transfer_info")
                if not infos or engine_rank >= len(infos):
                    return None
                return infos[engine_rank]

            from vime.utils.retry_utils import retry_until_deadline_sync

            return retry_until_deadline_sync(
                query,
                total_seconds=self.args.p2p_transfer_timeout,
                retry_if=_is_transient_control_error,
            )

        engine_cell_ids = [f"engine-{i}" for i in range(len(self.rollout_engines))]
        self._shards_by_rank = self._transport.connect(
            query_transfer_info, engine_cell_ids, self._assignments
        )
        self._buffers = TransferBuffers(
            2,  # double-buffered: pack the next bucket while the previous writes

            self.args.update_weight_buffer_size,
            device=self._transport.transfer_buffer_device,
            register_memory=self._transport.register_memory,
            unregister_memory=self._transport.unregister_memory,
            write_timeout=self.args.p2p_transfer_timeout,
        )

    def disconnect_rollout_engines(self) -> None:
        """Tear down the current fleet connection. Idempotent.

        A transport with uncertain writes can never unregister its memory, so
        its pinned buffers are abandoned (logged) rather than released; the
        transport itself is always closed (per-engine write threads) and dropped.
        """
        if self._buffers is not None:
            if self._transport is not None and self._transport.has_uncertain_writes:
                logger.warning(
                    "P2P transport has writes with unknown completion; abandoning "
                    "its pinned transfer buffers (they cannot be unregistered)"
                )
            else:
                self._buffers.release()
            self._buffers = None
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        if self.rollout_engines and self._driver_rank() == 0:
            for engine, target in zip(self.rollout_engines, self._targets, strict=True):
                if target.is_errored:
                    continue  # already shut down when it errored
                try:
                    _engine_call(engine, "destroy_weights_update_group", None)
                except Exception:
                    # Errored engines may be dead Ray actors by the time recover()
                    # reconnects; a stale engine's teardown must never block it.
                    logger.warning(
                        "could not tear down weight transfer on engine %d", target.engine_index, exc_info=True
                    )

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    # ---- update -------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        flush_cache = should_flush_cache(
            self.args.flush_cache_interval, self.weight_version, getattr(self.args, "update_weight_start_version", 0)
        )

        if self._driver_rank() == 0:
            for engine in self._healthy_engines():
                _engine_call(engine, "pause_generation", mode="abort" if flush_cache else "in_place")
            if flush_cache:
                for engine in self._healthy_engines():
                    _engine_call(engine, "flush_cache")
        _gloo_barrier()

        # Target session, then (mtp only) a draft session over the same buckets —
        # mirrors update_weight_from_distributed.py:113-123 (for mtp the source
        # stays the target weights; the engine swaps target at start_draft).
        self._run_session("start_weight_update")
        if self._mtp_draft_enabled():
            self._run_session("start_draft_weight_update")

        if self._driver_rank() == 0:
            for engine in self._healthy_engines():
                _engine_call(engine, "continue_generation")
        _gloo_barrier()

        errored = [t for t in self._targets if t.errored_at_version == self.weight_version]
        if errored and self._driver_rank() == 0:
            # Retire errored engines BEFORE the fail/continue decision so a
            # retired engine never serves stale weights: VLLMEngine.shutdown
            # unregisters the worker from the router and kills the server.
            for target in errored:
                try:
                    _engine_call(self.rollout_engines[target.engine_index], "shutdown")
                except Exception:
                    logger.exception("could not retire errored engine %d", target.engine_index)
        healthy = [t for t in self._targets if not t.is_errored]
        self.update_weight_metrics["p2p/errored_engines"] = sum(t.is_errored for t in self._targets)
        self.update_weight_metrics["p2p/healthy_engines"] = len(healthy)
        if not errored:
            return
        if self.args.p2p_on_engine_error == "continue":
            # The errored engine stays quarantined until the health monitor
            # retires it and recover() replaces + reconnects it; healthy engines
            # keep serving. Only dropping below the healthy floor fails.
            indices = [t.engine_index for t in errored]
            if len(healthy) < self.args.p2p_min_healthy_engines:
                raise RuntimeError(
                    f"P2P weight update: engines {indices} errored at version {self.weight_version}; "
                    f"healthy {len(healthy)} < --p2p-min-healthy-engines "
                    f"{self.args.p2p_min_healthy_engines}"
                )
            logger.warning(
                "P2P weight update: engines %s errored at version %d; continuing with %d healthy engines",
                indices,
                self.weight_version,
                len(healthy),
            )
            return
        raise RuntimeError(
            f"P2P weight update failed on engines {[t.engine_index for t in errored]}; "
            "they were not resumed (weights may be partially written). "
            f"First error: {errored[0].errored!r}"
        )

    def _mtp_draft_enabled(self) -> bool:
        """Same trigger as the distributed path's draft pass (mtp only; dspark is
        rejected at argument validation)."""
        return bool(
            self.args.enable_mtp_training
            and (self.args.vllm_speculative_config or {}).get("method") == "mtp"
        )

    def _healthy_engines(self) -> list:
        return [e for e, t in zip(self.rollout_engines, self._targets, strict=True) if not t.is_errored]

    def _run_session(self, start_method: str) -> None:
        """One weight-update session: start -> all buckets -> finish(version).

        Called once per version (target only) or twice (target, then draft, when
        mtp speculative decoding is active); the version does not change between
        the two sessions.

        start_method is "start_weight_update" for the target model or
        "start_draft_weight_update" for the draft (the worker swaps its update
        target and finish_weight_update resets it). weight_version is unchanged
        across the two sessions. A control-plane failure here (e.g. a stale
        engine without draft support) is not transient: the engine is quarantined.
        """
        if self._driver_rank() == 0:
            for engine, target in zip(self.rollout_engines, self._targets, strict=True):
                if target.is_errored:
                    continue
                try:
                    _engine_call(engine, start_method)
                except Exception as error:
                    target.mark_errored(error, self.weight_version)
        _quarantine_errored_engines(self._targets, self.weight_version)
        _gloo_barrier()

        for bucket in self._iterator.get_hf_weight_chunks(self._weights_getter()):
            self._send_bucket(dict(bucket))
        if self._buffers is not None:
            self._buffers.wait_for_all_writes()
        _quarantine_errored_engines(self._targets, self.weight_version)
        _gloo_barrier()

        if self._driver_rank() == 0:
            for engine, target in zip(self.rollout_engines, self._targets, strict=True):
                if target.is_errored:
                    continue
                _engine_call(engine, "finish_weight_update", weight_version=str(self.weight_version))
        _gloo_barrier()

    def _send_bucket(self, bucket: dict[str, torch.Tensor]) -> None:
        if self._buffers is None:
            return
        checksums = (
            {name: hash_tensor_sha256(tensor) for name, tensor in bucket.items()}
            if self.args.check_weight_transfer_checksum
            else None
        )
        shards = [
            shard
            for assignment in self._assignments
            for shard in self._shards_by_rank.get(assignment.rollout_engine_rank, [])
            if not self._targets[shard.rollout_engine_ind].is_errored
        ]
        if not shards:
            return
        buffer = self._buffers.acquire()
        # The bucket bytes are identical for every target (engines reshard with
        # their own loaders), so pack once and fan the writes out.
        manifest = pack_bucket_into_transfer_buffer(buffer, bucket)
        for entry in manifest:
            tensor = bucket[entry["name"]]
            entry["dtype"] = str(tensor.dtype).split(".")[-1]
            entry["shape"] = list(tensor.shape)
        total_nbytes = manifest[-1]["offset"] + manifest[-1]["nbytes"] if manifest else 0
        # run_write executes inside the transport's per-engine write thread, so
        # the load RPC is issued strictly after THIS engine's RDMA write landed.
        pending = self._transport.write(
            shards,
            buffer,
            total_nbytes,
            run_write=lambda shard, write, m=manifest: self._run_write_then_load(
                shard, write, m, checksums
            ),
        )
        self._buffers.release_after(
            buffer,
            [
                self._pending_write(future, shard)
                for future, shard in zip(pending, shards, strict=True)
            ],
        )

    def _pending_write(self, future, shard: RemoteShard):
        from .p2p import PendingWrite

        return PendingWrite(future=future, target=self._targets[shard.rollout_engine_ind])

    def _run_write_then_load(
        self,
        shard: RemoteShard,
        write: Callable[[], None],
        manifest: list[dict],
        checksums: dict[str, str] | None,
    ) -> None:
        target = self._targets[shard.rollout_engine_ind]
        try:
            write()
            self._load_on_engine(shard, manifest, checksums)
        except Exception as error:
            target.mark_errored(error, self.weight_version)
            raise

    def _load_on_engine(
        self, shard: RemoteShard, manifest: list[dict], checksums: dict[str, str] | None
    ) -> None:
        update_info = {
            "manifest": manifest,
            "checksums": checksums or {},
            "target_rank": shard.rollout_engine_rank,
        }
        received = _engine_call(
            self.rollout_engines[shard.rollout_engine_ind], "load_weights_from_staged", update_info
        )
        if isinstance(received, list):
            if not received:
                return  # non-node-0 actor: no HTTP, nothing to verify
            received = received[shard.rollout_engine_rank]
        if checksums is not None and received != checksums:
            raise RuntimeError(
                f"engine {shard.rollout_engine_ind} rank {shard.rollout_engine_rank} returned "
                "checksums that differ from the sender's"
            )


def _quarantine_errored_engines(targets: list[_EngineTarget], weight_version: int) -> None:
    """All trainer ranks quarantine the same engines (gloo all-gather of errored indices).

    miles _mark_cells_errored_on_any_rank equivalent: without this, one sender rank could
    observe a failure its peers did not, and the ranks would disagree about which engines
    are safe to keep serving.
    """
    locally_errored = {t.engine_index for t in targets if t.errored_at_version == weight_version}
    for engine_index in _union_over_gloo(locally_errored) - locally_errored:
        targets[engine_index].mark_errored(
            RuntimeError("engine errored on another trainer rank"), weight_version
        )


def _union_over_gloo(local: set[int]) -> set[int]:
    if not dist.is_initialized():
        return local
    gathered: list[set[int]] = [set() for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local, group=get_gloo_group())
    return set().union(local, *gathered)


def _is_transient_control_error(error: BaseException) -> bool:
    import json as _json

    import requests

    if isinstance(error, _json.JSONDecodeError) and not error.doc:
        return True
    return isinstance(error, (requests.ConnectionError, requests.Timeout))


def _gloo_barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())

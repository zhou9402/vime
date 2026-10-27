"""Mooncake RDMA transport for P2P weight updates (Design A: replica-free).

Adapted from miles' protocols/transports/mooncake.py. Where miles writes into
per-parameter addresses the engine publishes (requiring a trainer-side model
replica to reproduce the sharded layout), vime writes one packed HF-layout
bucket into a single engine-published staging buffer; the engine reshards with
its own loader afterwards (see docs/en/advanced/p2p-weight-transfer.md).

Weight writes are never retried: the weight-update session is no-retry by
design (miles' #4402 retry hardening covers Mooncake object-store ops only).

Mooncake is an optional dependency (``pip install vime[p2p]``); it is imported
lazily so this module loads without it.
"""

import logging
import os
import socket
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from .rollout_engine_rank_assignment import RolloutEngineRankAssignment

# Trainer transfer buffers are double-buffered: the next bucket is packed while
# the previous one is still being written. The engine staging buffer is NOT:
# the per-engine write thread serialises write -> load, so one slot suffices.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteShard:
    """Mooncake session and published staging buffer of one engine rank."""

    cell_id: str
    rollout_engine_ind: int
    rollout_engine_rank: int
    session_id: str
    staging_address: int
    capacity_nbytes: int


def pack_bucket_into_transfer_buffer(buffer: torch.Tensor, bucket: dict[str, torch.Tensor]) -> list[dict]:
    """Packs an HF-layout bucket contiguously into a transfer buffer; returns the manifest.

    The manifest (name/offset/nbytes per tensor) travels with the write so the engine's
    `load_weights_from_staged` can reshard the staged bytes with its own loader.
    """
    manifest = []
    offset = 0
    for name, tensor in bucket.items():
        nbytes = tensor.numel() * tensor.element_size()
        if offset + nbytes > buffer.numel():
            raise RuntimeError(f"bucket exceeds the transfer buffer at {name}")
        buffer[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape).copy_(tensor)
        manifest.append({"name": name, "offset": offset, "nbytes": nbytes})
        offset += nbytes
    return manifest


class MooncakeTransport:
    """Writes packed weight buckets from this trainer process's registered memory into rollout engines.

    The p2p updater keeps one for the whole trainer process: it registers its transfer buffers once, calls
    `connect` with each new set of rollout engines, then `write` for each rollout engine rank. Each rollout
    engine has its own write thread, so a stuck engine holds up only its own writes.

    Ordering contract: `connect` must only be called after every write issued through this transport has
    completed (e.g. after `TransferBuffers.wait_for_all_writes`); it raises while writes are in flight,
    because swapping executors must not strand native writes that still read a registered buffer.
    """

    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self, transfer_engine: Any | None = None) -> None:
        self._transfer_engine = transfer_engine if transfer_engine is not None else _create_transfer_engine()
        self._registered_tensors: dict[int, torch.Tensor] = {}
        self._has_uncertain_writes = False
        self._write_executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}
        self._in_flight_lock = threading.Lock()
        self._writes_in_flight = 0

    def connect(
        self,
        query_transfer_info: Callable[[int, int], dict[str, Any] | None],
        engine_cell_ids: Sequence[str],
        assignments: Sequence[RolloutEngineRankAssignment],
    ) -> dict[int, list[RemoteShard]]:
        """Queries each assigned engine rank's staging buffer; returns shards by rollout rank.

        `query_transfer_info(rollout_engine_ind, rollout_engine_rank)` wraps the engine's
        `get_p2p_transfer_info` RPC (F01-P1) and returns None for an engine that is being
        skipped. Starts new write threads: a rollout engine index now names a new engine,
        which must not queue behind the writes of the one it replaced.

        Must not be called while writes are in flight (see the class docstring).
        """
        with self._in_flight_lock:
            if self._writes_in_flight:
                raise RuntimeError(
                    "MooncakeTransport.connect() called with writes in flight; "
                    "wait for all writes (e.g. TransferBuffers.wait_for_all_writes) before reconnecting"
                )
        for write_executor in self._write_executors_by_rollout_engine_ind.values():
            write_executor.shutdown(wait=False, cancel_futures=True)
        self._write_executors_by_rollout_engine_ind = {}
        shards_by_rank: dict[int, list[RemoteShard]] = {}
        for assignment in assignments:
            shards = []
            for rollout_engine_ind in assignment.rollout_engine_indices:
                info = query_transfer_info(rollout_engine_ind, assignment.rollout_engine_rank)
                if info is None:
                    continue
                session_id = info.get("session_id")
                if session_id is None:
                    raise RuntimeError(
                        f"rollout engine {rollout_engine_ind} rank {assignment.rollout_engine_rank} "
                        "has no Mooncake session"
                    )
                shards.append(
                    RemoteShard(
                        cell_id=engine_cell_ids[rollout_engine_ind],
                        rollout_engine_ind=rollout_engine_ind,
                        rollout_engine_rank=assignment.rollout_engine_rank,
                        session_id=session_id,
                        staging_address=info["staging_address"],
                        capacity_nbytes=info["capacity_nbytes"],
                    )
                )
            shards_by_rank[assignment.rollout_engine_rank] = shards
        return shards_by_rank

    def register_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.register_memory(tensor.data_ptr(), _nbytes(tensor))
        if ret != 0:
            raise RuntimeError(
                f"Mooncake could not register {_nbytes(tensor)} bytes at {tensor.data_ptr():#x}: error {ret}"
            )

        self._registered_tensors[tensor.data_ptr()] = tensor

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        if self._has_uncertain_writes:
            raise RuntimeError(
                "Cannot release registered memory after a Mooncake write with unknown native completion"
            )
        ret = self._transfer_engine.unregister_memory(tensor.data_ptr())
        if ret != 0:
            raise RuntimeError(f"Mooncake could not unregister the memory at {tensor.data_ptr():#x}: error {ret}")
        del self._registered_tensors[tensor.data_ptr()]

    def write(
        self,
        remote_shards: Sequence[RemoteShard],
        bucket: torch.Tensor,
        bucket_nbytes: int,
        *,
        run_write: Callable[[RemoteShard, Callable[[], None]], None],
    ) -> list[Future[None]]:
        """Writes the first `bucket_nbytes` of `bucket` into every shard's staging buffer; one future per shard.

        Allocations must stay registered while native writes may still read them. Healthy receivers require
        unchanged bytes until their futures finish. Raises before writing anything if a shard's staging
        buffer is too small for the bucket.
        """
        for remote_shard in remote_shards:
            _assert_bucket_fits(remote_shard, bucket_nbytes)
        futures = []
        for remote_shard in remote_shards:
            with self._in_flight_lock:
                self._writes_in_flight += 1
            future = self._write_executor(remote_shard.rollout_engine_ind).submit(
                run_write, remote_shard, lambda shard=remote_shard: self._write_shard(shard, bucket, bucket_nbytes)
            )
            future.add_done_callback(self._write_finished)
            futures.append(future)
        return futures

    def close(self) -> None:
        """Shuts down the per-engine write threads. Call after the transfer buffers are released."""
        for write_executor in self._write_executors_by_rollout_engine_ind.values():
            write_executor.shutdown(wait=True)
        self._write_executors_by_rollout_engine_ind = {}

    def _write_finished(self, future: Future[None]) -> None:
        with self._in_flight_lock:
            self._writes_in_flight -= 1

    def _write_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._write_executors_by_rollout_engine_ind:
            self._write_executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mooncake-write-rollout-engine-{rollout_engine_ind}"
            )
        return self._write_executors_by_rollout_engine_ind[rollout_engine_ind]

    def _write_shard(self, remote_shard: RemoteShard, bucket: torch.Tensor, bucket_nbytes: int) -> None:
        try:
            ret = self._transfer_engine.batch_transfer_sync_write(
                remote_shard.session_id,
                [bucket.data_ptr()],
                [remote_shard.staging_address],
                [bucket_nbytes],
            )
            if ret < 0:
                raise RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")
        except Exception:
            self._has_uncertain_writes = True
            logger.exception("P2P transport write failed for session %s", remote_shard.session_id)
            raise


def _create_transfer_engine() -> Any:
    try:
        from mooncake.engine import TransferEngine
    except ImportError as error:
        raise ImportError(
            "P2P weight transfer requires the Mooncake transfer engine; "
            "install it with `pip install vime[p2p]` (mooncake-transfer-engine)"
        ) from error

    protocol = os.environ.get("MOONCAKE_PROTOCOL", "rdma")
    device = os.environ.get("MOONCAKE_DEVICE", "")
    transfer_engine = TransferEngine()
    ret = transfer_engine.initialize(_node_ip_address(), "P2PHANDSHAKE", protocol, device)
    if ret != 0:
        raise RuntimeError(
            f"Mooncake TransferEngine.initialize failed with error {ret} "
            f"(protocol={protocol!r}, device={device!r}); check MOONCAKE_PROTOCOL/MOONCAKE_DEVICE"
        )
    return transfer_engine


def _node_ip_address() -> str:
    try:
        import ray._private.services

        return ray._private.services.get_node_ip_address()
    except ImportError:
        return socket.gethostbyname(socket.gethostname())


def _assert_bucket_fits(remote_shard: RemoteShard, bucket_nbytes: int) -> None:
    target = f"rollout engine {remote_shard.rollout_engine_ind} rank {remote_shard.rollout_engine_rank}"
    if bucket_nbytes > remote_shard.capacity_nbytes:
        raise RuntimeError(
            f"the bucket is {bucket_nbytes} bytes but the staging buffer of {target} holds "
            f"{remote_shard.capacity_nbytes}; writing it would run past the target"
        )


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()

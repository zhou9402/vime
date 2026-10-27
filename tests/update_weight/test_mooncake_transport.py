"""Unit tests for the P2P trainer-side transport package (F01-P2).

Covers MooncakeTransport against an in-process fake TransferEngine (no Mooncake
install, no GPU): bucket-fit assertions, per-engine write-thread isolation,
the uncertain-writes memory guard, the transient-error classifier, plus the
transfer-buffer pipelining and the rank-assignment planner.
"""

import concurrent.futures
import ctypes
import threading
import time

import pytest
import torch

from vime.backends.megatron_utils.update_weight.p2p import (
    MooncakeTransport,
    RemoteShard,
    RolloutEngineRankAssignment,
    TransferBuffers,
    assign_rollout_engine_ranks_for_data_replica,
    pack_bucket_into_transfer_buffer,
)
from vime.backends.megatron_utils.update_weight.p2p.mooncake_transport import _create_transfer_engine

NUM_GPUS = 0


class FakeTransferEngine:
    def __init__(self):
        self.registered = {}
        self.writes = 0
        self.fail_with: int | None = None

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr: int) -> int:
        del self.registered[ptr]
        return 0

    def batch_transfer_sync_write(self, session_id, src_ptrs, dst_addrs, nbytes_list) -> int:
        self.writes += 1
        if self.fail_with is not None:
            return self.fail_with
        for src, dst, nbytes in zip(src_ptrs, dst_addrs, nbytes_list, strict=True):
            ctypes.memmove(dst, src, nbytes)
        return 0


class BlockingTransferEngine(FakeTransferEngine):
    """Engine 0 blocks until released; used to prove per-engine thread isolation."""

    def __init__(self, blocking_session: str, release: threading.Event):
        super().__init__()
        self.blocking_session = blocking_session
        self.release = release

    def batch_transfer_sync_write(self, session_id, src_ptrs, dst_addrs, nbytes_list) -> int:
        if session_id == self.blocking_session:
            self.release.wait(timeout=5)
        return super().batch_transfer_sync_write(session_id, src_ptrs, dst_addrs, nbytes_list)


def _shard(rollout_engine_ind: int, rank: int, capacity: int = 1 << 20, session: str | None = None) -> RemoteShard:
    staging = torch.empty(capacity, dtype=torch.uint8)
    # keep the tensor alive via the caller's registry
    _STAGING_KEEPALIVE.append(staging)
    return RemoteShard(
        cell_id=f"cell-{rollout_engine_ind}",
        rollout_engine_ind=rollout_engine_ind,
        rollout_engine_rank=rank,
        session_id=session or f"session-{rollout_engine_ind}",
        staging_address=staging.data_ptr(),
        capacity_nbytes=capacity,
    )


_STAGING_KEEPALIVE: list[torch.Tensor] = []


def _transport(fake_engine=None) -> MooncakeTransport:
    return MooncakeTransport(transfer_engine=fake_engine or FakeTransferEngine())


def _run_write(shard: RemoteShard, write) -> None:
    write()


# --- MooncakeTransport.connect -------------------------------------------------


def test_connect_queries_assigned_shards_and_skips_missing_engines():
    transport = _transport()
    assignments = [
        RolloutEngineRankAssignment(rollout_engine_rank=0, rollout_engine_indices=(0, 2)),
        RolloutEngineRankAssignment(rollout_engine_rank=1, rollout_engine_indices=(1,)),
    ]

    def query(engine_ind: int, rank: int):
        if engine_ind == 2:
            return None  # engine being skipped
        return {"session_id": f"s-{engine_ind}-{rank}", "staging_address": 0x1000, "capacity_nbytes": 128}

    shards = transport.connect(query, ["c0", "c1", "c2"], assignments)
    assert [s.session_id for s in shards[0]] == ["s-0-0"]
    assert [s.session_id for s in shards[1]] == ["s-1-1"]
    assert shards[0][0].cell_id == "c0"


def test_connect_asserts_session_id_present():
    transport = _transport()
    assignments = [RolloutEngineRankAssignment(rollout_engine_rank=0, rollout_engine_indices=(0,))]
    with pytest.raises(RuntimeError, match="no Mooncake session"):
        transport.connect(
            lambda ind, rank: {"session_id": None, "staging_address": 0, "capacity_nbytes": 1},
            ["c0"],
            assignments,
        )


# --- MooncakeTransport.write ----------------------------------------------------


def test_write_copies_bucket_into_staging_buffer():
    fake = FakeTransferEngine()
    transport = _transport(fake)
    bucket = torch.arange(256, dtype=torch.uint8)
    transport.register_memory(bucket)
    staging = torch.empty(512, dtype=torch.uint8)
    shard = RemoteShard("c0", 0, 0, "s0", staging.data_ptr(), staging.numel())

    futures = transport.write([shard], bucket, bucket.numel(), run_write=_run_write)
    for future in futures:
        future.result(timeout=5)
    assert torch.equal(staging[:256], bucket)


def test_write_rejects_oversized_bucket_before_writing():
    fake = FakeTransferEngine()
    transport = _transport(fake)
    bucket = torch.empty(4096, dtype=torch.uint8)
    shard = _shard(0, 0, capacity=1024)

    with pytest.raises(RuntimeError, match="run past the target"):
        transport.write([shard], bucket, bucket.numel(), run_write=_run_write)
    assert fake.writes == 0


def test_one_stuck_engine_does_not_block_others():
    release = threading.Event()
    blocking = BlockingTransferEngine("s-stuck", release)
    transport = _transport(blocking)
    bucket = torch.zeros(64, dtype=torch.uint8)
    transport.register_memory(bucket)
    stuck = _shard(0, 0, session="s-stuck")
    fast = _shard(1, 0, session="s-fast")

    try:
        futures = transport.write([stuck, fast], bucket, bucket.numel(), run_write=_run_write)
        futures[1].result(timeout=5)  # fast engine completes while engine 0 is stuck
        assert not futures[0].done()
    finally:
        release.set()
    futures[0].result(timeout=5)


def test_failed_write_marks_uncertain_and_blocks_unregister():
    fake = FakeTransferEngine()
    transport = _transport(fake)
    buffer = torch.empty(64, dtype=torch.uint8)
    transport.register_memory(buffer)
    fake.fail_with = -1
    shard = _shard(0, 0)

    futures = transport.write([shard], buffer, buffer.numel(), run_write=_run_write)
    with pytest.raises(RuntimeError, match="returned -1"):
        futures[0].result(timeout=5)
    with pytest.raises(RuntimeError, match="unknown native completion"):
        transport.unregister_memory(buffer)


def test_register_failure_raises():
    class RefusingEngine(FakeTransferEngine):
        def register_memory(self, ptr, nbytes):
            return -1

    transport = _transport(RefusingEngine())
    with pytest.raises(RuntimeError, match="could not register"):
        transport.register_memory(torch.empty(8, dtype=torch.uint8))


def test_lazy_mooncake_import_error_is_actionable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "mooncake", None)
    monkeypatch.setitem(__import__("sys").modules, "mooncake.engine", None)
    with pytest.raises(ImportError, match="pip install vime\\[p2p\\]"):
        _create_transfer_engine()


# --- connect ordering -------------------------------------------------------------


def test_connect_rejected_while_writes_in_flight():
    release = threading.Event()
    blocking = BlockingTransferEngine("s-stuck", release)
    transport = _transport(blocking)
    bucket = torch.zeros(64, dtype=torch.uint8)
    transport.register_memory(bucket)
    futures = transport.write([_shard(0, 0, session="s-stuck")], bucket, bucket.numel(), run_write=_run_write)
    try:
        with pytest.raises(RuntimeError, match="writes in flight"):
            transport.connect(lambda ind, rank: None, ["c0"], [])
    finally:
        release.set()
    for future in futures:
        future.result(timeout=5)
    # once every write finished, reconnecting is allowed again
    transport.connect(lambda ind, rank: None, ["c0"], [])


def test_close_shuts_down_write_executors():
    transport = _transport()
    bucket = torch.zeros(64, dtype=torch.uint8)
    transport.register_memory(bucket)
    futures = transport.write([_shard(0, 0)], bucket, bucket.numel(), run_write=_run_write)
    for future in futures:
        future.result(timeout=5)
    transport.close()
    assert transport._write_executors_by_rollout_engine_ind == {}


# --- rank assignment planner ------------------------------------------------------


def test_assignment_covers_every_target_exactly_once():
    engine_gpu_counts = [4, 4, 2]
    data_replica_size = 4
    seen = []
    for rank in range(data_replica_size):
        for assignment in assign_rollout_engine_ranks_for_data_replica(rank, data_replica_size, engine_gpu_counts):
            for engine_ind in assignment.rollout_engine_indices:
                seen.append((engine_ind, assignment.rollout_engine_rank))
    expected = [(ind, rank) for ind, count in enumerate(engine_gpu_counts) for rank in range(count)]
    assert sorted(seen) == sorted(expected)


def test_assignment_reuses_senders_per_engine_rank():
    # 2 data replicas, 3 engines of 1 GPU: rank 0 is prepared once and sent to
    # several engines; each replica sends a disjoint engine set.
    first = assign_rollout_engine_ranks_for_data_replica(0, 2, [1, 1, 1])
    second = assign_rollout_engine_ranks_for_data_replica(1, 2, [1, 1, 1])
    assert all(a.rollout_engine_rank == 0 for a in first + second)
    assert sorted(i for a in first for i in a.rollout_engine_indices) == [0, 2]
    assert sorted(i for a in second for i in a.rollout_engine_indices) == [1]


def test_pack_bucket_into_transfer_buffer():
    bucket = {
        "a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "b": torch.arange(4, dtype=torch.uint8),
    }
    total = sum(t.numel() * t.element_size() for t in bucket.values())
    buffer = torch.empty(total, dtype=torch.uint8)
    manifest = pack_bucket_into_transfer_buffer(buffer, bucket)
    assert manifest == [
        {"name": "a", "offset": 0, "nbytes": 24},
        {"name": "b", "offset": 24, "nbytes": 4},
    ]
    assert torch.equal(buffer[:24].view(torch.float32).reshape(2, 3), bucket["a"])
    assert torch.equal(buffer[24:], bucket["b"])
    with pytest.raises(RuntimeError, match="transfer buffer"):
        pack_bucket_into_transfer_buffer(torch.empty(8, dtype=torch.uint8), bucket)


def test_assignment_rejects_zero_gpu_engine():
    with pytest.raises(AssertionError, match="at least one"):
        assign_rollout_engine_ranks_for_data_replica(0, 1, [0])


# --- TransferBuffers ---------------------------------------------------------------


class _Target:
    def __init__(self):
        self.errored: BaseException | None = None

    @property
    def is_errored(self) -> bool:
        return self.errored is not None

    def mark_errored(self, error: BaseException) -> None:
        self.errored = error


def _make_buffers(num_buffers=2, nbytes=128):
    registered, unregistered = [], []
    buffers = TransferBuffers(
        num_buffers,
        nbytes,
        device=torch.device("cpu"),
        register_memory=registered.append,
        unregister_memory=unregistered.append,
        write_timeout=5.0,
    )
    return buffers, registered, unregistered


def _pending_write(fn, target) -> "object":
    from vime.backends.megatron_utils.update_weight.p2p import PendingWrite

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return PendingWrite(future=executor.submit(fn), target=target)


def test_buffers_round_robin_and_register_on_init():
    buffers, registered, _ = _make_buffers(num_buffers=2)
    assert len(registered) == 2
    first, second, third = buffers.acquire(), buffers.acquire(), buffers.acquire()
    assert first.data_ptr() == third.data_ptr() != second.data_ptr()


def test_buffers_wait_for_pending_write_before_reuse():
    buffers, _, _ = _make_buffers(num_buffers=1)
    buffer = buffers.acquire()
    gate = threading.Event()
    target = _Target()
    buffers.release_after(buffer, [_pending_write(lambda: gate.wait(timeout=5), target)])

    acquired = threading.Event()

    def second_acquire():
        buffers.acquire()
        acquired.set()

    thread = threading.Thread(target=second_acquire)
    thread.start()
    assert not acquired.wait(timeout=0.2), "buffer reused while a write still reads it"
    gate.set()
    thread.join(timeout=5)
    assert acquired.is_set()


def test_buffers_mark_failed_targets_and_release_unregisters():
    buffers, registered, unregistered = _make_buffers(num_buffers=1)
    buffer = buffers.acquire()
    target = _Target()
    buffers.release_after(buffer, [_pending_write(lambda: 1 / 0, target)])
    buffers.wait_for_all_writes()
    assert target.is_errored
    buffers.release()
    assert len(unregistered) == len(registered)


def test_release_refuses_while_writes_still_running():
    buffers, _, _ = _make_buffers(num_buffers=1, nbytes=8)
    buffer = buffers.acquire()
    never = threading.Event()
    buffers.release_after(buffer, [_pending_write(lambda: never.wait(timeout=10), _Target())])
    buffers._write_timeout = 0.05
    with pytest.raises(RuntimeError, match="still running"):
        buffers.release()
    never.set()

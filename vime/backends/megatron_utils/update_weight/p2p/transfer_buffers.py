"""Pipelined pinned host buffers for P2P weight sends.

Ported from miles' protocols/utils/transfer_buffers.py. The cell-updater
coupling is replaced by the ``WriteTarget`` protocol so this module stays
usable before the rollout-cell fault-tolerance layer (F01-P4) lands.
"""

import concurrent.futures
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

import torch


class WriteTarget(Protocol):
    """The per-write error-isolation handle the buffers wait on.

    F01-P3/P4's rollout-cell updater implements this; tests can use any object
    with the same shape.
    """

    @property
    def is_errored(self) -> bool: ...

    def mark_errored(self, error: BaseException) -> None: ...


@dataclass(frozen=True)
class PendingWrite:
    future: Future[None]
    target: WriteTarget


class TransferBuffers:
    """Memory a p2p sender loads weights into and a transport writes them to the rollout engines from.

    The protocol creates it once per process on the device the transport reads from and registers each buffer with
    the transport. For each load it calls `acquire`, loads into the buffer, starts the writes that read it, and hands
    them to `release_after`. A buffer is handed out again once healthy targets have finished reading it. Failed or
    timed-out targets stop participating, but their unfinished writes still keep the buffers registered until release.
    """

    def __init__(
        self,
        num_buffers: int,
        buffer_nbytes: int,
        *,
        device: torch.device,
        register_memory: Callable[[torch.Tensor], None],
        unregister_memory: Callable[[torch.Tensor], None],
        write_timeout: float,
    ) -> None:
        self.buffer_nbytes = buffer_nbytes
        self._unregister_memory = unregister_memory
        self._write_timeout = write_timeout
        self._buffers = [_allocate_transfer_buffer(buffer_nbytes, device) for _ in range(num_buffers)]
        for buffer in self._buffers:
            register_memory(buffer)
        self._pending_writes_by_buffer_index: list[list[PendingWrite]] = [[] for _ in self._buffers]
        self._next_buffer_index = 0

    def acquire(self) -> torch.Tensor:
        """Returns the next buffer once every healthy target has finished reading it."""
        buffer_index = self._next_buffer_index
        self._next_buffer_index = (buffer_index + 1) % len(self._buffers)
        self._wait_for_writes_reading([buffer_index])
        return self._buffers[buffer_index]

    def release_after(self, buffer: torch.Tensor, pending_writes: Sequence[PendingWrite]) -> None:
        """Records the writes and their connection-specific targets that read `buffer`."""
        (buffer_index,) = [index for index, candidate in enumerate(self._buffers) if candidate is buffer]
        self._pending_writes_by_buffer_index[buffer_index] += pending_writes

    def wait_for_all_writes(self) -> None:
        """Waits for healthy targets and marks targets with failed or timed-out writes."""
        self._wait_for_writes_reading(range(len(self._buffers)))

    def release(self) -> None:
        """Waits for every write still reading a buffer, then unregisters the buffers from the transport."""
        writes = [write.future for pending in self._pending_writes_by_buffer_index for write in pending]
        concurrent.futures.wait([write for write in writes if not write.done()], timeout=self._write_timeout)
        if any(not write.done() for write in writes):
            raise RuntimeError("Cannot release transfer buffers while Mooncake writes are still running")
        for buffer in self._buffers:
            self._unregister_memory(buffer)

    def _wait_for_writes_reading(self, buffer_indices: Sequence[int]) -> None:
        pending_writes = [write for index in buffer_indices for write in self._pending_writes_by_buffer_index[index]]
        try:
            _wait_for_healthy_targets(pending_writes, timeout=self._write_timeout)
        finally:
            for index in buffer_indices:
                self._pending_writes_by_buffer_index[index] = [
                    write for write in self._pending_writes_by_buffer_index[index] if not write.future.done()
                ]


def _wait_for_healthy_targets(pending_writes: list[PendingWrite], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for write in pending_writes:
        if write.target.is_errored:
            continue
        try:
            write.future.result(timeout=max(0, deadline - time.monotonic()))
        except Exception as error:
            write.target.mark_errored(error)


def _allocate_transfer_buffer(buffer_nbytes: int, device: torch.device) -> torch.Tensor:
    # host buffers are pinned in place: a pageable copy first would stay resident on aarch64 hosts
    return torch.empty(buffer_nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu")

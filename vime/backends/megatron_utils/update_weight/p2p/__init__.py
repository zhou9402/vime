"""Trainer-side transport for P2P/RDMA weight updates (Design A: replica-free).

The engine publishes ONE registered staging buffer per worker; the trainer
packs an HF-layout bucket into a pinned transfer buffer and RDMA-writes it into
the staging buffer, then the engine reshards with its own loader. No
trainer-side model replica. Disaggregated (non-colocate) deployments only.

Mooncake is an optional dependency (``pip install vime[p2p]``); it is imported
lazily inside ``MooncakeTransport`` so importing this package never requires it.
"""

from .mooncake_transport import MooncakeTransport, RemoteShard, pack_bucket_into_transfer_buffer
from .rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
    assign_rollout_engine_ranks_for_data_replica,
)
from .transfer_buffers import PendingWrite, TransferBuffers, WriteTarget

__all__ = [
    "MooncakeTransport",
    "PendingWrite",
    "RemoteShard",
    "RolloutEngineRankAssignment",
    "TransferBuffers",
    "WriteTarget",
    "assign_rollout_engine_ranks_for_data_replica",
    "pack_bucket_into_transfer_buffer",
]

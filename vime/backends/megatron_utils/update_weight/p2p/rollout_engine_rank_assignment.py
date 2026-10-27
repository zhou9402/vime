"""Which trainer data-replica rank sends weights to which rollout engine ranks.

Ported from miles' protocols/utils/rollout_engine_rank_assignment.py. Only the
pure planning function is ported; the miles wrapper takes miles' ParallelState
/ WeightUpdatePlacement, and F01-P3 wires the equivalent vime values in.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutEngineRankAssignment:
    """The weights of one rollout engine rank that a trainer rank sends, and the rollout engines it sends
    them to.

    These rollout engines have the same layout, so the trainer rank prepares the weights once for all of them.
    """

    rollout_engine_rank: int
    rollout_engine_indices: tuple[int, ...]


def assign_rollout_engine_ranks_for_data_replica(
    data_replica_rank: int, data_replica_size: int, engine_gpu_counts: Sequence[int]
) -> list[RolloutEngineRankAssignment]:
    """Returns the rollout engine ranks one trainer data-replica rank sends weights to in p2p weight updates.

    Data replicas are the trainer ranks that hold the same weights. Every (rollout engine, rollout engine
    rank) gets exactly one sender. First each data replica gets one, so all of them send in parallel; the
    rest go to a data replica already sending the same rollout engine rank, which prepares those weights
    once for several rollout engines. An empty list means this trainer rank sends nothing.
    """
    for rollout_engine_ind, gpu_count in enumerate(engine_gpu_counts):
        assert gpu_count > 0, f"rollout engine {rollout_engine_ind} has {gpu_count} GPUs; it needs at least one"
    targets = [
        (rollout_engine_ind, rollout_engine_rank)
        for rollout_engine_ind, gpu_count in enumerate(engine_gpu_counts)
        for rollout_engine_rank in range(gpu_count)
    ]
    rollout_engine_indices_by_data_replica = [defaultdict(list) for _ in range(data_replica_size)]
    for data_replica, (rollout_engine_ind, rollout_engine_rank) in zip(
        range(data_replica_size), targets, strict=False
    ):
        rollout_engine_indices_by_data_replica[data_replica][rollout_engine_rank].append(rollout_engine_ind)

    next_round_robin_data_replica = 0
    for rollout_engine_ind, rollout_engine_rank in targets[data_replica_size:]:
        rollout_engine_counts_by_data_replica = [
            len(rollout_engine_indices.get(rollout_engine_rank, ()))
            for rollout_engine_indices in rollout_engine_indices_by_data_replica
        ]
        if max(rollout_engine_counts_by_data_replica) > 0:
            _, data_replica = min(
                (rollout_engine_count, data_replica)
                for data_replica, rollout_engine_count in enumerate(rollout_engine_counts_by_data_replica)
                if rollout_engine_count > 0
            )
        else:
            data_replica = next_round_robin_data_replica % data_replica_size
            next_round_robin_data_replica += 1
        rollout_engine_indices_by_data_replica[data_replica][rollout_engine_rank].append(rollout_engine_ind)

    return [
        RolloutEngineRankAssignment(
            rollout_engine_rank=rollout_engine_rank, rollout_engine_indices=tuple(rollout_engine_indices)
        )
        for rollout_engine_rank, rollout_engine_indices in rollout_engine_indices_by_data_replica[
            data_replica_rank
        ].items()
    ]

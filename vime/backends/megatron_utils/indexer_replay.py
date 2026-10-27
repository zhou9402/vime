"""Trainer-side DSA indexer top-k replay fill (F07-8b).

Mirrors ``fill_routing_replay`` for the ``IndexerReplayManager`` in
``vime/utils/replay_base.py``: rollout-recorded ``rollout_indexer_topk``
payloads ([rows, num_streams, index_topk] int32 per sample, rows =
tokens[1:]) are padded/sliced to the local CP/SP token layout and registered
into per-indexer-layer replay queues.

Token-position rebase: unlike router expert ids, indexer indices are KV
*positions* within the sample, so when samples are packed (THD) each sample's
indices are offset by its packed start row (miles' fill_replay_data with
``indices_are_token_positions=True``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vime.utils.replay_base import indexer_replay_manager


def register_replay_list_sequential(replay_list, replay_data, **_kwargs):
    """Map replay streams to registered modules (from miles' replay/data.py).

    Each replay records ``replay_data[:, replay.stream_idx]`` if ``stream_idx``
    is set (sparse-layer layouts under PP/VPP), else 1:1 enumeration order.
    """
    for replay_idx, replay in enumerate(replay_list):
        stream_idx = replay.stream_idx if replay.stream_idx is not None else replay_idx
        if not 0 <= stream_idx < replay_data.shape[1]:
            raise AssertionError(
                f"replay stream_idx {stream_idx} out of range (replay_data has {replay_data.shape[1]} streams)"
            )
        replay.record(replay_data[:, stream_idx])


def _pad_rows(data: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return data
    pad_tensor = torch.full(
        (pad, *data.shape[1:]), fill_value=-1, device=data.device, dtype=data.dtype
    )
    return torch.cat([data, pad_tensor], dim=0)


def prepare_indexer_topk_for_replay(
    rollout_indexer_topk: list[torch.Tensor],
    tokens: list[torch.Tensor],
    *,
    data_pad_size_multiplier: int,
    sequence_parallel: bool,
    cp_size: int = 1,
    cp_rank: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> torch.Tensor:
    """Align indexer top-k rows with this rank's packed training tokens.

    Supports the common colocate layouts: CP=1, or zigzag CP chunking
    (non-allgather). allgather_cp is not supported yet (miles skips the
    token-position rebase there; punted until the layout is validated).
    """
    assert len(rollout_indexer_topk) == len(tokens) and len(tokens) > 0
    for values, token_ids in zip(rollout_indexer_topk, tokens, strict=True):
        assert values.shape[0] == token_ids.shape[0] - 1, f"{values.shape}, {token_ids.shape}"

    # One tail row per sample (the padded final token is masked from loss).
    padded = [_pad_rows(values, 1) for values in rollout_indexer_topk]

    offset = 0
    sliced = []
    for values in padded:
        token_len = values.shape[0]
        if cp_size == 1:
            local = values
        else:
            chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
            chunks = [
                (chunk_size * cp_rank, min(chunk_size * (cp_rank + 1), token_len)),
                (
                    chunk_size * (2 * cp_size - cp_rank - 1),
                    min(chunk_size * (2 * cp_size - cp_rank), token_len),
                ),
            ]
            local = torch.cat([values[start:end] for start, end in chunks if start < end], dim=0)
        # Rebase per-sample KV positions onto the packed training sequence.
        local = torch.where(local != -1, local + offset, local)
        offset += local.shape[0]
        sliced.append(local)

    replay_data = torch.cat(sliced, dim=0)
    pad_size = tp_size * data_pad_size_multiplier
    replay_data = _pad_rows(replay_data, (pad_size - replay_data.size(0) % pad_size) % pad_size)

    if sequence_parallel and tp_size > 1:
        seqlen = replay_data.size(0)
        assert seqlen % tp_size == 0
        replay_data = replay_data[seqlen // tp_size * tp_rank : seqlen // tp_size * (tp_rank + 1)]
    return replay_data


def fill_indexer_replay(args, data_iterator, num_microbatches, rollout_data) -> None:
    """Load rollout indexer top-k into the per-layer replay queues, in microbatch order."""
    data_key = indexer_replay_manager.data_key
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data when use_rollout_indexer_replay is set.")

    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    allgather_cp = getattr(args, "allgather_cp", False)
    if cp_size > 1 and allgather_cp:
        raise NotImplementedError("indexer replay with allgather_cp is not supported yet")
    tp_size = mpu.get_tensor_model_parallel_world_size()

    for iterator in data_iterator:
        iterator.reset()

    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens"])
        values = batch[data_key]
        tokens = batch["tokens"]
        if any(not torch.is_tensor(v) for v in values):
            raise NotImplementedError("disk-backed indexer replay (TensorRef) is not supported yet")
        replay_data = prepare_indexer_topk_for_replay(
            values,
            tokens,
            data_pad_size_multiplier=args.data_pad_size_multiplier,
            sequence_parallel=args.sequence_parallel,
            cp_size=cp_size,
            cp_rank=mpu.get_context_parallel_rank(),
            tp_size=tp_size,
            tp_rank=mpu.get_tensor_model_parallel_rank(),
        )
        register_replay_list_sequential(indexer_replay_manager.replays, replay_data)

    del rollout_data[data_key]
    for iterator in data_iterator:
        iterator.reset()


def set_stage(stage: str) -> None:
    """fallthrough | record | replay_forward | replay_backward."""
    indexer_replay_manager.stage = stage


def reset() -> None:
    indexer_replay_manager.clear_all()


def is_available() -> bool:
    return dist.is_available()

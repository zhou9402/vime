"""CPU tests for trainer-side indexer replay fill (F07-8b)."""

import argparse

import pytest
import torch

from vime.backends.megatron_utils.indexer_replay import (
    prepare_indexer_topk_for_replay,
    register_replay_list_sequential,
)
from vime.utils.replay_base import IndexerReplayManager


def _sample(rows, streams=2, topk=4, fill=0):
    # per-sample indexer top-k: values are positions within the sample
    base = torch.arange(rows * streams * topk, dtype=torch.int32).reshape(rows, streams, topk)
    return base % max(rows, 1) + fill


def test_prepare_rebases_token_positions_thd():
    # two packed samples: 3 and 2 tokens -> 2 and 1 replay rows (+1 tail pad each)
    values = [_sample(2), _sample(1)]
    tokens = [torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)]

    out = prepare_indexer_topk_for_replay(
        values, tokens, data_pad_size_multiplier=2, sequence_parallel=False
    )

    # rows: (2+1) + (1+1) = 5, padded to multiple of 2 (tp=1 * mult=2) -> 6
    assert out.shape == (6, 2, 4)
    # sample 0 rows: unchanged (offset 0), tail row -1
    assert (out[2] == -1).all()
    # sample 1 rows: rebased by its packed start (3)
    expected = torch.where(values[1][0] != -1, values[1][0] + 3, values[1][0])
    assert torch.equal(out[3], expected)
    assert (out[4] == -1).all() and (out[5] == -1).all()


def test_prepare_cp1_preserves_values_without_rebase_drift():
    values = [_sample(4)]
    tokens = [torch.zeros(5, dtype=torch.long)]
    out = prepare_indexer_topk_for_replay(
        values, tokens, data_pad_size_multiplier=4, sequence_parallel=False
    )
    assert torch.equal(out[:4], values[0])
    assert (out[4:] == -1).all()


def test_prepare_zigzag_cp2():
    rows = 8
    values = [_sample(rows)]
    tokens = [torch.zeros(rows + 1, dtype=torch.long)]
    out_r0 = prepare_indexer_topk_for_replay(
        values, tokens, data_pad_size_multiplier=1, sequence_parallel=False, cp_size=2, cp_rank=0
    )
    out_r1 = prepare_indexer_topk_for_replay(
        values, tokens, data_pad_size_multiplier=1, sequence_parallel=False, cp_size=2, cp_rank=1
    )
    # zigzag chunk formula (mirrors vime cp_utils): token_len=9, cp=2 -> chunk=3,
    # rank0 rows [0:3), rank1 rows [3:6)+[6:9); together they cover the sample.
    token_len = rows + 1
    chunk = (token_len + 2 * 2 - 1) // (2 * 2)
    r0 = values[0][0:chunk]
    r1 = torch.cat([values[0][chunk : 2 * chunk], values[0][2 * chunk : 3 * chunk]])
    # rebase offset is local to each rank's packed stream (single sample -> 0)
    assert torch.equal(out_r0[: r0.shape[0]], r0)
    assert torch.equal(out_r1[: r1.shape[0]], r1)
    assert (out_r0[r0.shape[0] :] == -1).all()
    assert (out_r1[r1.shape[0] :] == -1).all()


def test_register_sequential_stream_idx():
    manager = IndexerReplayManager()
    manager.enabled = True
    r0 = manager.create_replay(stream_idx=0)
    r2 = manager.create_replay(stream_idx=2)
    data = torch.arange(3 * 3 * 2, dtype=torch.int32).reshape(3, 3, 2)
    register_replay_list_sequential(manager.replays, data)
    assert torch.equal(r0.top_indices_list[0], data[:, 0])
    assert torch.equal(r2.top_indices_list[0], data[:, 2])


def test_register_sequential_out_of_range():
    manager = IndexerReplayManager()
    manager.enabled = True
    manager.create_replay(stream_idx=5)
    with pytest.raises(AssertionError, match="out of range"):
        register_replay_list_sequential(manager.replays, torch.zeros(2, 2, 2, dtype=torch.int32))


def test_fill_requires_data_key():
    from vime.backends.megatron_utils.indexer_replay import fill_indexer_replay

    args = argparse.Namespace()
    with pytest.raises(ValueError, match="rollout_indexer_topk is required"):
        fill_indexer_replay(args, [], [], {})


def test_fill_rejects_disk_backed():
    from vime.backends.megatron_utils.indexer_replay import fill_indexer_replay

    class FakeIterator:
        offset = 0
        micro_batch_indices = [[0]]

        def reset(self):
            pass

        def get_next(self, keys):
            return {"rollout_indexer_topk": ["tensorref-placeholder"], "tokens": [torch.zeros(3)]}

    args = argparse.Namespace(data_pad_size_multiplier=1, sequence_parallel=False, allgather_cp=False)
    pytest.importorskip("megatron.core")  # fill_indexer_replay imports mpu
    import vime.backends.megatron_utils.indexer_replay as ir

    class _FakeMpu:
        @staticmethod
        def get_context_parallel_world_size():
            return 1

        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 1

        @staticmethod
        def get_context_parallel_rank():
            return 0

        @staticmethod
        def get_tensor_model_parallel_rank():
            return 0

    import megatron.core

    orig = megatron.core.mpu
    megatron.core.mpu = _FakeMpu
    try:
        with pytest.raises(NotImplementedError, match="disk-backed"):
            fill_indexer_replay(args, [FakeIterator()], [1], {"rollout_indexer_topk": ["tensorref-placeholder"]})
    finally:
        megatron.core.mpu = orig

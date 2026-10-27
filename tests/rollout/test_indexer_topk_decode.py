"""CPU tests for the vLLM indexer-topk channel decode (F07-8a)."""

import argparse
import base64
import io

import numpy as np
import pytest
import torch

from vime.utils.types import Sample


def _wire(array: np.ndarray) -> np.ndarray:
    """Simulate the vLLM indexer-topk channel: base64 .npy on the choice,
    decoded to an ndarray by vllm_rollout before reaching Sample."""
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return np.load(io.BytesIO(base64.b64decode(base64.b64encode(buf.getvalue()))), allow_pickle=False)


def _args(streams=2, **kw):
    return argparse.Namespace(
        vllm_speculative_config=None,
        rollout_indexer_topk_num_streams=streams,
        use_rollout_indexer_replay=True,
        **kw,
    )


def _finish_meta(**extra):
    return {"finish_reason": {"type": "stop"}, **extra}


def test_indexer_topk_decode_shape_and_values():
    sample = Sample(tokens=[10, 11, 12, 13])  # rows = len(tokens)-1 = 3
    payload = np.arange(3 * 2 * 4, dtype=np.int32).reshape(3, 2, 4)
    sample.append_response_tokens(
        _args(),
        tokens=[],
        trainable=True,
        meta_info=_finish_meta(indexer_topk=_wire(payload)),
    )
    got = sample.materialize_rollout_indexer_topk()
    assert got.shape == (3, 2, 4)
    assert got.dtype == torch.int32
    assert torch.equal(got, torch.from_numpy(payload))


def test_indexer_topk_requires_num_streams():
    sample = Sample(tokens=[10, 11])
    payload = np.zeros((1, 2, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="rollout-indexer-topk-num-streams"):
        sample.append_response_tokens(
            _args(streams=None), tokens=[], trainable=True, meta_info=_finish_meta(indexer_topk=_wire(payload))
        )


def test_indexer_topk_row_mismatch_raises():
    sample = Sample(tokens=[10, 11, 12, 13])  # expects 3 rows
    payload = np.zeros((2, 2, 4), dtype=np.int32)  # 2 rows
    with pytest.raises(ValueError, match="shape does not match"):
        sample.append_response_tokens(
            _args(), tokens=[], trainable=True, meta_info=_finish_meta(indexer_topk=_wire(payload))
        )


def test_indexer_topk_continuation_appends():
    first = np.full((2, 2, 4), 7, dtype=np.int32)
    second = np.full((1, 2, 4), 9, dtype=np.int32)

    sample = Sample(tokens=[10, 11, 12])
    sample.append_response_tokens(
        _args(), tokens=[], trainable=True, meta_info=_finish_meta(indexer_topk=_wire(first))
    )
    assert sample.get_rollout_indexer_topk_length() == 2

    # continuation: one new token, start_len recorded by the rollout driver
    sample.append_response_tokens(
        _args(),
        tokens=[13],
        log_probs=[0.0],
        trainable=True,
        meta_info=_finish_meta(indexer_topk=_wire(second), indexer_topk_start_len=2),
    )
    got = sample.materialize_rollout_indexer_topk()
    assert got.shape == (3, 2, 4)
    assert (got[:2] == 7).all() and (got[2] == 9).all()


def test_indexer_topk_continuation_length_mismatch_raises():
    sample = Sample(tokens=[10, 11, 12])
    first = np.full((2, 2, 4), 7, dtype=np.int32)
    sample.append_response_tokens(
        _args(), tokens=[], trainable=True, meta_info=_finish_meta(indexer_topk=_wire(first))
    )
    bad = np.zeros((5, 2, 4), dtype=np.int32)  # wrong row count for start_len=2, tokens=4 -> 1 row
    with pytest.raises(ValueError, match="shape does not match"):
        sample.append_response_tokens(
            _args(),
            tokens=[13],
            log_probs=[0.0],
            trainable=True,
            meta_info=_finish_meta(indexer_topk=_wire(bad), indexer_topk_start_len=2),
        )


def test_indexer_topk_absent_is_noop():
    sample = Sample(tokens=[10, 11])
    sample.append_response_tokens(_args(), tokens=[], trainable=True, meta_info=_finish_meta())
    assert sample.rollout_indexer_topk is None
    assert sample.get_rollout_indexer_topk_length() == 0


def test_indexer_topk_accepts_base64_npy_string():
    sample = Sample(tokens=[10, 11, 12])
    payload = np.arange(2 * 2 * 4, dtype=np.int32).reshape(2, 2, 4)
    buf = io.BytesIO()
    np.save(buf, payload, allow_pickle=False)
    blob = base64.b64encode(buf.getvalue()).decode("ascii")
    sample.append_response_tokens(
        _args(), tokens=[], trainable=True, meta_info=_finish_meta(indexer_topk=blob)
    )
    assert torch.equal(sample.materialize_rollout_indexer_topk(), torch.from_numpy(payload))

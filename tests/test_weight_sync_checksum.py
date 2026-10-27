"""CPU unit tests for weight-sync checksums and compare evaluation (F02-2)."""

from __future__ import annotations

import sys
from pathlib import Path

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import pytest
import torch

from vime.backends.megatron_utils.update_weight.checksum_utils import (
    compute_send_checksums,
    hash_tensor_sha256,
    verify_transfer_checksums,
)
from vime.backends.vllm_utils.weight_checker import evaluate_compare_results

NUM_GPUS = 0


def test_hash_tensor_sha256_is_deterministic_and_content_sensitive():
    a = torch.arange(16, dtype=torch.float32)
    b = a.clone()
    assert hash_tensor_sha256(a) == hash_tensor_sha256(b)
    b.view(torch.uint8)[0] ^= 1
    assert hash_tensor_sha256(a) != hash_tensor_sha256(b)


def test_compute_send_checksums_rejects_duplicate_names_and_gpu_tensors():
    params = {"w": torch.zeros(4)}
    with pytest.raises(AssertionError, match="duplicate"):
        compute_send_checksums(params, ["w", "w"])
    if torch.cuda.is_available():
        with pytest.raises(AssertionError, match="cpu"):
            compute_send_checksums({"w": torch.zeros(4, device="cuda")}, ["w"])


def _engine_body(checksums_per_rank: dict[int, dict[str, str]]) -> dict:
    return {
        "ok": True,
        "supported": True,
        "results": [{"rank": rank, "checksums": checksums} for rank, checksums in checksums_per_rank.items()],
    }


def test_verify_transfer_checksums_passes_when_all_ranks_match():
    params = {"w": torch.randn(8), "model.layers.0.q.weight": torch.randn(8)}
    sent = compute_send_checksums(params, list(params))
    body = _engine_body({0: dict(sent), 1: dict(sent)})
    verify_transfer_checksums(sent_checksums=sent, engine_body=body)  # no raise


def test_verify_transfer_checksums_detects_one_bit_flip():
    params = {"w": torch.randn(8)}
    sent = compute_send_checksums(params, ["w"])
    flipped = params["w"].clone()
    flipped.view(torch.uint8)[0] ^= 1
    body = _engine_body({0: {"w": hash_tensor_sha256(flipped)}})
    with pytest.raises(RuntimeError, match="w@rank0"):
        verify_transfer_checksums(sent_checksums=sent, engine_body=body)


def test_verify_transfer_checksums_detects_missing_and_extra_tensors():
    sent = compute_send_checksums({"w": torch.zeros(4)}, ["w"])
    with pytest.raises(RuntimeError, match="w@rank0"):
        verify_transfer_checksums(sent_checksums=sent, engine_body=_engine_body({0: {}}))
    body = _engine_body({0: {**sent, "extra": hash_tensor_sha256(torch.zeros(4))}})
    with pytest.raises(RuntimeError, match="extra@rank0"):
        verify_transfer_checksums(sent_checksums=sent, engine_body=body)


def test_verify_transfer_checksums_requires_support_and_records():
    sent = compute_send_checksums({"w": torch.zeros(4)}, ["w"])
    with pytest.raises(RuntimeError, match="does not support"):
        verify_transfer_checksums(sent_checksums=sent, engine_body={"ok": True, "supported": False, "results": []})
    with pytest.raises(RuntimeError, match="failure"):
        verify_transfer_checksums(sent_checksums=sent, engine_body={"ok": False})
    with pytest.raises(RuntimeError, match="got none"):
        verify_transfer_checksums(sent_checksums=sent, engine_body={"ok": True, "supported": True, "results": []})
    dup = {"ok": True, "supported": True, "results": [{"rank": 0, "checksums": dict(sent)}, {"rank": 0, "checksums": {}}]}
    with pytest.raises(RuntimeError, match="conflicting"):
        verify_transfer_checksums(sent_checksums=sent, engine_body=dup)


def _compare_body(tensors_per_rank: dict[int, dict[str, str]]) -> dict:
    return {
        "ok": True,
        "supported": True,
        "results": [
            {"rank": rank, "tensors": {name: {"status": status} for name, status in tensors.items()}}
            for rank, tensors in tensors_per_rank.items()
        ],
    }


def test_evaluate_compare_results_passes_on_all_match():
    body = _compare_body({0: {"a": "match", "b": "match"}})
    assert evaluate_compare_results(body) == []


def test_evaluate_compare_results_raises_naming_mismatches():
    body = _compare_body({0: {"a": "match", "model.layers.0.w": "mismatch"}, 1: {"c": "missing"}})
    with pytest.raises(RuntimeError, match=r"model\.layers\.0\.w@rank0"):
        evaluate_compare_results(body)


def test_evaluate_compare_results_skip_list_downgrades():
    body = _compare_body({0: {"visual.proj.weight": "mismatch"}})
    downgraded = evaluate_compare_results(body, skip_list=["visual."])
    assert downgraded == ["visual.proj.weight@rank0"]
    body = _compare_body({0: {"visual.proj.weight": "mismatch", "w": "mismatch"}})
    with pytest.raises(RuntimeError, match="w@rank0"):
        evaluate_compare_results(body, skip_list=["visual."])


def test_evaluate_compare_results_refuses_unsupported_engine():
    with pytest.raises(RuntimeError, match="does not support"):
        evaluate_compare_results({"ok": True, "supported": False, "results": []})

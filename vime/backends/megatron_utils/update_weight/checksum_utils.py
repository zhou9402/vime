"""Checksum utilities for weight-sync verification.

Port of miles@ba28991834 backends/training_utils/weight_update/checksum_utils.py,
adapted to the vime/vLLM ``check_weights`` collective_rpc response shape:
``{"ok": True, "supported": True, "results": [{"rank": int, "checksums": {...}}]}``.
"""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def hash_tensor_sha256(tensor: torch.Tensor) -> str:
    """Real (cryptographic) hash: a mismatch here has to mean a bug."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


def compute_send_checksums(parameters: Mapping[str, torch.Tensor], names: Sequence[str]) -> dict[str, str]:
    assert len(names) == len(set(names)), "weight-update bucket contains duplicate tensor names"
    checksums = {}
    for name in names:
        tensor = parameters[name]
        assert tensor.device.type == "cpu", "Expected registered CPU send buffers"
        checksums[name] = hash_tensor_sha256(tensor)
    return checksums


def verify_transfer_checksums(*, sent_checksums: dict[str, str], engine_body: dict[str, Any]) -> None:
    """Compare trainer-side send checksums against the checksums every engine
    rank reported after the write landed. Raises on any mismatch."""
    if not engine_body.get("ok", False):
        raise RuntimeError(f"check_weights engine reported failure: {engine_body!r}")
    if not engine_body.get("supported", False):
        raise RuntimeError(
            "engine does not support check_weights checksums (vLLM check-weights patch missing?): "
            f"{engine_body!r}"
        )
    rank_checksums = _compute_received_checksums(engine_body)

    mismatched = sorted(
        f"{name}@rank{rank}"
        for rank, received in rank_checksums.items()
        for name in sent_checksums.keys() | received.keys()
        if sent_checksums.get(name) != received.get(name)
    )
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} tensors reached the rollout engine with checksums "
            f"different from what was sent: {mismatched}"
        )


def _compute_received_checksums(engine_body: dict[str, Any]) -> dict[int, dict[str, str]]:
    """Per-GPU-rank checksums from a check_weights RPC response. Every entry of
    ``results`` for the same rank must agree."""
    rank_checksums: dict[int, dict[str, str]] = {}
    for info in engine_body["results"]:
        rank = info["rank"]
        checksums = info["checksums"]
        if rank in rank_checksums:
            if rank_checksums[rank] != checksums:
                raise RuntimeError(f"conflicting checksum records for GPU rank {rank}: {engine_body!r}")
        else:
            rank_checksums[rank] = checksums
    if not rank_checksums:
        raise RuntimeError(f"expected checksum records, got none: {engine_body!r}")
    return rank_checksums

"""Unit tests for the vLLM-side P2P weight-transfer backend (F01-P1 patch).

Exercises `vllm.distributed.weight_transfer.p2p_engine.P2PWeightTransferEngine`
with a fake Mooncake TransferEngine (no RDMA, no GPU): the trainer side is
vime's shipped `pack_bucket_into_transfer_buffer`, and the "RDMA write" is a
numpy view alias over the engine's staging buffer (same bytes in-process).

Skips unless the patched vLLM (docker/patch/latest/vllm-p2p-weight-transfer.patch
applied) is importable.
"""

import ctypes
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

p2p_engine = pytest.importorskip(
    "vllm.distributed.weight_transfer.p2p_engine",
    reason="patched vLLM (vllm-p2p-weight-transfer.patch) not importable",
)
from vime.backends.megatron_utils.update_weight.p2p import pack_bucket_into_transfer_buffer

NUM_GPUS = 0

# The engine's load path pulls `disable_mtp_completeness_check`, which drags in
# vLLM's tool-parser registry; that needs exact pinned deps (xgrammar API). When
# it is not importable (local CPU envs), stub the no-op context manager instead.
try:
    from vllm.model_executor.model_loader.mtp_validation import (  # noqa: F401
        disable_mtp_completeness_check,
    )
except Exception:
    import contextlib
    import sys
    import types

    _stub = types.ModuleType("vllm.model_executor.model_loader.mtp_validation")
    _stub.disable_mtp_completeness_check = contextlib.nullcontext
    sys.modules[_stub.__name__] = _stub


class FakeTransferEngine:
    def __init__(self):
        self.registered = {}

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr: int) -> int:
        del self.registered[ptr]
        return 0


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4, 8))
        self.b = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        self.loaded: list[str] = []

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            with torch.no_grad():
                params[name].copy_(tensor)
            self.loaded.append(name)


def _new_engine(monkeypatch, capacity: int):
    fake = FakeTransferEngine()
    monkeypatch.setattr(p2p_engine, "_create_transfer_engine", lambda: (fake, "fake-session:1"))
    engine = p2p_engine.P2PWeightTransferEngine(
        config=SimpleNamespace(),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(rank=0), model_config=SimpleNamespace()),
        device=torch.device("cpu"),
        model=TinyModel(),
    )
    if capacity > 0:
        engine.init_transfer_engine(
            p2p_engine.P2PWeightTransferInitInfo(staging_buffer_size_bytes=capacity)
        )
    return engine, fake


def _bucket() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "w": torch.randn(4, 8, generator=generator),
        "b": torch.randn(4, generator=generator, dtype=torch.bfloat16),
    }


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _staging_alias(engine) -> torch.Tensor:
    """A writable torch view over the engine's staging buffer (in-process RDMA write)."""
    info = engine.get_p2p_transfer_info()
    c_array = (ctypes.c_char * info["capacity_nbytes"]).from_address(info["staging_address"])
    return torch.from_numpy(np.frombuffer(c_array, dtype=np.uint8))


def _write_and_load(engine, bucket, *, corrupt_checksums=False):
    staging = _staging_alias(engine)
    manifest = pack_bucket_into_transfer_buffer(staging, bucket)
    for entry in manifest:
        tensor = bucket[entry["name"]]
        entry["dtype"] = str(tensor.dtype).split(".")[-1]
        entry["shape"] = list(tensor.shape)
    checksums = {name: _sha256(t) for name, t in bucket.items()}
    if corrupt_checksums:
        checksums["w"] = "0" * 64
    return engine.load_weights_from_staged({"manifest": manifest, "checksums": checksums}), checksums


def test_transfer_info_publishes_session_and_capacity(monkeypatch):
    engine, _ = _new_engine(monkeypatch, 4096)
    info = engine.get_p2p_transfer_info()
    assert info["session_id"] == "fake-session:1"
    assert info["capacity_nbytes"] == 4096
    assert info["staging_address"] != 0


def test_staging_buffer_is_registered(monkeypatch):
    engine, fake = _new_engine(monkeypatch, 4096)
    info = engine.get_p2p_transfer_info()
    assert fake.registered == {info["staging_address"]: 4096}


def test_load_staged_weights_verified_and_loaded(monkeypatch):
    engine, _ = _new_engine(monkeypatch, 4096)
    bucket = _bucket()
    received, sent = _write_and_load(engine, bucket)
    assert received == sent
    assert torch.equal(engine.model.w.detach(), bucket["w"])
    assert torch.equal(engine.model.b.detach(), bucket["b"])
    assert engine.model.loaded == ["w", "b"]


def test_load_staged_rejects_checksum_mismatch(monkeypatch):
    engine, _ = _new_engine(monkeypatch, 4096)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _write_and_load(engine, _bucket(), corrupt_checksums=True)


def test_load_staged_rejects_manifest_past_capacity(monkeypatch):
    engine, _ = _new_engine(monkeypatch, 64)
    manifest = [{"name": "w", "offset": 32, "nbytes": 128, "dtype": "float32", "shape": [4, 8]}]
    with pytest.raises(RuntimeError, match="runs past"):
        engine.load_weights_from_staged({"manifest": manifest, "checksums": {}})


def test_uninitialized_engine_rejects_calls(monkeypatch):
    engine, _ = _new_engine(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="not initialized"):
        engine.get_p2p_transfer_info()


def test_init_rejects_nonpositive_buffer(monkeypatch):
    engine, _ = _new_engine(monkeypatch, 0)
    with pytest.raises(ValueError, match="positive"):
        engine.init_transfer_engine(
            p2p_engine.P2PWeightTransferInitInfo(staging_buffer_size_bytes=0)
        )


def test_shutdown_unregisters_staging_buffer(monkeypatch):
    engine, fake = _new_engine(monkeypatch, 4096)
    engine.shutdown()
    assert fake.registered == {}
    with pytest.raises(RuntimeError, match="not initialized"):
        engine.get_p2p_transfer_info()


def test_target_rank_filters_collective_rpc_fanout(monkeypatch):
    """A bucket targets exactly one rank; other workers no-op under the fan-out."""
    engine, _ = _new_engine(monkeypatch, 4096)
    bucket = _bucket()
    staging = _staging_alias(engine)
    manifest = pack_bucket_into_transfer_buffer(staging, bucket)
    for entry in manifest:
        tensor = bucket[entry["name"]]
        entry["dtype"] = str(tensor.dtype).split(".")[-1]
        entry["shape"] = list(tensor.shape)
    checksums = {name: _sha256(t) for name, t in bucket.items()}

    received = engine.load_weights_from_staged(
        {"manifest": manifest, "checksums": checksums, "target_rank": 7}
    )
    assert received == {}
    assert engine.model.loaded == [], "non-target rank must not load"

    received = engine.load_weights_from_staged(
        {"manifest": manifest, "checksums": checksums, "target_rank": 0}
    )
    assert received == checksums
    assert engine.model.loaded == ["w", "b"]

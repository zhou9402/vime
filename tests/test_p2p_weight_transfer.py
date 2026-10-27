"""P2P/RDMA weight-transfer contract verifier (F01-P0, Design A: replica-free).

Design A in one paragraph: each engine worker publishes ONE registered staging
buffer (session id, address, capacity). The trainer packs an HF-layout bucket
into a pinned transfer buffer, RDMA-writes it into the staging buffer, then the
engine reshards the staged bytes with its own loader and returns per-name
SHA-256 checksums for verification. No trainer-side model replica.

Weight writes are NEVER retried (miles: the weight-update session is no-retry
by design; the #4402 retry hardening covers Mooncake *object-store* ops only).
A failed write marks its target errored and the update moves on to healthy
targets; uncertain writes additionally pin the registered memory forever.

This file pins that contract with fakes only (no Mooncake, no vLLM, no GPU):
a fake TransferEngine copies bytes in-process with fault injection, and a fake
engine worker reshards staged bytes with a reference loader. The real pieces
must keep this green:
* F01-P1 (vLLM ``p2p`` weight-transfer backend, docker patch) implements the
  ``get_p2p_transfer_info`` / ``load_weights_from_staged`` worker contract.
* F01-P2 (trainer transport package) provides the real MooncakeTransport; the
  packing helper below is the shipped
  ``vime.backends.megatron_utils.update_weight.p2p.pack_bucket_into_transfer_buffer``.
* F01-P3 (UpdateWeightFromP2P) drives the same pack -> write -> load -> verify
  flow against real engines.
* F01-P4 adds the shared check_weights/quarantine hardening (with F02).

Deterministic: fixed-shape tensors, in-process byte copies, no network.
"""

import ctypes
import hashlib

import pytest
import torch

from vime.backends.megatron_utils.update_weight.p2p import pack_bucket_into_transfer_buffer

NUM_GPUS = 0


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class FakeTransferEngine:
    """In-process stand-in for mooncake.engine.TransferEngine.

    Copies bytes with ctypes.memmove; supports fault injection for the
    tamper/failure contract tests.
    """

    def __init__(self):
        self.registered = {}
        self.fail_with: int | None = None
        self._corrupt_next_write = False
        self.writes = 0

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr: int) -> int:
        del self.registered[ptr]
        return 0

    def corrupt_next_write(self):
        self._corrupt_next_write = True

    def batch_transfer_sync_write(self, session_id, src_ptrs, dst_addrs, nbytes_list) -> int:
        self.writes += 1
        if self.fail_with is not None:
            return self.fail_with
        for src, dst, nbytes in zip(src_ptrs, dst_addrs, nbytes_list, strict=True):
            assert src in self.registered, f"source {src:#x} is not registered"
            assert self.registered[src] >= nbytes
            ctypes.memmove(dst, src, nbytes)
            if self._corrupt_next_write:
                ctypes.memmove(dst, bytes([ctypes.c_uint8.from_address(dst).value ^ 0x1]), 1)
                self._corrupt_next_write = False
        return 0


class FakeEngineWorker:
    """Fake engine rank: one staging buffer + a reference loader.

    The reference loader plays the role of the F01-P1 vLLM backend: it consumes
    the packed staging buffer per the manifest and returns per-name checksums.
    """

    def __init__(self, params: dict[str, torch.Tensor], capacity_nbytes: int):
        self.params = params
        self.staging = torch.empty(capacity_nbytes, dtype=torch.uint8)
        self.session_id = f"fake-session-{id(self):x}"

    def get_p2p_transfer_info(self) -> dict:
        return {
            "session_id": self.session_id,
            "staging_address": self.staging.data_ptr(),
            "capacity_nbytes": self.staging.numel(),
        }

    def load_weights_from_staged(self, manifest: list[dict]) -> dict[str, str]:
        checksums = {}
        for entry in manifest:
            name, offset, nbytes = entry["name"], entry["offset"], entry["nbytes"]
            param = self.params[name]
            if param.numel() * param.element_size() != nbytes:
                raise RuntimeError(f"{name}: size mismatch")
            staged = self.staging[offset : offset + nbytes].view(param.dtype).view(param.shape)
            param.copy_(staged)
            checksums[name] = _sha256(param)
        return checksums


class _Target:
    """Error-isolation handle per engine (the F01-P3/P4 rollout-cell updater contract)."""

    def __init__(self):
        self.errored: BaseException | None = None

    @property
    def is_errored(self) -> bool:
        return self.errored is not None

    def mark_errored(self, error: BaseException) -> None:
        self.errored = error


class MiniP2PPipeline:
    """Test-local driver mimicking the F01-P3 updater: pack -> write -> load -> verify.

    Writes are never retried: a failed write marks the engine's target errored
    and the failure propagates; other (healthy) engines are still updated.
    """

    def __init__(self, transfer_engine: FakeTransferEngine, buffer_nbytes: int):
        self._transfer_engine = transfer_engine
        self.buffer = torch.empty(buffer_nbytes, dtype=torch.uint8, pin_memory=False)
        self._transfer_engine.register_memory(self.buffer.data_ptr(), buffer_nbytes)

    def update_engine(self, worker: FakeEngineWorker, bucket: dict[str, torch.Tensor], target: _Target) -> None:
        if target.is_errored:
            return
        info = worker.get_p2p_transfer_info()
        manifest = pack_bucket_into_transfer_buffer(self.buffer, bucket)
        total_nbytes = sum(entry["nbytes"] for entry in manifest)
        if total_nbytes > info["capacity_nbytes"]:
            raise RuntimeError("bucket exceeds the engine staging buffer")
        sent_checksums = {name: _sha256(tensor) for name, tensor in bucket.items()}

        ret = self._transfer_engine.batch_transfer_sync_write(
            info["session_id"], [self.buffer.data_ptr()], [info["staging_address"]], [total_nbytes]
        )
        if ret < 0:
            error = RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")
            target.mark_errored(error)
            raise error
        received = worker.load_weights_from_staged(manifest)
        if received != sent_checksums:
            error = RuntimeError(
                "checksum mismatch after P2P write: "
                f"{[n for n in sent_checksums if received.get(n) != sent_checksums[n]]}"
            )
            target.mark_errored(error)
            raise error


def _source_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "layers.0.mlp.gate_proj.weight": torch.randn(64, 32, generator=generator),
        "layers.0.self_attn.q_proj.weight": torch.randn(32, 32, generator=generator),
        "layers.1.experts.3.w1": torch.randn(16, 32, generator=generator, dtype=torch.bfloat16),
        "lm_head.weight": torch.randint(-100, 100, (17, 32), generator=generator, dtype=torch.int8),
    }


def _make_engine(params_source: dict[str, torch.Tensor], capacity_nbytes: int) -> FakeEngineWorker:
    return FakeEngineWorker(
        {name: torch.empty_like(tensor) for name, tensor in params_source.items()}, capacity_nbytes
    )


def test_p2p_weights_match_direct_copy():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())

    p2p_engines = [_make_engine(weights, total), _make_engine(weights, total)]
    pipeline = MiniP2PPipeline(FakeTransferEngine(), buffer_nbytes=total)
    for engine in p2p_engines:
        pipeline.update_engine(engine, weights, _Target())

    # NCCL-equivalent reference: a direct copy into the same engine params.
    reference_engine = _make_engine(weights, total)
    for name, tensor in weights.items():
        reference_engine.params[name].copy_(tensor)

    for engine in p2p_engines:
        for name, tensor in weights.items():
            assert torch.equal(engine.params[name], tensor), name
            assert torch.equal(engine.params[name], reference_engine.params[name]), name


def test_p2p_partial_bucket_updates_only_listed_params():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engine = _make_engine(weights, total)
    pipeline = MiniP2PPipeline(FakeTransferEngine(), buffer_nbytes=total)

    pipeline.update_engine(engine, weights, _Target())
    sentinel = torch.full_like(weights["lm_head.weight"], 7)
    engine.params["lm_head.weight"].copy_(sentinel)
    bucket = {k: v + 1 for k, v in weights.items() if k != "lm_head.weight"}
    pipeline.update_engine(engine, bucket, _Target())

    for name, tensor in bucket.items():
        assert torch.equal(engine.params[name], tensor), name
    assert torch.equal(engine.params["lm_head.weight"], sentinel)


def test_bit_flip_during_write_is_detected():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engine = _make_engine(weights, total)
    transfer_engine = FakeTransferEngine()
    transfer_engine.corrupt_next_write()
    pipeline = MiniP2PPipeline(transfer_engine, buffer_nbytes=total)
    target = _Target()

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        pipeline.update_engine(engine, weights, target)
    assert target.is_errored


def test_failed_write_marks_target_errored_and_is_never_retried():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engine = _make_engine(weights, total)
    transfer_engine = FakeTransferEngine()
    transfer_engine.fail_with = -800
    pipeline = MiniP2PPipeline(transfer_engine, buffer_nbytes=total)
    target = _Target()

    with pytest.raises(RuntimeError, match="returned -800"):
        pipeline.update_engine(engine, weights, target)
    assert target.is_errored
    assert transfer_engine.writes == 1, "weight writes are never retried, not even transient ones"

    # An errored target is skipped on later updates (partial-target quarantine).
    transfer_engine.fail_with = None
    pipeline.update_engine(engine, weights, target)
    assert transfer_engine.writes == 1


def test_failed_engine_does_not_block_healthy_engines():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engines = [_make_engine(weights, total), _make_engine(weights, total)]
    targets = [_Target(), _Target()]

    failing = FakeTransferEngine()
    failing.fail_with = -703
    MiniP2PPipeline(failing, buffer_nbytes=total)  # keep buffer-size parity; separate pipelines per sender
    pipeline = MiniP2PPipeline(FakeTransferEngine(), buffer_nbytes=total)

    # engine 0 fails on its sender; engine 1 is updated by a healthy sender
    pipeline_fail = MiniP2PPipeline(failing, buffer_nbytes=total)
    with pytest.raises(RuntimeError):
        pipeline_fail.update_engine(engines[0], weights, targets[0])
    pipeline.update_engine(engines[1], weights, targets[1])

    assert targets[0].is_errored
    assert not targets[1].is_errored
    for name, tensor in weights.items():
        assert torch.equal(engines[1].params[name], tensor), name


def test_bucket_exceeding_staging_capacity_is_rejected():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engine = _make_engine(weights, capacity_nbytes=total // 2)
    pipeline = MiniP2PPipeline(FakeTransferEngine(), buffer_nbytes=total)

    with pytest.raises(RuntimeError, match="staging buffer"):
        pipeline.update_engine(engine, weights, _Target())


def test_bucket_exceeding_transfer_buffer_is_rejected():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engine = _make_engine(weights, capacity_nbytes=total)
    pipeline = MiniP2PPipeline(FakeTransferEngine(), buffer_nbytes=total // 2)

    with pytest.raises(RuntimeError, match="transfer buffer"):
        pipeline.update_engine(engine, weights, _Target())

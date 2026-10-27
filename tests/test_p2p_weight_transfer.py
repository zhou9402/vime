"""P2P/RDMA weight-transfer verifier (F01): the real updater against fakes.

Drives `UpdateWeightFromP2P` (F01-P3) with:
* the shipped `MooncakeTransport` over an in-process fake TransferEngine
  (ctypes byte copies, fault injection — no Mooncake, no RDMA, no GPU), and
* fake vLLM engine objects implementing the F01-P1 worker contract
  (`init_weight_transfer_engine` / `get_p2p_transfer_info` /
  `load_weights_from_staged`) with a reference loader.

Contract pinned here: byte-equivalence with a direct copy, engine-side
resharding being the engine's business, per-name SHA-256 verification catching
a flipped bit, writes never retried, and failed engines quarantined while
healthy ones finish the update.

Deterministic: fixed-seed tensors, in-process copies, no network.
"""

import ctypes
import hashlib
from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.update_weight.p2p import MooncakeTransport
from vime.backends.megatron_utils.update_weight.update_weight_from_p2p import UpdateWeightFromP2P

NUM_GPUS = 0


class FakeTransferEngine:
    """In-process mooncake.engine.TransferEngine: memmove + fault injection."""

    def __init__(self):
        self.registered = {}
        self.fail_with: int | None = None
        self.fail_sessions: set[str] | None = None
        self.corrupt_sessions: set[str] = set()
        self.writes = 0

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr: int) -> int:
        del self.registered[ptr]
        return 0

    def corrupt_writes_to(self, session_id: str):
        self.corrupt_sessions.add(session_id)

    def batch_transfer_sync_write(self, session_id, src_ptrs, dst_addrs, nbytes_list) -> int:
        self.writes += 1
        if self.fail_with is not None and (self.fail_sessions is None or session_id in self.fail_sessions):
            return self.fail_with
        for src, dst, nbytes in zip(src_ptrs, dst_addrs, nbytes_list, strict=True):
            ctypes.memmove(dst, src, nbytes)
            if session_id in self.corrupt_sessions:
                ctypes.memmove(dst, bytes([ctypes.c_uint8.from_address(dst).value ^ 0x1]), 1)
        return 0


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class FakeEngine:
    """Fake vLLM engine (one worker rank): staging buffer + reference loader.

    The reference loader stands in for the F01-P1 backend: it consumes the
    staged bytes per the manifest and returns per-name checksums.
    """

    def __init__(self, params_source: dict[str, torch.Tensor], num_ranks: int = 1, node_rank: int = 0):
        self.node_rank = node_rank
        self.shutdown_called = False
        self.info_failures = 0  # transient control-plane failures to inject
        self.ranks = [
            {name: torch.empty_like(tensor) for name, tensor in params_source.items()}
            for _ in range(num_ranks)
        ]
        self.staging: list[torch.Tensor] = []
        self.resumed = False
        self.weight_version: str | None = None

    # -- F01-P1 worker contract (what VLLMEngine exposes over /collective_rpc) --

    def init_weight_transfer_engine(self, payload: dict):
        nbytes = payload["init_info"]["staging_buffer_size_bytes"]
        self.staging = [torch.empty(nbytes, dtype=torch.uint8) for _ in self.ranks]

    def get_p2p_transfer_info(self) -> list[dict]:
        if self.node_rank != 0:
            return []  # only the node-0 actor serves HTTP (VLLMEngine._collective_rpc)
        if self.info_failures:
            self.info_failures -= 1
            import requests

            raise requests.ConnectionError("simulated transient control-plane failure")
        return [
            {
                "session_id": f"fake-session-{id(self):x}-{rank}",
                "staging_address": staging.data_ptr(),
                "capacity_nbytes": staging.numel(),
            }
            for rank, staging in enumerate(self.staging)
        ]

    def load_weights_from_staged(self, update_info: dict) -> list[dict]:
        if self.node_rank != 0:
            return []
        # collective_rpc fan-out contract: one result per worker; non-target ranks no-op.
        target_rank = update_info.get("target_rank")
        results = []
        for rank in range(len(self.ranks)):
            if target_rank is not None and rank != target_rank:
                results.append({})
                continue
            staging = self.staging[rank]
            params = self.ranks[rank]
            checksums = {}
            for entry in update_info["manifest"]:
                name, offset, nbytes = entry["name"], entry["offset"], entry["nbytes"]
                staged = staging[offset : offset + nbytes]
                digest = hashlib.sha256(staged.numpy().tobytes()).hexdigest()
                expected = update_info.get("checksums", {}).get(name)
                if expected is not None and expected != digest:
                    raise RuntimeError(f"checksum mismatch for {name!r}")
                params[name].copy_(staged.view(params[name].dtype).view(params[name].shape))
                checksums[name] = digest
            results.append(checksums)
        return results

    # -- lifecycle (no-ops beyond bookkeeping) --

    def pause_generation(self, mode: str = "abort"):
        self.resumed = False

    def flush_cache(self):
        pass

    def start_weight_update(self):
        pass

    def finish_weight_update(self, weight_version: str):
        self.weight_version = weight_version

    def continue_generation(self):
        self.resumed = True

    def destroy_weights_update_group(self, _group):
        pass

    def shutdown(self):
        # VLLMEngine.shutdown: unregister from the router + kill the server.
        self.shutdown_called = True
        self.resumed = False


class _FakeIterator:
    def __init__(self, buckets: list[dict[str, torch.Tensor]]):
        self._buckets = buckets

    def get_hf_weight_chunks(self, _weights) -> list:
        return [list(bucket.items()) for bucket in self._buckets]


def _source_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "layers.0.mlp.gate_proj.weight": torch.randn(64, 32, generator=generator),
        "layers.0.self_attn.q_proj.weight": torch.randn(32, 32, generator=generator),
        "layers.1.experts.3.w1": torch.randn(16, 32, generator=generator, dtype=torch.bfloat16),
        "lm_head.weight": torch.randint(-100, 100, (17, 32), generator=generator, dtype=torch.int8),
    }


def _make_updater(weights, engines, fake_transfer_engine, *, checksum=True, buffer_nbytes=None, replicas=(0, 1)):
    total = sum(t.numel() * t.element_size() for t in weights.values())
    args = SimpleNamespace(
        update_weight_buffer_size=buffer_nbytes or total,
        p2p_transfer_timeout=5.0,
        check_weight_transfer_checksum=checksum,
        flush_cache_interval=1,
        rollout_num_gpus_per_engine=1,
        update_weight_start_version=0,
    )
    updater = UpdateWeightFromP2P(
        args,
        model=None,
        weights_getter=lambda: None,
        model_name="tiny",
        quantization_config=None,
        iterator=_FakeIterator([weights]),
        transport=MooncakeTransport(transfer_engine=fake_transfer_engine),
        data_replica_rank_and_size=replicas,
    )
    updater.connect_rollout_engines(engines, None, engine_gpu_counts=[len(e.ranks) for e in engines])
    return updater


def test_p2p_update_weights_match_direct_copy():
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    updater = _make_updater(weights, engines, FakeTransferEngine())
    updater.update_weights()

    reference = {name: torch.empty_like(t) for name, t in weights.items()}
    for name, tensor in weights.items():
        reference[name].copy_(tensor)

    for engine in engines:
        assert engine.resumed
        assert engine.weight_version == "1"
        for rank_params in engine.ranks:
            for name, tensor in weights.items():
                assert torch.equal(rank_params[name], tensor), name
                assert torch.equal(rank_params[name], reference[name]), name
    updater.disconnect_rollout_engines()


def test_p2p_two_ranks_per_engine_both_loaded():
    weights = _source_weights()
    engine = FakeEngine(weights, num_ranks=2)
    updater = _make_updater(weights, [engine], FakeTransferEngine())
    updater.update_weights()
    for rank_params in engine.ranks:
        for name, tensor in weights.items():
            assert torch.equal(rank_params[name], tensor), name


def test_bit_flip_during_write_is_detected_and_quarantined():
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    transfer_engine = FakeTransferEngine()
    updater = _make_updater(weights, engines, transfer_engine)
    # engine 0's buckets arrive corrupted (targets that engine's session only)
    transfer_engine.corrupt_writes_to(engines[0].get_p2p_transfer_info()[0]["session_id"])

    with pytest.raises(RuntimeError, match="P2P weight update failed on engines \\[0\\]"):
        updater.update_weights()
    assert not engines[0].resumed, "the corrupted engine must not resume"
    assert engines[0].shutdown_called, "errored engines are retired (router unregister + kill)"
    assert engines[1].resumed, "healthy engines finish the update"
    assert not engines[1].shutdown_called
    for name, tensor in weights.items():
        assert torch.equal(engines[1].ranks[0][name], tensor), name


def test_failed_write_is_never_retried_and_engine_is_quarantined():
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    transfer_engine = FakeTransferEngine()
    transfer_engine.fail_with = -800
    updater = _make_updater(weights, engines, transfer_engine)

    with pytest.raises(RuntimeError, match="engines \\[0, 1\\]"):
        updater.update_weights()
    # one write per engine's shard, no retries of the failed write
    assert transfer_engine.writes == 2
    assert not engines[0].resumed and not engines[1].resumed


def test_errored_engine_is_skipped_on_the_next_update():
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    transfer_engine = FakeTransferEngine()
    updater = _make_updater(weights, engines, transfer_engine)

    # only engine 0's writes fail; engine 1 finishes the first update
    transfer_engine.fail_with = -1
    transfer_engine.fail_sessions = {engines[0].get_p2p_transfer_info()[0]["session_id"]}
    with pytest.raises(RuntimeError, match="engines \\[0\\]"):
        updater.update_weights()
    assert engines[1].resumed
    writes_after_failure = transfer_engine.writes

    transfer_engine.fail_with = None
    updater.update_weights()  # succeeds: only engine 1 remains
    for name, tensor in weights.items():
        assert torch.equal(engines[1].ranks[0][name], tensor), name
    assert transfer_engine.writes > writes_after_failure
    assert not engines[0].resumed


def test_bucket_exceeding_staging_capacity_is_rejected():
    weights = _source_weights()
    total = sum(t.numel() * t.element_size() for t in weights.values())
    engine = FakeEngine(weights)
    # engine allocates staging at 2x the trainer's (too-small) buffer... so
    # instead give the trainer a buffer one byte too small for the bucket.
    updater = _make_updater(weights, [engine], FakeTransferEngine(), buffer_nbytes=total - 1)
    with pytest.raises(RuntimeError, match="transfer buffer"):
        updater.update_weights()
    assert not engine.resumed


def test_non_node0_engine_actors_are_tolerated():
    """Multi-node engines: non-node-0 actors answer [] over collective_rpc; the
    updater skips them for data and never crashes."""
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights, node_rank=1)]
    updater = _make_updater(weights, engines, FakeTransferEngine())
    updater.update_weights()
    for name, tensor in weights.items():
        assert torch.equal(engines[0].ranks[0][name], tensor), name
    assert not torch.equal(engines[1].ranks[0]["lm_head.weight"], weights["lm_head.weight"])
    assert engines[1].resumed


def test_bucket_is_packed_once_for_multiple_assignments(monkeypatch):
    """3 assignments (one 3-rank engine), 1 bucket -> exactly 1 pack call."""
    import vime.backends.megatron_utils.update_weight.update_weight_from_p2p as p2p_updater

    weights = _source_weights()
    engine = FakeEngine(weights, num_ranks=3)
    calls = 0
    real_pack = p2p_updater.pack_bucket_into_transfer_buffer

    def counting_pack(buffer, bucket):
        nonlocal calls
        calls += 1
        return real_pack(buffer, bucket)

    monkeypatch.setattr(p2p_updater, "pack_bucket_into_transfer_buffer", counting_pack)
    updater = _make_updater(weights, [engine], FakeTransferEngine())
    updater.update_weights()
    assert calls == 1
    for rank_params in engine.ranks:
        for name, tensor in weights.items():
            assert torch.equal(rank_params[name], tensor), name


def test_transient_control_plane_error_is_retried():
    """connect() queries may be retried (miles #4402); data writes never are."""
    weights = _source_weights()
    engines = [FakeEngine(weights)]
    engines[0].info_failures = 2  # first two get_p2p_transfer_info calls fail
    updater = _make_updater(weights, engines, FakeTransferEngine())
    updater.update_weights()
    for name, tensor in weights.items():
        assert torch.equal(engines[0].ranks[0][name], tensor), name


def test_quarantine_union_marks_engines_errored_on_other_ranks():
    """An engine that errored on any trainer rank is quarantined on all of them."""
    from vime.backends.megatron_utils.update_weight.update_weight_from_p2p import (
        _EngineTarget,
        _quarantine_errored_engines,
    )
    import vime.backends.megatron_utils.update_weight.update_weight_from_p2p as p2p_updater

    targets = [_EngineTarget(0), _EngineTarget(1)]
    targets[0].mark_errored(RuntimeError("boom"), weight_version=3)
    p2p_updater._union_over_gloo_orig = getattr(p2p_updater, "_union_over_gloo")
    try:
        p2p_updater._union_over_gloo = lambda local: local | {1}
        _quarantine_errored_engines(targets, weight_version=3)
    finally:
        p2p_updater._union_over_gloo = p2p_updater._union_over_gloo_orig
    assert targets[1].is_errored
    assert targets[1].errored_at_version == 3

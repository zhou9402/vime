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
        self.corrupt_after: dict[str, int] = {}
        self._writes_by_session: dict[str, int] = {}
        self.writes = 0

    def register_memory(self, ptr: int, nbytes: int) -> int:
        self.registered[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr: int) -> int:
        del self.registered[ptr]
        return 0

    def corrupt_writes_to(self, session_id: str, after_n_writes: int = 0):
        """Corrupt writes to `session_id` once it has received `after_n_writes` writes
        (per-session write counting keeps this deterministic under threaded writes)."""
        self.corrupt_sessions.add(session_id)
        self.corrupt_after[session_id] = after_n_writes

    def batch_transfer_sync_write(self, session_id, src_ptrs, dst_addrs, nbytes_list) -> int:
        self.writes += 1
        if self.fail_with is not None and (self.fail_sessions is None or session_id in self.fail_sessions):
            return self.fail_with
        count = self._writes_by_session.get(session_id, 0) + 1
        self._writes_by_session[session_id] = count
        for src, dst, nbytes in zip(src_ptrs, dst_addrs, nbytes_list, strict=True):
            ctypes.memmove(dst, src, nbytes)
            if session_id in self.corrupt_sessions and count > self.corrupt_after.get(session_id, 0):
                ctypes.memmove(dst, bytes([ctypes.c_uint8.from_address(dst).value ^ 0x1]), 1)
        return 0


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class FakeEngine:
    """Fake vLLM engine (one worker rank): staging buffer + reference loader.

    The reference loader stands in for the F01-P1 backend: it consumes the
    staged bytes per the manifest and returns per-name checksums.
    """

    def __init__(
        self,
        params_source: dict[str, torch.Tensor],
        num_ranks: int = 1,
        node_rank: int = 0,
        draft_params_source: dict[str, torch.Tensor] | None = None,
    ):
        self.node_rank = node_rank
        self.shutdown_called = False
        self.shutdown_calls = 0
        self.info_failures = 0  # transient control-plane failures to inject
        self.ranks = [
            {name: torch.empty_like(tensor) for name, tensor in params_source.items()}
            for _ in range(num_ranks)
        ]
        # separate store for the speculative draft (MTP) runner; must stay
        # disjoint from the target store (P6a)
        self.draft_ranks = [
            {name: torch.empty_like(tensor) for name, tensor in (draft_params_source or params_source).items()}
            for _ in range(num_ranks)
        ]
        self.supports_draft = True  # vLLM engine: supports_draft_weight_update
        self._draft_target = False  # set by start_draft_weight_update, reset by finish
        self.events: list = []  # ordered lifecycle + load calls, for order assertions
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
        # P6a: no payload field; the session picks the store (start_draft_weight_update
        # swapped the target, finish_weight_update resets it) — mirrors the vLLM worker.
        target_rank = update_info.get("target_rank")
        self.events.append(("load", self._draft_target, target_rank))
        results = []
        for rank in range(len(self.ranks)):
            if target_rank is not None and rank != target_rank:
                results.append({})
                continue
            staging = self.staging[rank]
            params = (self.draft_ranks if self._draft_target else self.ranks)[rank]
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
        self.events.append(("pause", mode))
        self.resumed = False

    def flush_cache(self):
        self.events.append(("flush",))

    def start_weight_update(self):
        self.events.append(("start",))

    def start_draft_weight_update(self):
        if not self.supports_draft:
            raise RuntimeError("P2PWeightTransferEngine does not support draft model weight updates.")
        self._draft_target = True
        self.events.append(("start_draft",))

    def finish_weight_update(self, weight_version: str):
        self._draft_target = False
        self.events.append(("finish", weight_version))
        self.weight_version = weight_version

    def continue_generation(self):
        self.events.append(("continue",))
        self.resumed = True

    def destroy_weights_update_group(self, _group):
        pass

    def shutdown(self):
        # VLLMEngine.shutdown: unregister from the router + kill the server.
        self.shutdown_called = True
        self.shutdown_calls += 1
        self.events.append(("shutdown",))
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


def _make_updater(weights, engines, fake_transfer_engine, *, checksum=True, buffer_nbytes=None,
                  replicas=(0, 1), **arg_overrides):
    total = sum(t.numel() * t.element_size() for t in weights.values())
    overrides = arg_overrides
    args_dict = dict(
        update_weight_buffer_size=buffer_nbytes or total,
        p2p_transfer_timeout=5.0,
        check_weight_transfer_checksum=checksum,
        flush_cache_interval=1,
        rollout_num_gpus_per_engine=1,
        update_weight_start_version=0,
        # P6a: speculative/draft mirrors the distributed path's condition
        enable_mtp_training=False,
        vllm_speculative_config=None,
        dspark_enabled=False,
        # P6b flags
        p2p_on_engine_error="fail",
        p2p_min_healthy_engines=1,
    )
    args_dict.update(overrides)
    args = SimpleNamespace(**args_dict)
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


# =============================================================================
# F01-P6 verifier (xfail until the P6a/P6b implementation lands)
# =============================================================================

_P6B = pytest.mark.xfail(strict=True, reason="F01-P6b not implemented")

_MTP_ARGS = dict(enable_mtp_training=True, vllm_speculative_config={"method": "mtp"})


def _load_events(engine):
    return [e for e in engine.events if e[0] == "load"]


# --- P6a: draft/MTP weights over P2P ------------------------------------------
# Two sessions per version, mirroring update_weight_from_distributed.py:113-123:
# start -> target buckets -> finish(v), then start_draft -> same buckets -> finish(v).
# The draft target swap happens worker-side (set_weight_update_target); the p2p
# payload carries no draft field.


def test_p6a_mtp_draft_weights_synced_in_second_session():
    """With mtp speculative config the target buckets are sent twice: once into the
    target store, once into the draft store (draft session swaps the target)."""
    weights = _source_weights()
    engine = FakeEngine(weights)
    updater = _make_updater(weights, [engine], FakeTransferEngine(), **_MTP_ARGS)
    updater.update_weights()

    target_loads = [e for e in _load_events(engine) if e[1] is False]
    draft_loads = [e for e in _load_events(engine) if e[1] is True]
    assert target_loads and draft_loads, "both a target session and a draft session"
    for name, tensor in weights.items():
        assert torch.equal(engine.ranks[0][name], tensor), name
        assert torch.equal(engine.draft_ranks[0][name], tensor), name


def test_p6a_call_order_two_sessions_per_version():
    """pause -> [flush] -> start -> target loads -> finish(v) -> start_draft ->
    target-bucket loads again -> finish(v) -> continue. start x1, finish x2."""
    weights = _source_weights()
    engine = FakeEngine(weights)
    updater = _make_updater(weights, [engine], FakeTransferEngine(), **_MTP_ARGS)
    updater.update_weights()

    kinds = [e[0] for e in engine.events]
    assert kinds[0] == "pause"
    assert kinds.count("start") == 1
    assert kinds.count("start_draft") == 1
    assert kinds.count("finish") == 2
    assert kinds[-1] == "continue"

    start = kinds.index("start")
    start_draft = kinds.index("start_draft")
    finishes = [i for i, k in enumerate(kinds) if k == "finish"]
    target_loads = [i for i, e in enumerate(engine.events) if e == ("load", False, 0)]
    draft_loads = [i for i, e in enumerate(engine.events) if e == ("load", True, 0)]
    assert target_loads and draft_loads
    assert start < min(target_loads) and max(target_loads) < finishes[0]
    assert start_draft == finishes[0] + 1, "draft session starts right after the target finish"
    assert start_draft < min(draft_loads) and max(draft_loads) < finishes[1]
    # both sessions close the same version
    assert engine.events[finishes[0]] == ("finish", "1")
    assert engine.events[finishes[1]] == ("finish", "1")


def test_p6a_stale_engine_without_draft_support_is_quarantined():
    """A stale engine build (no supports_draft_weight_update) fails loudly: its
    start_draft_weight_update raises, a non-transient control-plane error that
    quarantines the engine while healthy engines finish."""
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    engines[0].supports_draft = False
    updater = _make_updater(weights, engines, FakeTransferEngine(), **_MTP_ARGS)
    with pytest.raises(RuntimeError, match="engines \\[0\\]"):
        updater.update_weights()
    # target session landed fine on both; the draft session failed on engine 0
    for name, tensor in weights.items():
        assert torch.equal(engines[0].ranks[0][name], tensor), name
        assert torch.equal(engines[1].draft_ranks[0][name], tensor), name
    assert engines[0].shutdown_called
    assert engines[1].resumed


def test_p6a_no_spec_config_never_starts_a_draft_session():
    weights = _source_weights()
    engine = FakeEngine(weights)
    draft_snapshot = {n: t.clone() for n, t in engine.draft_ranks[0].items()}
    updater = _make_updater(weights, [engine], FakeTransferEngine())
    updater.update_weights()
    assert ("start_draft",) not in engine.events
    assert all(e[1] is False for e in _load_events(engine))
    for name, snapshot in draft_snapshot.items():
        assert torch.equal(engine.draft_ranks[0][name], snapshot), name


def test_p6a_draft_checksum_tamper_quarantines_engine():
    """A bit flip in the DRAFT session (target writes intact) is caught by the
    checksum verify and quarantines the engine exactly like a target error."""
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    transfer_engine = FakeTransferEngine()
    updater = _make_updater(weights, engines, transfer_engine, **_MTP_ARGS)
    session0 = engines[0].get_p2p_transfer_info()[0]["session_id"]
    # one bucket => one write per session: corrupt only the draft-session writes
    transfer_engine.corrupt_writes_to(session0, after_n_writes=1)
    with pytest.raises(RuntimeError, match="engines \\[0\\]"):
        updater.update_weights()
    for name, tensor in weights.items():
        assert torch.equal(engines[0].ranks[0][name], tensor), name  # target store is correct
    assert engines[0].shutdown_called
    assert engines[1].resumed
    for name, tensor in weights.items():
        assert torch.equal(engines[1].draft_ranks[0][name], tensor), name


def test_p6a_validation_accepts_mtp_and_rejects_dspark():
    """Pinned argument-validation contract (helper lives in p2p/arguments.py,
    called from vime_validate_args; reads vllm_speculative_config['method'] like
    the distributed path): p2p + mtp is accepted (trained MTP layers must reach
    the draft); p2p + dspark stays rejected with a clear message."""
    from vime.backends.megatron_utils.update_weight.p2p.arguments import (
        validate_p2p_spec_decode_args,
    )

    mtp = SimpleNamespace(
        update_weight_transport="p2p",
        enable_mtp_training=True,
        vllm_speculative_config={"method": "mtp"},
    )
    validate_p2p_spec_decode_args(mtp)  # must not raise

    dspark = SimpleNamespace(
        update_weight_transport="p2p",
        enable_mtp_training=False,
        vllm_speculative_config={"method": "dspark"},
    )
    with pytest.raises(ValueError, match="dspark"):
        validate_p2p_spec_decode_args(dspark)


# --- P6b: degrade instead of raise ----------------------------------------------


@_P6B
def test_p6b_continue_mode_quarantines_and_finishes_versions():
    """3 engines, engine 1 fails at v1, mode=continue: no raise; engines 0/2
    finish v1; engine 1 shut down exactly once; v2 skips engine 1 entirely."""
    weights = _source_weights()
    engines = [FakeEngine(weights) for _ in range(3)]
    transfer_engine = FakeTransferEngine()
    updater = _make_updater(
        weights, engines, transfer_engine, p2p_on_engine_error="continue", p2p_min_healthy_engines=1
    )
    transfer_engine.fail_with = -1
    transfer_engine.fail_sessions = {engines[1].get_p2p_transfer_info()[0]["session_id"]}
    updater.update_weights()  # no raise

    assert engines[1].shutdown_calls == 1
    assert engines[0].resumed and engines[2].resumed
    assert engines[0].weight_version == "1" and engines[2].weight_version == "1"

    transfer_engine.fail_with = None
    events_before = list(engines[1].events)
    updater.update_weights()
    assert engines[0].weight_version == "2" and engines[2].weight_version == "2"
    assert engines[1].events == events_before, "quarantined engine gets no calls at v2"
    assert engines[1].shutdown_calls == 1, "shutdown exactly once"


@_P6B
def test_p6b_continue_mode_raises_below_min_healthy():
    """2 of 3 engines fail with min_healthy=2 -> raise naming both indices + version."""
    weights = _source_weights()
    engines = [FakeEngine(weights) for _ in range(3)]
    transfer_engine = FakeTransferEngine()
    updater = _make_updater(
        weights, engines, transfer_engine, p2p_on_engine_error="continue", p2p_min_healthy_engines=2
    )
    transfer_engine.fail_with = -1
    transfer_engine.fail_sessions = {
        engines[0].get_p2p_transfer_info()[0]["session_id"],
        engines[1].get_p2p_transfer_info()[0]["session_id"],
    }
    with pytest.raises(RuntimeError, match=r"engines \[0, 1\].*version 1"):
        updater.update_weights()
    assert engines[2].resumed


def test_p6b_fail_mode_still_raises():
    """Default (fail) mode is unchanged: any error this version raises."""
    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    transfer_engine = FakeTransferEngine()
    updater = _make_updater(weights, engines, transfer_engine)  # default p2p_on_engine_error="fail"
    transfer_engine.fail_with = -1
    transfer_engine.fail_sessions = {engines[0].get_p2p_transfer_info()[0]["session_id"]}
    with pytest.raises(RuntimeError, match="engines \\[0\\]"):
        updater.update_weights()
    assert engines[1].resumed


@_P6B
def test_p6b_continue_quarantine_is_union_across_trainer_ranks(monkeypatch):
    """Engine 0's error is only 'seen on another rank' (mocked gloo union): under
    continue mode it is still quarantined and skipped at v2, without raising."""
    import vime.backends.megatron_utils.update_weight.update_weight_from_p2p as p2p_updater

    weights = _source_weights()
    engines = [FakeEngine(weights), FakeEngine(weights)]
    updater = _make_updater(weights, engines, FakeTransferEngine(), p2p_on_engine_error="continue")
    monkeypatch.setattr(p2p_updater, "_union_over_gloo", lambda local: local | {0})
    updater.update_weights()  # no raise; engine 0 quarantined via the union
    assert engines[0].shutdown_calls == 1
    assert engines[1].weight_version == "1"
    events_before = list(engines[0].events)
    updater.update_weights()
    assert engines[0].events == events_before, "union-quarantined engine gets no calls at v2"
    assert engines[1].weight_version == "2"


@_P6B
def test_p6b_continue_with_single_engine_rejected_at_validation():
    """continue with one engine is meaningless (min_healthy=1 can never hold):
    rejected at argument validation."""
    from vime.backends.megatron_utils.update_weight.p2p.arguments import (
        validate_p2p_degrade_args,
    )

    validate = validate_p2p_degrade_args
    ok = SimpleNamespace(p2p_on_engine_error="continue", p2p_min_healthy_engines=1, rollout_num_engines=3)
    validate(ok)  # must not raise
    single = SimpleNamespace(p2p_on_engine_error="continue", p2p_min_healthy_engines=1, rollout_num_engines=1)
    with pytest.raises(ValueError, match="single rollout engine"):
        validate(single)
    fail_mode = SimpleNamespace(p2p_on_engine_error="fail", p2p_min_healthy_engines=1, rollout_num_engines=1)
    validate(fail_mode)  # fail mode with one engine is fine

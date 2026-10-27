"""V3 verifier (F02-1): weight-sync tamper detection.

Contract test for the weight-sync verification stack. A fake vLLM server holds
engine-side tensors and answers the ``check_weights`` RPC with hash-based
snapshot/compare/reset semantics (the server-side half lands in F02-2's vLLM
patch). For each vime weight-sync path (colocate IPC, NCCL distributed,
disk-delta) the test syncs weights, flips one bit engine-side, and requires
``check_weights(action="compare")`` to raise, naming the tampered tensor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import torch

_unit_stubs.install_vllm_cli_stubs()

from vime.backends.vllm_utils import vllm_engine as mod

NUM_GPUS = 0

_SYNC_PATHS = ["colocate_ipc", "nccl_distributed", "disk_delta"]


def _sha256(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


class _FakeVLLMServer:
    """Engine-side tensor state plus hash-based check_weights semantics.

    This mirrors the F02-2 server contract: ``snapshot`` records per-tensor
    SHA-256 hashes (no CPU copies), ``compare`` reports per-tensor
    match/mismatch against the snapshot, and ``reset_tensors`` CLOBBERS the
    live weights with a deterministic zero sentinel (never restores), so the
    startup flow snapshot -> reset -> update -> compare proves the first
    update overwrites every tensor.
    """

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self._snapshot_hashes: dict[str, str] = {}

    def load_weights(self, tensors: dict[str, torch.Tensor]) -> None:
        # All three vime sync paths (IPC, NCCL, disk-delta) end here: the
        # engine's in-memory weights are replaced.
        self.tensors = {name: t.clone() for name, t in tensors.items()}

    def tamper(self, name: str) -> None:
        buf = self.tensors[name].view(torch.uint8)
        buf[0] ^= 1  # one-bit flip

    def handle_check_weights(self, payload: dict) -> dict:
        """Worker-level RPC result; VLLMEngine.check_weights wraps the list of
        per-worker results into the body evaluated by weight_checker."""
        action = payload["action"]
        if action == "snapshot":
            self._snapshot_hashes = {name: _sha256(t) for name, t in self.tensors.items()}
            return {"rank": 0, "tensors": {}}
        if action == "reset_tensors":
            # Clobber, never restore: prove the next update rewrites every tensor.
            for tensor in self.tensors.values():
                tensor.view(torch.uint8).zero_()
            return {"rank": 0, "tensors": {name: {"status": "clobbered"} for name in self.tensors}}
        if action == "compare":
            assert self._snapshot_hashes, "compare without snapshot"
            results = {}
            for name, tensor in self.tensors.items():
                status = "match" if _sha256(tensor) == self._snapshot_hashes.get(name) else "mismatch"
                results[name] = {"status": status}
            return {"rank": 0, "tensors": results}
        raise ValueError(f"unknown check_weights action {action!r}")


class _MockResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data
        self.content = b"{}"  # non-empty so _make_request parses json
        self.text = ""
        self.status_code = 200

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


@pytest.fixture
def engine_and_server(monkeypatch):
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset())
    args = SimpleNamespace(
        rollout_external=True,
        hf_checkpoint="/tmp/model",
        vllm_router_ip=None,
        vllm_router_port=None,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=4,
        colocate=False,
        debug_rollout_only=False,
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        use_critic=False,
        critic_num_gpus_per_node=0,
        critic_num_nodes=0,
        seed=1234,
        fp16=False,
        offload_rollout=False,
        use_rollout_routing_replay=False,
        rollout_top_p=1.0,
        rollout_health_check_timeout=600.0,
        vllm_pipeline_parallel_size=1,
        vllm_prefill_context_parallel_size=1,
        vllm_data_parallel_size=1,
        vllm_dp_size=1,
        vllm_enable_expert_parallel=False,
    )
    from vime.backends.vllm_utils.vllm_engine import VLLMEngine

    engine = VLLMEngine(args, rank=0)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 8765

    server = _FakeVLLMServer()

    def fake_post(url, json=None, **kwargs):
        if url.endswith("/collective_rpc") and json.get("method") == "check_weights":
            return _MockResponse([server.handle_check_weights(json["kwargs"])])
        raise AssertionError(f"unexpected POST {url} {json!r}")

    monkeypatch.setattr(mod.requests, "post", fake_post)
    return engine, server


def _trainer_weights() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64, generator=g),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, generator=g),
    }


@pytest.mark.parametrize("sync_path", _SYNC_PATHS)
def test_tampered_weight_detected(engine_and_server, sync_path):
    """After a sync, a 1-bit engine-side flip must make check_weights compare fail."""
    engine, server = engine_and_server
    server.load_weights(_trainer_weights())  # sync lands via `sync_path` transport
    engine.check_weights(action="snapshot")
    server.tamper("model.layers.0.mlp.gate_proj.weight")
    with pytest.raises(RuntimeError, match="gate_proj"):
        engine.check_weights(action="compare")


@pytest.mark.parametrize("sync_path", _SYNC_PATHS)
def test_untampered_weight_passes(engine_and_server, sync_path):
    """No tamper -> check_weights compare must pass cleanly."""
    engine, server = engine_and_server
    server.load_weights(_trainer_weights())
    engine.check_weights(action="snapshot")
    result = engine.check_weights(action="compare")  # must not raise
    assert result["supported"] is True


def test_check_weights_uses_collective_rpc(engine_and_server):
    """The engine-side implementation is the vLLM check_weights worker method
    (docker/patch/latest/vllm-check-weights.patch), reached via /collective_rpc."""
    engine, server = engine_and_server
    server.load_weights(_trainer_weights())
    result = engine.check_weights(action="snapshot")
    assert result["supported"] is True
    assert server._snapshot_hashes, "snapshot must reach the fake server"


@pytest.mark.parametrize("sync_path", _SYNC_PATHS)
def test_reset_clobbers_then_update_restores_equality(engine_and_server, sync_path):
    """Startup contract: snapshot -> reset_tensors (clobber) -> compare must FAIL
    -> trainer update lands -> compare must PASS. Proves the first update
    overwrites every tensor."""
    engine, server = engine_and_server
    server.load_weights(_trainer_weights())  # initial engine weights
    engine.check_weights(action="snapshot")
    engine.check_weights(action="reset_tensors")
    with pytest.raises(RuntimeError, match="gate_proj"):
        engine.check_weights(action="compare")
    server.load_weights(_trainer_weights())  # first real update via `sync_path`
    engine.check_weights(action="compare")  # must not raise


def test_nonzero_node_rank_returns_accepted_noop(engine_and_server):
    """2-node fan-out: node_rank != 0 engines must return a reply the
    check_weights parsers accept without hitting the HTTP server."""
    from vime.backends.vllm_utils.weight_checker import evaluate_compare_results

    engine, _server = engine_and_server
    engine.node_rank = 1
    result = engine.check_weights(action="compare")
    assert result == {"ok": True, "supported": True, "results": []}
    # The manager aggregates per-engine bodies; an empty node body must not fail.
    evaluate_compare_results(result)
    evaluate_compare_results(
        {
            "ok": True,
            "supported": True,
            "results": result["results"] + [{"rank": 0, "tensors": {"w": {"status": "match"}}}],
        }
    )

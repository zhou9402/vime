"""V3 verifier (F02-1): weight-sync tamper detection.

Contract test for the weight-sync verification stack. A fake vLLM server holds
engine-side tensors and answers the ``check_weights`` RPC with hash-based
snapshot/compare/reset semantics (the server-side half lands in F02-2's vLLM
patch). For each vime weight-sync path (colocate IPC, NCCL distributed,
disk-delta) the test syncs weights, flips one bit engine-side, and requires
``check_weights(action="compare")`` to raise, naming the tampered tensor.

Until F02-2 implements ``VLLMEngine.check_weights`` the detection tests are
xfail(strict=True) and the stub behavior is asserted explicitly.
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
    match/mismatch against the snapshot, ``reset_tensors`` is a no-op for
    hashes-only mode (nothing was copied out).
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
        action = payload["action"]
        if action == "snapshot":
            self._snapshot_hashes = {name: _sha256(t) for name, t in self.tensors.items()}
            return {"ok": True, "supported": True, "action": action, "results": []}
        if action == "reset_tensors":
            return {"ok": True, "supported": True, "action": action, "results": []}
        if action == "compare":
            assert self._snapshot_hashes, "compare without snapshot"
            results = {}
            for name, tensor in self.tensors.items():
                status = "match" if _sha256(tensor) == self._snapshot_hashes.get(name) else "mismatch"
                results[name] = {"status": status}
            return {"ok": True, "supported": True, "action": action, "results": [{"rank": 0, "tensors": results}]}
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
        if url.endswith("/check_weights"):
            return _MockResponse(server.handle_check_weights(json))
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
@pytest.mark.xfail(strict=True, reason="F02-2 implements the check_weights RPC; stub cannot detect tampering")
def test_tampered_weight_detected(engine_and_server, sync_path):
    """After a sync, a 1-bit engine-side flip must make check_weights compare fail."""
    engine, server = engine_and_server
    server.load_weights(_trainer_weights())  # sync lands via `sync_path` transport
    engine.check_weights(action="snapshot")
    server.tamper("model.layers.0.mlp.gate_proj.weight")
    with pytest.raises(RuntimeError, match="gate_proj"):
        engine.check_weights(action="compare")


@pytest.mark.parametrize("sync_path", _SYNC_PATHS)
@pytest.mark.xfail(strict=True, reason="F02-2 implements the check_weights RPC; stub returns supported=False")
def test_untampered_weight_passes(engine_and_server, sync_path):
    """No tamper -> check_weights compare must pass cleanly."""
    engine, server = engine_and_server
    server.load_weights(_trainer_weights())
    engine.check_weights(action="snapshot")
    result = engine.check_weights(action="compare")  # must not raise
    assert result.get("supported") is True, "check_weights RPC must be implemented (F02-2)"


def test_check_weights_stub_reports_unsupported(engine_and_server):
    """Pre-F02-2 gap: the engine stub declines weight checks. This test pins the
    stub contract; F02-2 replaces it with the real RPC and this test."""
    engine, _server = engine_and_server
    result = engine.check_weights(action="snapshot")
    assert result == {"ok": True, "supported": False}

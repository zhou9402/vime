"""CPU unit tests for vllm_engine LoRA enablement (F04-6).

Self-contained stubbing: works without vllm, and skips when an installed vllm
is too old for vime's pinned API (vllm.utils.system_utils).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest


def _install_vllm_engine_stubs() -> None:
    _unit_stubs.install_vllm_cli_stubs()
    # vllm_engine imports vllm.utils.system_utils directly; provide it when absent.
    try:
        import vllm.utils.system_utils  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.pop("vllm", None)  # drop a partial stub package if present
        vllm_mod = types.ModuleType("vllm")
        vllm_mod.__path__ = []
        utils_mod = types.ModuleType("vllm.utils")
        system_utils = types.ModuleType("vllm.utils.system_utils")
        system_utils.kill_process_tree = MagicMock()
        utils_mod.system_utils = system_utils
        vllm_mod.utils = utils_mod
        sys.modules.setdefault("vllm", vllm_mod)
        sys.modules["vllm.utils"] = utils_mod
        sys.modules["vllm.utils.system_utils"] = system_utils


_install_vllm_engine_stubs()

pytestmark = pytest.mark.skipif(
    _unit_stubs.real_module_available("vllm") and not _unit_stubs.real_module_available("vllm.utils.system_utils"),
    reason="installed vllm is incompatible with vime's pinned vllm API",
)

from vime.backends.vllm_utils import vllm_engine as mod

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def _patch_vllm_server_fields(monkeypatch):
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset())


@pytest.fixture
def vllm_args() -> SimpleNamespace:
    return SimpleNamespace(
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


def _lora_args(vllm_args, **overrides):
    values = dict(
        lora_rank=8,
        lora_adapter_path=None,
        lora_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
    )
    for key, value in (values | overrides).items():
        setattr(vllm_args, key, value)
    return vllm_args


def test_compute_server_args_lora_disabled_by_default(vllm_args):
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert "enable_lora" not in sa
    assert "lora_modules" not in sa


def test_compute_server_args_lora_enabled_without_disk_load(vllm_args, caplog):
    import logging

    sa, _ = mod._compute_server_args(
        _lora_args(vllm_args), rank=0, dist_init_addr=None, host="127.0.0.1", port=8000
    )
    assert sa["enable_lora"] is True
    assert sa["max_lora_rank"] == 8
    assert sa["max_loras"] == 1
    assert "lora_modules" not in sa  # the trainer pushes the adapter in the first weight sync


def test_compute_server_args_lora_disk_load_when_no_trainer_push(vllm_args):
    sa, _ = mod._compute_server_args(
        _lora_args(vllm_args, lora_adapter_path="/tmp/adapter", debug_rollout_only=True),
        rank=0,
        dist_init_addr=None,
        host="127.0.0.1",
        port=8000,
    )
    assert sa["lora_modules"] == [
        {"name": "vime_lora", "path": "/tmp/adapter", "base_model_name": "/tmp/model"}
    ]


def test_compute_server_args_lora_train_only_stays_off(vllm_args):
    sa, _ = mod._compute_server_args(
        _lora_args(vllm_args, lora_train_only=True), rank=0, dist_init_addr=None, host="127.0.0.1", port=8000
    )
    assert "enable_lora" not in sa


def test_build_subprocess_env_enables_runtime_lora_updating(vllm_args, monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)
    env = mod._build_subprocess_env(
        {"_args": _lora_args(vllm_args), "_visible_devices": "0"}
    )
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "1"
    env = mod._build_subprocess_env({"_args": _lora_args(vllm_args, lora_rank=0), "_visible_devices": "0"})
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in env

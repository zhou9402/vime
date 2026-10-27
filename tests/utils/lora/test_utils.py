"""Tests for vime.utils.lora.utils: gating predicates and adapter disk round-trip."""

import json
from types import SimpleNamespace

import torch

from vime.utils.lora.utils import (
    LORA_ADAPTER_NAME,
    build_lora_config,
    engine_loads_adapter_from_disk,
    get_adapter_target_modules,
    is_lora_enabled,
    is_lora_weight_name,
    lora_rollout_enabled,
    matches_lora_target,
    save_adapter_to_disk,
    targets_expert_leaves,
)


def _args(**overrides):
    args = dict(
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_adapter_path=None,
        lora_train_only=False,
        colocate=False,
        offload_rollout=False,
        lora_base_cpu_backup=False,
        check_weight_update_equal=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
    )
    return SimpleNamespace(**(args | overrides))


def test_adapter_name_is_vime_scoped():
    assert LORA_ADAPTER_NAME == "vime_lora"


def test_is_lora_enabled_gating():
    assert not is_lora_enabled(_args())
    assert is_lora_enabled(_args(lora_rank=8))
    assert is_lora_enabled(_args(lora_adapter_path="/tmp/x"))


def test_lora_rollout_enabled_respects_train_only():
    assert lora_rollout_enabled(_args(lora_rank=8))
    assert not lora_rollout_enabled(_args(lora_rank=8, lora_train_only=True))


def test_engine_loads_adapter_from_disk_only_without_trainer_push():
    assert not engine_loads_adapter_from_disk(_args(lora_adapter_path="/tmp/x"))
    assert engine_loads_adapter_from_disk(_args(lora_adapter_path="/tmp/x", debug_rollout_only=True))
    assert engine_loads_adapter_from_disk(_args(lora_adapter_path="/tmp/x", debug_skip_weight_update=True))


def test_weight_name_and_target_matching():
    assert is_lora_weight_name("model.layers.0.self_attn.q_proj.lora_A.weight")
    assert not is_lora_weight_name("model.layers.0.self_attn.q_proj.weight")
    assert matches_lora_target("model.layers.0.self_attn.q_proj", "q_proj")
    assert matches_lora_target("model.layers.0.self_attn.q_proj", "model.layers.*.self_attn.q_proj")
    assert not matches_lora_target("model.layers.0.self_attn.q_proj", "k_proj")


def test_targets_expert_leaves():
    assert targets_expert_leaves("all-linear")
    assert targets_expert_leaves(["model.layers.*.mlp.gate_up_proj"])
    assert not targets_expert_leaves(["model.layers.*.self_attn.q_proj"])


def test_get_adapter_target_modules_strips_peft_prefixes():
    names = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
    ]
    assert get_adapter_target_modules(names) == [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.self_attn.q_proj",
    ]


def test_build_lora_config_shape():
    config = build_lora_config(_args(lora_rank=8, lora_alpha=32, lora_dropout=0.1), target_modules=["q_proj"])
    assert config == {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def test_save_adapter_to_disk_roundtrip(tmp_path):
    tensors = {"model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 8)}
    config = build_lora_config(_args(lora_rank=4), target_modules=["q_proj"])
    save_adapter_to_disk(tmp_path, config, tensors)

    on_disk = json.loads((tmp_path / "adapter_config.json").read_text())
    assert on_disk == config

    import safetensors.torch

    loaded = safetensors.torch.load_file(str(tmp_path / "adapter_model.safetensors"))
    assert set(loaded) == set(tensors)
    assert torch.equal(loaded["model.layers.0.self_attn.q_proj.lora_A.weight"], tensors["model.layers.0.self_attn.q_proj.lora_A.weight"])

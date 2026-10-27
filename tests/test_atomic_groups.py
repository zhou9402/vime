"""CPU unit tests for atomic update groups + size bucketing (F02-4)."""

from __future__ import annotations

import sys
from pathlib import Path

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import pytest
import torch

from vime.backends.megatron_utils.update_weight.atomic_groups import get_hf_atomic_update_groups
from vime.backends.megatron_utils.update_weight.bucketing import (
    AtomicUpdateGroup,
    assemble_atomic_update_groups,
    pack_units_by_size,
)

NUM_GPUS = 0

_Q_LORA_GROUP = AtomicUpdateGroup(
    key="q_lora_a_proj",
    suffixes=(".self_attn.q_a_proj.weight", ".self_attn.kv_a_proj_with_mqa.weight"),
)


def _unit(name: str, numel: int = 8) -> list[tuple[str, torch.Tensor]]:
    return [(name, torch.zeros(numel))]


def test_q_lora_group_members_land_in_same_bucket():
    units = [
        _unit("model.layers.0.self_attn.q_a_proj.weight"),
        _unit("model.layers.0.mlp.gate_proj.weight"),
        _unit("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"),
    ]
    grouped = list(assemble_atomic_update_groups(iter(units), [_Q_LORA_GROUP]))
    chunks = list(pack_units_by_size(iter(grouped), max_bytes=1))  # force one unit per bucket
    names_per_chunk = [[name for name, _ in chunk] for chunk in chunks]
    qkv_chunks = [names for names in names_per_chunk if any("q_a_proj" in n for n in names)]
    assert qkv_chunks == [
        ["model.layers.0.self_attn.q_a_proj.weight", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]
    ]


def test_fp8_weight_and_scale_stay_together():
    # The converter emits weight+scale as one unit; pack_units_by_size never splits units.
    weight = ("model.layers.0.mlp.gate_proj.weight", torch.zeros(8))
    scale = ("model.layers.0.mlp.gate_proj.weight_scale_inv", torch.zeros(1))
    units = [[weight, scale], _unit("model.layers.0.mlp.up_proj.weight")]
    chunks = list(pack_units_by_size(iter(units), max_bytes=8 * 4 + 4))
    assert [[name for name, _ in chunk] for chunk in chunks] == [
        ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.weight_scale_inv"],
        ["model.layers.0.mlp.up_proj.weight"],
    ]


def test_incomplete_group_raises():
    units = [_unit("model.layers.0.self_attn.q_a_proj.weight")]
    with pytest.raises(AssertionError, match="Incomplete atomic update groups"):
        list(assemble_atomic_update_groups(iter(units), [_Q_LORA_GROUP]))


def test_absent_group_is_noop_for_unregistered_models():
    units = [_unit("model.layers.0.mlp.gate_proj.weight")]
    grouped = list(assemble_atomic_update_groups(iter(units), []))
    assert grouped == units
    # And no groups are registered for non-DeepSeek architectures.
    assert get_hf_atomic_update_groups("qwen3", q_lora_rank=None) == []
    assert get_hf_atomic_update_groups("glm4moe") == []


def test_registered_group_must_match_something():
    units = [_unit("model.layers.0.mlp.gate_proj.weight")]
    with pytest.raises(AssertionError, match="matched no params"):
        list(assemble_atomic_update_groups(iter(units), [_Q_LORA_GROUP]))


def test_get_hf_atomic_update_groups_only_for_deepseek_family_with_q_lora():
    groups = get_hf_atomic_update_groups("deepseekv3", q_lora_rank=1536)
    assert [g.key for g in groups] == ["q_lora_a_proj"]
    assert get_hf_atomic_update_groups("deepseekv3", q_lora_rank=None) == []
    assert get_hf_atomic_update_groups("qwen3", q_lora_rank=1536) == []
    assert [g.key for g in get_hf_atomic_update_groups("kimi-k3", q_lora_rank=1536)] == ["q_lora_a_proj"]


def test_duplicate_member_raises():
    units = [
        _unit("model.layers.0.self_attn.q_a_proj.weight"),
        _unit("model.layers.0.self_attn.q_a_proj.weight"),
    ]
    with pytest.raises(AssertionError, match="Duplicate member"):
        list(assemble_atomic_update_groups(iter(units), [_Q_LORA_GROUP]))


def test_oversized_unit_forms_its_own_bucket():
    big = [("w", torch.zeros(1024))]
    small = _unit("s")
    chunks = list(pack_units_by_size(iter([small, big, small]), max_bytes=64))
    assert [[name for name, _ in chunk] for chunk in chunks] == [["s"], ["w"], ["s"]]

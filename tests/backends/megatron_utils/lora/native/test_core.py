"""Tests for the native LoRA core: config, distributed helpers, naming, spec contracts.

CPU-only; TP=1 paths. Megatron is not imported on these paths (TP>1 branches
import lazily).
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vime.backends.megatron_utils.lora.native.config import LoRAConfig
from vime.backends.megatron_utils.lora.native.distributed import (
    ParallelGather,
    apply_lora_dropout,
    branch_input,
    reduce_row_parallel,
)
from vime.backends.megatron_utils.lora.native.naming import ConverterNaming, DeclaredNaming
from vime.backends.megatron_utils.lora.native.spec.base import (
    AttachContext,
    AttentionFamily,
    FixedTargets,
    LoRAArchSpec,
    ProjectionSpec,
    ShardLayout,
)

NUM_GPUS = 0


def _config(**overrides):
    values = dict(rank=4, alpha=16, dropout=0.0, target_modules=None, a_init_method="xavier")
    return LoRAConfig(**(values | overrides))


def _context(**overrides):
    values = dict(
        lora=_config(),
        transformer_config=SimpleNamespace(sequence_parallel=False, gated_linear_unit=True),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    return AttachContext(**(values | overrides))


class TestLoRAConfig:
    def test_scale(self):
        assert _config(rank=4, alpha=16).scale == 4.0

    def test_selects_none_means_all(self):
        assert _config().selects("model.layers.0.self_attn.q_proj")

    def test_selects_matches_targets(self):
        config = _config(target_modules=("q_proj",))
        assert config.selects("model.layers.0.self_attn.q_proj")
        assert not config.selects("model.layers.0.self_attn.k_proj")

    def test_from_args(self):
        args = SimpleNamespace(
            lora_rank=8, lora_alpha=32, lora_dropout=0.1, hf_lora_targets=["q_proj"], lora_A_init_method="kaiming"
        )
        config = LoRAConfig.from_args(args, select_all=False)
        assert (config.rank, config.alpha, config.dropout, config.a_init_method) == (8, 32, 0.1, "kaiming")
        assert config.target_modules == ("q_proj",)
        assert LoRAConfig.from_args(args, select_all=True).target_modules is None


class TestDistributed:
    def test_dropout_noop_when_zero_or_eval(self):
        x = torch.randn(4, 8)
        context = _context()
        assert torch.equal(apply_lora_dropout(x, context, training=True), x)
        context = _context(lora=_config(dropout=0.5))
        assert torch.equal(apply_lora_dropout(x, context, training=False), x)

    def test_branch_input_tp1_passthrough(self):
        x = torch.randn(4, 8)
        assert torch.equal(branch_input(x, nn.Linear(8, 8), _context()), x)

    def test_reduce_row_parallel_tp1_passthrough(self):
        x = torch.randn(4, 8)
        assert torch.equal(reduce_row_parallel(x, _context()), x)

    def test_parallel_gather_without_dist_returns_locals(self):
        assert not torch.distributed.is_initialized()
        gather = ParallelGather()
        a = torch.randn(2, 3)
        b = torch.randn(4)
        ra = gather.request(a, cat_dim=0, group="tp")
        rb = gather.request(b, cat_dim=0, group="ep")
        with pytest.raises(AssertionError, match="flush"):
            ra()
        gather.flush()
        assert torch.equal(ra(), a)
        assert torch.equal(rb(), b)
        with pytest.raises(AssertionError, match="unknown gather group"):
            gather.request(a, cat_dim=0, group="pp")


class TestNaming:
    def test_declared_naming_prefix_match(self):
        naming = DeclaredNaming(
            layer_prefix="model.layers.",
            hf_blocks_by_param_prefix={"self_attention.linear_qkv": "self_attn.qkv."},
        )
        layer = nn.Module()
        layer.layer_number = 5
        layer.self_attention = nn.Module()
        layer.self_attention.linear_qkv = nn.Linear(8, 8)
        prefix = naming.block_prefix(layer, layer.self_attention.linear_qkv.weight, leaves=[])
        assert prefix == "model.layers.4.self_attn.qkv."

    def test_converter_naming_uses_hf_converter(self, monkeypatch):
        layer = nn.Module()
        layer.layer_number = 3
        layer.self_attention = nn.Module()
        layer.self_attention.linear_qkv = nn.Linear(4, 12, bias=False)
        weight = layer.self_attention.linear_qkv.weight
        weight.tensor_model_parallel = True
        weight.partition_dim = 0

        fake_pkg = types.ModuleType("vime.backends.megatron_utils.megatron_to_hf")

        def convert_to_hf_names(args, model_name, name, param):
            assert param.is_meta
            assert list(param.shape) == [12 * 2, 4]  # global shape: tp_size * local shard
            assert name == "module.module.decoder.layers.2.self_attention.linear_qkv.weight"
            return [
                "model.layers.2.self_attn.q_proj.weight",
                "model.layers.2.self_attn.k_proj.weight",
                "model.layers.2.self_attn.v_proj.weight",
            ]

        fake_pkg.convert_to_hf_names = convert_to_hf_names
        monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.megatron_to_hf", fake_pkg)

        naming = ConverterNaming(
            args=SimpleNamespace(hf_checkpoint="/tmp/fake"), model_name="qwen3", tp_size=2, is_hf_name=lambda n: True
        )
        prefix = naming.block_prefix(layer, weight, leaves=["q_proj", "k_proj", "v_proj"])
        assert prefix == "model.layers.2.self_attn."


class TestSpecContracts:
    def test_enums_and_projection_spec(self):
        spec = ProjectionSpec(hf_leaf="q_proj", param_prefix="q", layout=ShardLayout.COLUMN)
        assert spec.layout is ShardLayout.COLUMN
        assert AttentionFamily("gqa") is AttentionFamily.GQA

    def test_attach_context_sequence_parallel_property(self):
        assert not _context().sequence_parallel
        gated = _context(transformer_config=SimpleNamespace(sequence_parallel=True, attention_output_gate=True))
        assert gated.sequence_parallel
        assert gated.output_gate
        full = AttachContext(
            lora=_config(),
            transformer_config=SimpleNamespace(sequence_parallel=True),
            tp_size=1,
            tp_rank=0,
            naming=None,
            full_sequence=True,
        )
        assert not full.sequence_parallel

    def test_arch_spec_attaches_and_validates(self):
        attention = SimpleNamespace(
            hf_block="self_attn",
            supported_targets=frozenset({"q_proj", "o_proj"}),
            validate=lambda config, *, tp_size: None,
        )
        mlp = SimpleNamespace(supported_targets=frozenset({"gate_proj"}))
        spec = LoRAArchSpec(name="toy", attention=attention, mlp=mlp, fixed_targets=FixedTargets(serving=list))
        assert spec.attaches("model.layers.*.self_attn.q_proj")
        assert spec.attaches("model.layers.3.mlp.gate_proj")
        assert not spec.attaches("model.layers.*.self_attn.k_proj")
        spec.validate(_context())
        with pytest.raises(AssertionError, match="gated"):
            spec.validate(_context(transformer_config=SimpleNamespace(sequence_parallel=False, gated_linear_unit=False)))

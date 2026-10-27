"""Tests for native-LoRA architecture specs (GQA attention, split gate/up MLP) and merging — no GPU.

Port-equivalent of miles tests/.../native/test_specs.py and the non-MoE part of
test_merge.py; miles' fixture-based versions need its pydantic args layer, so
these use bare blocks + the naming stub.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

from naming_stub import named

from vime.backends.megatron_utils.lora.native.config import LoRAConfig
from vime.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from vime.backends.megatron_utils.lora.native.modules.linear import iter_adapters
from vime.backends.megatron_utils.lora.native.spec.attention import GQAAttentionSpec
from vime.backends.megatron_utils.lora.native.spec.base import AttachContext
from vime.backends.megatron_utils.lora.native.spec.mlp import SplitGateUpMLPSpec

NUM_GPUS = 0

HIDDEN, HEADS, GROUPS, HEAD_DIM, INTER = 8, 4, 2, 2, 6


class _MCoreLinear(nn.Module):
    """MCore linear contract: forward returns (out, bias)."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = _MCoreLinear((HEADS + 2 * GROUPS) * HEAD_DIM, HIDDEN)
        self.linear_proj = _MCoreLinear(HIDDEN, HEADS * HEAD_DIM)
        self.num_attention_heads_per_partition = HEADS
        self.num_query_groups_per_partition = GROUPS
        self.hidden_size_per_attention_head = HEAD_DIM


class _MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = _MCoreLinear(2 * INTER, HIDDEN)
        self.linear_fc2 = _MCoreLinear(HIDDEN, INTER)


def _context(targets=None, prefix="model.layers.0.self_attn."):
    context = AttachContext(
        lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=targets),
        transformer_config=SimpleNamespace(hidden_size=HIDDEN, sequence_parallel=False, num_query_groups=GROUPS),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    return named(context, prefix)


def _layer(block):
    layer = nn.Module()
    layer.layer_number = 1
    layer.block = block
    return layer


class TestGQAAttentionSpec:
    def test_attach_and_zero_init_is_noop(self):
        block, layer = _AttentionBlock(), None
        layer = _layer(block)
        spec = GQAAttentionSpec()
        assert spec.applies_to(block)
        count = spec.attach(block, layer, _context())
        assert count == 2  # fused qkv + o_proj

        x = torch.randn(3, HIDDEN)
        base_out, _ = block.linear_qkv(x)
        proj_in = torch.randn(3, HEADS * HEAD_DIM)
        base_proj, _ = block.linear_proj(proj_in)
        adapters = list(iter_adapters([layer]))
        assert len(adapters) == 2
        assert {a.hf_prefix for a in adapters} == {"model.layers.0.self_attn."}
        # B is zero-initialized: forward must be unchanged
        out, _ = block.linear_qkv(x)
        torch.testing.assert_close(out, base_out)
        proj, _ = block.linear_proj(proj_in)
        torch.testing.assert_close(proj, base_proj)

    def test_nonzero_adapter_changes_and_trains(self):
        block = _AttentionBlock()
        layer = _layer(block)
        GQAAttentionSpec().attach(block, layer, _context())
        x = torch.randn(3, HIDDEN)
        base_out, _ = _AttentionBlock().linear_qkv(x)  # different weights; just shape ref
        base_out, _ = block.linear_qkv(x)

        with torch.no_grad():
            for adapter in iter_adapters([layer]):
                for name, param in adapter.named_parameters():
                    param.normal_(0, 0.1)
        out, _ = block.linear_qkv(x)
        assert not torch.allclose(out, base_out)

        proj_out, _ = block.linear_proj(torch.randn(3, HEADS * HEAD_DIM))
        loss = out.square().sum() + proj_out.square().sum()
        loss.backward()
        grads = {
            name: param.grad for adapter in iter_adapters([layer]) for name, param in adapter.named_parameters()
        }
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads.values())

    def test_target_selection_skips_unselected(self):
        block = _AttentionBlock()
        layer = _layer(block)
        context = _context(targets=("model.layers.*.self_attn.o_proj",))
        count = GQAAttentionSpec().attach(block, layer, context)
        assert count == 1
        adapters = list(iter_adapters([layer]))
        assert len(adapters) == 1

    def test_validate_rejects_too_few_query_groups(self):
        import pytest

        config = SimpleNamespace(num_query_groups=1)
        with pytest.raises(AssertionError, match="query group"):
            GQAAttentionSpec().validate(config, tp_size=2)


class TestSplitGateUpMLPSpec:
    def test_attach_forward_backward(self):
        block = _MLPBlock()
        layer = _layer(block)
        spec = SplitGateUpMLPSpec()
        assert spec.applies_to(block)
        count = spec.attach(block, layer, _context(prefix="model.layers.0.mlp."))
        assert count == 2  # fused [gate; up] + down

        x = torch.randn(3, HIDDEN)
        base_fc1, _ = block.linear_fc1(x)
        adapters = list(iter_adapters([layer]))
        assert {a.hf_prefix for a in adapters} == {"model.layers.0.mlp."}
        out, _ = block.linear_fc1(x)
        torch.testing.assert_close(out, base_fc1)  # zero-init no-op

        with torch.no_grad():
            for adapter in adapters:
                for param in adapter.parameters():
                    param.normal_(0, 0.1)
        out, _ = block.linear_fc1(x)
        assert not torch.allclose(out, base_fc1)
        fc2_out, _ = block.linear_fc2(torch.randn(3, INTER))
        (out.square().sum() + fc2_out.square().sum()).backward()
        assert all(p.grad is not None for a in adapters for p in a.parameters())


class TestMerge:
    def test_merged_weights_view_adds_deltas(self):
        block = _MLPBlock()
        layer = _layer(block)
        SplitGateUpMLPSpec().attach(block, layer, _context(prefix="model.layers.0.mlp."))
        with torch.no_grad():
            for adapter in iter_adapters([layer]):
                for param in adapter.parameters():
                    param.normal_(0, 0.1)

        weights = {
            "linear_fc1.weight": block.linear_fc1.weight,
            "linear_fc2.weight": block.linear_fc2.weight,
            "other": torch.randn(2, 2),
        }
        merged = merge_lora_into_weights([layer], weights)
        assert len(merged) == 3
        assert torch.equal(merged["other"], weights["other"])
        assert not torch.allclose(merged["linear_fc1.weight"], weights["linear_fc1.weight"])
        # merge is additive in fp32 then cast back
        fc1_delta = [d for a in iter_adapters([layer]) for w, d in a.weight_deltas() if w is block.linear_fc1.weight]
        assert len(fc1_delta) == 1
        torch.testing.assert_close(
            merged["linear_fc1.weight"],
            (weights["linear_fc1.weight"].float() + fc1_delta[0]()).to(weights["linear_fc1.weight"].dtype),
        )

    def test_merge_rejects_unknown_host(self):
        block = _MLPBlock()
        layer = _layer(block)
        SplitGateUpMLPSpec().attach(block, layer, _context(prefix="model.layers.0.mlp."))
        import pytest

        with pytest.raises(AssertionError, match="not exported"):
            merge_lora_into_weights([layer], {"unrelated": torch.randn(2, 2)})

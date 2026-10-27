"""Kimi-K3 converter round-trip + cross-check vs the mbridge bridge tables
(vime_plugins/mbridge/kimi_k3.py, verbatim from miles@ba28991834).

The bridge's _DIRECT/_ATTENTION/_MLP/_OTHER mapping dicts are vendored below
(BRIDGE_*); update them in the same PR if the bridge changes.
"""

import pytest
import torch

from tests.test_hf_to_megatron import DictReader
from vime.backends.megatron_utils.hf_to_megatron.kimi_k3 import kimi_k3_hf_tensor
from vime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core

_P = "language_model.model"

# Vendored from KimiK3Bridge (miles@ba28991834): DSv4-style direct names.
BRIDGE_DIRECT = {
    "embedding.word_embeddings.weight": f"{_P}.embed_tokens.weight",
    "decoder.final_layernorm.weight": f"{_P}.norm.weight",
    "output_layer.weight": "language_model.lm_head.weight",
}

# Bridge _ATTENTION_MAPPING + _MLP_MAPPING + residual names (single-key entries only;
# multi-key splits/merges are checked in the round-trip test below).
BRIDGE_LAYER_SINGLE = {
    "input_layernorm.weight": "input_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "self_attention_res_norm.weight": "self_attention_res_norm.weight",
    "self_attention_res_proj.weight": "self_attention_res_proj.weight",
    "mlp_res_norm.weight": "mlp_res_norm.weight",
    "mlp_res_proj.weight": "mlp_res_proj.weight",
    "self_attention.linear_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attention.linear_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attention.linear_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attention.linear_attn.q_conv1d.weight": "self_attn.q_conv1d.weight",
    "self_attention.linear_attn.k_conv1d.weight": "self_attn.k_conv1d.weight",
    "self_attention.linear_attn.v_conv1d.weight": "self_attn.v_conv1d.weight",
    "self_attention.linear_attn.A_log": "self_attn.A_log",
    "self_attention.linear_attn.dt_bias": "self_attn.dt_bias",
    "self_attention.linear_attn.f_a_proj.weight": "self_attn.f_a_proj.weight",
    "self_attention.linear_attn.f_b_proj.weight": "self_attn.f_b_proj.weight",
    "self_attention.linear_attn.b_proj.weight": "self_attn.b_proj.weight",
    "self_attention.linear_attn.g_proj.weight": "self_attn.g_proj.weight",
    "self_attention.linear_attn.norm.weight": "self_attn.o_norm.weight",
    "self_attention.linear_attn.out_proj.weight": "self_attn.o_proj.weight",
    "self_attention.g_proj.weight": "self_attn.g_proj.weight",
    "self_attention.o_proj.weight": "self_attn.o_proj.weight",
    "self_attention.q_a_proj.weight": "self_attn.q_a_proj.weight",
    "self_attention.q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "self_attention.q_b_proj.weight": "self_attn.q_b_proj.weight",
    "self_attention.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "self_attention.kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
    "self_attention.kv_b_proj.weight": "self_attn.kv_b_proj.weight",
    "mlp.linear_fc2.weight": "mlp.down_proj.weight",
    "mlp.router.weight": "block_sparse_moe.gate.weight",
    "mlp.router.expert_bias": "block_sparse_moe.gate.e_score_correction_bias",
    "mlp.fc1_latent_proj.weight": "block_sparse_moe.routed_expert_down_proj.weight",
    "mlp.routed_expert_norm.weight": "block_sparse_moe.routed_expert_norm.weight",
    "mlp.fc2_latent_proj.weight": "block_sparse_moe.routed_expert_up_proj.weight",
    "mlp.shared_experts.linear_fc2.weight": "block_sparse_moe.shared_experts.down_proj.weight",
}

# Known naming-convention difference vs the bridge: it lists
# "mlp.experts.linear_fc2.weight" with an {expert_id} placeholder; vime (like miles)
# carries the expert index as a name suffix (linear_fc2.weight3). Both land on
# block_sparse_moe.experts.N.w2.weight; covered by the round-trip test.

_CONFIG = __import__("types").SimpleNamespace(model_type="kimi_k3", tie_word_embeddings=False)


def test_kimi_k3_roundtrip_bitwise_all_branches():
    # K3 layers are either KDA (linear_attn.*) or MLA (full attention), never both;
    # self_attn.g_proj.weight exists on both layer types, so put them on different layers.
    kda_names = [n for n in sorted(BRIDGE_LAYER_SINGLE) if "linear_attn" in n]
    mla_names = [n for n in sorted(BRIDGE_LAYER_SINGLE) if "linear_attn" not in n]
    layer_names = (
        [f"decoder.layers.0.{n}" for n in kda_names]
        + [f"decoder.layers.1.{n}" for n in mla_names]
        + [
            "decoder.layers.0.output_attn_res_norm.weight",
            "decoder.layers.0.output_attn_res_proj.weight",
            "decoder.layers.0.mlp.linear_fc1.weight",
            "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            "decoder.layers.0.mlp.experts.linear_fc1.weight3",
            "decoder.layers.0.mlp.experts.linear_fc2.weight3",
        ]
    )
    names = [f"module.module.{k}" for k in BRIDGE_DIRECT] + [f"module.module.{rest}" for rest in layer_names]

    torch.manual_seed(0)
    params = {}
    for name in names:
        if any(name.endswith(s) for s in ("A_log", "dt_bias", "norm.weight", "expert_bias")) or "layernorm" in name:
            shape = (8,)
        elif "conv1d" in name:
            shape = (8, 4)
        else:
            shape = (16, 8)
        params[name] = torch.randn(shape)

    hf_tensors = {}
    for name, param in params.items():
        for hf_name, tensor in _convert_to_hf_core(None, "kimik3config", name, param):
            assert hf_name not in hf_tensors, f"duplicate HF tensor {hf_name}"
            hf_tensors[hf_name] = tensor

    reader = DictReader(hf_tensors)
    for name, param in params.items():
        loaded = kimi_k3_hf_tensor(name, reader, _CONFIG)
        assert loaded.shape == param.shape, f"{name}: {tuple(loaded.shape)} != {tuple(param.shape)}"
        assert torch.equal(loaded, param), f"{name}: not bitwise"


@pytest.mark.parametrize("rest,hf_suffix", sorted(BRIDGE_LAYER_SINGLE.items()))
def test_kimi_k3_matches_bridge_layer_names(rest, hf_suffix):
    name = f"module.module.decoder.layers.2.{rest}"
    hf_key = f"{_P}.layers.2.{hf_suffix}"
    param = torch.randn(4, 4)

    exported = _convert_to_hf_core(None, "kimik3config", name, param)
    assert [k for k, _ in exported] == [hf_key]

    class RecordingReader(DictReader):
        def __init__(self, tensors):
            super().__init__(tensors)
            self.read = []

        def get_tensor(self, n):
            self.read.append(n)
            return super().get_tensor(n)

    reader = RecordingReader({hf_key: param})
    assert torch.equal(kimi_k3_hf_tensor(name, reader, _CONFIG), param)
    assert reader.read == [hf_key]


@pytest.mark.parametrize("rest,hf_key", sorted(BRIDGE_DIRECT.items()))
def test_kimi_k3_matches_bridge_direct_names(rest, hf_key):
    param = torch.randn(4, 4)
    name = f"module.module.{rest}"
    exported = _convert_to_hf_core(None, "kimik3config", name, param)
    assert [k for k, _ in exported] == [hf_key]
    assert torch.equal(kimi_k3_hf_tensor(name, DictReader({hf_key: param}), _CONFIG), param)


def test_kimi_k3_split_merge_rules_match_bridge():
    """Bridge pairs (gate,up) / (w1,w3) order must match the exporter's chunk order."""
    fc1 = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    exported = dict(_convert_to_hf_core(None, "kimik3config", "module.module.decoder.layers.0.mlp.linear_fc1.weight", fc1))
    assert list(exported) == [
        f"{_P}.layers.0.mlp.gate_proj.weight",
        f"{_P}.layers.0.mlp.up_proj.weight",
    ]
    assert torch.equal(exported[f"{_P}.layers.0.mlp.gate_proj.weight"], fc1[:8])
    assert torch.equal(exported[f"{_P}.layers.0.mlp.up_proj.weight"], fc1[8:])

    expert = dict(
        _convert_to_hf_core(None, "kimik3config", "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3", fc1)
    )
    prefix = f"{_P}.layers.0.block_sparse_moe.experts.3"
    assert list(expert) == [f"{prefix}.w1.weight", f"{prefix}.w3.weight"]

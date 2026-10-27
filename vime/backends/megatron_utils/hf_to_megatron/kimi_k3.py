"""HF->Megatron loader for Kimi-K3 (model_type "kimi_k3").

Inverse of vime.backends.megatron_utils.megatron_to_hf.kimi_k3 — the two are
kept in lockstep by the round-trip and bridge cross-check tests.
"""

from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers

_PREFIX = "language_model.model"

_DIRECT = {
    "embedding.word_embeddings.weight": f"{_PREFIX}.embed_tokens.weight",
    "decoder.final_layernorm.weight": f"{_PREFIX}.norm.weight",
    "output_layer.weight": "language_model.lm_head.weight",
}

_LAYER = {
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

_OUTPUT_HEAD = {
    "output_attn_res_norm.weight": f"{_PREFIX}.output_attn_res_norm.weight",
    "output_attn_res_proj.weight": f"{_PREFIX}.output_attn_res_proj.weight",
}


def kimi_k3_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if name in _DIRECT:
        return reader.get_tensor(_DIRECT[name])

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match is None:
        raise KeyError(f"Unsupported Kimi-K3 Megatron parameter {name!r}")
    layer, rest = match.groups()
    prefix = f"{_PREFIX}.layers.{layer}"

    expert = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if expert:
        projection, idx = expert.groups()
        expert_prefix = f"{prefix}.block_sparse_moe.experts.{idx}"
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{expert_prefix}.w1.weight"),
                reader.get_tensor(f"{expert_prefix}.w3.weight"),
            )
        return reader.get_tensor(f"{expert_prefix}.w2.weight")

    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    if rest == "mlp.shared_experts.linear_fc1.weight":
        shared = f"{prefix}.block_sparse_moe.shared_experts"
        return merge_gate_up(
            reader.get_tensor(f"{shared}.gate_proj.weight"),
            reader.get_tensor(f"{shared}.up_proj.weight"),
        )
    if rest in _OUTPUT_HEAD:
        return reader.get_tensor(_OUTPUT_HEAD[rest])
    if rest in _LAYER:
        return reader.get_tensor(f"{prefix}.{_LAYER[rest]}")
    raise KeyError(f"Unsupported Kimi-K3 Megatron layer parameter {rest!r}")

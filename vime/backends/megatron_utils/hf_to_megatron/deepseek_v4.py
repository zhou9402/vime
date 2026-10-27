"""HF->Megatron loader for DeepSeek-V4 (model_type "deepseek_v4").

Inverse of vime.backends.megatron_utils.megatron_to_hf.deepseekv4 — the two are
kept in lockstep by the round-trip tests in tests/test_hf_to_megatron.py.
"""

from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers
from .deepseek import _direct

_V4_DIRECT = {
    "decoder.hc_head_fn": "model.hc_head_fn",
    "decoder.hc_head_base": "model.hc_head_base",
    "decoder.hc_head_scale": "model.hc_head_scale",
}

_ALPHA_SEGMENTS = {"alpha_pre": 0, "alpha_post": 1, "alpha_res": 2}


def _layer_tensor(layer: int, rest: str, reader: SafetensorReader) -> torch.Tensor:
    prefix = f"model.layers.{layer}"
    mapping = {
        "self_attention.linear_q_down_proj.weight": "self_attn.wq_a.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.linear_q_up_proj.weight": "self_attn.wq_b.weight",
        "self_attention.linear_kv_proj.weight": "self_attn.wkv.weight",
        "self_attention.kv_layernorm.weight": "self_attn.kv_norm.weight",
        "self_attention.linear_o_group_proj": "self_attn.wo_a.weight",
        "self_attention.linear_proj.weight": "self_attn.wo_b.weight",
        "self_attention.core_attention.attn_sink": "self_attn.attn_sink",
        "self_attention.core_attention.compressor.ape": "self_attn.compressor.ape",
        "self_attention.core_attention.compressor.linear_wkv.weight": "self_attn.compressor.wkv.weight",
        "self_attention.core_attention.compressor.linear_wgate.weight": "self_attn.compressor.wgate.weight",
        "self_attention.core_attention.compressor.norm.weight": "self_attn.compressor.norm.weight",
        "self_attention.core_attention.indexer.linear_wq_b.weight": "self_attn.indexer.wq_b.weight",
        "self_attention.core_attention.indexer.linear_wk.weight": "self_attn.indexer.wk.weight",
        "self_attention.core_attention.indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
        "self_attention.core_attention.indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
        "self_attention.core_attention.indexer.linear_weights_proj.weight": "self_attn.indexer.weights_proj.weight",
        "self_attention.core_attention.indexer.compressor.ape": "self_attn.indexer.compressor.ape",
        "self_attention.core_attention.indexer.compressor.linear_wkv.weight": "self_attn.indexer.compressor.wkv.weight",
        "self_attention.core_attention.indexer.compressor.linear_wgate.weight": "self_attn.indexer.compressor.wgate.weight",
        "self_attention.core_attention.indexer.compressor.norm.weight": "self_attn.indexer.compressor.norm.weight",
        "input_layernorm.weight": "input_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "mlp.router.tid2eid": "mlp.topk.tid2eid",
        "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
        "self_attention_hyper_connection.bias": "hc_attn_base",
        "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
        "mlp_hyper_connection.bias": "hc_ffn_base",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")

    for suffix, segment in _ALPHA_SEGMENTS.items():
        for stem, hf_name in (
            ("self_attention_hyper_connection", "hc_attn_scale"),
            ("mlp_hyper_connection", "hc_ffn_scale"),
        ):
            if rest == f"{stem}.{suffix}":
                # The checkpoint packs [pre, post, res] into one tensor; Megatron stores scalars.
                return reader.get_tensor(f"{prefix}.{hf_name}").reshape(-1)[segment : segment + 1]

    match = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if match:
        projection, expert = match.groups()
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                reader.get_tensor(f"{prefix}.mlp.experts.{expert}.up_proj.weight"),
            )
        return reader.get_tensor(f"{prefix}.mlp.experts.{expert}.down_proj.weight")

    if rest == "mlp.shared_experts.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.shared_experts.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.shared_experts.up_proj.weight"),
        )
    if rest == "mlp.shared_experts.linear_fc2.weight":
        return reader.get_tensor(f"{prefix}.mlp.shared_experts.down_proj.weight")
    if rest == "mlp.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    if rest == "mlp.linear_fc2.weight":
        return reader.get_tensor(f"{prefix}.mlp.down_proj.weight")
    raise KeyError(f"Unsupported DeepSeek-V4 Megatron layer parameter {rest!r}")


def deepseek_v4_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if (tensor := _direct(name, reader, config)) is not None:
        return tensor
    if name in _V4_DIRECT:
        return reader.get_tensor(_V4_DIRECT[name])

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported DeepSeek-V4 Megatron parameter {name!r}")
    layer, rest = int(match.group(1)), match.group(2)
    return _layer_tensor(layer, rest, reader)

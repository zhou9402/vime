"""Consistency test: live MXFP8 export allowlist vs offline converter blocklist.

For Qwen3-dense, Qwen3-MoE and DeepSeek-V3 name mappings, every HF weight the
live export quantizer (quantizer_mxfp8) quantizes must be accepted by
``should_use_mxfp8`` (the offline converter / rollout-schema blocklist), and
every accepted HF weight in the standard param set must be quantized live.
Name-level only: real converter name mappings, synthetic small tensors.

The TE-backed mxfp8_quantize is monkeypatched with the quant_ref oracle.
"""

from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.megatron_to_hf import convert_to_hf
from vime.backends.megatron_utils.megatron_to_hf.processors import quantizer_mxfp8
from vime.utils.mxfp8 import should_use_mxfp8
from vime.utils.quant_ref import mxfp8_quantize_ref

MXFP8_CONFIG = {"quant_method": "mxfp8", "fmt": "e4m3", "activation_scheme": "dynamic"}

HIDDEN = 64
ARGS = SimpleNamespace(
    hidden_size=HIDDEN,
    num_attention_heads=4,
    num_query_groups=4,
    kv_channels=None,
    num_layers=1,
    vocab_size=32,
    extra_high_precision_layers_megatron=(),
    num_layers_at_start_in_bf16=0,
    num_layers_at_end_in_bf16=0,
)

# (megatron name, rows, cols) per family. Rows/cols respect each converter's
# reshape/split requirements (qkv: (vpg + 2) * groups * head_dim rows).
_COMMON_DENSE = [
    ("module.module.embedding.word_embeddings.weight", 32, HIDDEN),
    ("module.module.output_layer.weight", 32, HIDDEN),
    ("module.module.decoder.final_layernorm.weight", HIDDEN, None),
    ("module.module.decoder.layers.0.self_attention.linear_proj.weight", HIDDEN, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.linear_qkv.weight", 192, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight", HIDDEN, None),
    ("module.module.decoder.layers.0.mlp.linear_fc1.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.mlp.linear_fc2.weight", HIDDEN, 64),
    ("module.module.decoder.layers.0.pre_mlp_layernorm.weight", HIDDEN, None),
]

_MOE_EXTRA = [
    ("module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0", 128, HIDDEN),
    ("module.module.decoder.layers.0.mlp.experts.linear_fc2.weight0", HIDDEN, 64),
    ("module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.mlp.shared_experts.linear_fc2.weight", HIDDEN, 64),
    ("module.module.decoder.layers.0.mlp.router.weight", 8, HIDDEN),
]

_DEEPSEEK_V3 = [
    ("module.module.embedding.word_embeddings.weight", 32, HIDDEN),
    ("module.module.output_layer.weight", 32, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.linear_proj.weight", HIDDEN, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.linear_q_proj.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.linear_q_down_proj.weight", HIDDEN, 64),
    ("module.module.decoder.layers.0.self_attention.linear_q_up_proj.weight", 128, 64),
    ("module.module.decoder.layers.0.self_attention.linear_q_up_proj.layer_norm_weight", 64, None),
    ("module.module.decoder.layers.0.self_attention.linear_kv_down_proj.weight", HIDDEN, 64),
    ("module.module.decoder.layers.0.self_attention.linear_kv_up_proj.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.wq_b.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.wk.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.weights_proj.weight", 4, HIDDEN),
    ("module.module.decoder.layers.0.self_attention.k_norm.weight", 128, None),
    ("module.module.decoder.layers.0.mlp.linear_fc1.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.mlp.linear_fc2.weight", HIDDEN, 64),
    ("module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0", 128, HIDDEN),
    ("module.module.decoder.layers.0.mlp.experts.linear_fc2.weight0", HIDDEN, 64),
    ("module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight", 128, HIDDEN),
    ("module.module.decoder.layers.0.mlp.shared_experts.linear_fc2.weight", HIDDEN, 64),
    ("module.module.decoder.layers.0.mlp.router.weight", 8, HIDDEN),
    ("module.module.decoder.layers.0.input_layernorm.weight", HIDDEN, None),
]

CASES = {
    "qwen3-dense": ("qwen3-0.6b", _COMMON_DENSE),
    "qwen3-moe": ("qwen3moeforcausallm", _COMMON_DENSE + _MOE_EXTRA),
    "deepseek-v3": ("deepseek-v3", _DEEPSEEK_V3),
}


@pytest.fixture(autouse=True)
def patch_quantizer(monkeypatch):
    monkeypatch.setattr(quantizer_mxfp8, "mxfp8_quantize", mxfp8_quantize_ref)


def _to_param(rows, cols):
    if cols is None:  # 1D norm/bias
        return torch.randn(rows, dtype=torch.bfloat16)
    return torch.randn(rows, cols, dtype=torch.bfloat16)


@pytest.mark.parametrize("case", list(CASES))
def test_live_allowlist_matches_offline_blocklist(case):
    model_name, params = CASES[case]
    for megatron_name, rows, cols in params:
        param = _to_param(rows, cols)
        out = convert_to_hf(ARGS, model_name, megatron_name, param, quantization_config=dict(MXFP8_CONFIG))
        out_names = [n for n, _ in out]
        for hf_name, tensor in out:
            if not hf_name.endswith(".weight"):
                continue
            quantized = hf_name.replace(".weight", ".weight_scale") in out_names
            accepted_by_blocklist = should_use_mxfp8(hf_name, tuple(tensor.shape), torch.bfloat16)
            assert quantized == accepted_by_blocklist, (
                f"{model_name}: {megatron_name} -> {hf_name}: live quantizer "
                f"{'quantized' if quantized else 'skipped'} it but should_use_mxfp8 "
                f"{'accepts' if accepted_by_blocklist else 'rejects'} it"
            )
            if quantized:
                scale = dict(out)[hf_name.replace(".weight", ".weight_scale")]
                assert scale.dtype == torch.uint8
                assert scale.shape[-1] == tensor.shape[-1] // 32

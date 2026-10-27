"""CPU tests for the MXFP8 Megatron->HF export quantizer.

The TE-backed ``mxfp8_quantize`` is monkeypatched with the deterministic
``quant_ref`` oracle so these tests run without TransformerEngine or a GPU;
the numerics of the oracle itself are pinned in test_low_precision_quant_ref.py.

Port of miles@ba28991834 miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.megatron_to_hf import processors
from vime.backends.megatron_utils.megatron_to_hf.processors import quantizer_mxfp8
from vime.utils import quant_ref


def _ref_quantize(weight):
    # TE pads rows for alignment; the oracle works on unpadded rows, which is
    # what the exported tensors use after slicing, so shapes/values match.
    return quant_ref.mxfp8_quantize_ref(weight)


@pytest.fixture(autouse=True)
def patch_quantizer(monkeypatch):
    monkeypatch.setattr(quantizer_mxfp8, "mxfp8_quantize", _ref_quantize)


MXFP8_CONFIG = {"quant_method": "mxfp8", "fmt": "e4m3", "activation_scheme": "dynamic"}


def _args(**overrides):
    base = dict(
        extra_high_precision_layers_megatron=(),
        num_layers=4,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(args, megatron_name, hf_name, shape=(8, 64)):
    weight = torch.randn(*shape, dtype=torch.bfloat16)
    out = quantizer_mxfp8.quantize_params_mxfp8(args, megatron_name, [(hf_name, weight)], MXFP8_CONFIG)
    return out, weight


def test_dense_attention_weight_is_quantized():
    args = _args()
    out, weight = _run(
        args,
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    )
    assert [n for n, _ in out] == [
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.o_proj.weight_scale",
    ]
    qweight, scale = out[0][1], out[1][1]
    assert qweight.dtype == torch.float8_e4m3fn and qweight.shape == weight.shape
    assert scale.dtype == torch.uint8 and scale.shape == (8, 2)
    dequant = quant_ref.mxfp8_dequant_ref(qweight, scale)
    rel = ((dequant - weight.float()).abs() / weight.float().abs().clamp_min(1e-6))
    assert (rel <= 2**-4 + 1e-6).all()


def test_expert_weight_quantized_and_scale_skipped():
    args = _args()
    weight = torch.randn(8, 64, dtype=torch.bfloat16)
    out = quantizer_mxfp8.quantize_params_mxfp8(
        args,
        "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight0",
        [
            ("model.layers.1.mlp.experts.0.gate_up_proj.weight", weight),
            ("model.layers.1.mlp.experts.0.gate_up_proj.weight_scale", torch.ones(8)),
        ],
        MXFP8_CONFIG,
    )
    names = [n for n, _ in out]
    # The incoming bf16 *_scale tensor is dropped; the emitted scale is uint8.
    assert names == [
        "model.layers.1.mlp.experts.0.gate_up_proj.weight",
        "model.layers.1.mlp.experts.0.gate_up_proj.weight_scale",
    ]
    assert out[1][1].dtype == torch.uint8


def test_non_whitelisted_weight_passes_through():
    args = _args()
    out, weight = _run(
        args,
        "module.module.decoder.layers.0.self_attention.linear_proj.weight_extra",
        "model.layers.0.foo.weight",
    )
    assert len(out) == 1 and out[0][1] is weight


def test_non_decoder_weight_passes_through():
    args = _args()
    out, weight = _run(args, "module.module.embedding.word_embeddings.weight", "model.embed_tokens.weight")
    assert len(out) == 1 and out[0][1] is weight


def test_extra_high_precision_layers_megatron_skip():
    args = _args(extra_high_precision_layers_megatron=("layers.0.self_attention",))
    out, weight = _run(
        args,
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    )
    assert len(out) == 1 and out[0][1] is weight


def test_first_last_layers_bf16_skip():
    args = _args(num_layers=4, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1)
    out_first, w = _run(
        args,
        "module.module.decoder.layers.0.mlp.linear_fc1.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    )
    assert len(out_first) == 1 and out_first[0][1] is w
    out_last, w = _run(
        args,
        "module.module.decoder.layers.3.mlp.linear_fc1.weight",
        "model.layers.3.mlp.gate_up_proj.weight",
    )
    assert len(out_last) == 1 and out_last[0][1] is w
    out_mid, _ = _run(
        args,
        "module.module.decoder.layers.1.mlp.linear_fc1.weight",
        "model.layers.1.mlp.gate_up_proj.weight",
    )
    assert out_mid[1][1].dtype == torch.uint8


def test_mtp_layer_quantized():
    args = _args()
    out, _ = _run(
        args,
        "module.module.mtp.layers.0.transformer_layer.mlp.linear_fc1.weight",
        "model.mtp.layers.0.mlp.gate_up_proj.weight",
    )
    assert out[1][0].endswith(".weight_scale")


def test_dispatcher_routes_mxfp8():
    weight = torch.randn(8, 64, dtype=torch.bfloat16)
    out = processors.quantize_params(
        _args(),
        "module.module.decoder.layers.0.mlp.linear_fc1.weight",
        [("model.layers.0.mlp.gate_up_proj.weight", weight)],
        dict(MXFP8_CONFIG),
    )
    assert out[1][0] == "model.layers.0.mlp.gate_up_proj.weight_scale"


def test_dispatcher_raises_on_nvfp4_until_ported():
    weight = torch.randn(8, 64, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="NVFP4"):
        processors.quantize_params(
            _args(),
            "module.module.decoder.layers.0.mlp.linear_fc1.weight",
            [("model.layers.0.mlp.gate_up_proj.weight", weight)],
            {"quant_method": "modelopt_fp4"},
        )

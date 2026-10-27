"""CPU tests for the NVFP4 Megatron->HF export quantizer.

TE-backed ``nvfp4_quantize_1d(_pair)`` is monkeypatched with oracle-based
wrappers so tests run on CPU; oracle numerics pinned in
test_low_precision_quant_ref.py, TE equivalence (GPU) in test_nvfp4_te_vs_oracle.py.

Port of miles@ba28991834 miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py.
"""

from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.megatron_to_hf import processors
from vime.backends.megatron_utils.megatron_to_hf.processors import quantizer_nvfp4
from vime.utils import quant_ref

NVFP4_CONFIG = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": []}


def _quantize_1d_ref(weight):
    return quant_ref.nvfp4_quantize_ref(weight.float())


def _quantize_1d_pair_ref(first, second):
    combined = torch.cat((first.float(), second.float()))
    q, s, g = quant_ref.nvfp4_quantize_ref(combined)
    rows = first.shape[0]
    return (q[:rows], s[:rows], g), (q[rows:], s[rows:], g.clone())


@pytest.fixture(autouse=True)
def patch_quantizers(monkeypatch):
    monkeypatch.setattr(quantizer_nvfp4, "nvfp4_quantize_1d", _quantize_1d_ref)
    monkeypatch.setattr(quantizer_nvfp4, "nvfp4_quantize_1d_pair", _quantize_1d_pair_ref)


def _args(**overrides):
    base = dict(
        extra_high_precision_layers_megatron=(),
        num_layers=4,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=0,
        fp4_param=False,
        fp4_param_gather=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_routed_expert_pair_quantized_with_shared_global_scale():
    gate = torch.randn(20, 48, dtype=torch.bfloat16)
    up = torch.randn(20, 48, dtype=torch.bfloat16) * 100
    out = quantizer_nvfp4.quantize_params_nvfp4(
        _args(),
        "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight0",
        [
            ("model.layers.1.mlp.experts.0.gate_proj.weight", gate),
            ("model.layers.1.mlp.experts.0.up_proj.weight", up),
        ],
        NVFP4_CONFIG,
    )
    names = [n for n, _ in out]
    assert names == [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale_2",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight_scale",
        "model.layers.1.mlp.experts.0.up_proj.weight_scale_2",
    ]
    tensors = dict(out)
    gq = tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]
    gs = tensors["model.layers.1.mlp.experts.0.gate_proj.weight_scale"]
    g2 = tensors["model.layers.1.mlp.experts.0.gate_proj.weight_scale_2"]
    u2 = tensors["model.layers.1.mlp.experts.0.up_proj.weight_scale_2"]
    # vLLM modelopt_fp4 on-disk contract: uint8 packed [M, K/2], e4m3 [M, K/16],
    # fp32 global; gate/up share one global scale.
    assert gq.dtype == torch.uint8 and gq.shape == (20, 24)
    assert gs.dtype == torch.float8_e4m3fn and gs.shape == (20, 3)
    assert g2.dtype == torch.float32 and torch.equal(g2, u2)
    dequant = quant_ref.nvfp4_dequant_ref(gq, gs, g2)
    err = (dequant - gate.float()).abs() / gate.float().abs().clamp_min(1.0)
    assert (err < 1.0).all()


def test_incomplete_gate_up_pair_raises():
    with pytest.raises(ValueError, match="gate/up"):
        quantizer_nvfp4.quantize_params_nvfp4(
            _args(),
            "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight0",
            [("model.layers.1.mlp.experts.0.gate_proj.weight", torch.randn(16, 32, dtype=torch.bfloat16))],
            NVFP4_CONFIG,
        )


def test_fc2_expert_quantized_singly():
    weight = torch.randn(33, 16, dtype=torch.bfloat16)  # M % 16 != 0
    out = quantizer_nvfp4.quantize_params_nvfp4(
        _args(),
        "module.module.decoder.layers.2.mlp.experts.linear_fc2.weight3",
        [("model.layers.2.mlp.experts.3.down_proj.weight", weight)],
        NVFP4_CONFIG,
    )
    tensors = dict(out)
    assert tensors["model.layers.2.mlp.experts.3.down_proj.weight"].shape == (33, 8)


def test_dense_and_shared_expert_pass_through():
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    for megatron_name in (
        "module.module.decoder.layers.0.mlp.linear_fc1.weight",
        "module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
    ):
        out = quantizer_nvfp4.quantize_params_nvfp4(
            _args(), megatron_name, [("some.weight", weight)], NVFP4_CONFIG
        )
        assert len(out) == 1 and out[0][1] is weight, megatron_name


def test_mtp_layer_stays_bf16():
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    out = quantizer_nvfp4.quantize_params_nvfp4(
        _args(),
        "module.module.mtp.layers.0.mlp.experts.linear_fc1.weight0",
        [("model.layers.4.mlp.experts.0.gate_proj.weight", weight)],
        NVFP4_CONFIG,
    )
    assert len(out) == 1 and out[0][1] is weight


def test_ignore_rules_skip():
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    config = dict(NVFP4_CONFIG, ignore=["re:.*experts\\.0\\..*"])
    out = quantizer_nvfp4.quantize_params_nvfp4(
        _args(),
        "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight0",
        [("model.layers.1.mlp.experts.0.down_proj.weight", weight)],
        config,
    )
    assert len(out) == 1 and out[0][1] is weight


def test_first_last_layers_and_extra_high_precision_skip():
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    args = _args(num_layers_at_end_in_bf16=1)
    out = quantizer_nvfp4.quantize_params_nvfp4(
        args,
        "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
        [("model.layers.3.mlp.experts.0.down_proj.weight", weight)],
        NVFP4_CONFIG,
    )
    assert len(out) == 1 and out[0][1] is weight

    args = _args(extra_high_precision_layers_megatron=("layers.1.mlp.experts",))
    out = quantizer_nvfp4.quantize_params_nvfp4(
        args,
        "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight0",
        [("model.layers.1.mlp.experts.0.down_proj.weight", weight)],
        NVFP4_CONFIG,
    )
    assert len(out) == 1 and out[0][1] is weight


def test_dispatcher_routes_nvfp4_variants():
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    for config in (
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        {"quant_method": "modelopt_fp4"},
        {"quant_method": "nvfp4"},
    ):
        out = processors.quantize_params(
            _args(),
            "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight0",
            [("model.layers.1.mlp.experts.0.down_proj.weight", weight)],
            config,
        )
        names = [n for n, _ in out]
        assert "model.layers.1.mlp.experts.0.down_proj.weight_scale_2" in names, config

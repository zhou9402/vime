"""CPU tests for the DSv4 small ops (F07-3a)."""

import pytest
import torch

from vime_plugins.models.deepseek_v4.arguments import is_dsv4_model, normalize_dsv4_args
from vime_plugins.models.deepseek_v4.ops.norm import RMSNorm
from vime_plugins.models.deepseek_v4.ops.precision_aligned_ops import linear_bf16_fp32
from vime_plugins.models.deepseek_v4.ops.qat import fp8_simulate


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    norm = RMSNorm(16)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    out = norm(x)
    x32 = x.float()
    expected = (norm.weight * x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    assert torch.equal(out, expected)
    assert norm.weight.dtype == torch.float32


def test_linear_bf16_fp32_fwd_bwd():
    torch.manual_seed(0)
    x = torch.randn(6, 8, dtype=torch.float32, requires_grad=True)
    w = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    out = linear_bf16_fp32(x, w)
    assert out.dtype == torch.float32
    expected = torch.mm(x.to(torch.bfloat16).float(), w.to(torch.bfloat16).t().float())
    assert torch.allclose(out, expected, rtol=0, atol=0)
    out.sum().backward()
    assert x.grad.shape == x.shape and w.grad.shape == w.shape
    assert torch.equal(w.grad, torch.ones(6, 4).t() @ x.to(torch.bfloat16).float())


def test_fp8_simulate_torch_fallback_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    out = fp8_simulate(x, 128)
    assert out.shape == x.shape and out.dtype == x.dtype
    # dequantized values stay within fp8 e4m3 relative error of the input
    rel = ((out.float() - x.float()).abs() / x.float().abs().clamp(min=1e-3)).max()
    assert rel < 2**-3 + 1e-2


def test_rope_inverse_is_identity():
    pytest.importorskip("triton")  # megatron.core import requires triton
    from vime_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, precompute_freqs_cis

    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, dtype=torch.float32)
    freqs = precompute_freqs_cis(16, 8, 0, 10000, 4, 32, 1)
    rotated = apply_rotary_emb(x.clone(), freqs)
    restored = apply_rotary_emb(rotated, freqs, inverse=True)
    assert torch.allclose(restored, x, rtol=1e-5, atol=1e-5)


def test_is_dsv4_model_and_normalize():
    import argparse

    args = argparse.Namespace(
        spec=["vime_plugins.models.deepseek_v4.deepseek_v4", "get_dsv4_spec"],
        dsv4_impl="megatron",
        tensor_model_parallel_size=1,
        dsa_kernel_backend="none",
        enable_hyper_connections=False,
        experimental_attention_variant=None,
    )
    assert is_dsv4_model(args)
    normalize_dsv4_args(args)
    assert args.enable_hyper_connections is True
    assert args.experimental_attention_variant == "dsv4_hybrid"

    args.spec = None
    assert not is_dsv4_model(args)

    bad = argparse.Namespace(
        spec=["vime_plugins.models.deepseek_v4.deepseek_v4", "get_dsv4_spec"],
        dsv4_impl="megatron",
        tensor_model_parallel_size=2,
        dsa_kernel_backend="none",
    )
    with pytest.raises(ValueError, match="tensor-model-parallel-size 1"):
        normalize_dsv4_args(bad)

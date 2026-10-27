"""The tilelang fp8 act-quant kernel against a torch reference: per-128-block scales, optional
power-of-two (MXFP) scale rounding, and the fused inplace quant+dequant."""

import sys

import pytest
import torch

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from vime.kernels.quant.fp8_act_quant import act_quant  # noqa: E402


def test_scales_are_amax_over_448():
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    y, s = act_quant(x, block_size=128)
    assert y.dtype == torch.float8_e4m3fn
    amax = x.float().reshape(64, 2, 128).abs().amax(dim=-1).clamp(min=1e-4)
    torch.testing.assert_close(s, amax / 448.0, rtol=0, atol=0)
    # roundtrip
    s_up = s.repeat_interleave(128, dim=-1)
    torch.testing.assert_close(y.float() * s_up, x.float(), rtol=6e-2, atol=6e-2)


def test_inplace_is_fused_quant_dequant():
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    out = act_quant(x.clone(), block_size=128, inplace=True)
    _, s = act_quant(x.clone(), block_size=128)
    expected = ((x.float() / s.repeat_interleave(128, dim=-1)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=1e-3)


def test_power_of_two_scales():
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    _, s = act_quant(x, block_size=128, scale_fmt="e8m0")
    log2_s = torch.log2(s)
    torch.testing.assert_close(log2_s, log2_s.round(), rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

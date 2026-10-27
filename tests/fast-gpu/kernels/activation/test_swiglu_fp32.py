"""The fp32 SwiGLU Triton kernel against a float64 torch reference, forward and backward,
including the per-token scale."""

import sys

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from vime.kernels.activation.swiglu_fp32 import swiglu_fp32  # noqa: E402


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _swiglu_reference(fc1_out: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    gate, up = fc1_out.chunk(2, dim=-1)
    out = F.silu(gate) * up
    return out if scale is None else out * scale[:, None]


@pytest.mark.parametrize("with_scale", [False, True])
def test_swiglu_matches_float64_reference(with_scale):
    torch.manual_seed(0)
    tokens, width = 257, 384
    fc1_out = torch.randn(tokens, 2 * width, device="cuda", dtype=torch.bfloat16)
    scale = torch.rand(tokens, device="cuda", dtype=torch.float32) + 0.5 if with_scale else None
    grad = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)

    fc1_kernel = fc1_out.clone().requires_grad_()
    scale_kernel = scale.clone().requires_grad_() if with_scale else None
    out = swiglu_fp32(fc1_kernel, scale_kernel)
    out.backward(grad)

    fc1_ref = fc1_out.double().requires_grad_()
    scale_ref = scale.double().requires_grad_() if with_scale else None
    out_ref = _swiglu_reference(fc1_ref, scale_ref)
    out_ref.backward(grad.double())

    assert out.dtype == fc1_out.dtype
    assert _rel_err(out, out_ref) < 5e-3
    assert _rel_err(fc1_kernel.grad, fc1_ref.grad) < 5e-3
    if with_scale:
        assert _rel_err(scale_kernel.grad, scale_ref.grad) < 5e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

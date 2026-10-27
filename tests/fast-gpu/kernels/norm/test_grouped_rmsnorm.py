"""Grouped RMSNorm against a float64 torch reference, forward and backward."""

import sys

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from vime.kernels.norm.grouped_rmsnorm import block_c, grouped_rmsnorm_bwd_kernel, grouped_rmsnorm_fwd  # noqa: E402


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _reference(x2d: torch.Tensor, weight: torch.Tensor, n: int, eps: float):
    T, W = x2d.shape
    C = W // n
    xg = x2d.reshape(T, n, C)
    var = (xg * xg).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    normed = xg * rstd * (1.0 + weight.reshape(n, C))
    return normed.reshape(T, W), rstd.reshape(T, n)


class _GroupedRmsNorm(torch.autograd.Function):
    """Autograd wrapper that drives grouped_rmsnorm_bwd_kernel for dx."""

    @staticmethod
    def forward(ctx, x2d, weight, n, eps):
        normed, rstd = grouped_rmsnorm_fwd(x2d, weight, n, eps)
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.n = n
        return normed

    @staticmethod
    def backward(ctx, dnormed):
        x2d, weight, rstd = ctx.saved_tensors
        T, W = x2d.shape
        dx = torch.empty_like(x2d)
        C = W // ctx.n
        grouped_rmsnorm_bwd_kernel[(T * ctx.n,)](x2d, weight, rstd, dnormed.contiguous(), dx, T, N=ctx.n, C=C, BLOCK_C=block_c(C))
        return dx, None, None, None


def test_grouped_rmsnorm_fwd_matches_float64_reference():
    torch.manual_seed(0)
    T, n, C = 129, 4, 96  # non-power-of-2 C exercises masking
    x = torch.randn(T, n * C, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n * C, device="cuda", dtype=torch.float32) * 0.1
    eps = 1e-6

    normed, rstd = grouped_rmsnorm_fwd(x, weight, n, eps)
    normed_ref, rstd_ref = _reference(x.double(), weight.double(), n, eps)

    assert normed.dtype == torch.float32 and rstd.dtype == torch.float32
    assert _rel_err(normed, normed_ref) < 1e-4
    assert _rel_err(rstd, rstd_ref) < 1e-4


def test_grouped_rmsnorm_bwd_matches_float64_reference():
    torch.manual_seed(0)
    T, n, C = 129, 4, 96
    eps = 1e-6
    x = torch.randn(T, n * C, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n * C, device="cuda", dtype=torch.float32) * 0.1
    grad = torch.randn(T, n * C, device="cuda", dtype=torch.float32)

    x_kernel = x.clone().requires_grad_()
    out = _GroupedRmsNorm.apply(x_kernel, weight, n, eps)
    out.backward(grad)

    x_ref = x.double().requires_grad_()
    out_ref, _ = _reference(x_ref, weight.double(), n, eps)
    out_ref.backward(grad.double())

    assert _rel_err(x_kernel.grad, x_ref.grad) < 5e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

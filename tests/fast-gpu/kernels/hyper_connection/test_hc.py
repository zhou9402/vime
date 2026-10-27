"""The hc_mix_inject / hc_combine Triton kernels against float64 torch references, forward and backward."""

import sys

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from vime.kernels.hyper_connection.hc import hc_combine, hc_mix_inject  # noqa: E402


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _mix_inject_reference(x2d, weight, w_down, w_up, w_inject, n, eps):
    T, W = x2d.shape
    C = W // n
    xg = x2d.reshape(T, n, C)
    rstd = 1.0 / torch.sqrt((xg * xg).mean(dim=-1, keepdim=True) + eps)
    normed = (xg * rstd * (1.0 + weight.reshape(n, C))).reshape(T, W)
    z1 = F.linear(normed, w_down) / n
    gate = torch.sigmoid(F.linear(F.silu(z1), w_up))
    mixed = (gate.reshape(T, n, C) * normed.reshape(T, n, C)).mean(dim=1)
    h_post = 2.0 * torch.sigmoid(F.linear(normed, w_inject) / n) if w_inject is not None else None
    return mixed, h_post


def _combine_reference(residual2d, y2d, h_post, n):
    T, W = residual2d.shape
    C = W // n
    return residual2d + h_post.reshape(T, n, 1) * y2d.reshape(T, 1, C)


def _inputs(T, n, C, seed=0, with_inject=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = n * C
    x = torch.randn(T, W, device="cuda", dtype=torch.bfloat16, generator=g)
    weight = torch.randn(W, device="cuda", dtype=torch.float32, generator=g) * 0.1
    w_down = torch.randn(C, W, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w_up = torch.randn(W, C, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    w_inject = torch.randn(n, W, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 if with_inject else None
    return x, weight, w_down, w_up, w_inject


@pytest.mark.parametrize("with_inject", [False, True])
def test_hc_mix_inject_matches_reference(with_inject):
    n, C, T, eps = 4, 96, 37, 1e-6
    x, weight, w_down, w_up, w_inject = _inputs(T, n, C, with_inject=with_inject)

    tensors = [x, weight, w_down, w_up] + ([w_inject] if with_inject else [])
    k = [t.clone().requires_grad_() for t in tensors]
    r = [t.double().clone().requires_grad_() for t in tensors]

    mixed_k, h_post_k = hc_mix_inject(k[0], k[1], k[2], k[3], k[4] if with_inject else None, n, eps)
    mixed_r, h_post_r = _mix_inject_reference(r[0], r[1], r[2], r[3], r[4] if with_inject else None, n, eps)

    assert _rel_err(mixed_k, mixed_r) < 5e-3
    if with_inject:
        assert _rel_err(h_post_k, h_post_r) < 5e-3

    g = torch.Generator(device="cuda").manual_seed(1)
    dmixed = torch.randn(T, C, device="cuda", generator=g, dtype=torch.bfloat16)
    loss_k = (mixed_k * dmixed).sum()
    loss_r = (mixed_r * dmixed.double()).sum()
    if with_inject:
        dh = torch.randn(T, n, device="cuda", generator=g)
        loss_k = loss_k + (h_post_k * dh.to(h_post_k.dtype)).sum()
        loss_r = loss_r + (h_post_r * dh.double()).sum()
    loss_k.backward()
    loss_r.backward()

    for i, tk in enumerate(k):
        assert tk.grad.dtype == tensors[i].dtype
        assert _rel_err(tk.grad, r[i].grad) < 5e-3


def test_hc_combine_matches_reference():
    n, C, T = 4, 96, 37
    g = torch.Generator(device="cuda").manual_seed(2)
    residual = torch.randn(T, n * C, device="cuda", dtype=torch.bfloat16, generator=g)
    y = torch.randn(T, C, device="cuda", dtype=torch.bfloat16, generator=g)
    h_post = torch.randn(T, n, device="cuda", dtype=torch.float32, generator=g)

    rk = residual.clone().requires_grad_()
    yk = y.clone().requires_grad_()
    hk = h_post.clone().requires_grad_()
    out_k = hc_combine(rk, yk, hk, n)

    rr = residual.double().requires_grad_()
    yr = y.double().requires_grad_()
    hr = h_post.double().requires_grad_()
    out_r = _combine_reference(rr, yr, hr, n)

    assert _rel_err(out_k, out_r) < 5e-3

    dout = torch.randn(T, n * C, device="cuda", dtype=torch.bfloat16, generator=g)
    out_k.backward(dout)
    out_r.backward(dout.double())
    for tk, tr in [(rk, rr), (yk, yr), (hk, hr)]:
        assert _rel_err(tk.grad, tr.grad) < 5e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

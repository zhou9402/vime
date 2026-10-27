"""The fp8 blockwise cast kernel against a torch reference: 128x128 blocks, scale = amax/448."""

import sys

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from vime.kernels.quant.fp8_blockwise import fp8_blockwise_cast  # noqa: E402


def _reference(x: torch.Tensor, block_m=128, block_n=128):
    M, N = x.shape
    xb = x.float()
    y = torch.empty_like(xb)
    s = torch.empty((M + block_m - 1) // block_m, (N + block_n - 1) // block_n, dtype=torch.float32, device=x.device)
    for i, bi in enumerate(range(0, M, block_m)):
        for j, bj in enumerate(range(0, N, block_n)):
            block = xb[bi : bi + block_m, bj : bj + block_n]
            amax = max(block.abs().max().item(), 1e-10)
            s[i, j] = amax / 448.0
            y[bi : bi + block_m, bj : bj + block_n] = (block / s[i, j]).clamp(-448.0, 448.0)
    return y, s


@pytest.mark.parametrize("contiguous", [True, False])
def test_fp8_blockwise_matches_reference(contiguous):
    torch.manual_seed(0)
    M, N = 257, 384  # non-multiples of the 128 block exercise masking
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    if not contiguous:
        x = torch.randn(M, 2 * N, device="cuda", dtype=torch.bfloat16)[:, ::2]

    y, s = fp8_blockwise_cast(x)
    y_ref, s_ref = _reference(x)

    assert y.dtype == torch.float8_e4m3fn
    assert s.shape == ((M + 127) // 128, (N + 127) // 128)
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    # roundtrip dequant vs original
    s_up = s.repeat_interleave(128, dim=0)[:M].repeat_interleave(128, dim=1)[:, :N]
    torch.testing.assert_close(y.float() * s_up, x.float(), rtol=6e-2, atol=6e-2)
    torch.testing.assert_close(y.float(), y_ref.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

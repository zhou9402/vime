"""GPU test: TE-backed nvfp4_quantize_1d / nvfp4_quantize_1d_pair vs the oracle.

Skipped unless CUDA and TransformerEngine are available. Ground truth that the
pure-torch reference (vime.utils.quant_ref.nvfp4_quantize_ref) and the torch
QDQ fallback (nvfp4_qdq_torch) match TE's NVFP4Quantizer numerics.

Run on a GPU box: python -m pytest tests/test_nvfp4_te_vs_oracle.py -x -q
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine.pytorch")

from vime.backends.megatron_utils.kernels.nvfp4_qdq_torch import NVFP4QDQConfig, fused_nvfp4_qdq  # noqa: E402
from vime.utils import nvfp4, quant_ref  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "NVTE_NVFP4_4OVER6",
        "NVTE_NVFP4_4OVER6_E4M3_USE_256",
        "NVTE_NVFP4_4OVER6_ERR_MODE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")


@pytest.mark.parametrize("shape", [(16, 16), (20, 32), (33, 48), (1, 16 * 5), (100, 16)])  # M % 16 != 0, K = 16 * odd
def test_te_nvfp4_matches_oracle(shape):
    torch.manual_seed(0)
    weight = (torch.randn(*shape) * torch.logspace(-2, 2, shape[0]).unsqueeze(1)).bfloat16().cuda()

    qweight, block_scale, global_scale = nvfp4.nvfp4_quantize_1d(weight)
    q_ref, bs_ref, gs_ref = quant_ref.nvfp4_quantize_ref(weight.float().cpu())

    assert qweight.shape == q_ref.shape and qweight.dtype == torch.uint8
    assert block_scale.dtype == torch.float8_e4m3fn
    assert torch.equal(block_scale.float().cpu(), bs_ref.float())
    assert torch.allclose(global_scale.cpu(), gs_ref, rtol=0, atol=0)
    # Exact e2m1 code equality (both pipelines are deterministic RNE).
    assert torch.equal(quant_ref._e2m1_unpack(qweight.cpu()), quant_ref._e2m1_unpack(q_ref))


@pytest.mark.parametrize("shape", [(20, 32), (33, 48)])
def test_torch_qdq_matches_te(shape):
    torch.manual_seed(0)
    weight = (torch.randn(*shape) * 10).bfloat16().cuda()

    qweight, block_scale, global_scale = nvfp4.nvfp4_quantize_1d(weight)
    te_dequant = quant_ref.nvfp4_dequant_ref(qweight.cpu(), block_scale.cpu(), global_scale.cpu()).to(weight.dtype)
    torch_dequant = fused_nvfp4_qdq(weight, NVFP4QDQConfig())
    torch.testing.assert_close(torch_dequant.cpu(), te_dequant, rtol=0, atol=0)


def test_te_pair_shares_global_scale():
    torch.manual_seed(0)
    gate = (torch.randn(24, 48)).bfloat16().cuda()
    up = (torch.randn(24, 48) * 100).bfloat16().cuda()
    (gq, gs, g2), (uq, us, u2) = nvfp4.nvfp4_quantize_1d_pair(gate, up)
    assert torch.equal(g2, u2)
    # The shared global scale covers the larger (up) amax.
    combined = torch.cat((gate, up))
    _, _, expected_g2 = nvfp4.nvfp4_quantize_1d(combined)
    assert torch.allclose(g2.cpu(), expected_g2.cpu(), rtol=0, atol=0)

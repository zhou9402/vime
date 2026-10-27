"""GPU test: TE-backed mxfp8_quantize vs the pure-torch quant_ref oracle.

Skipped unless CUDA and TransformerEngine are available; this is the ground
truth check that vime.utils.mxfp8.mxfp8_quantize matches the deterministic
reference numerics pinned in test_low_precision_quant_ref.py.

Run on a GPU box: python -m pytest tests/test_mxfp8_te_vs_oracle.py -x -q
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_engine.pytorch")

from vime.utils import mxfp8, quant_ref  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
        (64, 128),
        (40, 96),  # M not a multiple of 32 (TE row padding path)
        (7, 32 * 3),  # K = 32 * odd
        (130, 32 * 5),
    ],
)
def test_te_mxfp8_matches_oracle(shape):
    torch.manual_seed(0)
    weight = (torch.randn(*shape) * torch.logspace(-2, 2, shape[0]).unsqueeze(1)).bfloat16().cuda()

    qweight, scale = mxfp8.mxfp8_quantize(weight)
    qweight_ref, scale_ref = quant_ref.mxfp8_quantize_ref(weight.float().cpu())

    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.uint8
    assert qweight.shape == weight.shape
    assert scale.shape == scale_ref.shape

    # Exact match on the UE8M0 exponents and on the e4m3 code values.
    assert torch.equal(scale.cpu(), scale_ref)
    assert torch.equal(qweight.float().cpu(), qweight_ref.float())

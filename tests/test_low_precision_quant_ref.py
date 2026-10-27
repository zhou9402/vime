"""CPU tests for the low-precision quantization reference oracle (vime/utils/quant_ref.py).

The oracle pins the numerics contract for the MXFP8/NVFP4 port: production
quantizers (TransformerEngine-backed ``vime.utils.mxfp8`` and the NVFP4 fake-QAT
path) are validated against these reference implementations, so the references
themselves are locked down here with golden vectors and roundtrip error bounds.

Port of miles@ba28991834 numerics (miles/utils/mxfp8.py, miles/utils/nvfp4.py).
"""

import torch

from vime.utils import quant_ref


def test_e2m1_round_grid_and_saturation():
    x = torch.tensor([0.0, 0.4, 0.6, 1.25, 1.75, 2.5, 5.0, 7.0, -0.6, -3.4, -100.0])
    got = quant_ref._round_to_e2m1(x)
    # Ties resolve to the even magnitude code: 1.25 -> 1.0 (code 2), 2.5 -> 2.0 (code 4).
    expected = torch.tensor([0.0, 0.5, 0.5, 1.0, 2.0, 2.0, 4.0, 6.0, -0.5, -3.0, -6.0])
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_e2m1_round_ties_to_even():
    # Exact midpoints round to the even code (IEEE RNE): 0.25 -> 0 (code 0),
    # 0.75 -> 1.0 (code 2), 1.25 -> 1.0 (code 2), 5.0 tie between 4 and 6 -> 4.
    x = torch.tensor([0.25, 0.75, 1.25, 5.0])
    got = quant_ref._round_to_e2m1(x)
    expected = torch.tensor([0.0, 1.0, 1.0, 4.0])
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_e2m1_pack_unpack_roundtrip():
    q = torch.tensor([[0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, -6.0]])
    packed = quant_ref._e2m1_pack(q)
    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 4)
    # Nibble codes: sign<<3 | magnitude code; low nibble = even column.
    assert packed.tolist() == [[0x10, 0x3A, 0xD4, 0xF6]]
    unpacked = quant_ref._e2m1_unpack(packed)
    torch.testing.assert_close(unpacked, q, rtol=0, atol=0)


def test_block_fp8_dequant_matches_manual():
    weight = torch.ones(256, 256, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.full((2, 2), 0.5)
    out = quant_ref.block_fp8_dequant(weight, scale, (128, 128))
    assert out.shape == (256, 256)
    torch.testing.assert_close(out, torch.full((256, 256), 0.5), rtol=0, atol=0)


def test_block_fp8_dequant_partial_edge_blocks():
    weight = torch.ones(130, 200, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.arange(8, dtype=torch.float32).view(2, 4) + 1.0  # ceil(130/128)=2, ceil(200/128)... no: bk
    # block (128, 64): scale shape (2, 4)
    out = quant_ref.block_fp8_dequant(weight, scale, (128, 64))
    assert out.shape == (130, 200)
    assert out[0, 0] == 1.0 and out[129, 0] == 5.0 and out[0, 199] == 4.0 and out[129, 199] == 8.0


def test_mxfp8_quantize_ref_golden():
    weight = torch.zeros(2, 32)
    weight[0, 0] = 448.0  # amax == e4m3 max -> scale exponent 127 (2^0)
    weight[1, 0] = 896.0  # amax = 2 * 448 -> scale exponent 128 (2^1)
    weight[1, 1] = -1.0
    qweight, scale = quant_ref.mxfp8_quantize_ref(weight)
    assert qweight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.uint8
    assert scale.shape == (2, 1)
    assert scale[0, 0].item() == 127
    assert scale[1, 0].item() == 128
    assert qweight[0, 0].float().item() == 448.0
    assert qweight[1, 0].float().item() == 448.0
    assert qweight[1, 1].float().item() == -0.5  # -1.0 / 2^1


def test_mxfp8_roundtrip_error_bound():
    torch.manual_seed(0)
    weight = torch.randn(64, 128) * torch.logspace(-3, 3, 64).unsqueeze(1)
    qweight, scale = quant_ref.mxfp8_quantize_ref(weight)
    dequant = quant_ref.mxfp8_dequant_ref(qweight, scale)
    rel = ((dequant - weight).abs() / weight.abs().clamp_min(1e-6))
    # e4m3 has 3 mantissa bits: relative quantization error <= 2^-4 per element.
    assert (rel <= 2**-4 + 1e-6).all(), f"max rel err {rel.max()}"


def test_mxfp8_roundtrip_all_zero_group():
    weight = torch.zeros(4, 64)
    qweight, scale = quant_ref.mxfp8_quantize_ref(weight)
    dequant = quant_ref.mxfp8_dequant_ref(qweight, scale)
    assert not torch.isnan(dequant).any()
    assert (dequant == 0).all()


def test_nvfp4_quantize_ref_golden():
    # amax = 2688 = 448 * 6 -> encode scale 1.0 (global decode = 1.0);
    # block amax 2688 -> block scale e4m3_rn(2688/6) = 448, so 2688 -> e2m1 6.0.
    weight = torch.zeros(1, 16)
    weight[0, 0] = 2688.0
    weight[0, 1] = -1344.0  # -> -3.0 on the e2m1 grid after block scaling
    qweight, block_scale, global_scale = quant_ref.nvfp4_quantize_ref(weight)
    assert qweight.dtype == torch.uint8 and qweight.shape == (1, 8)
    assert block_scale.dtype == torch.float8_e4m3fn and block_scale.shape == (1, 1)
    torch.testing.assert_close(global_scale, torch.tensor(1.0), rtol=0, atol=0)
    torch.testing.assert_close(block_scale.float(), torch.full((1, 1), 448.0), rtol=0, atol=0)
    # e2m1(6.0) = 0b0111 low nibble, e2m1(-3.0) = 0b1101 high nibble.
    assert qweight[0, 0].item() == 0xD7
    dequant = quant_ref.nvfp4_dequant_ref(qweight, block_scale, global_scale)
    assert dequant[0, 0].item() == 2688.0 and dequant[0, 1].item() == -1344.0


def test_nvfp4_roundtrip_error_bound():
    torch.manual_seed(0)
    weight = torch.randn(32, 64) * 100.0
    qweight, block_scale, global_scale = quant_ref.nvfp4_quantize_ref(weight)
    dequant = quant_ref.nvfp4_dequant_ref(qweight, block_scale, global_scale)
    # Per-element bound on the e2m1 grid: |err| <= max(0.25, |v| / 2) in grid
    # units (round-to-nearest, ties to even; values below 0.25 flush to zero).
    scaled = weight / global_scale
    v = scaled / block_scale.float().repeat_interleave(16, dim=1)
    err = ((dequant - weight) / (global_scale * block_scale.float().repeat_interleave(16, dim=1))).abs()
    bound = torch.maximum(torch.full_like(v, 0.25), v.abs() / 2)
    assert (err <= bound + 1e-5).all(), f"max violation {(err - bound).max()}"
    # Error bound is tight for elements that land exactly on grid points.
    on_grid = torch.tensor([1.0, 2.0, 6.0, -0.5]).repeat(16).view(1, 64) * 10.0
    q2, s2, g2 = quant_ref.nvfp4_quantize_ref(on_grid)
    d2 = quant_ref.nvfp4_dequant_ref(q2, s2, g2)
    torch.testing.assert_close(d2, on_grid, rtol=1e-6, atol=1e-5)


def test_nvfp4_all_zero_weight():
    weight = torch.zeros(2, 32)
    qweight, block_scale, global_scale = quant_ref.nvfp4_quantize_ref(weight)
    dequant = quant_ref.nvfp4_dequant_ref(qweight, block_scale, global_scale)
    assert not torch.isnan(dequant).any()
    assert (dequant == 0).all()

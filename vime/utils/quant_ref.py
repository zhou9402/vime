"""Pure-torch reference implementations for low-precision quantization.

These functions are the deterministic oracle for the MXFP8/NVFP4 port
(see docs/en/advanced/low-precision.md). They favor clarity and exact,
CPU-reproducible numerics over speed. Production quantizers
(``vime.utils.mxfp8``, TransformerEngine) are tested against them.

Port of miles@ba28991834 numerics (miles/utils/mxfp8.py, miles/utils/nvfp4.py,
miles/kernels/quant/nvfp4_qdq.py docstring contract), reimplemented without
TransformerEngine / CuTe dependencies.
"""

import torch

MXFP8_GROUP_SIZE = 32
NVFP4_GROUP_SIZE = 16

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0

# E2M1 grid (sign excluded): 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# 3-bit magnitude codes aligned with _E2M1_LEVELS (e2m1: [ee][m], bias 1,
# subnormal E=0 gives m * 0.5).
_E2M1_CODES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)


def _round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even, saturating cast to float8_e4m3fn, returned as fp32."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even, saturating cast to the e2m1 grid, returned as fp32."""
    levels = _E2M1_LEVELS.to(x.device)
    sign = torch.sign(x)
    ax = x.abs().clamp_max(FP4_E2M1_MAX)
    # Distance to each level; argmin with round-half-to-even on the level index.
    dist = (ax.unsqueeze(-1) - levels).abs()
    idx = dist.argmin(dim=-1)
    # Exact ties (x.25 boundaries) resolve to the even code, matching IEEE RNE.
    min_dist = dist.min(dim=-1, keepdim=True).values
    tied = (dist == min_dist).to(torch.int64).sum(dim=-1) > 1
    if tied.any():
        # argmin returns the lower of the two tied adjacent levels; the even
        # code is the lower one when it is even, otherwise the upper one.
        even_idx = torch.where(idx % 2 == 0, idx, idx + 1)
        idx = torch.where(tied, even_idx, idx)
    return sign * levels[idx]


def _e2m1_pack(q: torch.Tensor) -> torch.Tensor:
    """Pack an fp32 tensor of e2m1 values into uint8, two values per byte along the last dim."""
    levels = _E2M1_LEVELS.to(q.device)
    codes = _E2M1_CODES.to(q.device)
    sign_bit = (q < 0).to(torch.uint8) << 3
    idx = (q.abs().unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1)
    nibbles = sign_bit | codes[idx]
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"Last dim {q.shape[-1]} must be even to pack e2m1 pairs.")
    nibbles = nibbles.view(*q.shape[:-1], q.shape[-1] // 2, 2)
    return (nibbles[..., 0] | (nibbles[..., 1] << 4)).contiguous()


def _e2m1_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _e2m1_pack; returns fp32 e2m1 values with doubled last dim."""
    levels = _E2M1_LEVELS.to(packed.device)
    lo = packed & 0x0F
    hi = packed >> 4
    nibbles = torch.stack((lo, hi), dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
    sign = torch.where(nibbles & 0x08 != 0, -1.0, 1.0)
    mag = levels[(nibbles & 0x07).to(torch.int64)]
    return sign * mag


def block_fp8_dequant(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize block-wise FP8 (e.g. [128,128] blocks with fp32 scales) to fp32.

    Drop-in, dependency-free replacement for
    ``sglang.srt.layers.quantization.fp8_utils.block_quant_dequant``.
    ``scale`` has shape ``(ceil(M/bm), ceil(K/bk))``; partial edge blocks allowed.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    bm, bk = block_size
    m, k = weight.shape
    if scale.shape != ((m + bm - 1) // bm, (k + bk - 1) // bk):
        raise ValueError(f"Scale shape {tuple(scale.shape)} does not match weight {(m, k)} / block {block_size}")
    rows = torch.arange(m, device=weight.device) // bm
    cols = torch.arange(k, device=weight.device) // bk
    return weight.to(torch.float32) * scale.to(torch.float32)[rows][:, cols]


def mxfp8_quantize_ref(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference rowwise MXFP8 quantize: e4m3 values + ue8m0 (uint8) scales.

    Per 32-element group along the last dim:
    ``scale = 2**ceil(log2(amax / 448))`` (exponent stored biased by 127),
    ``q = RN_e4m3(w / scale)``. Groups with amax == 0 use exponent byte 0.
    Returns ``(qweight float8_e4m3fn, scale uint8)`` with scale shape
    ``(..., K // 32)``.
    """
    w = weight.to(torch.float32)
    k = w.shape[-1]
    if k % MXFP8_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {MXFP8_GROUP_SIZE} for MXFP8.")
    grouped = w.view(*w.shape[:-1], k // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE)
    amax = grouped.abs().amax(dim=-1)
    # ue8m0 exponent: smallest e with 2^(e-127) * 448 >= amax.
    exp = torch.ceil(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny) / FP8_E4M3_MAX)) + 127.0
    exp = torch.where(amax == 0, torch.zeros_like(exp), exp).clamp(0.0, 254.0)
    scale_u8 = exp.to(torch.uint8)
    scale = torch.pow(2.0, exp - 127.0)
    q = _round_to_e4m3(grouped / scale.unsqueeze(-1)).view_as(w)
    return q.to(torch.float8_e4m3fn), scale_u8


def mxfp8_dequant_ref(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 (e4m3 values, uint8 ue8m0 scales) to fp32."""
    k = qweight.shape[-1]
    if scale.shape[-1] != k // MXFP8_GROUP_SIZE:
        raise ValueError(f"Scale last dim {scale.shape[-1]} != K//32 for K={k}")
    scale_f = torch.pow(2.0, scale.to(torch.float32) - 127.0)
    grouped = qweight.to(torch.float32).view(*qweight.shape[:-1], k // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE)
    return (grouped * scale_f.unsqueeze(-1)).view_as(qweight)


def nvfp4_quantize_ref(
    weight: torch.Tensor,
    nvfp4_e4m3_max: float = FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantize of a 2D weight, mirroring TE numerics.

    Returns ``(qweight uint8 packed 2-per-byte along K, block_scale
    float8_e4m3fn [M, K//16], global_decode_scale fp32 scalar)`` where
    ``global_decode_scale = amax / (nvfp4_e4m3_max * 6)`` (1.0 when amax == 0).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    w = weight.to(torch.float32)
    m, k = w.shape
    if k % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {k} must be divisible by {NVFP4_GROUP_SIZE} for NVFP4.")

    global_amax = w.abs().max()
    encode = (nvfp4_e4m3_max * FP4_E2M1_MAX) / global_amax
    encode = torch.where(global_amax == 0, torch.ones_like(encode), encode)
    global_decode = 1.0 / encode

    scaled = w * encode
    grouped = scaled.view(m, k // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    block_amax = grouped.abs().amax(dim=-1)
    # Block scale: e4m3 RN of amax / 6 so that values land within the e2m1 grid.
    block_scale_f = _round_to_e4m3(block_amax / FP4_E2M1_MAX)
    block_scale_f = torch.where(block_amax == 0, torch.ones_like(block_scale_f), block_scale_f)
    q = _round_to_e2m1(grouped / block_scale_f.unsqueeze(-1)).view(m, k)
    return (
        _e2m1_pack(q),
        block_scale_f.to(torch.float8_e4m3fn),
        global_decode.reshape(()).to(torch.float32),
    )


def nvfp4_dequant_ref(
    qweight: torch.Tensor,
    block_scale: torch.Tensor,
    global_decode_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize NVFP4 triple back to fp32 [M, K]."""
    q = _e2m1_unpack(qweight)
    m, k = q.shape
    grouped = q.view(m, k // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    scaled = grouped * block_scale.to(torch.float32).unsqueeze(-1)
    return scaled.view(m, k) * global_decode_scale.to(torch.float32)


def nvfp4_quantize_4over6_ref(
    weight: torch.Tensor,
    error_mode: str = "MAE",
    nvfp4_e4m3_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """4-over-6 reference: per 16-element block, pick the better of the
    standard (amax/6) and candidate (amax/4) block scales by the error mode.
    Returns the dequantized fp32 tensor (QDQ), brute force per block."""
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")
    w = weight.to(torch.float32)
    m, k = w.shape
    global_amax = w.abs().max()
    encode = (nvfp4_e4m3_max * FP4_E2M1_MAX) / global_amax
    encode = torch.where(global_amax == 0, torch.ones_like(encode), encode)
    scaled = w * encode
    grouped = scaled.view(m, k // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    block_amax = grouped.abs().amax(dim=-1)

    dequants = []
    for div in (FP4_E2M1_MAX, 4.0):
        bs = _round_to_e4m3(block_amax / div).clamp(max=float(nvfp4_e4m3_max))
        bs = torch.where(block_amax == 0, torch.ones_like(bs), bs)
        q = _round_to_e2m1(grouped / bs.unsqueeze(-1))
        dequants.append(q * bs.unsqueeze(-1))
    if error_mode.upper() == "MSE":
        errs = [((d - grouped) ** 2).sum(dim=-1) for d in dequants]
    else:
        errs = [(d - grouped).abs().sum(dim=-1) for d in dequants]
    pick_candidate = (errs[1] < errs[0]).unsqueeze(-1)
    chosen = torch.where(pick_candidate, dequants[1], dequants[0])
    return chosen.view(m, k) / encode

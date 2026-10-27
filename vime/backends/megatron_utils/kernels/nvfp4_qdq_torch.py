"""Pure-torch fused NVFP4 quantize->dequantize (QDQ) for fake QAT.

Portable fallback for miles' CuTe-DSL kernel
(miles/kernels/quant/nvfp4_qdq.py, SM10x-only, not ported). Implements the
same numerics contract with plain torch ops, so it runs on any device; it is
slower than the fused kernel but bit-comparable against the
``vime.utils.quant_ref`` oracle and (without 4over6) against TE's
``NVFP4Quantizer``.

Config surface mirrors the kernel's env contract:
``NVTE_USE_FAST_MATH`` (must be 0), ``NVTE_NVFP4_4OVER6``,
``NVTE_NVFP4_4OVER6_E4M3_USE_256``, ``NVTE_NVFP4_4OVER6_ERR_MODE`` (MAE/MSE).

4-over-6 is implemented per 16-element block like the kernel: two candidate
block scales (RN_e4m3(amax/6) and RN_e4m3(amax/4)) are evaluated and the
lower-error one wins per block (MAE/MSE per the config).
"""

import os
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import torch

from vime.utils import quant_ref
from vime.utils.nvfp4 import NVFP4_GROUP_SIZE, nvfp4_global_encode_scale_te


class NVFP4QDQErrorMode(IntEnum):
    MAE = 0
    MSE = 1


@dataclass(frozen=True)
class NVFP4QDQConfig:
    use_4over6: bool = False
    e4m3_max: int = 448
    error_mode: NVFP4QDQErrorMode = NVFP4QDQErrorMode.MAE

    def __post_init__(self):
        if self.e4m3_max not in (256, 448):
            raise ValueError(f"e4m3_max must be 256 or 448, got {self.e4m3_max}")
        if self.e4m3_max == 256 and not self.use_4over6:
            raise ValueError("e4m3_max=256 only applies to the 4over6 path.")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "weights", "all")


@lru_cache(maxsize=1)
def current_nvfp4_qdq_config() -> NVFP4QDQConfig:
    """QDQ config resolved once per process from NVTE_* env vars.

    Mirrors the INT4 path, which reads its env once at hook install time.
    Tests must call ``current_nvfp4_qdq_config.cache_clear()`` after changing
    the env (see reset_nvfp4_qdq_config_for_tests).
    """
    if _env_flag("NVTE_USE_FAST_MATH"):
        raise RuntimeError("NVFP4 fake QAT requires NVTE_USE_FAST_MATH=0 (exact RN numerics).")
    use_4over6 = _env_flag("NVTE_NVFP4_4OVER6")
    e4m3_max = 448
    if use_4over6 and _env_flag("NVTE_NVFP4_4OVER6_E4M3_USE_256"):
        e4m3_max = 256
    error_mode = NVFP4QDQErrorMode[os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper()]
    return NVFP4QDQConfig(use_4over6=use_4over6, e4m3_max=e4m3_max, error_mode=error_mode)


def reset_nvfp4_qdq_config_for_tests() -> None:
    """Drop the cached QDQ config so the next call re-reads NVTE_* env vars."""
    current_nvfp4_qdq_config.cache_clear()


def _quantize_dequantize(weight: torch.Tensor, scale_div: float, e4m3_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """QDQ of a [M, K] fp32 tensor at fixed block-scale divisor; returns (dequant, block_scale_f)."""
    m, k = weight.shape
    grouped = weight.view(m, k // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    block_amax = grouped.abs().amax(dim=-1)
    block_scale_f = quant_ref._round_to_e4m3(block_amax / scale_div).clamp(max=float(e4m3_max))
    block_scale_f = torch.where(block_amax == 0, torch.ones_like(block_scale_f), block_scale_f)
    q = quant_ref._round_to_e2m1(grouped / block_scale_f.unsqueeze(-1))
    return (q * block_scale_f.unsqueeze(-1)).view(m, k), block_scale_f


def fused_nvfp4_qdq(weight: torch.Tensor, config: NVFP4QDQConfig | None = None) -> torch.Tensor:
    """Detached NVFP4 quantize->dequantize of a 2D weight, TE numerics."""
    if config is None:
        config = current_nvfp4_qdq_config()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if weight.dim() != 2:
        raise ValueError(f"NVFP4 QDQ expects a 2D weight, got shape {tuple(weight.shape)}")
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} for NVFP4.")

    w = weight.to(torch.float32)
    global_amax = w.abs().max()
    encode = nvfp4_global_encode_scale_te(global_amax, config.e4m3_max)
    scaled = w * encode

    dequant_standard, _ = _quantize_dequantize(scaled, quant_ref.FP4_E2M1_MAX, config.e4m3_max)
    if config.use_4over6:
        dequant_candidate, _ = _quantize_dequantize(scaled, 4.0, config.e4m3_max)
        # The kernel selects the divisor per 16-element block, not per row.
        m, k = scaled.shape
        group_shape = (m, k // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
        err_standard = (dequant_standard - scaled).view(group_shape)
        err_candidate = (dequant_candidate - scaled).view(group_shape)
        if config.error_mode == NVFP4QDQErrorMode.MSE:
            err_standard = (err_standard**2).sum(dim=-1, keepdim=True)
            err_candidate = (err_candidate**2).sum(dim=-1, keepdim=True)
        else:
            err_standard = err_standard.abs().sum(dim=-1, keepdim=True)
            err_candidate = err_candidate.abs().sum(dim=-1, keepdim=True)
        pick_candidate = (err_candidate < err_standard).view(m, k // NVFP4_GROUP_SIZE, 1)
        dequant = torch.where(
            pick_candidate,
            dequant_candidate.view(group_shape),
            dequant_standard.view(group_shape),
        ).view(m, k)
    else:
        dequant = dequant_standard
    return (dequant / encode).to(weight.dtype)


class _FusedNVFP4QDQSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, config):
        return fused_nvfp4_qdq(weight.detach(), config)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_nvfp4_quantization_ste(weight: torch.Tensor, config: NVFP4QDQConfig | None = None) -> torch.Tensor:
    """Straight-through NVFP4 fake quantization; preserves Megatron main_grad."""
    quantized = _FusedNVFP4QDQSTE.apply(weight, config)
    if hasattr(weight, "main_grad"):
        quantized.main_grad = weight.main_grad
    return quantized

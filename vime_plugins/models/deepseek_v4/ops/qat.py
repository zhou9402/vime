import torch

try:  # fast path: share the DeepSeek FP8 kernels (A2 quant kernels + TileKernels)
    from tile_kernels.quant import per_token_cast_back

    from vime.kernels.quant.fp8_act_quant import act_quant
except ImportError:  # CPU CI / quant kernels not yet ported: pure-torch fallback
    per_token_cast_back = None
    act_quant = None


def _fp8_simulate_torch(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pure-torch per-token-block FP8 (E4M3) cast + dequant with UE8M0 scaling."""
    x_c = x.contiguous()
    n = x_c.size(-1)
    blocks = x_c.view(-1, n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))  # ue8m0: power-of-two scale
    y = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (y.float() * scale).view_as(x_c).to(x.dtype)


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    The fast path routes the cast (via :func:`act_quant`) and cast-back through
    ``deepseek-ai/TileKernels`` so we share the same FP8 kernels with the rest
    of the DeepSeek stack; without them a pure-torch equivalent is used.
    """
    if act_quant is None:
        return _fp8_simulate_torch(x, block_size)
    x_c = x.contiguous()
    y, scale = act_quant(x_c, block_size, "ue8m0")

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = per_token_cast_back((y_flat, scale_flat), "bf16" if x.dtype == torch.bfloat16 else "fp32", block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply

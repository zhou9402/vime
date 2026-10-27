# Triton kernel bundle (`vime.kernels`)

`vime/kernels/` vendors standalone, backend-neutral Triton kernels ported from miles. They carry no
Megatron, process-group, or engine coupling; model plugins under `vime_plugins/` import them.

Current ops:

| Module | Entry points | Notes |
|---|---|---|
| `vime/kernels/activation/swiglu_fp32.py` | `swiglu_fp32` | fp32-accumulated SwiGLU with optional per-token scale (fwd+bwd) |
| `vime/kernels/norm/grouped_rmsnorm.py` | `grouped_rmsnorm_fwd`, `grouped_rmsnorm_fwd_kernel`, `grouped_rmsnorm_bwd_kernel` | grouped RMSNorm; device helpers shared by hyper-connection kernels |
| `vime/kernels/position/rope.py` | `apply_rotary_emb` | in-place RoPE on `[b, s, d]` or `[b, s, h, d]`, autograd-safe |
| `vime/kernels/softmax/token_log_softmax.py` | `row_statistics`, `write_logits_grad`, `zero_unscored_rows` | sharded per-token log-softmax stats/grad with vocab-padding support |
| `vime/kernels/quant/fp8_blockwise.py` | `fp8_blockwise_cast` | 128x128-block fp8e4m3 cast + scales |
| `vime/kernels/quant/fp8_act_quant.py` | `act_quant` | tilelang fp8 act-quant, optional MXFP power-of-2 scales, fused inplace quant+dequant |
| `vime/kernels/quant/fake_quant.py` | `fake_quant_fp8`, `fake_quant_fp4`, `fake_quant_compressed_kv` | straight-through fake quantization |
| `vime/kernels/hyper_connection/hc.py` | `hc_mix_inject`, `hc_combine` | fused hyper-connection mix/inject and combine |
| `vime/kernels/hyper_connection/mhc.py` | `mhc_mix`, `mhc_aggregate` | multi-stream hyper-connection mix/aggregate |

`vime/backends/megatron_utils/kernels/fp8_kernel.py` is an older copy of `fp8_blockwise.py` kept for
slime-mirror compatibility; new code should use `vime.kernels.quant.fp8_blockwise`.

Every op has a torch-reference test under `tests/fast-gpu/kernels/<area>/` (GPU-only; tests skip on
CPU-only hosts). See `vime/kernels/README.md` for the directory conventions.

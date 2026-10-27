# Triton kernel 包（`vime.kernels`）

`vime/kernels/` 收录了从 miles 移植的独立、后端无关的 Triton kernel。它们不依赖 Megatron、进程组或推理引擎；
模型插件（`vime_plugins/`）直接 import 使用。

当前算子：

| 模块 | 入口 | 说明 |
|---|---|---|
| `vime/kernels/activation/swiglu_fp32.py` | `swiglu_fp32` | fp32 累加 SwiGLU，可选 per-token scale（含反向） |
| `vime/kernels/norm/grouped_rmsnorm.py` | `grouped_rmsnorm_fwd`、`grouped_rmsnorm_fwd_kernel`、`grouped_rmsnorm_bwd_kernel` | 分组 RMSNorm；device helper 供 hyper-connection kernel 复用 |
| `vime/kernels/position/rope.py` | `apply_rotary_emb` | 原地 RoPE，支持 `[b, s, d]` / `[b, s, h, d]`，带 autograd |
| `vime/kernels/softmax/token_log_softmax.py` | `row_statistics`、`write_logits_grad`、`zero_unscored_rows` | 分词 log-softmax 统计/梯度，支持 vocab padding 分片 |
| `vime/kernels/quant/fp8_blockwise.py` | `fp8_blockwise_cast` | 128x128 分块 fp8e4m3 cast + scale |
| `vime/kernels/quant/fp8_act_quant.py` | `act_quant` | tilelang fp8 act-quant，可选 MXFP 2 的幂 scale，支持 inplace 融合 quant+dequant |
| `vime/kernels/quant/fake_quant.py` | `fake_quant_fp8`、`fake_quant_fp4`、`fake_quant_compressed_kv` | 直通梯度 fake quantization |
| `vime/kernels/hyper_connection/hc.py` | `hc_mix_inject`、`hc_combine` | 融合 hyper-connection mix/inject 与 combine |
| `vime/kernels/hyper_connection/mhc.py` | `mhc_mix`、`mhc_aggregate` | 多流 hyper-connection mix/aggregate |

`vime/backends/megatron_utils/kernels/fp8_kernel.py` 是 `fp8_blockwise.py` 的旧拷贝，为兼容 slime 镜像而保留；
新代码请使用 `vime.kernels.quant.fp8_blockwise`。

每个算子在 `tests/fast-gpu/kernels/<area>/` 下有 torch 参考实现对照测试（仅 GPU；无 GPU 主机上自动 skip）。
目录约定见 `vime/kernels/README.md`。

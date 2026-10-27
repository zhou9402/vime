# MFU 指标与指标历史回归门

本页记录从 miles 移植的性能计量与 CI 回归门工具（来源 `miles@ba28991834`）。

## FLOPs 模型与 V5 oracle

`vime/utils/flops_model.py` 根据 Hugging Face config 解析计算 decoder-only
模型的前向与训练 FLOPs，覆盖 dense GQA、MoE、MLA、Gated DeltaNet / KDA
线性注意力、DSA/QSA 稀疏注意力 indexer、DeepSeek-V4 压缩注意力以及
Nemotron-H Mamba2 混合结构。MTP 层按 miles 的设计刻意不计入；如需
vime 专属的 MTP 项，作为后续工作单独记录。

`tests/ci/flops_oracle/flops_utils_miles_ba28991834.py` 是原样拷贝的 miles
实现，作为 oracle。验证器在 8 个模型架构配置上比较任意实现与 oracle，
相对误差达到 1e-6 即判失败：

```bash
python tests/ci/flops_oracle/verify_flops.py --impl vime/utils/flops_model.py
```

退出码 0 表示通过（实现文件尚不存在时输出 SKIP 并通过）；退出码 1
表示不一致。

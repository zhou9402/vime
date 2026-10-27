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

## MFU 指标

`log_perf_data` 每个 rollout 记录 `perf/actor_train_mfu`（以及
`perf/mfu_peak_tflops` 和 `perf/actor_train_tflops_hf`），同时保留不变的
slime 原有 `perf/actor_train_tflops`。MFU 分子来自 HF-config FLOPs 模型
（对 `args.hf_checkpoint` 调用 `model_flops_from_hf_config`，按 checkpoint
缓存）；分母优先取 `--mfu-peak-tflops`，否则查
`vime/utils/device_flops.py` 内置设备表（NVIDIA A100/H100/H200/GH200/
B200/B300/GB200/GB300 与 AMD MI300X/MI325X/MI355X）。两者都不可得时不记录
`perf/mfu*` 指标。

## 指标历史存储

`tests/ci/metric_history/storage/` 定义指标历史契约（`MetricHistoryStore`、
`RunIdentity`、`RunProvenance`）及 SQLite 后端。baseline 按
`(test_path, backend, suite)` 隔离；比较坐标是 `(metric_key, step)`；
信任标记在 run 级别，可通过 `mark_untrusted`（按 `run_id`、`ci_run_id`
或 `commit_sha`）撤销。provenance 列使用 CI 中性命名（`ci_run_id` =
Buildkite build 号，`ci_run_attempt` = 重试次数，`pipeline` = pipeline 名）。

# HF→Megatron 加载模式（builtin / mbridge）

`--hf-to-megatron-load-mode {builtin,mbridge}` 选择当 `--load` 指向 HF 目录时，
如何把 HuggingFace 权重加载进 Megatron 模型。

- `builtin`（默认）：vime 手写的转换器
  `vime/backends/megatron_utils/hf_to_megatron/`，按 checkpoint 的
  `config.model_type` 分发。
- `mbridge`：外部 [mbridge](https://github.com/ISEEKYAN/mbridge) 包
  （固定 commit `89eb1088`，与 miles 一致），配合 `vime_plugins/mbridge/` 中的
  bridge 注册。没有 builtin 转换器的 day-0 模型
  （DeepSeek-V4/V4.1、GLM5-Next、Kimi-K3、Qwen3.8-Next）需要此模式。

与 miles 的对应关系：miles 的 `--megatron-to-hf-mode raw` 对应
`--hf-to-megatron-load-mode mbridge`；miles 的 `bridge` 模式（NVIDIA
Megatron-Bridge）在 vime 中不支持。Megatron→HF 导出路径不受此 flag 影响。

mbridge 模式需要固定版本的包：`pip install --no-deps
git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c`
（vime Docker 镜像已预装）。

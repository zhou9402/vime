# LoRA（进行中）

vime 正在从 miles 移植 LoRA 支持（`miles@ba28991834`，移植编号 F04）。本页面随每个 PR 逐步完善。

当前状态：

- **F04-0**：等价性校验器 `tests/test_lora_equivalence.py`——在 2 层 tiny 模型上，合并 LoRA 后的 logits 必须与功能性施加 adapter 的 logits 一致；adapter 导出/导入与 trainer→engine 同步必须逐位一致（含篡改检测）。运行：`python -m pytest tests/test_lora_equivalence.py -x -q`。

训练侧 LoRA（Megatron native 路径）与 vLLM adapter 加载/同步将在后续 PR 落地。

- **F04-1**：引擎无关的辅助包 `vime/utils/lora/`（`utils.py`：`LORA_ADAPTER_NAME="vime_lora"`、门控谓词、`save_adapter_to_disk`、`build_lora_config`；`hf_lora_targets.py`：按模型类型解析 HF 命名空间的 LoRA target 布局）。测试：`tests/utils/lora/`。

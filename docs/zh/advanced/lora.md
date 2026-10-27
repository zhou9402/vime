# LoRA（进行中）

vime 正在从 miles 移植 LoRA 支持（`miles@ba28991834`，移植编号 F04）。本页面随每个 PR 逐步完善。

当前状态：

- **F04-0**：等价性校验器 `tests/test_lora_equivalence.py`——在 2 层 tiny 模型上，合并 LoRA 后的 logits 必须与功能性施加 adapter 的 logits 一致；adapter 导出/导入与 trainer→engine 同步必须逐位一致（含篡改检测）。运行：`python -m pytest tests/test_lora_equivalence.py -x -q`。

训练侧 LoRA（Megatron native 路径）与 vLLM adapter 加载/同步将在后续 PR 落地。

- **F04-1**：引擎无关的辅助包 `vime/utils/lora/`（`utils.py`：`LORA_ADAPTER_NAME="vime_lora"`、门控谓词、`save_adapter_to_disk`、`build_lora_config`；`hf_lora_targets.py`：按模型类型解析 HF 命名空间的 LoRA target 布局）。测试：`tests/utils/lora/`。

- **F04-2**：新增 CLI 参数（`--lora-rank/--lora-alpha/--lora-dropout/--lora-a-init-method/--lora-type/--target-modules/--exclude-modules/--lora-adapter-path/--lora-sync-from-tensor/--lora-base-cpu-backup/--lora-train-only/--experts-shared-outer-loras/--check-lora-weight-equal`，经 `vime/utils/arguments.py` 的 `add_lora_arguments`），以及 `vime/utils/lora/arguments.py` 校验逻辑和移植的 `vime/utils/hf_utils/`（`load_hf_config`、`HfWeightMapping`）。第一阶段仅支持 attention + 稠密 MLP LoRA——expert 投影与 multi-LoRA 会被明确报错拒绝。

- **F04-3a**：native LoRA 核心——`vime/backends/megatron_utils/lora/native/{config,distributed,naming}.py` 与 `spec/base.py`（`LoRAConfig`、TP=1 直通即可运行的 TP/EP/SP 通信辅助、`ConverterNaming`/`DeclaredNaming`，经新增的 `megatron_to_hf.convert_to_hf_names` 桥接、架构 spec 契约）。测试：`tests/backends/megatron_utils/lora/native/test_core.py`。

- **F04-3b**：native LoRA 模块——`native/modules/linear.py`（`NativeLoRAAdapter`、`LoRALinear`、`LoRASplitQKV`/`FC1`、forward hook 挂接）、`native/modules/lm_head.py`、`native/spec/layout.py`（模块布局/fused attach）。测试：`test_modules.py`（QKV 排列、初始化分布、fused-norm 宿主契约）。

- **F04-3c**：架构 spec——`native/spec/{attention,mlp,lm_head}.py`（GQA/带门控 GQA/MLA 注意力布局、split+fused gate/up MLP）与 `native/merge.py`（用于全量 HF 导出的 `MergedWeights` 视图）。测试：`test_specs.py`（裸模块挂接、零初始化等价、梯度流、target 选择、合并 delta）。

- **F04-3d**：MoE adapter——`native/modules/moe.py`（`LoRAGroupedFC1/FC2`、shared-experts adapter）与 `native/spec/moe.py`；移植 miles 的 `test_merge.py`（合并权重 delta 必须复现前向 delta，含 MLA 与分组 experts）。rollout 侧 expert LoRA 仍由 F04-2 的校验关闭。

- **F04-3e**：注册表与入口——`native/registry.py`（`MODEL_SPECS`、`resolve_adapter_targets`、`validate_native_lora_args`、`preflight_native_lora`）、`native/lora.py`（`apply_native_lora`、`wrap_model_provider_with_lora`）、包导出与 `megatron_to_hf.converter_model_name` 桥接。测试：`test_registry.py`（架构守卫、target 契约、attach 遍历、运行配置校验）。

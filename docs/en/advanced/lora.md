# LoRA (work in progress)

vime is porting LoRA support from miles (`miles@ba28991834`, port F04). This page is built up per PR.

Current status:

- **F04-0**: equivalence verifier `tests/test_lora_equivalence.py` — merged-LoRA logits must equal adapter-applied logits on a 2-layer tiny model; adapter export/import and trainer→engine sync must be bitwise identical (tamper detection included). Run: `python -m pytest tests/test_lora_equivalence.py -x -q`.

Training-side LoRA (Megatron, native path) and vLLM adapter loading/sync land in subsequent PRs.

- **F04-1**: engine-agnostic helper package `vime/utils/lora/` (`utils.py`: `LORA_ADAPTER_NAME="vime_lora"`, gating predicates, `save_adapter_to_disk`, `build_lora_config`; `hf_lora_targets.py`: per-model-type HF-namespace target layouts and resolution). Tests: `tests/utils/lora/`.

- **F04-2**: CLI flags `--lora-rank/--lora-alpha/--lora-dropout/--lora-a-init-method/--lora-type/--target-modules/--exclude-modules/--lora-adapter-path/--lora-sync-from-tensor/--lora-base-cpu-backup/--lora-train-only/--experts-shared-outer-loras/--check-lora-weight-equal` (via `add_lora_arguments` in `vime/utils/arguments.py`), plus `vime/utils/lora/arguments.py` validation and the ported `vime/utils/hf_utils/` (`load_hf_config`, `HfWeightMapping`). Phase 1: attention + dense-MLP LoRA only — expert projections and multi-LoRA are rejected with a clear error.

- **F04-3a**: native LoRA core — `vime/backends/megatron_utils/lora/native/{config,distributed,naming}.py` + `spec/base.py` (`LoRAConfig`, TP/EP/SP collectives with TP=1 passthrough, `ConverterNaming`/`DeclaredNaming` via the new `megatron_to_hf.convert_to_hf_names` shim, architecture spec contracts). Tests: `tests/backends/megatron_utils/lora/native/test_core.py`.

- **F04-3b**: native LoRA modules — `native/modules/linear.py` (`NativeLoRAAdapter`, `LoRALinear`, `LoRASplitQKV`/`FC1`, forward-hook attach), `native/modules/lm_head.py`, `native/spec/layout.py` (module layouts / fused attach). Tests: `test_modules.py` (QKV permutation, init distributions, fused-norm host contract).

- **F04-3c**: architecture specs — `native/spec/{attention,mlp,lm_head}.py` (GQA/gated-GQA/MLA attention layouts, split+fused gate/up MLP) and `native/merge.py` (`MergedWeights` view for full-model HF export). Tests: `test_specs.py` (attach on bare blocks, zero-init no-op, grad flow, target selection, merge deltas).

- **F04-3d**: MoE adapters — `native/modules/moe.py` (`LoRAGroupedFC1/FC2`, shared-experts adapter) + `native/spec/moe.py`; ported miles' `test_merge.py` (merged-weight deltas must reproduce forward deltas, incl. MLA and grouped experts). Rollout-side expert LoRA remains gated off (F04-2 validation).

- **F04-3e**: registry + entry point — `native/registry.py` (`MODEL_SPECS` for llama/qwen2/qwen3(_moe)/glm4(_moe)/deepseek_v3(_v32)/kimi_k2/k25/k3/qwen3_next/qwen3_5(_moe)/glm_moe_dsa/glm5_next/inkling/mimo, `resolve_adapter_targets`, `validate_native_lora_args`, `preflight_native_lora`), `native/lora.py` (`apply_native_lora`, `wrap_model_provider_with_lora`), package exports, `megatron_to_hf.converter_model_name` shim. Tests: `test_registry.py` (architecture guards, target contract, attach walk, run-contract validation).

- **F04-4**: Megatron train-side glue — `vime/backends/megatron_utils/lora/utils.py` (grad reduce for replicated adapter params, adapter discovery, save/restore) and hooks in `vime/backends/megatron_utils/model.py`: provider wrapped with `wrap_model_provider_with_lora` for the actor, `finalize_model_grads` gains the marked-adapter TP/EP sum when LoRA is on, and `--lora-adapter-path` is restored after the base checkpoint (its iteration overrides rollout numbering). Note: colocate `--offload-train` + LoRA fails fast because vime's pinned Megatron lacks the fork's `disable_*_cpu_backup` ddp_config flags.

- **F04-5**: adapter-only checkpoints — `save_checkpoint` in `vime/backends/megatron_utils/checkpoint.py` routes LoRA models to `save_lora_checkpoint`, writing `<save>/iter_NNNNNNN/adapter/` (per-rank `adapter_megatron_rank*.pt`, optional `training_state_rank*.pt`, `adapter_config.json`) and updating `latest_checkpointed_iteration.txt`; the frozen base is never re-saved. Resume = `--load <base>` + `--lora-adapter-path <save>/iter_NNNNNNN/adapter`. Tests: `tests/backends/megatron_utils/test_lora_checkpoint.py`.

- **Review fixes**: LoRA checkpoints no longer touch Megatron's tracker — progress lives in `<save>/latest_lora_iteration.txt`; resume via `--lora-adapter-path <save>/iter_NNNNNNN/adapter` or auto-resolved from `--save`. `--lora-rank>0` + `--offload-train` is rejected at arg validation. LoRA arg validation raises `ValueError` (not bare asserts). `vime/utils/hf_utils/` is a verbatim miles copy (modulo import prefix + the DeepseekV4 sglang branch → `NotImplementedError`).

- **F04-6**: vLLM engine enablement — `_compute_server_args` sets `enable_lora/max_lora_rank/max_loras` (and `lora_modules` only for debug disk-load); the engine subprocess gets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`; rollout requests route to the adapter via `rollout_model_name` (served name `vime_lora`) in `vllm_rollout.py`. Tests: `tests/utils/test_vllm_engine_lora.py`, `rollout_model_name` predicate test.

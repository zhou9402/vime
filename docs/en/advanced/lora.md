# LoRA (work in progress)

vime is porting LoRA support from miles (`miles@ba28991834`, port F04). This page is built up per PR.

Current status:

- **F04-0**: equivalence verifier `tests/test_lora_equivalence.py` — merged-LoRA logits must equal adapter-applied logits on a 2-layer tiny model; adapter export/import and trainer→engine sync must be bitwise identical (tamper detection included). Run: `python -m pytest tests/test_lora_equivalence.py -x -q`.

Training-side LoRA (Megatron, native path) and vLLM adapter loading/sync land in subsequent PRs.

- **F04-1**: engine-agnostic helper package `vime/utils/lora/` (`utils.py`: `LORA_ADAPTER_NAME="vime_lora"`, gating predicates, `save_adapter_to_disk`, `build_lora_config`; `hf_lora_targets.py`: per-model-type HF-namespace target layouts and resolution). Tests: `tests/utils/lora/`.

- **F04-2**: CLI flags `--lora-rank/--lora-alpha/--lora-dropout/--lora-a-init-method/--lora-type/--target-modules/--exclude-modules/--lora-adapter-path/--lora-sync-from-tensor/--lora-base-cpu-backup/--lora-train-only/--experts-shared-outer-loras/--check-lora-weight-equal` (via `add_lora_arguments` in `vime/utils/arguments.py`), plus `vime/utils/lora/arguments.py` validation and the ported `vime/utils/hf_utils/` (`load_hf_config`, `HfWeightMapping`). Phase 1: attention + dense-MLP LoRA only — expert projections and multi-LoRA are rejected with a clear error.

- **F04-3a**: native LoRA core — `vime/backends/megatron_utils/lora/native/{config,distributed,naming}.py` + `spec/base.py` (`LoRAConfig`, TP/EP/SP collectives with TP=1 passthrough, `ConverterNaming`/`DeclaredNaming` via the new `megatron_to_hf.convert_to_hf_names` shim, architecture spec contracts). Tests: `tests/backends/megatron_utils/lora/native/test_core.py`.

"""LoRA argument validation for vime (port of miles/utils/lora/arguments.py @ ba28991834).

Differences from miles:
- vLLM replaces SGLang: no attention-DP pinned-adapter constraints, no sglang_* flags.
- vime has no Megatron-Bridge path, so only the native ("raw") resolution runs;
  the Bridge/multi-LoRA variants are follow-ups (see MILES_PORT_PLAN.md C2).
- Phase 1 rollout supports attention + dense-MLP LoRA only; expert projections
  are rejected here with a clear error (training-side MoE code stays, gated).
"""

import logging
from typing import Any

from vime.utils.hf_utils.config import load_hf_config
from vime.utils.hf_utils.weight_mapping import HfWeightMapping
from vime.utils.lora.hf_lora_targets import (
    LORA_TARGET_GROUPS,
    exclude_hf_lora_targets,
    expand_hf_lora_targets,
    expand_packed_hf_lora_targets,
    get_hf_lora_targets,
    parse_lora_targets,
    resolve_hf_lora_targets,
)
from vime.utils.lora.utils import is_lora_enabled, lora_rollout_enabled, matches_lora_target

logger = logging.getLogger(__name__)


def validate_lora_args(args):
    # Tolerate hand-built namespaces (unit tests, recovery paths) that skip the CLI parser.
    args.lora_rank = getattr(args, "lora_rank", 0)
    args.lora_adapter_path = getattr(args, "lora_adapter_path", None)
    # Derived/non-CLI fields other LoRA helpers read unconditionally.
    args.hf_lora_targets = getattr(args, "hf_lora_targets", None)
    args.lora_adapter_targets = getattr(args, "lora_adapter_targets", None)
    args.lora_B_init_method = getattr(args, "lora_B_init_method", None) or "zero"
    if not hasattr(args, "debug_skip_weight_update"):
        args.debug_skip_weight_update = False
    _validate_multi_lora_unsupported(args)
    if not is_lora_enabled(args):
        return
    if args.train_backend != "megatron":
        raise ValueError("LoRA injection is not implemented for FSDP; use --train-backend megatron")
    if args.lora_rank <= 0:
        raise ValueError("LoRA requires a positive --lora-rank, including when loading an adapter")
    if getattr(args, "offload_train", False):
        raise ValueError(
            "LoRA colocate offload requires Megatron ddp_config fields "
            "['disable_param_buffers_cpu_backup', 'disable_grad_buffers_cpu_backup'], which vime's "
            "pinned Megatron does not provide. Disable --offload-train for colocated LoRA, or port "
            "the fork's param_and_grad_buffer cpu-backup flags."
        )
    if lora_rollout_enabled(args):
        if getattr(args, "colocate", False):
            if not getattr(args, "save", None) and not getattr(args, "lora_rollout_cache_dir", None):
                raise ValueError(
                    "LoRA rollout sync needs an adapter directory on the colocated node's "
                    "filesystem: set --save or --lora-rollout-cache-dir."
                )
        elif not getattr(args, "lora_rollout_cache_dir", None):
            raise ValueError(
                "LoRA rollout sync without --colocate requires --lora-rollout-cache-dir pointing "
                "at a filesystem readable by every engine node (node-local --save is not shared)."
            )
    hf_config = load_hf_config(args.hf_checkpoint)
    args.hf_lora_targets, args.lora_adapter_targets = _resolve_lora_targets(args, hf_config)
    _reject_expert_targets(args.hf_lora_targets)


def validate_lora_args_after_megatron(args):
    """LoRA checks that read Megatron's resolved args (e.g. the expert TP size it defaults from TP)."""
    if is_lora_enabled(args):
        from vime.backends.megatron_utils.lora.native import validate_native_lora_args

        validate_native_lora_args(args)


def _validate_multi_lora_unsupported(args: Any) -> None:
    """Multi-LoRA slots are a follow-up (F04-F1); the flag is not registered yet."""
    args.multi_lora = False
    if getattr(args, "multi_lora_n_adapters", 0) != 0:
        raise ValueError(
            "Multi-LoRA is not supported in vime yet (follow-up F04-F1). "
            "Do not pass --multi-lora-n-adapters."
        )


def _reject_expert_targets(hf_targets: list[str]) -> None:
    experts = [target for target in hf_targets if "expert" in target]
    if experts:
        raise ValueError(
            "MoE-expert LoRA is not supported in this phase: the vLLM rollout side serves "
            "attention + dense-MLP adapters only. Restrict --target-modules (e.g. 'attn') or "
            f"--exclude-modules so no expert projections are selected; got {experts}."
        )


def _resolve_lora_targets(args, hf_config):
    """Return final HF and adapter targets without changing the CLI selectors in `args`."""
    hf_mapping = HfWeightMapping.from_config(hf_config)
    hf_modules = sorted(name.removesuffix(".weight") for name in hf_mapping.parameter_names)
    targets = parse_lora_targets(args.target_modules)
    exclusions = parse_lora_targets(args.exclude_modules) or []
    explicit_targets = []
    if targets is not None:
        expanded = expand_packed_hf_lora_targets(targets, hf_modules)
        explicit_targets = [
            target for target in dict.fromkeys(targets + expanded) if target not in (*LORA_TARGET_GROUPS, "all-linear")
        ]
        targets = expanded
    if exclusions:
        conflicts = {
            module
            for module in hf_modules + explicit_targets + exclusions
            if any(matches_lora_target(module, target) for target in explicit_targets)
            and any(matches_lora_target(module, exclude) for exclude in exclusions)
        }
        if conflicts:
            raise ValueError(f"Explicit LoRA targets overlap --exclude-modules: {sorted(conflicts)}")
    targets = resolve_hf_lora_targets(hf_config.to_dict(), target_modules=targets)
    targets = exclude_hf_lora_targets(targets, exclusions)

    if all(any(matches_lora_target(module, target) for module in hf_modules) for target in targets + exclusions):
        hf_targets = targets
        if exclusions:
            selected = [
                module for module in hf_modules if any(matches_lora_target(module, target) for target in targets)
            ]
            hf_targets = exclude_hf_lora_targets(selected, exclusions)
    else:
        layout = get_hf_lora_targets(hf_config.to_dict())
        hf_targets = exclude_hf_lora_targets(expand_hf_lora_targets(targets, layout), exclusions)

    # vime is native-mode only: adapter targets always go through the native resolver.
    from vime.backends.megatron_utils.lora.native import resolve_adapter_targets

    adapter_targets = resolve_adapter_targets(
        hf_config.to_dict(),
        hf_targets,
        hf_modules=hf_modules,
        lora_type=args.lora_type,
        experts_shared_outer_loras=args.experts_shared_outer_loras,
    )
    return hf_targets, adapter_targets

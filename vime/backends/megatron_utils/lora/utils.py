"""LoRA utilities for the Megatron backend (native path; vime port).

Port of miles/backends/megatron_utils/lora/utils.py @ ba28991834 with the
Megatron-Bridge pieces dropped (not ported) and miles' ``args.backend.*`` /
``--megatron-to-hf-mode`` indirections flattened to vime's plain args.
"""

import dataclasses
import json
import logging
import os
import shutil
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vime.backends.megatron_utils.lora.native import load_lora_adapter_hf
from vime.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
from vime.utils.lora.utils import (  # noqa: F401  (re-exported)
    build_lora_config,
    is_lora_enabled,
    lora_rollout_enabled,
)

logger = logging.getLogger(__name__)

_marked_lora_grad_params_cache: dict[int, list] = {}


def reduce_marked_lora_grads(model: Sequence[torch.nn.Module]) -> None:
    """Sum partial grads of replicated LoRA params over their tagged group ("tp"|"ep"), before the DP reduce-scatter."""
    from megatron.core import parallel_state as ps

    key = id(model[0]) if model else 0
    marked = _marked_lora_grad_params_cache.get(key)
    if marked is None:
        marked = []
        for chunk in model:
            for param in chunk.parameters():
                group_name = getattr(
                    param, "_lora_grad_sum_group", None
                )  # config-access-exempt: _lora_grad_sum_group is optional backend-attached tensor metadata
                if group_name is not None and param.requires_grad:
                    marked.append((param, group_name))
        _marked_lora_grad_params_cache[key] = marked
    if not marked:
        return
    groups = {
        "tp": (ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_world_size()),
        "ep": (ps.get_expert_model_parallel_group(), ps.get_expert_model_parallel_world_size()),
    }
    for group_name in ("tp", "ep"):
        group, size = groups[group_name]
        if size <= 1:
            continue
        grads = []
        for param, g_name in marked:
            if g_name != group_name:
                continue
            grad = getattr(
                param, "main_grad", None
            )  # config-access-exempt: main_grad is optional backend-attached tensor metadata
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads.append(grad)
        # set iteration order follows address-derived hashes and need not agree across ranks
        for dt in sorted({g.dtype for g in grads}, key=str):
            gs = [g for g in grads if g.dtype == dt]
            if len(gs) == 1:
                dist.all_reduce(gs[0], op=dist.ReduceOp.SUM, group=group)
                continue
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            for g, red in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs), strict=False):
                g.copy_(red)


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Whether the model was built with LoRA; set at build so every rank agrees, including a pipeline
    stage the target selection left without adapters."""
    return model[0].lora_attached


def adapter_named_parameters(model: Sequence[torch.nn.Module]) -> list[dict[str, torch.nn.Parameter]]:
    """Every adapter parameter of each model chunk, found by the NativeLoRAAdapter modules that own them.

    Virtual pipeline chunks reuse local parameter names, so the parameters stay keyed per chunk.
    """
    adapter_param_ids: set[int] = set()
    for model_chunk in model:
        for module in model_chunk.modules():
            if isinstance(module, NativeLoRAAdapter):
                adapter_param_ids.update(id(param) for param in module.parameters())
    return [
        {name: param for name, param in model_chunk.named_parameters() if id(param) in adapter_param_ids}
        for model_chunk in model
    ]


def _adapter_shard_params(model: Sequence[torch.nn.Module]) -> dict[str, torch.nn.Parameter]:
    """A rank's adapter parameters under their native-shard names."""
    params_by_chunk = adapter_named_parameters(model)
    if len(params_by_chunk) == 1:
        return params_by_chunk[0]
    # virtual pipeline chunks are prefixed model{i}, as Megatron's own checkpoints key them
    return {
        f"model{chunk_index}.{name}": param
        for chunk_index, params in enumerate(params_by_chunk)
        for name, param in params.items()
    }


_param_grad_buffer_patched = False


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer so adapter buffers survive torch_memory_saver.pause(tag="default").

    In colocate mode with offload_train, pause() offloads default-region GPU memory. During
    LoRA training the base weights are frozen, so DDP only creates buffers for adapter params;
    the miles Megatron fork keeps those in the "param_buffer" region via the ddp_config flags
    ``disable_param_buffers_cpu_backup`` / ``disable_grad_buffers_cpu_backup``.

    vime's pinned Megatron does not have those flags (verified against NVIDIA/Megatron-LM
    @4b4acac9 + docker/patch/latest/megatron.patch), so this patch fails fast instead of
    silently running with offload-unsafe buffers.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        # Megatron reads these flags from ddp_config (its first ctor argument).
        ddp_config = kwargs.get("ddp_config", args[0] if args else None)
        field_names = {f.name for f in dataclasses.fields(type(ddp_config))}
        required = {"disable_param_buffers_cpu_backup", "disable_grad_buffers_cpu_backup"}
        missing = required - field_names
        if missing:
            raise RuntimeError(
                "LoRA colocate offload requires Megatron ddp_config fields "
                f"{sorted(missing)}, which vime's pinned Megatron does not provide. "
                "Disable --offload-train for colocated LoRA, or port the fork's "
                "param_and_grad_buffer cpu-backup flags."
            )
        ddp_config.disable_param_buffers_cpu_backup = True
        ddp_config.disable_grad_buffers_cpu_backup = True
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    _param_grad_buffer_patched = True
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (disable cpu backup)")


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def _write_checkpoint_dir(save_dir, write_fn) -> None:
    """Write ``save_dir`` atomically: build in a sibling .tmp dir, then rename."""
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = save_dir.with_name(save_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    write_fn(temporary)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    os.replace(temporary, save_dir)


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Collectively save native adapter shards, training state, and the HF adapter config."""
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    def write_shards(checkpoint_dir: Path):
        adapter_state = {name: param.detach().cpu() for name, param in _adapter_shard_params(model).items()}
        training_state = None
        if optimizer is not None:
            save_optimizer = not args.no_save_optim
            training_state = {
                "iteration": iteration,
                "optimizer": optimizer.state_dict() if save_optimizer else None,
                "parameter_state": (
                    [sub.get_parameter_state_dp_reshardable() for sub in _distributed_optimizers(optimizer)]
                    if save_optimizer
                    else None
                ),
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            }

        if global_rank == 0:
            config = build_lora_config(args, target_modules=args.lora_adapter_targets)
            config.update(
                experts_shared_outer_loras=bool(args.experts_shared_outer_loras),
                format="megatron_rank_sharded",
            )
            (checkpoint_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
        torch.save(adapter_state, checkpoint_dir / f"adapter_megatron_rank{global_rank}.pt")
        if training_state is not None:
            torch.save(training_state, checkpoint_dir / f"training_state_rank{global_rank}.pt")

    _write_checkpoint_dir(save_dir, write_shards)
    return str(save_dir)


LATEST_LORA_ITERATION_FILE = "latest_lora_iteration.txt"


def resolve_lora_adapter_path(args: Namespace) -> str | None:
    """The adapter dir to resume from: explicit --lora-adapter-path wins, else the
    latest adapter checkpoint under --save (tracked in latest_lora_iteration.txt,
    separate from Megatron's tracker so --load == --save still resolves the base)."""
    if getattr(args, "lora_adapter_path", None) is not None:
        return args.lora_adapter_path
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return None
    tracker = Path(save_dir) / LATEST_LORA_ITERATION_FILE
    if not tracker.exists():
        return None
    iteration = int(tracker.read_text().strip())
    adapter_dir = Path(save_dir) / f"iter_{iteration:07d}" / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"{tracker} points at {adapter_dir}, which does not exist; the adapter checkpoint "
            "is incomplete or was removed. Pass --lora-adapter-path explicitly."
        )
    logger.info(f"Auto-resolved LoRA adapter checkpoint: {adapter_dir}")
    return str(adapter_dir)


def restore_lora_adapter(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    *,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    """Restore the resolved adapter checkpoint into the attached adapters; return the saved iteration, if any.

    The directory's format decides what can be restored: per-rank native shards carry the
    optimizer and scheduler state, an HF/PEFT adapter carries weights only. Restoring the
    optimizer is required unless ``--no-load-optim``; a checkpoint that cannot provide it fails.
    """
    adapter_dir = Path(resolve_lora_adapter_path(args)).resolve()
    shard = _native_adapter_shard(adapter_dir)
    if shard is not None:
        _load_native_adapter_shard(model, shard)
        training_state = _load_training_state(adapter_dir)
    else:
        assert (adapter_dir / "adapter_model.safetensors").exists(), (
            f"--lora-adapter-path {adapter_dir} holds neither per-rank native shards "
            "(adapter_megatron_rank*.pt) nor an HF adapter (adapter_model.safetensors)"
        )
        load_lora_adapter_hf(model, str(adapter_dir))
        training_state = None

    if optimizer is None:
        return None
    if args.no_load_optim:
        # the fp32 masters were built before the adapter load; without this the first step() reverts it
        optimizer.reload_model_params()
    else:
        assert (
            training_state is not None and training_state["optimizer"] is not None
        ), f"{adapter_dir} has no optimizer state to resume; pass --no-load-optim to start a fresh optimizer"
        assert "parameter_state" in training_state, (
            f"{adapter_dir} predates saving DistributedOptimizer parameter state (fp32 masters, moments); "
            "pass --no-load-optim to resume its adapter with a fresh optimizer"
        )
        optimizer.load_state_dict(training_state["optimizer"])
        for sub, parameter_state in zip(
            _distributed_optimizers(optimizer), training_state["parameter_state"], strict=True
        ):
            sub.load_parameter_state_from_dp_reshardable(parameter_state)
        logger.info("Restored optimizer state from LoRA checkpoint")
    if training_state is None:
        return None
    if opt_param_scheduler is not None and training_state["opt_param_scheduler"] is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
    return training_state["iteration"]


def _native_adapter_shard(adapter_dir: Path) -> Path | None:
    """This rank's native adapter shard, or None when the directory holds none at all."""
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    shard = adapter_dir / f"adapter_megatron_rank{global_rank}.pt"
    if shard.exists():
        return shard
    assert not any(adapter_dir.glob("adapter_megatron_rank*.pt")), (
        f"{adapter_dir} holds per-rank adapter shards but none for global rank {global_rank}; "
        "it was saved under a different parallel layout."
    )
    from megatron.core import parallel_state as ps

    legacy = adapter_dir / (
        f"adapter_megatron_tp{ps.get_tensor_model_parallel_rank()}_pp{ps.get_pipeline_model_parallel_rank()}.pt"
    )
    if legacy.exists():
        logger.warning(f"Using legacy tp/pp-named adapter shard {legacy}; only valid when EP<=TP")
        return legacy
    return None


def _load_native_adapter_shard(model: Sequence[torch.nn.Module], shard: Path) -> None:
    state_dict = torch.load(shard, map_location="cpu", weights_only=True)
    adapter_params = _adapter_shard_params(model)
    missing = adapter_params.keys() - state_dict.keys()
    unexpected = state_dict.keys() - adapter_params.keys()
    if missing or unexpected:
        raise RuntimeError(
            f"Adapter checkpoint {shard} does not match the model's adapter parameters: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, param in adapter_params.items():
        param.data.copy_(state_dict[name].to(device=param.device))
    logger.info(f"Loaded {len(adapter_params)} adapter tensors from Megatron-native checkpoint: {shard}")


def _load_training_state(adapter_dir: Path) -> dict | None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None
    # Optimizer state dicts hold non-tensor objects (step counts, param group metadata).
    return torch.load(state_path, map_location="cpu", weights_only=False)


def _distributed_optimizers(optimizer: Any) -> list:
    """The DistributedOptimizers under ``optimizer``, whose ``state_dict()`` leaves out the per-parameter state."""
    from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer

    subs = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    return [sub for sub in subs if isinstance(sub, DistributedOptimizer)]

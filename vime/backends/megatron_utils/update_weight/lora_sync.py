"""Disk-mode LoRA adapter sync (F04-7).

Every weight-sync step, the trainer exports the native adapter in HF/PEFT form
(``export_lora_hf_named``, collective over TP), rank 0 writes it atomically to a
directory, and each rollout engine hot-loads it through vLLM's
``/v1/load_lora_adapter`` with ``load_inplace=True`` — no vLLM patch required
(gated server-side by ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``, set in F04-6).

Topology (enforced at arg validation, F04-2): with ``--colocate``, the default
``<save>/rollout_adapter`` works because trainer and engines share the node's
filesystem; without ``--colocate``, ``--lora-rollout-cache-dir`` is required and
must point at a filesystem readable by every engine node. LoRA + ``--offload-train``
(colocate CPU-backup of adapter buffers) needs the miles Megatron fork's
``disable_*_cpu_backup`` ddp_config flags — a named follow-up.

``--check-lora-weight-equal`` writes a per-tensor sha256 manifest and verifies the
on-disk artifact matches what was exported (vLLM has no engine-side weight-hash
hook; a tampered artifact fails before engines load it).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from vime.backends.megatron_utils.lora.native import export_lora_hf_named
from vime.utils.lora.utils import LORA_ADAPTER_NAME, build_lora_config, lora_rollout_enabled

logger = logging.getLogger(__name__)

_ADAPTER_DIR_NAME = "rollout_adapter"


def lora_rollout_adapter_dir(args: Namespace) -> Path:
    """Directory the engines load the adapter from (validated at arg time; this is the runtime belt)."""
    override = getattr(args, "lora_rollout_cache_dir", None)
    if override:
        return Path(override)
    if not getattr(args, "save", None):
        raise ValueError(
            "LoRA rollout sync needs an adapter directory readable by every engine: set --save "
            "(colocate) or --lora-rollout-cache-dir."
        )
    return Path(args.save) / _ADAPTER_DIR_NAME


def adapter_checksums(tensors: dict) -> dict[str, str]:
    # fp32 upcast keeps the hash dtype-agnostic across the bf16 safetensors round-trip.
    return {
        name: hashlib.sha256(t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()
        for name, t in sorted(tensors.items())
    }


def _write_adapter_dir_atomic(target: Path, config: dict, tensors) -> None:
    from vime.utils.lora.utils import save_adapter_to_disk

    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_adapter_to_disk(temporary, config, tensors)
    if target.exists():
        shutil.rmtree(target)
    os.replace(temporary, target)


def _verify_written_adapter(target: Path, expected: dict[str, str]) -> None:
    import safetensors.torch

    written = safetensors.torch.load_file(str(target / "adapter_model.safetensors"))
    actual = adapter_checksums(written)
    if actual != expected:
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        mismatched = [k for k in expected.keys() & actual.keys() if expected[k] != actual[k]]
        raise RuntimeError(
            f"LoRA adapter sync integrity check failed for {target}: "
            f"missing={sorted(missing)} extra={sorted(extra)} mismatched={sorted(mismatched)}"
        )


def push_lora_adapter(args: Namespace, model, rollout_engines) -> None:
    """Export the adapter, publish it to disk, and hot-load it on every engine.

    Called from every trainer rank near the end of ``update_weights`` (engines are
    paused); the export is collective, the disk write and engine fan-out are rank 0.
    """
    if not lora_rollout_enabled(args):
        return

    tensors = dict(export_lora_hf_named(model))
    if not tensors:
        raise RuntimeError("LoRA rollout sync exported zero adapter tensors; the model has no adapters attached")

    rank = dist.get_rank() if dist.is_initialized() else 0
    target = lora_rollout_adapter_dir(args)
    if rank == 0:
        config = build_lora_config(args, target_modules=args.lora_adapter_targets)
        expected = adapter_checksums(tensors) if getattr(args, "check_lora_weight_equal", False) else None
        _write_adapter_dir_atomic(target, config, tensors)
        if expected is not None:
            (target / "adapter_sha256.json").write_text(json.dumps(expected, indent=2))
            _verify_written_adapter(target, expected)
        logger.info("[lora-sync] wrote %d adapter tensors to %s", len(tensors), target)
    if dist.is_initialized():
        from vime.utils.distributed_utils import get_gloo_group

        dist.barrier(group=get_gloo_group())
    if rank != 0:
        return

    import ray

    ray.get([engine.load_lora_adapter.remote(LORA_ADAPTER_NAME, str(target)) for engine in rollout_engines])
    logger.info("[lora-sync] engines hot-loaded adapter %s from %s", LORA_ADAPTER_NAME, target)

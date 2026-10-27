"""Opt-in HF->Megatron weight loading through the external mbridge package.

Active only with ``--hf-to-megatron-load-mode mbridge``. Importing
``vime_plugins.mbridge`` registers the day-0 model bridges into mbridge's
global registry, then ``mbridge.AutoBridge`` dispatches on the checkpoint's
``config.model_type``.
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import nullcontext
from pathlib import Path

from transformers import AutoConfig

logger = logging.getLogger(__name__)

MBRIDGE_PIN = "89eb10887887bc74853f89a4de258c0702932a1c"


def mbridge_available() -> bool:
    return importlib.util.find_spec("mbridge") is not None


def _model_registry() -> dict:
    """mbridge's _MODEL_REGISTRY is private; pin must match. Raise a clear error on drift."""
    from mbridge.core import bridge as mbridge_core_bridge

    registry = getattr(mbridge_core_bridge, "_MODEL_REGISTRY", None)
    if registry is None:
        raise RuntimeError(
            "mbridge.core.bridge._MODEL_REGISTRY is missing — the installed mbridge does not match "
            f"the pinned commit {MBRIDGE_PIN}. Reinstall: pip install --no-deps "
            f"git+https://github.com/ISEEKYAN/mbridge.git@{MBRIDGE_PIN}"
        )
    return registry


def _registered_model_types() -> set[str]:
    import vime_plugins.mbridge  # noqa: F401  (registers vime bridge subclasses)

    return set(_model_registry())


def supports_mbridge_loading(path: str | Path) -> bool:
    if not mbridge_available():
        return False
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config.model_type in _registered_model_types()


def load_hf_weights_with_mbridge(args, model, path: str | Path) -> None:
    if not mbridge_available():
        raise ModuleNotFoundError(
            "--hf-to-megatron-load-mode mbridge requires the mbridge package. "
            f"Install it pinned: pip install --no-deps git+https://github.com/ISEEKYAN/mbridge.git@{MBRIDGE_PIN}",
            name="mbridge",
        )

    import torch

    import vime_plugins.mbridge  # noqa: F401  (registers vime bridge subclasses)
    from mbridge import AutoBridge

    # miles wraps the load in torch_memory_saver.disable() when offload_train is
    # on, so staging copies are not managed by the memory saver.
    bypass = nullcontext()
    if getattr(args, "offload_train", False) and importlib.util.find_spec("torch_memory_saver") is not None:
        from torch_memory_saver import torch_memory_saver

        bypass = torch_memory_saver.disable()

    logger.info(f"Load checkpoint from HuggingFace model into Megatron via mbridge (path={path})")
    with bypass:
        bridge = AutoBridge.from_pretrained(str(path), trust_remote_code=True)
        bridge.load_weights(model, str(path), memory_efficient=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

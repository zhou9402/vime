"""vime-native LoRA: adapters attached directly to the mcore model.

Megatron is imported inside functions so argument validation can import this package.
"""

from vime.backends.megatron_utils.lora.native.lora import (
    apply_native_lora,
    export_lora_hf_named,
    load_lora_adapter_hf,
    wrap_model_provider_with_lora,
)
from vime.backends.megatron_utils.lora.native.registry import (
    preflight_native_lora,
    resolve_adapter_targets,
    validate_native_lora_args,
)

__all__ = [
    "apply_native_lora",
    "export_lora_hf_named",
    "load_lora_adapter_hf",
    "preflight_native_lora",
    "resolve_adapter_targets",
    "validate_native_lora_args",
    "wrap_model_provider_with_lora",
]

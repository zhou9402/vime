"""Shared harness for HF<->Megatron converter verification (F07 day-0 models).

Every converter PR for a day-0 model family plugs into these helpers with a
tiny 2-layer JSON config from ``tiny_configs/`` and asserts:

1. ``assert_roundtrip_bitwise`` — Megatron -> HF -> Megatron is bitwise
   identical for every parameter (loader(exporter(w)) == w).
2. ``assert_hf_logits_preserved`` — a real (tiny, randomly initialized)
   HuggingFace model's weights survive HF -> Megatron -> HF bitwise, and the
   reconstructed model produces allclose logits (explicit tolerance below).

Logits tolerance: fp32 CPU forward, weights round-trip bitwise so logits
should match to numerical noise; rtol=1e-5 / atol=1e-5.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch

_TINY_CONFIG_DIR = Path(__file__).resolve().parent / "tiny_configs"

LOGITS_RTOL = 1e-5
LOGITS_ATOL = 1e-5


class DictReader:
    """SafetensorReader-compatible view over an in-memory tensor dict."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def load_tiny_config(name: str) -> types.SimpleNamespace:
    with (_TINY_CONFIG_DIR / f"{name}.json").open() as f:
        return json.load(f, object_hook=lambda d: types.SimpleNamespace(**d))


def export_args(config) -> types.SimpleNamespace:
    text = getattr(config, "text_config", config)
    return types.SimpleNamespace(
        kv_channels=getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads,
        hidden_size=text.hidden_size,
        num_attention_heads=text.num_attention_heads,
        num_query_groups=getattr(text, "num_key_value_heads", text.num_attention_heads),
        num_layers=text.num_hidden_layers,
        q_lora_rank=getattr(text, "q_lora_rank", None),
        vocab_size=text.vocab_size,
    )


def dense_megatron_param_names(config) -> list[str]:
    """Canonical Megatron parameter names for a dense LLaMA-style model."""
    text = getattr(config, "text_config", config)
    names = [
        "module.module.embedding.word_embeddings.weight",
        "module.module.output_layer.weight",
        "module.module.decoder.final_layernorm.weight",
    ]
    for layer in range(text.num_hidden_layers):
        prefix = f"module.module.decoder.layers.{layer}."
        names += [
            prefix + "self_attention.linear_qkv.weight",
            prefix + "self_attention.linear_proj.weight",
            prefix + "self_attention.linear_qkv.layer_norm_weight",
            prefix + "self_attention.q_layernorm.weight",
            prefix + "self_attention.k_layernorm.weight",
            prefix + "mlp.linear_fc1.weight",
            prefix + "mlp.linear_fc1.layer_norm_weight",
            prefix + "mlp.linear_fc2.weight",
        ]
    return names


def assert_roundtrip_bitwise(loader, exporter, args, config, params: dict[str, torch.Tensor]) -> None:
    """For every Megatron param: export to HF names, reload, require bitwise equality."""
    hf_tensors: dict[str, torch.Tensor] = {}
    for name, param in params.items():
        for hf_name, tensor in exporter(args, name, param):
            assert hf_name not in hf_tensors, f"duplicate HF tensor {hf_name}"
            hf_tensors[hf_name] = tensor
    reader = DictReader(hf_tensors)
    for name, param in params.items():
        loaded = loader(name, reader, config)
        assert loaded.shape == param.shape, f"{name}: shape {tuple(loaded.shape)} != {tuple(param.shape)}"
        assert torch.equal(loaded, param), f"{name}: Megatron->HF->Megatron round trip is not bitwise"


def hf_config_from_tiny(config):
    from transformers import AutoConfig

    kwargs = {k: v for k, v in vars(config).items() if k != "model_type"}
    return AutoConfig.for_model(config.model_type, **kwargs)


def assert_hf_logits_preserved(
    config,
    loader,
    exporter,
    args,
    megatron_param_names: list[str],
    *,
    seed: int = 0,
) -> None:
    """HF -> Megatron -> HF on a real tiny model must preserve logits.

    Builds a randomly initialized HF model (fixed seed), maps every weight
    into Megatron space via ``loader`` and back via ``exporter``, reloads the
    round-tripped weights into a fresh model instance, and compares logits.
    """
    from transformers import AutoModelForCausalLM

    hf_config = hf_config_from_tiny(config)

    torch.manual_seed(seed)
    reference = AutoModelForCausalLM.from_config(hf_config)
    reference.eval()

    hf_state = reference.state_dict()
    reader = DictReader(hf_state)
    round_tripped: dict[str, torch.Tensor] = {}
    for name in megatron_param_names:
        param = loader(name, reader, config)
        for hf_name, tensor in exporter(args, name, param):
            assert hf_name not in round_tripped, f"duplicate HF tensor {hf_name}"
            round_tripped[hf_name] = tensor

    for hf_name, tensor in round_tripped.items():
        assert torch.equal(tensor, hf_state[hf_name]), f"{hf_name}: HF->Megatron->HF is not bitwise"

    reconstructed = AutoModelForCausalLM.from_config(hf_config)
    reconstructed.load_state_dict({**hf_state, **round_tripped})
    reconstructed.eval()

    input_ids = torch.arange(16).view(2, 8) % config.vocab_size
    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = reconstructed(input_ids).logits
    assert torch.allclose(actual, expected, rtol=LOGITS_RTOL, atol=LOGITS_ATOL)

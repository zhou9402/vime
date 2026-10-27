"""LoRA argument validation (vime port of miles tests/fast/utils/lora/test_arguments.py)."""

import argparse
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
from transformers import LlamaConfig

_tests_root = Path(__file__).resolve().parents[2]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs  # noqa: E402

_unit_stubs.install_vllm_cli_stubs()

from vime.utils.lora.arguments import _resolve_lora_targets, validate_lora_args  # noqa: E402


def _stub_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """vime.backends.megatron_utils.lora.native lands in F04-3; stub the lazy import."""
    native = types.ModuleType("vime.backends.megatron_utils.lora.native")
    native.resolve_adapter_targets = lambda config, hf_targets, **kwargs: list(hf_targets)
    native.validate_native_lora_args = lambda args: None
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.lora.native", native)


class TestLoraTargetOrdering:
    def test_exclusions_expand_to_stable_hf_module_order_without_changing_selectors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolved adapter targets have reproducible order after exclusions expand module names."""
        _stub_native(monkeypatch)
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        args = Namespace(
            target_modules="all-linear",
            exclude_modules="q_proj",
            lora_type="lora",
            experts_shared_outer_loras=False,
        )

        hf_targets, adapter_targets = _resolve_lora_targets(args, config)

        expected = [
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.1.mlp.down_proj",
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.1.self_attn.v_proj",
        ]
        assert hf_targets == expected
        assert adapter_targets == expected
        assert args.target_modules == "all-linear"
        assert args.exclude_modules == "q_proj"


def _write_hf_checkpoint(tmp_path, config_dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    return str(tmp_path)


def _llama_checkpoint_dict():
    return LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    ).to_dict()


class TestValidateLoraArgs:
    def _args(self, tmp_path, **overrides):
        args = dict(
            lora_rank=0,
            lora_adapter_path=None,
            target_modules=None,
            exclude_modules=None,
            lora_type="lora",
            experts_shared_outer_loras=False,
            train_backend="megatron",
            colocate=True,
            save=str(tmp_path / "ckpt"),
            lora_rollout_cache_dir=None,
            lora_train_only=False,
            hf_checkpoint=_write_hf_checkpoint(tmp_path, _llama_checkpoint_dict()),
        )
        return Namespace(**(args | overrides))

    def test_disabled_is_noop_and_sets_defaults(self, tmp_path):
        args = self._args(tmp_path)
        validate_lora_args(args)
        assert args.multi_lora is False
        assert args.lora_B_init_method == "zero"
        assert args.hf_lora_targets is None

    def test_requires_megatron_backend(self, tmp_path):
        args = self._args(tmp_path, lora_rank=8, train_backend="fsdp")
        with pytest.raises(ValueError, match="megatron"):
            validate_lora_args(args)

    def test_requires_positive_rank(self, tmp_path):
        args = self._args(tmp_path, lora_adapter_path="/tmp/adapter")
        with pytest.raises(ValueError, match="--lora-rank"):
            validate_lora_args(args)

    def test_resolves_targets_on_dense_model(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _stub_native(monkeypatch)
        args = self._args(tmp_path, lora_rank=8)
        validate_lora_args(args)
        assert "model.layers.*.self_attn.q_proj" in args.hf_lora_targets
        assert args.lora_adapter_targets == args.hf_lora_targets

    def test_offload_train_is_rejected(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _stub_native(monkeypatch)
        args = self._args(tmp_path, lora_rank=8, offload_train=True)
        with pytest.raises(ValueError, match="disable_param_buffers_cpu_backup"):
            validate_lora_args(args)

    def test_noncolocate_rollout_requires_shared_cache_dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _stub_native(monkeypatch)
        args = self._args(tmp_path, lora_rank=8, colocate=False, save=str(tmp_path / "ckpt"))
        with pytest.raises(ValueError, match="--lora-rollout-cache-dir"):
            validate_lora_args(args)
        # shared mount satisfies it
        args = self._args(tmp_path, lora_rank=8, colocate=False, lora_rollout_cache_dir="/shared/lora")
        validate_lora_args(args)

    def test_colocate_rollout_requires_save_or_cache_dir(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _stub_native(monkeypatch)
        args = self._args(tmp_path, lora_rank=8, colocate=True, save=None)
        with pytest.raises(ValueError, match="--lora-rollout-cache-dir"):
            validate_lora_args(args)
        args = self._args(tmp_path, lora_rank=8, colocate=True, save=None, lora_rollout_cache_dir="/node/local")
        validate_lora_args(args)

    def test_multi_lora_flag_is_rejected(self, tmp_path):
        args = self._args(tmp_path, multi_lora_n_adapters=4)
        with pytest.raises(ValueError, match="F04-F1"):
            validate_lora_args(args)

    def test_expert_targets_are_rejected(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _stub_native(monkeypatch)
        moe_dict = _llama_checkpoint_dict()
        # qwen3_moe layout: packed expert projections must be gated off in phase 1.
        moe_dict.update(model_type="qwen3_moe", num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16)
        args = self._args(tmp_path, lora_rank=8, target_modules="all-linear")
        (tmp_path / "moe").mkdir(exist_ok=True)
        args.hf_checkpoint = _write_hf_checkpoint(tmp_path / "moe", moe_dict)
        with pytest.raises(ValueError, match="MoE-expert LoRA is not supported"):
            validate_lora_args(args)


class TestCliFlags:
    def test_lora_flags_parse_with_defaults(self):
        try:
            from vime.utils.arguments import get_vime_extra_args_provider
        except ModuleNotFoundError as e:
            pytest.skip(f"vime.arguments provider unavailable in this environment ({e}); covered by CI stubs")

        parser = argparse.ArgumentParser()
        provider = get_vime_extra_args_provider(None)
        parser = provider(parser)
        args, _ = parser.parse_known_args(["--rollout-batch-size", "1"])
        assert args.lora_rank == 0
        assert args.lora_alpha == 16
        assert args.lora_dropout == 0.0
        assert args.lora_a_init_method == "xavier"
        assert args.lora_type == "lora"
        assert args.target_modules is None
        assert args.lora_adapter_path is None
        assert args.lora_train_only is False
        assert args.lora_base_cpu_backup is False
        assert args.check_lora_weight_equal is False

        args, _ = parser.parse_known_args(
            ["--rollout-batch-size", "1", "--lora-rank", "8", "--target-modules", "attn,mlp", "--lora-train-only"]
        )
        assert args.lora_rank == 8
        assert args.target_modules == "attn,mlp"
        assert args.lora_train_only is True

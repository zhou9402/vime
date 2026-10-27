"""Tests for vime.backends.megatron_utils.lora.utils — CPU-only, megatron stubbed."""

import dataclasses
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vime.backends.megatron_utils.lora.native.config import LoRAConfig
from vime.backends.megatron_utils.lora.native.modules.linear import LoRALinear
from vime.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout
from vime.backends.megatron_utils.lora.utils import (
    _adapter_shard_params,
    adapter_named_parameters,
    is_lora_model,
    patch_param_grad_buffer_for_colocate_mode_lora,
    reduce_marked_lora_grads,
    restore_lora_adapter,
    save_lora_checkpoint,
)
import vime.backends.megatron_utils.lora.utils as lora_utils

NUM_GPUS = 0


def _model_with_adapter(rank=2):
    """A model chunk with one LoRALinear adapter attached to a frozen linear."""
    context = AttachContext(
        lora=LoRAConfig(rank=rank, alpha=4, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    model = nn.Module()
    model.block = nn.Module()
    model.block.linear = nn.Linear(6, 4, bias=False)
    model.block.linear.weight.requires_grad_(False)
    adapter = LoRALinear(
        hf_prefix="model.layers.0.mlp.",
        projection=ProjectionSpec("up_proj", "up", ShardLayout.COLUMN),
        reference=model.block.linear.weight,
        context=context,
        in_features=6,
        out_features=4,
    )
    model.block.lora_up_adapter = adapter
    return model


class TestAdapterDiscovery:
    def test_is_lora_model_reads_the_build_flag(self):
        model = _model_with_adapter()
        model.lora_attached = True
        assert is_lora_model([model])
        model.lora_attached = False
        assert not is_lora_model([model])

    def test_adapter_named_parameters_finds_only_adapter_params(self):
        model = _model_with_adapter()
        (params,) = adapter_named_parameters([model])
        assert set(params) == {"block.lora_up_adapter.up_A", "block.lora_up_adapter.up_B"}
        assert all(p.requires_grad for p in params.values())

    def test_adapter_shard_params_prefixes_virtual_pipeline_chunks(self):
        chunks = [_model_with_adapter(), _model_with_adapter()]
        assert set(_adapter_shard_params(chunks)) == {
            "model0.block.lora_up_adapter.up_A",
            "model0.block.lora_up_adapter.up_B",
            "model1.block.lora_up_adapter.up_A",
            "model1.block.lora_up_adapter.up_B",
        }


class TestReduceMarkedGrads:
    def _stub_parallel_state(self, monkeypatch):
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        ps = types.ModuleType("megatron.core.parallel_state")
        ps.get_tensor_model_parallel_group = lambda: None
        ps.get_tensor_model_parallel_world_size = lambda: 1
        ps.get_expert_model_parallel_group = lambda: None
        ps.get_expert_model_parallel_world_size = lambda: 1
        core.parallel_state = ps
        megatron.core = core
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", ps)

    def test_no_marked_params_is_noop(self, monkeypatch):
        self._stub_parallel_state(monkeypatch)
        reduce_marked_lora_grads([_model_with_adapter()])
        reduce_marked_lora_grads([])  # empty model list is fine

    def test_single_process_groups_skip_collectives(self, monkeypatch):
        self._stub_parallel_state(monkeypatch)

        model = _model_with_adapter()
        param = model.block.lora_up_adapter.up_A
        param._lora_grad_sum_group = "tp"
        param.grad = torch.ones_like(param)
        reduce_marked_lora_grads([model])
        assert torch.equal(param.grad, torch.ones_like(param))


class TestParamGradBufferPatch:
    @pytest.fixture(autouse=True)
    def _reset_patch(self, monkeypatch):
        monkeypatch.setattr(lora_utils, "_param_grad_buffer_patched", False)

    def _install_fake_megatron_buffer(self, monkeypatch, ddp_fields):
        DDPConfig = dataclasses.make_dataclass(
            "DDPConfig", [(field, bool, dataclasses.field(default=False)) for field in ddp_fields]
        )

        class FakeBuffer:
            def __init__(self, ddp_config):
                self.ddp_config = ddp_config

        mod = types.ModuleType("megatron.core.distributed.param_and_grad_buffer")
        mod._ParamAndGradBuffer = FakeBuffer
        monkeypatch.setitem(sys.modules, "megatron.core.distributed.param_and_grad_buffer", mod)
        return FakeBuffer, DDPConfig

    def test_fails_fast_without_fork_flags(self, monkeypatch):
        FakeBuffer, DDPConfig = self._install_fake_megatron_buffer(monkeypatch, ddp_fields=[])
        patch_param_grad_buffer_for_colocate_mode_lora()
        with pytest.raises(RuntimeError, match="disable_param_buffers_cpu_backup"):
            FakeBuffer(DDPConfig())

    def test_sets_flags_when_supported(self, monkeypatch):
        FakeBuffer, DDPConfig = self._install_fake_megatron_buffer(
            monkeypatch, ddp_fields=["disable_param_buffers_cpu_backup", "disable_grad_buffers_cpu_backup"]
        )
        patch_param_grad_buffer_for_colocate_mode_lora()
        patch_param_grad_buffer_for_colocate_mode_lora()  # idempotent
        buffer = FakeBuffer(DDPConfig())
        assert buffer.ddp_config.disable_param_buffers_cpu_backup
        assert buffer.ddp_config.disable_grad_buffers_cpu_backup


class TestCheckpointRoundTrip:
    def _args(self, tmp_path, **overrides):
        values = dict(
            lora_rank=2,
            lora_alpha=4,
            lora_dropout=0.0,
            lora_adapter_targets=["model.layers.*.mlp.up_proj"],
            experts_shared_outer_loras=False,
            no_save_optim=False,
            no_load_optim=False,
            lora_adapter_path=None,
            save=None,
        )
        return SimpleNamespace(**(values | overrides))

    def test_resolve_adapter_path_explicit_wins_and_missing_is_none(self, tmp_path):
        args = self._args(tmp_path, lora_adapter_path="/tmp/explicit", save=str(tmp_path))
        assert lora_utils.resolve_lora_adapter_path(args) == "/tmp/explicit"
        assert lora_utils.resolve_lora_adapter_path(self._args(tmp_path, save=str(tmp_path))) is None

    def test_resume_both_forms(self, tmp_path):
        model = _model_with_adapter()
        with torch.no_grad():
            for param in model.block.lora_up_adapter.parameters():
                param.normal_(0, 0.5)
        args = self._args(tmp_path)
        adapter_dir = tmp_path / "ckpt" / "iter_0000007" / "adapter"
        save_lora_checkpoint([model], args, str(adapter_dir))
        (tmp_path / "ckpt" / "latest_lora_iteration.txt").write_text("7\n")

        # form 1: explicit --lora-adapter-path
        fresh = _model_with_adapter()
        restore_lora_adapter(
            [fresh], self._args(tmp_path, lora_adapter_path=str(adapter_dir)), optimizer=None, opt_param_scheduler=None
        )
        # form 2: auto-resolve from --save
        fresh_auto = _model_with_adapter()
        restore_lora_adapter(
            [fresh_auto], self._args(tmp_path, save=str(tmp_path / "ckpt")), optimizer=None, opt_param_scheduler=None
        )
        for name, saved in adapter_named_parameters([model])[0].items():
            assert torch.equal(adapter_named_parameters([fresh])[0][name], saved)
            assert torch.equal(adapter_named_parameters([fresh_auto])[0][name], saved)

    def test_auto_resolve_fails_loudly_on_missing_dir(self, tmp_path):
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / "latest_lora_iteration.txt").write_text("7\n")
        with pytest.raises(FileNotFoundError, match="iter_0000007"):
            lora_utils.resolve_lora_adapter_path(self._args(tmp_path, save=str(tmp_path / "ckpt")))

    def test_save_writes_shards_and_config(self, tmp_path):
        model = _model_with_adapter()
        args = self._args(tmp_path)
        save_lora_checkpoint([model], args, str(tmp_path / "adapter"), optimizer=None, iteration=7)
        shard = tmp_path / "adapter" / "adapter_megatron_rank0.pt"
        assert shard.exists()
        state = torch.load(shard, map_location="cpu", weights_only=True)
        assert set(state) == {"block.lora_up_adapter.up_A", "block.lora_up_adapter.up_B"}
        config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
        assert config["r"] == 2 and config["format"] == "megatron_rank_sharded"
        assert not (tmp_path / "adapter" / "training_state_rank0.pt").exists()  # no optimizer given
        assert not (tmp_path / "adapter.tmp").exists()  # atomic rename completed

    def test_save_then_restore_bitwise(self, tmp_path):
        model = _model_with_adapter()
        with torch.no_grad():
            for param in model.block.lora_up_adapter.parameters():
                param.normal_(0, 0.5)
        args = self._args(tmp_path)
        save_lora_checkpoint([model], args, str(tmp_path / "adapter"))

        fresh = _model_with_adapter()  # B is zero-init: differs from saved
        restore_args = self._args(tmp_path, lora_adapter_path=str(tmp_path / "adapter"))
        assert restore_lora_adapter([fresh], restore_args, optimizer=None, opt_param_scheduler=None) is None
        for (name, saved), (_, loaded) in zip(
            adapter_named_parameters([model])[0].items(), adapter_named_parameters([fresh])[0].items()
        ):
            assert name == _
            assert torch.equal(saved, loaded)

    def test_restore_rejects_layout_mismatch(self, tmp_path):
        model = _model_with_adapter()
        args = self._args(tmp_path)
        save_lora_checkpoint([model], args, str(tmp_path / "adapter"))
        wider = _model_with_adapter(rank=4)
        with pytest.raises(RuntimeError, match=r"does not match|must match"):
            restore_lora_adapter(
                [wider], self._args(tmp_path, lora_adapter_path=str(tmp_path / "adapter")), optimizer=None,
                opt_param_scheduler=None,
            )

    def test_restore_rejects_unknown_layout(self, tmp_path, monkeypatch):
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        ps = types.ModuleType("megatron.core.parallel_state")
        ps.get_tensor_model_parallel_rank = lambda: 0
        ps.get_pipeline_model_parallel_rank = lambda: 0
        core.parallel_state = ps
        megatron.core = core
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", ps)
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(AssertionError, match="neither per-rank native shards"):
            restore_lora_adapter(
                [_model_with_adapter()], self._args(tmp_path, lora_adapter_path=str(empty)), optimizer=None,
                opt_param_scheduler=None,
            )

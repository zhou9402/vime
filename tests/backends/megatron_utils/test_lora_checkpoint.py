"""LoRA branch routing in vime.backends.megatron_utils.checkpoint — CPU-only, megatron stubbed.

Port-equivalent of miles tests/fast/backends/megatron_utils/test_lora_checkpoint_helpers.py:
a LoRA model saves adapter-only under <save>/iter_NNNNNNN/adapter/ (plus the tracker file);
a non-LoRA model goes through the plain Megatron save.
"""

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

NUM_GPUS = 0


@pytest.fixture()
def checkpoint_mod(monkeypatch):
    """Import vime.backends.megatron_utils.checkpoint with megatron.training stubbed."""
    megatron = types.ModuleType("megatron")
    training = types.ModuleType("megatron.training")
    ckpt = types.ModuleType("megatron.training.checkpointing")
    global_vars = types.ModuleType("megatron.training.global_vars")

    state = {}

    def fake_save(iteration, model, optimizer, opt_param_scheduler, **kwargs):
        state["megatron_save_called"] = True

    ckpt.load_checkpoint = lambda *a, **k: None
    ckpt.save_checkpoint = fake_save
    global_vars.get_args = lambda: state["args"]
    training.checkpointing = ckpt
    training.global_vars = global_vars
    megatron.training = training
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", ckpt)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars)

    import importlib

    import vime.backends.megatron_utils.checkpoint as mod

    mod = importlib.reload(mod)  # re-bind _save_checkpoint_megatron to this test's stub
    monkeypatch.setattr(mod, "get_args", lambda: state["args"])
    return mod, state


def _args(tmp_path, **overrides):
    values = dict(
        save=str(tmp_path / "ckpt"),
        async_save=False,
        offload_train=False,
        use_stateless_adam=False,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_adapter_targets=["model.layers.*.mlp.up_proj"],
        experts_shared_outer_loras=False,
        no_save_optim=False,
        no_load_optim=False,
    )
    return SimpleNamespace(**(values | overrides))


def _toy_model(lora_attached, tmp_path):
    model = nn.Module()
    model.lora_attached = lora_attached
    return [model]


class TestLoraCheckpointRouting:
    def test_lora_model_saves_adapter_only(self, checkpoint_mod, tmp_path, monkeypatch):
        mod, state = checkpoint_mod
        state["args"] = _args(tmp_path)

        calls = {}
        monkeypatch.setattr(
            mod.dist, "get_rank", lambda: 0, raising=False
        ) if hasattr(mod, "dist") else monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        import vime.backends.megatron_utils.lora.utils as lora_utils

        def fake_save_lora(model, args, save_dir, **kwargs):
            calls["save_dir"] = save_dir
            calls["kwargs"] = kwargs

        monkeypatch.setattr(lora_utils, "save_lora_checkpoint", fake_save_lora)

        mod.save_checkpoint(7, _toy_model(True, tmp_path), optimizer=None, opt_param_scheduler=None)

        assert calls["save_dir"].endswith("iter_0000007/adapter")
        assert "megatron_save_called" not in state
        # LoRA progress is tracked separately; Megatron's tracker is untouched.
        assert (tmp_path / "ckpt" / "latest_lora_iteration.txt").read_text().strip() == "7"
        assert not (tmp_path / "ckpt" / "latest_lora_iteration.tmp").exists()
        assert not (tmp_path / "ckpt" / "latest_checkpointed_iteration.txt").exists()

    def test_non_lora_model_uses_megatron_save(self, checkpoint_mod, tmp_path):
        mod, state = checkpoint_mod
        state["args"] = _args(tmp_path)
        mod.save_checkpoint(3, _toy_model(False, tmp_path), optimizer=None, opt_param_scheduler=None)
        assert state["megatron_save_called"]
        assert not (tmp_path / "ckpt" / "iter_0000003" / "adapter").exists()

    def test_lora_branch_accepts_and_ignores_megatron_save_kwargs(self, checkpoint_mod, tmp_path, monkeypatch, caplog):
        """Parity: the kwargs vime's model.save() forwards must not break or be silently dropped."""
        mod, state = checkpoint_mod
        state["args"] = _args(tmp_path)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        import vime.backends.megatron_utils.lora.utils as lora_utils

        calls = {}
        monkeypatch.setattr(
            lora_utils, "save_lora_checkpoint", lambda model, args, save_dir, **kw: calls.update(kw)
        )

        mod.save_checkpoint(
            5,
            _toy_model(True, tmp_path),
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
        )
        # adapter save receives exactly the training state, nothing Megatron-specific
        assert set(calls) == {"optimizer", "opt_param_scheduler", "iteration"}
        assert calls["iteration"] == 5
        assert (tmp_path / "ckpt" / "latest_lora_iteration.txt").read_text().strip() == "5"

    def test_is_lora_model_tolerates_plain_models(self, checkpoint_mod):
        mod, _ = checkpoint_mod
        assert not mod._is_lora_model([nn.Module()])
        assert mod._is_lora_model(_toy_model(True, None))

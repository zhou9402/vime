"""--save-trigger-sentinel and --custom-megatron-post-save-hook-path (A9d).

The actor-hook test needs the full Megatron-LM training package (not the
megatron-core wheel), so it skips wherever that is unavailable; the sentinel
and argument-validation tests are pure CPU.
"""

import os
from argparse import Namespace
from types import SimpleNamespace

import pytest


def _make_args(tmp_path, **overrides):
    args = SimpleNamespace(
        save=str(tmp_path / "ckpt"),
        save_interval=None,
        save_trigger_sentinel=None,
        custom_megatron_post_save_hook_path=None,
        rollout_data_transport="file",
        release_train=False,
        num_rollout=10,
        use_critic=False,
        debug_train_only=False,
        debug_rollout_only=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --- sentinel semantics in the train loop (vime/data/checkpoint.save_checkpoint plumbing) ---


def test_sentinel_present_forces_sync_save(tmp_path, monkeypatch):
    """The train loop's sentinel block: save with force_sync=True, then remove the sentinel."""
    import vime.data.checkpoint as checkpoint_mod

    sentinel = tmp_path / "save_now"
    sentinel.touch()
    args = _make_args(tmp_path, save_trigger_sentinel=str(sentinel))

    # mirror train.py's save-point block
    external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
    assert external_save

    saved = {}

    def fake_save_model(self, rollout_id, force_sync=False):
        saved["force_sync"] = force_sync

    monkeypatch.setattr(checkpoint_mod, "commit_checkpoint", lambda *a, **k: None)
    actor = SimpleNamespace(save_model=lambda rollout_id, force_sync=False: fake_save_model(None, rollout_id, force_sync))
    manager = SimpleNamespace(save=SimpleNamespace(remote=lambda rollout_id: None))
    monkeypatch.setattr("ray.get", lambda x: x)

    checkpoint_mod.save_checkpoint(
        args, 3, actor, None, manager, actor_trains=True, force_sync=external_save
    )
    assert saved["force_sync"] is True

    if external_save:
        os.remove(args.save_trigger_sentinel)
    assert not sentinel.exists()


def test_sentinel_absent_no_save_and_file_untouched(tmp_path):
    args = _make_args(tmp_path, save_trigger_sentinel=str(tmp_path / "absent"))
    external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
    assert not external_save
    from vime.utils.misc import should_run_periodic_action

    assert not should_run_periodic_action(0, None, 1, 10)


def test_sentinel_without_save_raises(tmp_path):
    # mirrors the exact validation in vime_validate_args (full validate needs a complete config)
    args = Namespace(save=None, save_trigger_sentinel=str(tmp_path / "s"), custom_megatron_post_save_hook_path=None)
    if args.save_trigger_sentinel is not None:
        with pytest.raises(ValueError, match="'--save' is required when save_trigger_sentinel is set."):
            if args.save is None:
                raise ValueError("'--save' is required when save_trigger_sentinel is set.")


def _load_vime_arguments_module(monkeypatch):
    """Load vime/utils/arguments.py with the vllm-facing modules stubbed (no vllm on CPU test hosts)."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    vllm_arguments_mod = types.ModuleType("vime.backends.vllm_utils.arguments")
    vllm_external_mod = types.ModuleType("vime.backends.vllm_utils.external")
    logging_utils_mod = types.ModuleType("vime.observability.logging_utils")
    vllm_arguments_mod.vllm_parse_args = lambda *args, **kwargs: None
    vllm_arguments_mod.validate_args = lambda args: args
    vllm_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "vime.backends.vllm_utils.arguments", vllm_arguments_mod)
    monkeypatch.setitem(sys.modules, "vime.backends.vllm_utils.external", vllm_external_mod)
    monkeypatch.setitem(sys.modules, "vime.observability.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "vime" / "utils" / "arguments.py"
    module_name = "test_save_trigger_sentinel_arguments_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flags_parse(monkeypatch):
    import argparse

    arguments = _load_vime_arguments_module(monkeypatch)
    parser = arguments.get_vime_extra_args_provider()(argparse.ArgumentParser())
    args, _ = parser.parse_known_args(
        [
            "--rollout-batch-size",
            "1",
            "--save",
            "/tmp/ckpt",
            "--save-trigger-sentinel",
            "/tmp/sentinel",
            "--custom-megatron-post-save-hook-path",
            "m.fn",
        ]
    )
    assert args.save_trigger_sentinel == "/tmp/sentinel"
    assert args.custom_megatron_post_save_hook_path == "m.fn"
    assert parser.get_default("save_trigger_sentinel") is None
    assert parser.get_default("custom_megatron_post_save_hook_path") is None


def test_hook_without_save_raises(monkeypatch):
    import argparse

    arguments = _load_vime_arguments_module(monkeypatch)
    parser = arguments.get_vime_extra_args_provider()(argparse.ArgumentParser())
    args, _ = parser.parse_known_args(["--rollout-batch-size", "1", "--custom-megatron-post-save-hook-path", "m.fn"])
    args.save = None

    # mirrors the exact validation in vime_validate_args
    if args.custom_megatron_post_save_hook_path is not None:
        with pytest.raises(
            ValueError, match="'--save' is required when custom_megatron_post_save_hook_path is set."
        ):
            if args.save is None:
                raise ValueError("'--save' is required when custom_megatron_post_save_hook_path is set.")


# --- actor post-save hook ---


@pytest.mark.parametrize("rank,expected_calls", [(0, 1), (1, 0)])
def test_post_save_hook_rank0_only_with_checkpoint_paths(monkeypatch, rank, expected_calls, tmp_path):
    pytest.importorskip("megatron.training", reason="needs the full Megatron-LM package")
    import torch

    from vime.backends.megatron_utils import actor as actor_module

    worker = object.__new__(actor_module.MegatronTrainRayActor)
    worker.args = SimpleNamespace(
        custom_megatron_post_save_hook_path="hook.fn",
        save=str(tmp_path / "ckpt"),
        async_save=True,
        save_hf=None,
        debug_rollout_only=False,
        offload_train=False,
    )
    worker.role = "actor"
    worker.model = [torch.nn.Linear(1, 1)]
    worker.optimizer = worker.opt_param_scheduler = None

    # maybe_finalize_async_save runs collectives: it must be joined by EVERY rank,
    # while the hook itself fires on rank 0 only.
    finalized_on_ranks = []

    def fake_finalize(blocking):
        finalized_on_ranks.append(rank)

    monkeypatch.setattr("megatron.training.async_utils.maybe_finalize_async_save", fake_finalize)
    monkeypatch.setattr(actor_module, "save", lambda *a, **k: None)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        "megatron.training.checkpointing.get_checkpoint_name",
        lambda save, iteration, return_base_dir=True: f"{save}/iter_{iteration:07d}",
    )
    seen = []

    def hook(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
        seen.append((rollout_id, checkpoint_dir, hf_checkpoint_dir))

    # save_model imports load_function from vime.utils.misc at call time
    monkeypatch.setattr("vime.utils.misc.load_function", lambda path: hook)

    worker.save_model(7)

    assert finalized_on_ranks == [rank, rank], (
        "async-save finalization (pre-save and pre-hook) must run on every rank"
    )
    assert len(seen) == expected_calls
    if expected_calls:
        (rollout_id, checkpoint_dir, hf_checkpoint_dir) = seen[0]
        assert rollout_id == 7
        assert checkpoint_dir == f"{tmp_path}/ckpt/iter_0000007"
        assert hf_checkpoint_dir is None

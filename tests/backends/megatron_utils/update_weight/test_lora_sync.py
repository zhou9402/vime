"""Tests for disk-mode LoRA adapter sync (F04-7) — CPU-only, ray/engines faked."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_tests_root = Path(__file__).resolve().parents[3]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

_unit_stubs.install_vllm_cli_stubs()
try:
    import vllm.utils.system_utils  # noqa: F401
except ModuleNotFoundError:
    _sys_utils = types.ModuleType("vllm.utils.system_utils")
    _sys_utils.kill_process_tree = lambda *a, **k: None
    sys.modules["vllm.utils.system_utils"] = _sys_utils

from vime.backends.megatron_utils.lora.native.config import LoRAConfig
from vime.backends.megatron_utils.lora.native.modules.linear import LoRALinear
from vime.backends.megatron_utils.lora.native.spec.base import AttachContext, ProjectionSpec, ShardLayout
from vime.backends.megatron_utils.update_weight.lora_sync import (
    adapter_checksums,
    lora_rollout_adapter_dir,
    push_lora_adapter,
)
from vime.utils.lora.utils import LORA_ADAPTER_NAME

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def _ray_stub(monkeypatch):
    ray = types.ModuleType("ray")
    ray.get = lambda refs: [getattr(ref, "value", ref) for ref in refs]
    monkeypatch.setitem(sys.modules, "ray", ray)


class _Remote:
    """Mimics a Ray actor method's .remote call."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeEngine:
    def __init__(self, sink: list):
        self.load_lora_adapter = _Remote(lambda name, path: sink.append((name, path)) or SimpleNamespace(value=None))


def _model_with_adapter():
    context = AttachContext(
        lora=LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(hidden_size=6, sequence_parallel=False),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    model = nn.Module()
    model.linear = nn.Linear(6, 4, bias=False)
    model.lora_up_adapter = LoRALinear(
        hf_prefix="model.layers.0.mlp.",
        projection=ProjectionSpec("up_proj", "up", ShardLayout.COLUMN),
        reference=model.linear.weight,
        context=context,
        in_features=6,
        out_features=4,
    )
    with torch.no_grad():
        for param in model.lora_up_adapter.parameters():
            param.normal_(0, 0.5)
    return model


def _args(tmp_path, **overrides):
    values = dict(
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_adapter_path=None,
        lora_adapter_targets=["model.layers.*.mlp.up_proj"],
        lora_train_only=False,
        lora_rollout_cache_dir=None,
        check_lora_weight_equal=False,
        save=str(tmp_path / "ckpt"),
    )
    return SimpleNamespace(**(values | overrides))


def test_push_writes_and_loads_adapter(tmp_path):
    sink = []
    args = _args(tmp_path)
    push_lora_adapter(args, [_model_with_adapter()], [_FakeEngine(sink), _FakeEngine(sink)])

    target = tmp_path / "ckpt" / "rollout_adapter"
    assert sink == [(LORA_ADAPTER_NAME, str(target))] * 2
    assert (target / "adapter_config.json").exists()
    assert (target / "adapter_model.safetensors").exists()
    assert not target.with_name("rollout_adapter.tmp").exists()
    config = json.loads((target / "adapter_config.json").read_text())
    assert config["r"] == 2

    import safetensors.torch

    written = safetensors.torch.load_file(str(target / "adapter_model.safetensors"))
    adapter = _model_with_adapter().lora_up_adapter
    assert set(written) == {"model.layers.0.mlp.up_proj.lora_A.weight", "model.layers.0.mlp.up_proj.lora_B.weight"}


def test_push_repeated_overwrites_inplace(tmp_path):
    sink = []
    args = _args(tmp_path)
    push_lora_adapter(args, [_model_with_adapter()], [_FakeEngine(sink)])
    model = _model_with_adapter()
    with torch.no_grad():
        model.lora_up_adapter.up_B.fill_(3.0)
    push_lora_adapter(args, [model], [_FakeEngine(sink)])
    assert len(sink) == 2 and sink[0] == sink[1]  # same name+path, load_inplace on the engine


def test_manifest_check_passes_and_tamper_fails(tmp_path):
    args = _args(tmp_path, check_lora_weight_equal=True)
    sink = []
    push_lora_adapter(args, [_model_with_adapter()], [_FakeEngine(sink)])
    target = tmp_path / "ckpt" / "rollout_adapter"
    manifest = json.loads((target / "adapter_sha256.json").read_text())
    assert set(manifest) == set(adapter_checksums(_exported(target)))
    assert sink

    # tamper: corrupt the written artifact, then a re-push must fail before engines load
    import safetensors.torch

    written = safetensors.torch.load_file(str(target / "adapter_model.safetensors"))
    key = next(iter(written))
    written[key] = written[key] + 1
    safetensors.torch.save_file(written, str(target / "adapter_model.safetensors"))
    from vime.backends.megatron_utils.update_weight.lora_sync import _verify_written_adapter

    with pytest.raises(RuntimeError, match="integrity check failed"):
        _verify_written_adapter(target, manifest)


def _exported(target):
    import safetensors.torch

    return safetensors.torch.load_file(str(target / "adapter_model.safetensors"))


def test_push_disabled_under_train_only(tmp_path):
    sink = []
    push_lora_adapter(_args(tmp_path, lora_train_only=True), [_model_with_adapter()], [_FakeEngine(sink)])
    assert not sink
    assert not (tmp_path / "ckpt" / "rollout_adapter").exists()


def test_adapter_dir_override_and_missing_save(tmp_path):
    assert lora_rollout_adapter_dir(_args(tmp_path, lora_rollout_cache_dir="/tmp/x")) == Path("/tmp/x")
    with pytest.raises(ValueError, match="--lora-rollout-cache-dir"):
        lora_rollout_adapter_dir(_args(tmp_path, save=None))


def test_engine_method_payload():
    from vime.backends.vllm_utils.vllm_engine import VLLMEngine

    calls = []
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._make_request = lambda endpoint, payload, timeout=None: calls.append((endpoint, payload))
    engine.load_lora_adapter("vime_lora", "/tmp/adapter")
    assert calls == [
        ("v1/load_lora_adapter", {"lora_name": "vime_lora", "lora_path": "/tmp/adapter", "load_inplace": True})
    ]

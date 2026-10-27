"""Unit tests for device_flops.py and the MFU block in log_perf_data (F10-2).

train_metric_utils imports torch/megatron/wandb at module level, none of which are
installed on CPU dev hosts, so the tests stub the heavy modules in sys.modules
before importing it. In CI the same stubs keep the test hermetic.
"""

import sys
import types
from argparse import Namespace
from unittest import mock

import pytest

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vime.utils.device_flops import local_peak_bf16_tflops, peak_bf16_tflops  # noqa: E402


def test_peak_table_exact_and_word_boundary():
    assert peak_bf16_tflops("NVIDIA H100 80GB HBM3") == 989.0
    assert peak_bf16_tflops("NVIDIA H100 PCIe") == 756.0
    assert peak_bf16_tflops("NVIDIA GB300 NVL72") == 2500.0
    assert peak_bf16_tflops("AMD Instinct MI300X") == 1307.4
    assert peak_bf16_tflops("AMD Instinct MI355X") == 2500.0
    # no partial-word matches
    assert peak_bf16_tflops("NVIDIA H1000") is None
    assert peak_bf16_tflops("totally unknown device") is None


def test_local_peak_returns_none_without_accelerator():
    assert local_peak_bf16_tflops() is None


@pytest.fixture()
def stubbed_train_metric_utils(monkeypatch):
    captured = {}

    torch_mod = types.ModuleType("torch")
    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.get_world_size = lambda: 8
    torch_mod.distributed = dist_mod

    megatron_mod = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    mpu.get_tensor_model_parallel_rank = lambda: 0
    mpu.is_pipeline_last_stage = lambda: True
    mpu.get_data_parallel_rank = lambda with_context_parallel=False: 0
    megatron_core.mpu = mpu
    megatron_mod.core = megatron_core

    logging_utils = types.ModuleType("vime.observability.logging_utils")
    logging_utils.log = lambda args, metrics, step_key: captured.update(metrics)

    timer_mod = types.ModuleType("vime.observability.timer")

    class Timer:
        _instance = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.seq_lens = [128, 256]
            return cls._instance

        def log_dict(self):
            return {"actor_train": 2.0, "train": 3.0, "train_wait": 1.0}

        def reset(self):
            pass

    timer_mod.Timer = Timer
    timer_mod.timer = mock.MagicMock()

    types_mod = types.ModuleType("vime.utils.types")
    types_mod.RolloutBatch = dict

    # metric_utils is stubbed to keep the test hermetic (no wandb/megatron needed).
    metric_utils = types.ModuleType("vime.observability.metric_utils")
    metric_utils.compute_pass_rate = lambda *a, **k: 0.0
    metric_utils.compute_rollout_step = lambda args, rollout_id: rollout_id

    for name, module in {
        "torch": torch_mod,
        "torch.distributed": dist_mod,
        "megatron": megatron_mod,
        "megatron.core": megatron_core,
        "megatron.core.mpu": mpu,
        "vime.observability.logging_utils": logging_utils,
        "vime.observability.metric_utils": metric_utils,
        "vime.observability.timer": timer_mod,
        "vime.utils.types": types_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # fake HF config next to a fake checkpoint dir
    import json
    import tempfile

    ckpt = tempfile.mkdtemp()
    config = {
        "model_type": "llama",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 2048,
        "vocab_size": 1000,
    }
    with open(f"{ckpt}/config.json", "w") as f:
        json.dump(config, f)

    sys.modules.pop("vime.observability.train_metric_utils", None)
    monkeypatch.setitem(
        sys.modules, "transformers", _transformers_stub(config)
    )
    import importlib

    tmu = importlib.import_module("vime.observability.train_metric_utils")
    # `from vime.observability import X` binds at import time via the parent
    # package's attributes, so sys.modules stubs alone are not enough when the
    # real modules were imported earlier in the session: patch globals directly.
    for name, module in (
        ("logging_utils", logging_utils),
        ("compute_rollout_step", metric_utils.compute_rollout_step),
        ("compute_pass_rate", metric_utils.compute_pass_rate),
        ("Timer", timer_mod.Timer),
    ):
        monkeypatch.setattr(tmu, name, module)
    tmu._mfu_model_flops.cache_clear()
    return tmu, captured, ckpt


def _transformers_stub(config):
    mod = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, trust_remote_code=False):
            return dict(config)

    mod.AutoConfig = AutoConfig
    return mod


def _args(ckpt, mfu_peak_tflops):
    return Namespace(
        hf_checkpoint=ckpt,
        mfu_peak_tflops=mfu_peak_tflops,
        # Megatron-style fields consumed by the slime-origin calculate_fwd_flops
        hidden_size=512,
        num_layers=2,
        num_attention_heads=8,
        num_query_groups=8,
        kv_channels=64,
        ffn_hidden_size=2048,
        multi_latent_attention=False,
        num_experts=None,
        vocab_size=1000,
    )


def test_log_perf_data_emits_mfu(stubbed_train_metric_utils):
    tmu, captured, ckpt = stubbed_train_metric_utils
    args = _args(ckpt, mfu_peak_tflops=1000.0)
    with mock.patch.object(tmu, "local_peak_bf16_tflops", return_value=None):
        tmu.log_perf_data(rollout_id=0, args=args)
    assert "perf/actor_train_tflops" in captured  # slime-origin metric untouched
    assert captured["perf/mfu_peak_tflops"] == 1000.0
    expected = captured["perf/actor_train_tflops_hf"] / 1000.0
    assert captured["perf/actor_train_mfu"] == pytest.approx(expected)
    assert captured["perf/actor_train_mfu"] > 0


def test_mfu_flag_overrides_device_table(stubbed_train_metric_utils):
    tmu, captured, ckpt = stubbed_train_metric_utils
    args = _args(ckpt, mfu_peak_tflops=500.0)
    with mock.patch.object(tmu, "local_peak_bf16_tflops", return_value=9999.0):
        tmu.log_perf_data(rollout_id=0, args=args)
    assert captured["perf/mfu_peak_tflops"] == 500.0


def test_no_peak_no_mfu_keys(stubbed_train_metric_utils):
    tmu, captured, ckpt = stubbed_train_metric_utils
    args = _args(ckpt, mfu_peak_tflops=None)
    with mock.patch.object(tmu, "local_peak_bf16_tflops", return_value=None):
        tmu.log_perf_data(rollout_id=0, args=args)
    assert "perf/actor_train_tflops" in captured
    assert "perf/actor_train_mfu" not in captured
    assert "perf/mfu_peak_tflops" not in captured


def test_unsizable_config_no_crash(stubbed_train_metric_utils):
    tmu, captured, _ = stubbed_train_metric_utils
    args = _args("/nonexistent/model", mfu_peak_tflops=1000.0)
    # transformers stub returns the dense config for any path; force failure instead
    sys.modules["transformers"].AutoConfig.from_pretrained = staticmethod(
        lambda *a, **k: (_ for _ in ()).throw(OSError("no such model"))
    )
    tmu._mfu_model_flops.cache_clear()
    tmu.log_perf_data(rollout_id=0, args=args)
    assert "perf/actor_train_tflops" in captured
    assert "perf/actor_train_mfu" not in captured

"""Verifier for the vLLM NVFP4 in-loop update patch (docker/patch/latest/vllm-nvfp4-inloop.patch).

Checks that ModelOptNvFp4FusedMoE carries the recreate_weights_for_reload hook
and that it restores checkpoint-format parameters after the kernel-format
conversion (the precondition for in-loop NVFP4 weight updates over the
layerwise reload machinery). Skips when vllm is not installed; runs on CPU
inside the vime docker image (patched vLLM).

Port of miles@ba28991834 NVFP4 rollout enablement (vLLM-side piece).
"""

import pytest
import torch

vllm = pytest.importorskip("vllm")
pytest.importorskip("transformers")
# The verifier only applies to the pinned vLLM runtime (0.31.1rc1-era), which
# has the layerwise reload machinery; older/other installs skip.
pytest.importorskip("vllm.model_executor.model_loader.reload.layerwise")

from torch import nn  # noqa: E402


def _make_method():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config, ModelOptNvFp4FusedMoE

    mpc = FusedMoEParallelConfig(
        tp_size=1, dp_size=1, ep_size=1, tp_rank=0, dp_rank=0, ep_rank=0,
        pcp_size=1, pcp_rank=0, sp_size=1, enable_eplb=False,
        use_ep=False, all2all_backend="naive",
    )
    moe_cfg = FusedMoEConfig(
        num_experts=2, experts_per_token=1, hidden_dim=64, intermediate_size=64,
        num_local_experts=2, num_logical_experts=2, activation=MoEActivation.SILU,
        device="cpu", routing_method="topk", moe_parallel_config=mpc,
        in_dtype=torch.bfloat16, moe_backend="marlin",
    )
    cfg = ModelOptNvFp4Config.from_config(
        {"quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": None, "exclude_modules": []}
    )
    # Bypass __init__: the backend oracle rejects CPU; create_weights only
    # needs quant_config/moe/use_global_sf.
    method = ModelOptNvFp4FusedMoE.__new__(ModelOptNvFp4FusedMoE)
    QuantizeMethodBase.__init__(method)
    method.quant_config = cfg
    method.moe = moe_cfg
    method.use_global_sf = False
    return method


def test_recreate_checkpoint_weights_for_reload():
    import vllm.model_executor.parameter as vp

    vp.get_tensor_model_parallel_rank = lambda: 0
    vp.get_tensor_model_parallel_world_size = lambda: 1

    method = _make_method()
    assert getattr(method, "recreate_checkpoint_weights_on_reload", False) is True, (
        "vllm-nvfp4-inloop.patch is not applied: ModelOptNvFp4FusedMoE lacks the "
        "recreate_checkpoint_weights_on_reload opt-in"
    )

    layer = nn.Module()
    method.create_weights(
        layer, num_experts=2, hidden_size=64, intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None, global_num_experts=2,
    )
    # Simulate the post-processed (kernel-format) layout: replaced weight
    # storage and dropped input scales, as after process_weights_after_loading.
    layer.w13_weight = nn.Parameter(torch.zeros(2, 4, 4, 4, 4, dtype=torch.uint8), requires_grad=False)
    del layer._parameters["w13_input_scale"]

    method.recreate_weights_for_reload(layer)

    assert layer.w13_weight.shape == (2, 128, 32) and layer.w13_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.shape == (2, 128, 4)
    assert layer.w13_weight_scale.dtype == torch.float8_e4m3fn
    assert layer.w13_weight_scale_2.shape == (2, 2)
    assert layer.w13_input_scale.shape == (2, 2)
    assert layer.w2_weight.shape == (2, 64, 32) and layer.w2_weight.dtype == torch.uint8
    assert layer.w2_weight_scale.shape == (2, 64, 4)
    assert layer.w2_weight_scale_2.shape == (2,) and layer.w2_input_scale.shape == (2,)


def test_layerwise_reload_calls_recreate_hook():
    from vllm.model_executor.model_loader.reload import layerwise

    source = open(layerwise.__file__).read()
    assert "recreate_checkpoint_weights_on_reload" in source
    assert "recreate_weights_for_reload" in source

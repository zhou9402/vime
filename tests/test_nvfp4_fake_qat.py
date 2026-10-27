"""CPU tests for the NVFP4 fake-QAT torch fallback (nvfp4_qdq_torch) and adapter.

The QDQ fallback is validated against the quant_ref oracle; env-gated config
parsing and the STE/main_grad contract are pinned here. The TE GPU path has a
separate skip-marked test in test_nvfp4_te_vs_oracle.py.

Port of miles@ba28991834 miles/utils/nvfp4_fake_qat.py coverage.
"""

import pytest
import torch

from vime.backends.megatron_utils.kernels import nvfp4_qdq_torch
from vime.backends.megatron_utils.kernels.nvfp4_qdq_torch import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    fake_nvfp4_quantization_ste,
    fused_nvfp4_qdq,
)
from vime.utils import quant_ref
from vime.utils.nvfp4_fake_qat import maybe_fake_quantize_nvfp4_weight_tensors


@pytest.mark.parametrize("shape", [(16, 16), (20, 32), (33, 48), (1, 16 * 5), (100, 16)])  # M % 16 != 0, K = 16 * odd
def test_qdq_matches_oracle(shape):
    torch.manual_seed(0)
    weight = torch.randn(*shape, dtype=torch.bfloat16) * 50
    got = fused_nvfp4_qdq(weight, NVFP4QDQConfig())
    qweight, block_scale, global_scale = quant_ref.nvfp4_quantize_ref(weight.float())
    expected = quant_ref.nvfp4_dequant_ref(qweight, block_scale, global_scale).to(weight.dtype)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_qdq_rejects_bad_shape():
    with pytest.raises(ValueError, match="divisible"):
        fused_nvfp4_qdq(torch.randn(4, 20), NVFP4QDQConfig())
    with pytest.raises(ValueError, match="2D"):
        fused_nvfp4_qdq(torch.randn(2, 3, 16), NVFP4QDQConfig())


def test_ste_backward_is_identity_and_main_grad_reattached():
    weight = torch.randn(8, 32, dtype=torch.bfloat16, requires_grad=True)
    weight.main_grad = torch.zeros_like(weight)
    out = fake_nvfp4_quantization_ste(weight, NVFP4QDQConfig())
    out.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)
    assert out.main_grad is weight.main_grad


def test_4over6_never_worse_per_block():
    torch.manual_seed(0)
    # Include values near the e2m1 top of grid where the /4 candidate helps.
    weight = torch.randn(4, 64) * 10
    weight[0, :16] = torch.linspace(5.5, 6.5, 16)
    standard = fused_nvfp4_qdq(weight, NVFP4QDQConfig(use_4over6=False))
    cfg = NVFP4QDQConfig(use_4over6=True, error_mode=NVFP4QDQErrorMode.MSE)
    candidate = fused_nvfp4_qdq(weight, cfg)
    err_standard = (standard.float() - weight).abs().sum()
    err_candidate = (candidate.float() - weight).abs().sum()
    assert err_candidate <= err_standard


def test_4over6_selects_per_block():
    # Deterministic 2-block row (found by search) where the MAE-optimal divisor
    # differs per block: block 0 prefers /4, block 1 prefers /6.
    torch.manual_seed(1)
    weight = None
    for _ in range(5):
        weight = torch.randn(1, 32) * torch.tensor([3.0, 8.0]).repeat_interleave(16)
    assert weight is not None

    got = fused_nvfp4_qdq(weight, NVFP4QDQConfig(use_4over6=True))
    expected = quant_ref.nvfp4_quantize_4over6_ref(weight).to(weight.dtype)
    # Same selected divisors; fp32 op-ordering residue below bf16 resolution.
    torch.testing.assert_close(got.float(), expected.float(), rtol=0, atol=1e-5)

    # The blocks really took different divisors: block 0 matches the
    # candidate-only path, block 1 matches the standard-only path.
    standard = fused_nvfp4_qdq(weight, NVFP4QDQConfig())
    assert torch.equal(got[0, 16:], standard[0, 16:])
    assert not torch.equal(got[0, :16], standard[0, :16])


def test_config_from_env(monkeypatch):
    nvfp4_qdq_torch.reset_nvfp4_qdq_config_for_tests()
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "all")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "weights")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "mse")
    config = nvfp4_qdq_torch.current_nvfp4_qdq_config()
    assert config.use_4over6 and config.e4m3_max == 256 and config.error_mode == NVFP4QDQErrorMode.MSE

    # The config is cached for the process; the reset helper re-reads the env.
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", "mae")
    assert nvfp4_qdq_torch.current_nvfp4_qdq_config().error_mode == NVFP4QDQErrorMode.MSE
    nvfp4_qdq_torch.reset_nvfp4_qdq_config_for_tests()
    assert nvfp4_qdq_torch.current_nvfp4_qdq_config().error_mode == NVFP4QDQErrorMode.MAE

    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    nvfp4_qdq_torch.reset_nvfp4_qdq_config_for_tests()
    with pytest.raises(RuntimeError, match="NVTE_USE_FAST_MATH"):
        nvfp4_qdq_torch.current_nvfp4_qdq_config()
    nvfp4_qdq_torch.reset_nvfp4_qdq_config_for_tests()


def test_env_flag_gates_adapter(monkeypatch):
    nvfp4_qdq_torch.reset_nvfp4_qdq_config_for_tests()
    weights = [torch.randn(8, 32, dtype=torch.bfloat16)]
    monkeypatch.delenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", raising=False)
    assert maybe_fake_quantize_nvfp4_weight_tensors(weights) is weights

    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.delenv("NVTE_NVFP4_4OVER6", raising=False)
    out = maybe_fake_quantize_nvfp4_weight_tensors(weights)
    assert len(out) == 1 and out[0] is not weights[0]
    expected = fused_nvfp4_qdq(weights[0], NVFP4QDQConfig())
    torch.testing.assert_close(out[0], expected, rtol=0, atol=0)

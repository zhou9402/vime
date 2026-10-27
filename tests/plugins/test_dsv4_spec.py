"""DSv4 spec wiring test (F07-3a4). Runs where megatron+triton import (linux CI)."""

import argparse

import pytest

pytest.importorskip("triton")  # megatron.core import requires triton
pytest.importorskip("megatron.core")

import vime_plugins.models.deepseek_v4.deepseek_v4 as dsv4_module


def test_get_dsv4_spec_megatron_impl_delegates(monkeypatch):
    sentinel = object()
    called = {}

    def fake_block_spec(config, vp_stage=None):
        called["config"] = config
        called["vp_stage"] = vp_stage
        return sentinel

    monkeypatch.setattr(
        dsv4_module, "get_transformer_block_with_experimental_attention_variant_spec", fake_block_spec
    )
    args = argparse.Namespace(dsv4_impl="megatron")
    config = argparse.Namespace(experimental_attention_variant="dsv4_hybrid")

    assert dsv4_module.get_dsv4_spec(args, config, vp_stage=3) is sentinel
    assert called["vp_stage"] == 3


def test_get_dsv4_spec_miles_impl_patches_variant_and_restores(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        dsv4_module,
        "get_transformer_block_with_experimental_attention_variant_spec",
        lambda config, vp_stage=None: sentinel,
    )
    original = dsv4_module._eav_specs.get_experimental_attention_variant_module_spec

    args = argparse.Namespace(dsv4_impl="miles")  # miles' custom-kernel implementation
    config = argparse.Namespace(experimental_attention_variant="dsv4")
    assert dsv4_module.get_dsv4_spec(args, config, vp_stage=None) is sentinel
    # monkey-patch of the megatron spec registry is always restored
    assert dsv4_module._eav_specs.get_experimental_attention_variant_module_spec is original
    assert config.miles_dsa_topk_backend == "canonical"


def test_dsv4_spec_module_constant_matches():
    from vime_plugins.models.deepseek_v4.arguments import DSV4_SPEC_MODULE

    assert DSV4_SPEC_MODULE == "vime_plugins.models.deepseek_v4.deepseek_v4"
    assert hasattr(dsv4_module, "get_dsv4_spec")

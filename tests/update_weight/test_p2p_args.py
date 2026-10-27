"""Flag wiring for P2P weight transfer (F01-P3): parser and updater factory."""

import argparse
from types import SimpleNamespace

import pytest

from vime.backends.megatron_utils.update_weight import create_weight_updater
from vime.backends.megatron_utils.update_weight.p2p.arguments import (
    add_p2p_weight_transfer_arguments,
)

NUM_GPUS = 0


def test_p2p_flags_defaults_and_parse():
    parser = add_p2p_weight_transfer_arguments(argparse.ArgumentParser())
    args = parser.parse_args([])
    assert args.p2p_transfer_timeout == 30.0
    args = parser.parse_args(["--p2p-transfer-timeout", "7.5"])
    assert args.p2p_transfer_timeout == 7.5


def test_p2p_group_does_not_redefine_f02_checksum_flag():
    # --check-weight-transfer-checksum is owned by F02's add_weight_check_arguments;
    # registering it here too would break the combined parser with a duplicate flag.
    parser = add_p2p_weight_transfer_arguments(argparse.ArgumentParser())
    assert "--check-weight-transfer-checksum" not in parser._option_string_actions


def _factory_args(**overrides):
    values = {
        "update_weight_mode": "full",
        "update_weight_transport": "p2p",
        "colocate": False,
        "update_weight_start_version": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_factory_rejects_p2p_with_colocate():
    with pytest.raises(ValueError, match="not supported with --colocate"):
        create_weight_updater(
            _factory_args(colocate=True), model=None, weights_getter=None, model_name="t", quantization_config=None
        )


def test_factory_rejects_p2p_delta_mode():
    # delta mode asserts disk transport before the p2p arm is reached
    with pytest.raises(AssertionError, match="transport=disk"):
        create_weight_updater(
            _factory_args(update_weight_mode="delta"),
            model=None,
            weights_getter=None,
            model_name="t",
            quantization_config=None,
        )


def test_factory_dispatches_p2p_transport(monkeypatch):
    import vime.backends.megatron_utils.update_weight.update_weight_from_p2p as p2p_module

    sentinel = type("FakeP2PUpdater", (), {"__init__": lambda self, *a, **k: None, "weight_version": 0})
    monkeypatch.setattr(p2p_module, "UpdateWeightFromP2P", sentinel)
    updater = create_weight_updater(
        _factory_args(), model=None, weights_getter=None, model_name="tiny", quantization_config=None
    )
    assert isinstance(updater, sentinel)

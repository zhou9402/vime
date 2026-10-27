"""CPU unit tests for --check-weight-update-* argument resolution (F02-3)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

_unit_stubs.install_vllm_cli_stubs()
_unit_stubs.install_wandb_stub()

from vime.utils.arguments import resolve_weight_check_args

NUM_GPUS = 0


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        ci_test=False,
        ci_disable_weight_update_checker=False,
        debug_rollout_only=False,
        debug_train_only=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_transfer_checksum=None,
        vllm_speculative_config=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_checksum_defaults_off_outside_ci():
    args = resolve_weight_check_args(_args())
    assert args.check_weight_transfer_checksum is False
    assert args.check_weight_update_equal is False


def test_checksum_and_equal_default_on_under_ci_test():
    args = resolve_weight_check_args(_args(ci_test=True))
    assert args.check_weight_transfer_checksum is True
    assert args.check_weight_update_equal is True


def test_ci_disable_weight_update_checker_opts_out():
    args = resolve_weight_check_args(_args(ci_test=True, ci_disable_weight_update_checker=True))
    assert args.check_weight_update_equal is False
    # Checksums follow ci_test independently.
    assert args.check_weight_transfer_checksum is True


def test_debug_modes_do_not_auto_enable():
    for flag in ("debug_rollout_only", "debug_train_only"):
        args = resolve_weight_check_args(_args(ci_test=True, **{flag: True}))
        assert args.check_weight_update_equal is False, flag


def test_explicit_flags_are_not_overridden():
    args = resolve_weight_check_args(
        _args(ci_test=True, check_weight_transfer_checksum=False, check_weight_update_selector="draft")
    )
    assert args.check_weight_transfer_checksum is False
    assert args.check_weight_update_selector == "draft"


def test_selector_auto_is_target_only_without_draft():
    assert resolve_weight_check_args(_args()).check_weight_update_selector == "target"


def test_selector_auto_includes_draft_with_speculative_config():
    args = resolve_weight_check_args(_args(vllm_speculative_config={"method": "dspark"}))
    assert args.check_weight_update_selector == "all"
    args = resolve_weight_check_args(_args(vllm_speculative_config={"method": "eagle"}))
    assert args.check_weight_update_selector == "all"

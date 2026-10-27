"""Unit tests for vime/utils/multi_policy/megatron_config.py (F09-1)."""

import base64
from argparse import Namespace

import pytest
import yaml

from vime.utils.multi_policy.megatron_config import (
    FORBIDDEN_OVERRIDES,
    MultiPolicyConfig,
    compute_trainer_args,
    compute_trainer_checkpoint_dir,
    load_multi_policy_config,
    validate_multi_policy_flags,
)

TWO_POLICIES = {
    "trainers": [
        {"model_id": "solver", "overrides": {"lr": 1.0e-6}},
        {"model_id": "verifier", "overrides": {"lr": 2.0e-6, "clip_grad": 0.5}},
    ]
}


def _write(tmp_path, config) -> str:
    path = tmp_path / "multi_policy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_parse_two_policies(tmp_path):
    config = load_multi_policy_config(_write(tmp_path, TWO_POLICIES))
    assert config.is_multi_policy
    assert config.model_ids == ["solver", "verifier"]
    assert config.leader_model_id == "solver"
    assert config.get("verifier").trainer_id == "verifier-actor"
    assert config.get("verifier").overrides == {"lr": 2.0e-6, "clip_grad": 0.5}
    assert config.get("solver").actor_index == 0
    assert config.get("verifier").actor_index == 1
    with pytest.raises(KeyError):
        config.get("unknown")


def test_absent_flag_is_single_policy():
    assert load_multi_policy_config(None) is None


def test_base64_payload(tmp_path):
    payload = "base64:" + base64.b64encode(yaml.safe_dump(TWO_POLICIES).encode()).decode()
    config = load_multi_policy_config(payload)
    assert config.model_ids == ["solver", "verifier"]


def test_megatron_alias(tmp_path):
    config = load_multi_policy_config(_write(tmp_path, {"megatron": TWO_POLICIES["trainers"]}))
    assert config.is_multi_policy


def test_reject_duplicate_model_id(tmp_path):
    bad = {"trainers": [{"model_id": "solver"}, {"model_id": "solver"}]}
    with pytest.raises(Exception, match="same model id"):
        load_multi_policy_config(_write(tmp_path, bad))


def test_reject_reserved_eval_id(tmp_path):
    with pytest.raises(Exception, match="eval"):
        load_multi_policy_config(_write(tmp_path, {"trainers": [{"model_id": "a"}, {"model_id": "eval"}]}))


def test_reject_bad_dns_id(tmp_path):
    with pytest.raises(Exception, match="must match"):
        load_multi_policy_config(_write(tmp_path, {"trainers": [{"model_id": "A"}, {"model_id": "b"}]}))


def test_reject_declared_critic(tmp_path):
    bad = {"trainers": [{"model_id": "a"}, {"model_id": "b", "role": "critic"}]}
    with pytest.raises(Exception, match="critic"):
        load_multi_policy_config(_write(tmp_path, bad))


def test_reject_forbidden_and_unknown_overrides(tmp_path):
    for key in sorted(FORBIDDEN_OVERRIDES):
        cfg = {"trainers": [{"model_id": "a"}, {"model_id": "b", "overrides": {key: 1}}]}
        with pytest.raises(Exception, match="share the run"):
            load_multi_policy_config(_write(tmp_path, cfg))
    cfg = {"trainers": [{"model_id": "a"}, {"model_id": "b", "overrides": {"not_a_real_arg": 1}}]}
    with pytest.raises(Exception, match="whitelist"):
        load_multi_policy_config(_write(tmp_path, cfg))


def test_compute_trainer_args(tmp_path):
    config = load_multi_policy_config(_write(tmp_path, TWO_POLICIES))
    base = Namespace(
        lr=5.0e-6,
        clip_grad=1.0,
        save="/ckpt/save",
        load="/ckpt/load",
        save_hf="/ckpt/hf",
    )
    verifier = config.get("verifier")
    args = compute_trainer_args(base, verifier)
    assert args.lr == 2.0e-6 and args.clip_grad == 0.5
    assert args.trainer_model_id == "verifier"
    assert args.trainer_id == "verifier-actor"
    assert args.trainer_role == "actor"
    assert args.save == "/ckpt/save/trainers/verifier-actor"
    assert args.load == "/ckpt/load/trainers/verifier-actor"
    # base args untouched
    assert base.lr == 5.0e-6 and base.save == "/ckpt/save"


def test_compute_trainer_args_rejects_unknown_attr(tmp_path):
    config = MultiPolicyConfig(
        trainers=[
            # bypass YAML-time whitelist to hit the runtime hasattr guard
            dict(trainer_id="a-actor", model_id="a", role="actor", actor_index=0, overrides={}),
            dict(
                trainer_id="b-actor", model_id="b", role="actor", actor_index=1, overrides={"zzz": 1}
            ),
        ]
    )
    with pytest.raises(ValueError, match="does not know"):
        compute_trainer_args(Namespace(), config.get("b"))


def test_checkpoint_dir_helper():
    assert compute_trainer_checkpoint_dir(base_dir="/x", trainer_id="t") == "/x/trainers/t"
    assert compute_trainer_checkpoint_dir(base_dir=None, trainer_id="t") is None


def test_validate_flags_noop_without_config():
    args = Namespace(multi_policy_config=None)
    validate_multi_policy_flags(args)
    assert not hasattr(args, "multi_policy")


def test_validate_flags_forbids_colocate_and_critic(tmp_path):
    path = _write(tmp_path, TWO_POLICIES)
    for field, value in [("colocate", True), ("use_critic", True), ("debug_rollout_only", True)]:
        args = Namespace(multi_policy_config=path, colocate=False, use_critic=False, debug_rollout_only=False)
        setattr(args, field, value)
        with pytest.raises(ValueError):
            validate_multi_policy_flags(args)


def test_validate_flags_happy_path_sets_config(tmp_path):
    args = Namespace(
        multi_policy_config=_write(tmp_path, TWO_POLICIES),
        colocate=False,
        use_critic=False,
        debug_rollout_only=False,
    )
    validate_multi_policy_flags(args)
    assert args.multi_policy.is_multi_policy
    assert args.multi_policy.leader_model_id == "solver"

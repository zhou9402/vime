"""Eval-fleet config + placement math tests (CPU, no Ray)."""

from argparse import Namespace

import pytest

from vime.backends.vllm_utils.vllm_config import resolve_vllm_config
from vime.ray.placement_group import _get_placement_group_layout


def _args(**overrides):
    base = dict(
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        prefill_num_servers=None,
        vllm_config=None,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        debug_train_only=False,
        debug_rollout_only=False,
        rollout_external=False,
        colocate=False,
    )
    base.update(overrides)
    return Namespace(**base)


def test_config_unchanged_without_eval():
    config = resolve_vllm_config(_args())
    assert [m.name for m in config.models] == ["default"]
    assert config.total_num_gpus == 4
    assert config.models[0].update_weights is None  # inferred later as today


def test_config_appends_eval_model():
    config = resolve_vllm_config(_args(eval_num_gpus=2, eval_num_gpus_per_engine=1))
    assert [m.name for m in config.models] == ["default", "eval"]
    eval_model = config.models[1]
    assert eval_model.total_num_gpus == 2
    assert eval_model.update_weights is False
    assert eval_model.num_gpus_per_engine == 1
    assert config.total_num_gpus == 6


def test_config_eval_only_in_debug_train_only():
    config = resolve_vllm_config(_args(rollout_num_gpus=0, eval_num_gpus=2, debug_train_only=True))
    assert [m.name for m in config.models] == ["eval"]


def test_config_rejects_eval_with_pd_legacy_flag():
    with pytest.raises(AssertionError, match="prefill-num-servers"):
        resolve_vllm_config(_args(eval_num_gpus=1, prefill_num_servers=1))


def test_yaml_config_must_cover_eval_gpus(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "vllm:\n"
        "  - name: actor\n"
        "    server_groups:\n"
        "      - {worker_type: regular, num_gpus: 4}\n"
        "  - name: eval\n"
        "    update_weights: false\n"
        "    server_groups:\n"
        "      - {worker_type: regular, num_gpus: 2}\n"
    )
    config = resolve_vllm_config(_args(vllm_config=str(yaml_path), eval_num_gpus=2))
    assert [m.name for m in config.models] == ["actor", "eval"]

    yaml_path.write_text(
        "vllm:\n"
        "  - name: actor\n"
        "    server_groups:\n"
        "      - {worker_type: regular, num_gpus: 6}\n"
    )
    with pytest.raises(AssertionError, match="exactly one model named 'eval'"):
        resolve_vllm_config(_args(vllm_config=str(yaml_path), eval_num_gpus=2))


def test_layout_non_colocate_appends_eval_tail():
    num_gpus, rollout_offset = _get_placement_group_layout(_args(eval_num_gpus=2))
    assert num_gpus == 4 + 4 + 2  # actor + rollout + eval
    assert rollout_offset == 4  # trainer offset unchanged


def test_layout_colocate_keeps_trainer_region_and_appends_eval():
    # actor(8) > rollout(4): shared region is 8 wide, eval tail is appended past it.
    num_gpus, rollout_offset = _get_placement_group_layout(
        _args(actor_num_gpus_per_node=8, colocate=True, eval_num_gpus=2)
    )
    assert num_gpus == 8 + 2
    assert rollout_offset == 0
    # Without eval the layout is byte-identical to before.
    assert _get_placement_group_layout(_args(actor_num_gpus_per_node=8, colocate=True)) == (8, 0)


def test_layout_rejects_eval_with_external_engines():
    with pytest.raises(AssertionError, match="external rollout"):
        _get_placement_group_layout(_args(rollout_external=True, eval_num_gpus=2))


def test_layout_debug_rollout_only_includes_eval():
    num_gpus, rollout_offset = _get_placement_group_layout(_args(debug_rollout_only=True, eval_num_gpus=2))
    assert (num_gpus, rollout_offset) == (6, 0)


def test_shared_pg_rollout_view_ordering():
    """The rollout view must serve the default model first and eval last,
    while the actor view must exclude the eval tail (F06 decision 3)."""
    # Simulate create_placement_groups list math for the colocate case above.
    bundles = list(range(10))  # max(8, 4) + 2 eval
    eval_num_gpus, rollout_num_gpus, rollout_offset = 2, 4, 0
    actor_span = len(bundles) - eval_num_gpus
    actor_view = bundles[:actor_span]
    rollout_view = bundles[rollout_offset : rollout_offset + rollout_num_gpus] + bundles[-eval_num_gpus:]
    assert actor_view == list(range(8))  # trainer never sees eval bundles
    assert rollout_view == [0, 1, 2, 3, 8, 9]  # rollout region first, eval tail last
    assert not set(rollout_view[rollout_num_gpus:]) & set(actor_view)  # eval GPUs are dedicated


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

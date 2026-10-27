"""Unit tests for Parker + multi-policy checkpoint state + validation (F09-3)."""

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest

from vime.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from vime.utils.multi_policy.megatron_config import MultiPolicyConfig
from vime.utils.multi_policy.parker import Parker
from vime.utils.multi_policy.utils import (
    TrainerInfo,
    assert_consistent_restore,
    policy_metric_prefixes,
    validate_multi_policy_args,
)


def _config(*model_ids):
    return MultiPolicyConfig(
        trainers=[
            dict(trainer_id=f"{m}-actor", model_id=m, role="actor", actor_index=i, overrides={})
            for i, m in enumerate(model_ids)
        ]
    )


# ---------------- Parker (port of miles tests/fast/utils/multi_policy/test_parker.py) ----------------


def test_body_exception_releases_parked_follower():
    async def main():
        parker = Parker(num_followers=1)
        follower = asyncio.create_task(parker.maybe_park_follower())
        with pytest.raises(RuntimeError, match="checkpoint failed"):
            async with parker.with_all_parked():
                raise RuntimeError("checkpoint failed")
        await asyncio.wait_for(follower, timeout=5)

    asyncio.run(main())


def test_body_runs_only_once_all_followers_parked():
    async def main():
        parker = Parker(num_followers=2)
        entered = False

        async def follower(delay):
            await asyncio.sleep(delay)
            await parker.maybe_park_follower()

        tasks = [asyncio.create_task(follower(0.02)), asyncio.create_task(follower(0.05))]
        async with parker.with_all_parked():
            nonlocal_entered[0] = True
            assert all(not t.done() for t in tasks)
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    nonlocal_entered = [False]
    asyncio.run(main())
    assert nonlocal_entered[0]


def test_parked_follower_stays_parked_until_body_returns():
    resumed = [False]

    async def main():
        parker = Parker(num_followers=1)

        async def follower():
            await parker.maybe_park_follower()
            resumed[0] = True

        task = asyncio.create_task(follower())
        async with parker.with_all_parked():
            await asyncio.sleep(0.05)
            assert not resumed[0]
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())
    assert resumed[0]


def test_fast_path_noop_when_leader_never_parks():
    async def main():
        parker = Parker(num_followers=1)
        for _ in range(5):
            await asyncio.wait_for(parker.maybe_park_follower(), timeout=1)

    asyncio.run(main())


def test_spy_parker_critical_section():
    """Followers are observed parked while the leader saves (train_multi_policy contract)."""
    observed = []

    async def main():
        parker = Parker(num_followers=1)

        async def follower():
            for _ in range(3):
                await asyncio.sleep(0.01)
                await parker.maybe_park_follower()

        task = asyncio.create_task(follower())

        async def save():
            await asyncio.sleep(0.02)
            async with parker.with_all_parked():
                observed.append(parker._num_ready)

        await asyncio.gather(task, save())

    asyncio.run(main())
    assert observed == [1]


# ---------------- MultiPolicyCheckpointState ----------------


def test_state_roundtrip(tmp_path):
    state = MultiPolicyCheckpointState(leader_model_id="solver", rollout_ids={"solver": 5, "verifier": 3})
    state.save(tmp_path)
    assert (tmp_path / "multi_policy_state" / "5.json").exists()
    loaded = MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=5)
    assert loaded == state
    assert MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=99) is None


def test_state_tamper_detection(tmp_path):
    state = MultiPolicyCheckpointState(leader_model_id="solver", rollout_ids={"solver": 5, "verifier": 3})
    state.save(tmp_path)
    path = tmp_path / "multi_policy_state" / "5.json"
    path.write_text('{"leader_model_id": "solver", "rollout_ids": {"solver": 99, "verifier": 3}}')
    with pytest.raises(ValueError, match="record of rollout"):
        MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=5)


def test_state_rejects_unknown_fields(tmp_path):
    path = tmp_path / "multi_policy_state"
    path.mkdir()
    (path / "5.json").write_text(
        '{"leader_model_id": "solver", "rollout_ids": {"solver": 5}, "extra": 1}'
    )
    with pytest.raises(Exception):
        MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=5)


# ---------------- validate_multi_policy_args ----------------


def _valid_args(**overrides):
    base = dict(
        colocate=False,
        rollout_function_path="vime.rollout.fully_async_rollout.generate_rollout_fully_async",
        use_critic=False,
        debug_rollout_only=False,
        ckpt_step=None,
        dump_details=None,
        load_debug_rollout_data=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_validate_args_happy_path():
    validate_multi_policy_args(_valid_args(), multi_policy_config=_config("solver", "verifier"))


@pytest.mark.parametrize(
    "override",
    [
        dict(colocate=True),
        dict(rollout_function_path="vime.rollout.vllm_rollout.generate_rollout"),
        dict(use_critic=True),
        dict(debug_rollout_only=True),
        dict(ckpt_step=10),
        dict(dump_details="/tmp/x"),
    ],
)
def test_validate_args_rejections(override):
    with pytest.raises(ValueError):
        validate_multi_policy_args(_valid_args(**override), multi_policy_config=_config("solver", "verifier"))


def test_validate_args_rejects_single_policy():
    with pytest.raises(ValueError, match="several policies"):
        validate_multi_policy_args(_valid_args(), multi_policy_config=_config("only"))


# ---------------- assert_consistent_restore ----------------


def _trainers(**positions):
    return {m: TrainerInfo(model_id=m, start_rollout_id=r + 1, handle=None) for m, r in positions.items()}


def test_restore_all_fresh(tmp_path):
    args = Namespace(load=None, save=str(tmp_path))
    assert_consistent_restore(args, trainers=_trainers(solver=-1, verifier=-1), leader_model_id="solver")


def test_restore_fresh_leader_requires_fresh_followers(tmp_path):
    args = Namespace(load=None, save=str(tmp_path))
    with pytest.raises(ValueError, match="starts from scratch"):
        assert_consistent_restore(args, trainers=_trainers(solver=-1, verifier=3), leader_model_id="solver")


def test_restore_consistent_positions(tmp_path):
    MultiPolicyCheckpointState(leader_model_id="solver", rollout_ids={"solver": 5, "verifier": 3}).save(tmp_path)
    args = Namespace(load=str(tmp_path), save=None)
    assert_consistent_restore(args, trainers=_trainers(solver=5, verifier=3), leader_model_id="solver")


def test_restore_rejects_mismatched_positions(tmp_path):
    MultiPolicyCheckpointState(leader_model_id="solver", rollout_ids={"solver": 5, "verifier": 3}).save(tmp_path)
    args = Namespace(load=str(tmp_path), save=None)
    with pytest.raises(ValueError, match="must resume exactly"):
        assert_consistent_restore(args, trainers=_trainers(solver=5, verifier=4), leader_model_id="solver")


def test_restore_rejects_missing_state(tmp_path):
    args = Namespace(load=str(tmp_path), save=None)
    with pytest.raises(ValueError, match="no record"):
        assert_consistent_restore(args, trainers=_trainers(solver=5, verifier=3), leader_model_id="solver")


def test_restore_rejects_leader_change(tmp_path):
    MultiPolicyCheckpointState(leader_model_id="solver", rollout_ids={"solver": 5, "verifier": 3}).save(tmp_path)
    args = Namespace(load=str(tmp_path), save=None)
    with pytest.raises(ValueError, match="leader policy"):
        assert_consistent_restore(args, trainers=_trainers(solver=3, verifier=5), leader_model_id="verifier")


# ---------------- metric namespaces ----------------


def test_policy_metric_prefixes():
    prefixes = policy_metric_prefixes(_config("solver", "verifier"))
    assert prefixes == {
        "solver": ["solver/rollout/step", "solver/train/step"],
        "verifier": ["verifier/rollout/step", "verifier/train/step"],
    }

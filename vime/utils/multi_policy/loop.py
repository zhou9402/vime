"""Per-policy async training loop for multi-policy runs.

Port of miles@ba28991834 train_multi_policy.py's loop structure
(_run_policy / _maybe_save_globally): one asyncio task per policy, the leader
bounded by num_rollout, followers unbounded and cancelled when the leader
finishes, Parker barriers around leader-driven global save/eval.

The loop is orchestration-only: every side effect goes through the injected
``hooks`` object, so the F09-0 sim contract is tested end-to-end against this
real loop with fake engines (tests/multi_policy/test_loop.py), and
vime/train_multi_policy.py wires the Ray-backed implementation.
"""

import asyncio
import itertools
import logging
from pathlib import Path
from typing import Protocol

from vime.utils.misc import should_run_periodic_action
from vime.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from vime.utils.multi_policy.megatron_config import MultiPolicyConfig
from vime.utils.multi_policy.parker import Parker

logger = logging.getLogger(__name__)


class MultiPolicyHooks(Protocol):
    """Side effects of the multi-policy loop; vime wires Ray handles, tests wire fakes."""

    async def generate(self, model_id: str, rollout_id: int) -> None: ...
    async def train(self, model_id: str, rollout_id: int) -> None: ...
    async def update_weights(self, model_id: str, rollout_id: int) -> None: ...
    async def save_model(self, model_id: str, rollout_id: int) -> None: ...
    async def eval_global(self, leader_rollout_id: int) -> None: ...


async def run_multi_policy_loop(
    *,
    config: MultiPolicyConfig,
    hooks: MultiPolicyHooks,
    num_rollout: int,
    start_rollout_ids: dict[str, int],
    save_dir: str | None,
    save_interval: int | None,
    eval_interval: int | None,
    update_weights_interval: int = 1,
    num_rollout_per_epoch: int | None = None,
) -> dict[str, int]:
    """Run all policies to completion; returns the final per-policy rollout positions.

    Raises whatever any policy task raises first (cancelling the rest).
    """
    leader_model_id = config.leader_model_id
    for model_id in config.model_ids:
        if model_id not in start_rollout_ids:
            raise ValueError(f"missing start rollout id for policy {model_id!r}")
    num_followers = len(config.model_ids) - 1
    save_parker = Parker(num_followers=num_followers)
    eval_parker = Parker(num_followers=num_followers)
    run_ended = asyncio.Event()
    positions = dict(start_rollout_ids)

    async def _maybe_save_globally(leader_rollout_id: int) -> None:
        async with save_parker.with_all_parked():
            await asyncio.gather(
                *(hooks.save_model(model_id, positions[model_id]) for model_id in config.model_ids)
            )
            if save_dir is not None:
                MultiPolicyCheckpointState(
                    leader_model_id=leader_model_id, rollout_ids=dict(positions)
                ).save(Path(save_dir))
            logger.info("multi-policy global save at leader rollout %d: %s", leader_rollout_id, positions)

    async def _run_policy(model_id: str) -> None:
        is_leader = model_id == leader_model_id
        rounds = (
            range(start_rollout_ids[model_id], num_rollout)
            if is_leader
            else itertools.count(start_rollout_ids[model_id])
        )
        for rollout_id in rounds:
            if run_ended.is_set():
                break
            await hooks.generate(model_id, rollout_id)
            await hooks.train(model_id, rollout_id)
            positions[model_id] = rollout_id
            if (rollout_id + 1) % update_weights_interval == 0:
                await hooks.update_weights(model_id, rollout_id)
            if is_leader:
                if save_interval and should_run_periodic_action(
                    rollout_id, save_interval, num_rollout_per_epoch, num_rollout
                ):
                    await _maybe_save_globally(rollout_id)
                if eval_interval and should_run_periodic_action(
                    rollout_id, eval_interval, num_rollout_per_epoch, num_rollout
                ):
                    async with eval_parker.with_all_parked():
                        await hooks.eval_global(rollout_id)
            else:
                await save_parker.maybe_park_follower()
                await eval_parker.maybe_park_follower()
        if is_leader:
            run_ended.set()

    tasks = [asyncio.create_task(_run_policy(model_id)) for model_id in config.model_ids]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    first_error = next((task.exception() for task in done if not task.cancelled()), None)
    run_ended.set()
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if first_error is not None:
        raise first_error
    return positions

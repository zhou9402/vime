"""Multi-policy training entry point.

Port of miles@ba28991834 train_multi_policy.py. Trains several policies
(declared via --multi-policy-config) against a shared rollout fleet: each
policy gets its own Megatron trainer group and its own updatable serving
server, and drains only its own samples (Sample.trainer_model_id) from the
shared fully-async queue. The leader policy's rollout index bounds the run and
drives global save/eval through Parker barriers.

Run with: python train_multi_policy.py ... --multi-policy-config policies.yaml
"""

import asyncio
import logging

import ray

from vime.ray.placement_group import (
    _create_placement_group,
    allocate_train_group,
    create_rollout_manager,
)
from vime.utils.arguments import parse_args
from vime.utils.cleanup import Cleanup
from vime.utils.multi_policy.loop import run_multi_policy_loop
from vime.utils.multi_policy.utils import (
    TrainerInfo,
    assert_consistent_restore,
    validate_multi_policy_args,
)

logger = logging.getLogger(__name__)


class _RayHooks:
    """MultiPolicyHooks over Ray handles (manager + per-policy trainer groups)."""

    def __init__(self, manager, trainers: dict, update_lock: asyncio.Lock):
        self._manager = manager
        self._trainers = trainers
        # Serialize weight updates so concurrent policies cannot interleave
        # pause/update/continue on their (disjoint) engine sets.
        self._update_lock = update_lock
        self._data_refs = {}

    async def generate(self, model_id: str, rollout_id: int) -> None:
        data_ref = await asyncio.to_thread(
            ray.get, self._manager.generate.remote(rollout_id, model_id=model_id)
        )
        self._data_refs[model_id] = data_ref

    async def train(self, model_id: str, rollout_id: int) -> None:
        refs = self._trainers[model_id].handle.async_train(rollout_id, self._data_refs[model_id])
        await asyncio.to_thread(ray.get, refs)
        await asyncio.to_thread(ray.get, self._manager.training_completed.remote(rollout_id))

    async def update_weights(self, model_id: str, rollout_id: int) -> None:
        async with self._update_lock:
            await asyncio.to_thread(self._trainers[model_id].handle.update_weights)

    async def save_model(self, model_id: str, rollout_id: int) -> None:
        await asyncio.to_thread(self._trainers[model_id].handle.save_model, rollout_id, True)

    async def eval_global(self, leader_rollout_id: int) -> None:
        await asyncio.to_thread(ray.get, self._manager.eval.remote(leader_rollout_id))


def train_multi_policy(args, startup) -> dict[str, int]:
    config = args.multi_policy
    validate_multi_policy_args(args, multi_policy_config=config)
    manager = startup.manager

    trainers: dict[str, TrainerInfo] = {}
    for trainer_config in config.trainers:
        model_id = trainer_config.model_id
        trainer_args = _trainer_args(args, trainer_config)
        num_gpus = trainer_args.actor_num_nodes * trainer_args.actor_num_gpus_per_node
        pg = _create_placement_group(num_gpus)
        group = allocate_train_group(trainer_args, trainer_args.actor_num_nodes,
                                     trainer_args.actor_num_gpus_per_node, pg, role="actor")
        start_ids = group.create(rollout_manager=manager)
        start_rollout_id = start_ids[0] if start_ids else args.start_rollout_id
        if any(sid != start_rollout_id for sid in start_ids or []):
            raise RuntimeError(f"policy {model_id!r} ranks disagree on start_rollout_id: {start_ids}")
        trainers[model_id] = TrainerInfo(model_id=model_id, start_rollout_id=start_rollout_id, handle=group)
        logger.info("policy %s: trainer group ready, start_rollout_id=%d", model_id, start_rollout_id)

    assert_consistent_restore(args, trainers=trainers, leader_model_id=config.leader_model_id)

    # Initial weight publish per policy before any generation.
    for model_id in config.model_ids:
        trainers[model_id].handle.update_weights()
    ray.get(manager.training_ready.remote())

    async def _run():
        return await run_multi_policy_loop(
            config=config,
            hooks=_RayHooks(manager, trainers, asyncio.Lock()),
            num_rollout=args.num_rollout,
            start_rollout_ids={m: t.start_rollout_id for m, t in trainers.items()},
            save_dir=args.save,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            num_rollout_per_epoch=startup.num_rollout_per_epoch,
        )

    return asyncio.run(_run())


def _trainer_args(args, trainer_config):
    from vime.utils.multi_policy.megatron_config import compute_trainer_args

    return compute_trainer_args(args, trainer_config)


def main():
    args, restore_plan = parse_args(return_restore_plan=True)
    if getattr(args, "multi_policy", None) is None:
        raise ValueError("train_multi_policy.py requires --multi-policy-config; use train.py for single policy")
    startup = create_rollout_manager(args, restore_plan=restore_plan)
    try:
        train_multi_policy(startup.args, startup)
    except BaseException:
        startup.close(failed=True)
        raise
    startup.close(failed=False)


if __name__ == "__main__":
    with Cleanup():
        main()

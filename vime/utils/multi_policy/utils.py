"""Multi-policy validation / restore-consistency helpers.

Port of miles@ba28991834 miles/utils/multi_policy/utils.py (validation and
restore-consistency subset). Trainer creation (miles' create_trainers over
BaseWorkerHandle) is not ported here; F09-4 wires vime's allocate_train_group.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vime.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from vime.utils.multi_policy.megatron_config import MultiPolicyConfig

logger = logging.getLogger(__name__)

FULLY_ASYNC_ROLLOUT_FUNCTION = "vime.rollout.fully_async_rollout.generate_rollout_fully_async"


@dataclass(frozen=True)
class TrainerInfo:
    model_id: str
    start_rollout_id: int
    handle: Any  # RayTrainGroup driver handle


def validate_multi_policy_args(args, *, multi_policy_config: MultiPolicyConfig) -> None:
    """Hard requirements for train_multi_policy.py; see also validate_multi_policy_flags."""
    if not multi_policy_config.is_multi_policy:
        raise ValueError(
            f"train_multi_policy.py trains several policies, but --multi-policy-config names "
            f"only {multi_policy_config.model_ids}; run train.py instead"
        )
    if args.colocate:
        raise ValueError("multi policy training does not support --colocate")
    if args.rollout_function_path != FULLY_ASYNC_ROLLOUT_FUNCTION:
        raise ValueError(
            f"multi policy training {multi_policy_config.model_ids} requires fully-async rollout "
            f"({FULLY_ASYNC_ROLLOUT_FUNCTION}), got {args.rollout_function_path}: every other rollout "
            f"mode drives one policy per rollout round"
        )
    if args.use_critic:
        raise ValueError("multi policy training does not support --use-critic")
    if args.debug_rollout_only:
        raise ValueError(
            "multi policy training does not support --debug-rollout-only: it sizes the placement group for one "
            "trainer, while every policy still asks for a slice of its own"
        )
    if getattr(args, "ckpt_step", None) is not None:
        raise ValueError(
            "multi policy training does not support --ckpt-step: every policy stands at an iteration of its own"
        )
    _assert_no_debug_rollout_data_flags(args)


def _assert_no_debug_rollout_data_flags(args) -> None:
    set_flags = [
        flag
        for flag, value in (
            ("--dump-details", getattr(args, "dump_details", None)),
            ("--load-debug-rollout-data", getattr(args, "load_debug_rollout_data", None)),
        )
        if value
    ]
    if set_flags:
        raise ValueError(
            f"multi policy training does not support {set_flags}: those paths are keyed by rollout id alone, "
            f"and every policy counts its own rollouts, so the policies would overwrite and replay each "
            f"other's data."
        )


def assert_consistent_restore(args, *, trainers: dict[str, TrainerInfo], leader_model_id: str) -> None:
    leader_rollout_id = trainers[leader_model_id].start_rollout_id - 1
    if leader_rollout_id < 0:
        fresh = [model_id for model_id, trainer in trainers.items() if trainer.start_rollout_id != 0]
        if fresh:
            raise ValueError(
                f"the leader policy {leader_model_id!r} starts from scratch, but {fresh} restored a "
                f"checkpoint; the data source is global, so it would be replayed from zero into policies "
                f"that already trained on it"
            )
        return

    state_dir = args.load or args.save
    if state_dir is None:
        raise ValueError(
            f"the leader policy {leader_model_id!r} restored rollout {leader_rollout_id} without --load or "
            f"--save, so where the other policies stood cannot be read back"
        )
    state = MultiPolicyCheckpointState.load(Path(state_dir), leader_rollout_id=leader_rollout_id)
    if state is None:
        raise ValueError(
            f"resuming at rollout {leader_rollout_id} but {state_dir} holds no record of where the other "
            f"policies stood; the checkpoint was not written by a multi policy run, so the policies cannot "
            f"be proven to resume at consistent positions"
        )
    if state.leader_model_id != leader_model_id:
        raise ValueError(
            f"the checkpoint was written with {state.leader_model_id!r} as the leader policy, but this run "
            f"makes {leader_model_id!r} the leader; the global rollout index would change meaning"
        )

    recorded = state.rollout_ids
    restored = {model_id: trainer.start_rollout_id - 1 for model_id, trainer in trainers.items()}
    if recorded != restored:
        raise ValueError(
            f"the record at {state_dir} says the policies stood at {recorded}, but this run restored "
            f"{restored}; every policy runs at its own pace, but each one must resume exactly where the "
            f"global checkpoint recorded it"
        )
    logger.info(f"Restored multi policy run at {recorded} (leader {leader_model_id})")


def policy_metric_prefixes(multi_policy_config: MultiPolicyConfig) -> dict[str, list[str]]:
    """Per-policy metric namespaces: ``{model_id}/rollout/step`` and ``{model_id}/train/step``.

    miles registers these with its tracking backend; vime emits prefixed metric
    keys directly (F09-4), so this only centralizes the naming.
    """
    return {
        model_id: [f"{model_id}/rollout/step", f"{model_id}/train/step"]
        for model_id in multi_policy_config.model_ids
    }

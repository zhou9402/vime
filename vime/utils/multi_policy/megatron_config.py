"""Multi-policy trainer configuration (``--multi-policy-config``).

Port of miles@ba28991834 ``miles/backends/megatron_utils/megatron_config.py``
(subset: trainer schema, validation, per-trainer args/checkpoint dirs). The
miles critic synthesis, parser-based override re-verification, and cell
accounting are intentionally not ported; vime keeps its own critic wiring.

The YAML file (or ``base64:`` payload) declares one entry per policy:

    trainers:
      - model_id: solver
        overrides: {lr: 1.0e-6}
      - model_id: verifier
        overrides: {lr: 2.0e-6}

``model_id`` is the cross-cutting join key: the custom generate function stamps
it into ``Sample.trainer_model_id``, the queue shards by it, the per-policy
serving server is bound to it, and metrics are namespaced by it.
"""

import base64
import copy
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

import pydantic
import yaml

ACTOR_ROLE = "actor"
CRITIC_ROLE = "critic"
TrainerRole = Literal["actor", "critic"]
TRAINER_CHECKPOINT_DIRNAME = "trainers"
RESERVED_MODEL_ID = "eval"
TRAINER_ID_MAX_LENGTH = 63
MODEL_ID_PATTERN = re.compile(r"\A[a-z0-9]([-a-z0-9]*[a-z0-9])?\Z")

# Knobs a policy may override relative to the run's base args. Deliberately a
# static whitelist (miles re-verifies against a throwaway Megatron parser; that
# couples config parsing to argv and is not ported).
PER_POLICY_OVERRIDES: frozenset[str] = frozenset(
    {
        "hf_checkpoint",
        "ref_load",
        "optimizer",
        "lr",
        "min_lr",
        "lr_decay_style",
        "lr_warmup_iters",
        "lr_warmup_fraction",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "clip_grad",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "sequence_parallel",
        "use_kl_loss",
        "kl_loss_coef",
        "kl_loss_type",
        "entropy_coef",
        "eps_clip",
        "eps_clip_high",
    }
)

# These must stay identical across policies in v1 (shared rollout batch
# semantics), so overriding them per policy is rejected.
FORBIDDEN_OVERRIDES: frozenset[str] = frozenset(
    {
        "global_batch_size",
        "micro_batch_size",
        "max_tokens_per_gpu",
        "use_dynamic_batch_size",
        "advantage_estimator",
    }
)


class _FrozenStrict(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class RawTrainerConfig(_FrozenStrict):
    model_id: str
    role: TrainerRole = ACTOR_ROLE
    trainer_id: str | None = None
    overrides: dict[str, Any] = {}


class _RawMultiPolicyConfig(_FrozenStrict):
    trainers: list[RawTrainerConfig] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("trainers", "megatron")
    )


class MultiPolicyTrainerConfig(_FrozenStrict):
    trainer_id: str
    model_id: str
    role: TrainerRole
    actor_index: int
    overrides: dict[str, Any]

    @classmethod
    def resolve(cls, raw: RawTrainerConfig, *, actor_index: int) -> "MultiPolicyTrainerConfig":
        return cls(
            trainer_id=raw.trainer_id if raw.trainer_id is not None else f"{raw.model_id}-{raw.role}",
            model_id=raw.model_id,
            role=raw.role,
            actor_index=actor_index,
            overrides=_resolve_overrides(raw.overrides, model_id=raw.model_id),
        )


class MultiPolicyConfig(_FrozenStrict):
    trainers: list[MultiPolicyTrainerConfig]

    @pydantic.model_validator(mode="after")
    def _validate(self) -> "MultiPolicyConfig":
        if not self.trainers:
            raise ValueError("--multi-policy-config must declare at least one trainer")
        if not all(t.role == ACTOR_ROLE for t in self.trainers):
            raise ValueError(
                "--multi-policy-config declaring a critic is not supported; vime wires critics from --use-critic"
            )
        model_ids = self.model_ids
        bad = [i for i in model_ids if MODEL_ID_PATTERN.match(i) is None]
        if bad:
            raise ValueError(f"multi-policy model ids {bad} must match {MODEL_ID_PATTERN.pattern}")
        if RESERVED_MODEL_ID in model_ids:
            raise ValueError(
                f"--multi-policy-config declares the model id {RESERVED_MODEL_ID!r}, "
                "which names the shared eval rollouts"
            )
        actor_ids = [t.model_id for t in self.trainers]
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError(
                f"--multi-policy-config declares several actors for the same model id ({actor_ids}); "
                "trainers are keyed by model id"
            )
        trainer_ids = [t.trainer_id for t in self.trainers]
        if len(set(trainer_ids)) != len(trainer_ids):
            raise ValueError(f"--multi-policy-config trainer ids must be unique, got {trainer_ids}")
        too_long = [t for t in trainer_ids if len(t) > TRAINER_ID_MAX_LENGTH]
        if too_long:
            raise ValueError(f"multi-policy trainer ids {too_long} exceed {TRAINER_ID_MAX_LENGTH} chars")
        return self

    @property
    def model_ids(self) -> list[str]:
        return list(dict.fromkeys(t.model_id for t in self.trainers))

    @property
    def leader_model_id(self) -> str:
        return self.trainers[0].model_id

    @property
    def is_multi_policy(self) -> bool:
        return len(self.model_ids) > 1

    def get(self, model_id: str) -> MultiPolicyTrainerConfig:
        for trainer in self.trainers:
            if trainer.model_id == model_id:
                return trainer
        raise KeyError(f"Unknown trainer model id {model_id!r}, known ids: {self.model_ids}")


def resolve_config_text(value: str) -> str:
    """Read a ``--multi-policy-config`` value: a YAML file path or ``base64:`` payload."""
    if value.startswith("base64:"):
        return base64.b64decode(value[len("base64:") :]).decode("utf-8")
    return Path(value).read_text(encoding="utf-8")


def load_multi_policy_config(value: str | None) -> MultiPolicyConfig | None:
    """Parse ``--multi-policy-config``; ``None`` (absent) means a single-policy run."""
    if value is None:
        return None
    raw = _RawMultiPolicyConfig.model_validate(yaml.safe_load(resolve_config_text(value)))
    return MultiPolicyConfig(
        trainers=[MultiPolicyTrainerConfig.resolve(t, actor_index=i) for i, t in enumerate(raw.trainers)]
    )


def compute_trainer_checkpoint_dir(*, base_dir: str | None, trainer_id: str) -> str | None:
    if base_dir is None:
        return None
    return str(Path(base_dir) / TRAINER_CHECKPOINT_DIRNAME / trainer_id)


def compute_trainer_args(args: Namespace, trainer: MultiPolicyTrainerConfig) -> Namespace:
    """Deep-copy the run args and apply one trainer's identity and whitelisted overrides."""
    ans = copy.deepcopy(args)
    ans.trainer_id = trainer.trainer_id
    ans.trainer_model_id = trainer.model_id
    ans.trainer_role = trainer.role
    ans.trainer_actor_index = trainer.actor_index
    for key, value in trainer.overrides.items():
        if not hasattr(ans, key):
            raise ValueError(
                f"--multi-policy-config trainer {trainer.trainer_id!r} overrides {key!r}, "
                "which this run's argument parser does not know"
            )
        setattr(ans, key, value)
    for attr in ("save", "load", "save_hf"):
        if hasattr(ans, attr):
            setattr(
                ans, attr, compute_trainer_checkpoint_dir(base_dir=getattr(ans, attr), trainer_id=trainer.trainer_id)
            )
    if trainer.role == CRITIC_ROLE:
        ans.loss_type = "value_loss"
    return ans


def validate_multi_policy_flags(args: Namespace) -> None:
    """Called from ``vime_validate_args``; no-op unless --multi-policy-config is set."""
    config = load_multi_policy_config(getattr(args, "multi_policy_config", None))
    if config is None:
        return
    if not config.is_multi_policy:
        raise ValueError("--multi-policy-config declares a single policy; use train.py instead")
    if getattr(args, "colocate", False):
        raise ValueError(
            "multi-policy training forbids --colocate in v1: per-policy weight updates would fight over "
            "the shared GPU set"
        )
    if getattr(args, "use_critic", False):
        raise ValueError("multi-policy training does not support --use-critic")
    if getattr(args, "debug_rollout_only", False):
        raise ValueError("multi-policy training forbids --debug-rollout-only")
    args.multi_policy = config


def _resolve_overrides(overrides: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    unknown = set(overrides) - PER_POLICY_OVERRIDES
    forbidden = set(overrides) & FORBIDDEN_OVERRIDES
    if forbidden:
        raise ValueError(
            f"--multi-policy-config model {model_id!r} overrides {sorted(forbidden)}; "
            "every policy has to share the run's rollout batch arguments"
        )
    if unknown:
        raise ValueError(
            f"--multi-policy-config model {model_id!r} overrides {sorted(unknown)}, which are not in the "
            f"per-policy whitelist; extend PER_POLICY_OVERRIDES in {__file__} to allow them"
        )
    return dict(overrides)

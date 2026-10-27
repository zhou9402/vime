from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vime.utils.types import Sample

if TYPE_CHECKING:
    from argparse import Namespace

    from vime.data.data_source import DataSource
    from vime.data.transport import DiskPayloadRef
    from vime.rollout.vllm_rollout import GenerateState


@dataclass
class RolloutFnTrainOutput:
    # Accept Sample groups or a straw manifest referencing the selected batch.
    samples: list[list[Sample]] | DiskPayloadRef
    metrics: dict[str, Any] = None
    on_accepted: Callable[[], None] | None = field(default=None, repr=False, compare=False)
    sample_refs: list[list[DiskPayloadRef]] | None = field(default=None, repr=False, compare=False)


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    # engine weight version, None before the first weight update
    weight_version: int | None = None
    # which policy model asked for this rollout; None when the run trains one policy
    trainer_model_id: str | None = None

    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    invocation_id: str = field(default_factory=lambda: uuid4().hex)
    generate_state: GenerateState | None = None
    weight_version: str | None = None
    hf_dir: str | None = None

    @property
    def evaluation(self):
        return True


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


class BaseRolloutFn(abc.ABC):
    def __init__(self, input: RolloutFnConstructorInput) -> None:
        self.constructor_input = input

    @abc.abstractmethod
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> None:
        return None

    async def dispose(self) -> None:
        return None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    from vime.data.transport import DiskPayloadRef, RawRolloutRef

    output = fn(*args, **kwargs, evaluation=evaluation)
    if isinstance(output, RawRolloutRef):
        if evaluation:
            raise TypeError("An accepted training collection cannot be used as evaluation output")
        return output

    if evaluation:
        if isinstance(output, RolloutFnEvalOutput):
            return output
        if isinstance(output, dict):
            return RolloutFnEvalOutput(data=output)
    else:
        if isinstance(output, RolloutFnTrainOutput):
            if not isinstance(output.samples, (list, DiskPayloadRef)):
                raise TypeError("Training output must contain Samples or a supported rollout reference")
            return output
        if isinstance(output, (list, DiskPayloadRef)):
            return RolloutFnTrainOutput(samples=output)
    raise TypeError(f"Unsupported rollout output: {type(output).__name__}")


def finalize_rollout_groups(args, rollout_id, groups, metrics=None, *, controller=None):
    """Order selected groups, run the batch hook once, and publish the batch."""
    from vime.data.transport import RolloutGroupRef, load_rollout_samples, pack_rollout_payload
    from vime.utils.misc import load_function

    groups.sort(
        key=lambda group: (group.index if isinstance(group, RolloutGroupRef) else next(iter_samples(group)).index) or 0
    )
    dropped = set()
    incoming = set()
    if args.rollout_data_transport != "straw":
        incoming = {
            sample._queue_receipt["position"] for sample in iter_samples(groups) if hasattr(sample, "_queue_receipt")
        }
    if args.rollout_sample_filter_path is not None:
        # A custom batch hook can mutate arbitrary Sample fields. Its result
        # must be serialized again; without a hook, retain existing group refs.
        groups = load_rollout_samples(groups)
        incoming = {
            sample._queue_receipt["position"] for sample in iter_samples(groups) if hasattr(sample, "_queue_receipt")
        }
        load_function(args.rollout_sample_filter_path)(args, groups)
        selected = {
            sample._queue_receipt["position"] for sample in iter_samples(groups) if hasattr(sample, "_queue_receipt")
        }
        dropped = incoming - selected
    samples = pack_rollout_payload(groups, args, rollout_id) if args.rollout_data_transport == "straw" else groups
    if dropped and args.rollout_data_transport == "straw":
        import ray

        decision = pack_rollout_payload(
            {"positions": sorted(dropped), "reason": "rollout_sample_filter", "output": samples}, args, rollout_id
        )
        ray.get(controller.record_dispositions.remote(decision.manifest))
    elif incoming and args.rollout_data_transport != "straw":
        import ray

        decision = pack_rollout_payload(
            {"positions": sorted(incoming), "reason": "legacy object-store delivery after batch filter"},
            args,
            rollout_id,
        )
        ray.get(controller.record_dispositions.remote(decision.manifest))
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)


def iter_samples(value):
    """Visit Sample leaves without changing custom generation's nested shape."""
    if isinstance(value, Sample):
        yield value
    else:
        for child in value:
            yield from iter_samples(child)

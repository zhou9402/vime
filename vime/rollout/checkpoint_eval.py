"""Run eval on a backend of your own.

Subclass ``CheckpointEvalFn`` and point ``--eval-function-path`` at it. Every eval
point hands you a finished HF checkpoint directory and takes back results; what
happens in between is yours — a vLLM server you launched, a remote scoring
service, a black box with an HTTP API.

The trainer exports the snapshot, dispatches it without blocking the training loop,
logs the result at the snapshot's step, and reclaims the directory afterwards. Raise
``EvalSkip(reason)`` to skip a point with attribution instead of counting as a crash.

Requires a snapshot source (``--eval-hf-dir``) and is mutually
exclusive with the dedicated eval fleet (``--eval-num-gpus``), which already owns
weight delivery. ``examples/eval/external_vllm_eval_fn.py`` is a working
implementation.

Port of miles@ba28991834 miles/rollout/checkpoint_eval.py.
"""

import abc
import copy
import inspect
import logging
from collections.abc import Sequence

from vime.rollout.base_types import BaseRolloutFn, RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnInput
from vime.utils.misc import load_function

__all__ = [
    "retarget_args",
    "EvalSkip",
    "CheckpointEvalFn",
    "is_checkpoint_eval_fn",
]

logger = logging.getLogger(__name__)


def retarget_args(*, args, router_ip: str, router_port: int, engine_gpu_counts: Sequence[int]):
    """Shallow-copy ``args`` with the router address swapped for eval.

    vLLM generate functions read ``args.vllm_router_ip`` / ``args.vllm_router_port``
    on every request, so a retargeted copy runs the standard eval path against a
    different set of engines unchanged. ``engine_gpu_counts`` is recorded for
    callers that size their own concurrency off the fleet shape.
    """
    retargeted = copy.copy(args)
    retargeted.vllm_router_ip = router_ip
    retargeted.vllm_router_port = router_port
    retargeted.eval_engine_gpu_counts = list(engine_gpu_counts)
    return retargeted


class EvalSkip(Exception):
    """Raise from a ``CheckpointEvalFn`` to skip this eval point with an attributable
    reason (logged as ``eval/skipped_{reason}``) instead of counting as a crash."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CheckpointEvalFn(BaseRolloutFn, abc.ABC):
    """Contract for eval backends that consume HF checkpoint snapshots.

    ``__init__`` takes a ``RolloutFnConstructorInput`` and prepares the backend —
    launch a server, attach to one, open a client. ``evaluate_checkpoint`` then runs
    one eval point against the snapshot at ``checkpoint_dir``.
    """

    @abc.abstractmethod
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput: ...

    async def __call__(self, input: RolloutFnInput) -> RolloutFnEvalOutput:
        assert input.evaluation, "CheckpointEvalFn only serves eval; keep the train fn on --rollout-function-path"
        assert input.hf_dir is not None, (
            "no snapshot was dispatched — checkpoint eval fns require a snapshot source (--eval-hf-dir)"
        )
        return await self.evaluate_checkpoint(input.hf_dir, input)

    async def dispose(self) -> None:  # noqa: B027 — optional hook, deliberately a no-op default
        """Tear down anything launched in ``__init__``. Called when the eval runner shuts down."""


def is_checkpoint_eval_fn(eval_function_path: str | None) -> bool:
    """Whether ``--eval-function-path`` points at a black-box checkpoint backend."""
    if eval_function_path is None:
        return False
    try:
        eval_fn = load_function(eval_function_path)
    except (ImportError, AttributeError):
        # Existence of the path is enforced when the rollout manager loads it;
        # here an unimportable path just means "not a checkpoint eval fn".
        return False
    return inspect.isclass(eval_fn) and issubclass(eval_fn, CheckpointEvalFn)

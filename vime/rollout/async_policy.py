"""Disposition of fully-async rollout groups that training will not consume.

Port of miles's ``--async-unused-samples-handler`` (miles @ ba28991834,
``miles/rollout/fully_async_rollout.py`` ``_recycle``/``_drop_unused``), built
on vime's queue leases instead of miles's in-memory retry deque:

* ``retry`` requeues the group into the same data source
  (``QueueReader.add_samples`` / the in-memory buffer). ABORTED groups keep
  their partial content (vime's continuation semantics); STALE groups are
  reset to their prompt first, otherwise they would be re-delivered as ready
  groups and re-judged stale forever.
* ``drop`` discards the group through the transport's release/reject path.

When the flag is unset the default is reason-dependent: ABORTED groups retry
(vime's historical behavior), STALE groups drop (miles parity).
"""

from __future__ import annotations

import logging
from enum import Enum

from vime.data.transport import discard_rollout_group
from vime.rollout.base_types import iter_samples
from vime.utils.types import Sample

logger = logging.getLogger("vime.rollout.unused_samples")


class UnusedReason(str, Enum):
    ABORTED = "aborted"
    STALE = "stale"


_DEFAULT_PER_REASON = {UnusedReason.ABORTED: "retry", UnusedReason.STALE: "drop"}


def unused_samples_handler(args, reason: UnusedReason) -> str:
    handler = getattr(args, "async_unused_samples_handler", None)
    if handler is None:
        return _DEFAULT_PER_REASON[reason]
    return handler


def reset_group_for_retry(group) -> None:
    """Strip generated content so a group re-enters the queue as a fresh prompt."""
    for sample in iter_samples(group):
        sample.tokens = sample.tokens[: len(sample.tokens) - sample.response_length]
        sample.response = ""
        sample.response_length = 0
        sample.reward = None
        sample.loss_mask = None
        sample.rollout_log_probs = None
        sample.weight_versions = []
        sample.status = Sample.Status.PENDING


def prompt_retry_stubs(group) -> list[Sample]:
    """Prompt-only copies of a group's samples.

    Used when the group is payload-backed (a straw ``RolloutGroupRef``): the
    payload must be loaded to recover prompts, but the returned stubs keep
    only what regeneration needs (prompt, label, multimodal inputs, index and
    queue source positions), so the heavy generated artifacts (tokens,
    logprobs, tensors) are dropped with the payload reference immediately.
    """
    stubs = []
    for sample in iter_samples(group):
        stub = Sample(
            index=sample.index,
            group_index=sample.group_index,
            prompt=sample.prompt,
            label=sample.label,
            metadata=sample.metadata,
            multimodal_inputs=sample.multimodal_inputs,
        )
        for attr in ("_queue_source_positions", "_queue_branch"):
            if hasattr(sample, attr):
                setattr(stub, attr, getattr(sample, attr))
        stubs.append(stub)
    return stubs


def handle_unused_group(args, group, reason: UnusedReason, *, data_source=None, controller=None) -> str:
    """Apply the configured handler; returns the disposition (``retry``/``drop``)."""
    handler = unused_samples_handler(args, reason)
    if handler == "retry":
        if data_source is None or not hasattr(data_source, "add_samples"):
            raise ValueError(f"--async-unused-samples-handler=retry needs a requeueable data source ({reason})")
        if reason is UnusedReason.STALE:
            reset_group_for_retry(group)
        data_source.add_samples([group])
        return "retry"
    discard_rollout_group(group, args, reason.value, controller=controller)
    return "drop"


def async_capacity_limit(args) -> int | None:
    """Group-level look-ahead cap from ``--async-capacity-factor`` (None = off).

    Bounds how far generation may run ahead of training: completed + in-flight
    groups never exceed ``int(factor * rollout_batch_size)``. Callers that can
    compute their concurrency floor must assert the cap is at least
    ``max(rollout_batch_size, floor)`` so warmup cannot starve.
    """
    factor = getattr(args, "async_capacity_factor", None)
    if factor is None:
        return None
    return int(factor * args.rollout_batch_size)

"""Submission pacing for fully-async rollout (group- vs sample-level).

Port of miles@ba28991834 ``miles/rollout/submission_scheduler.py``. The
module is engine-agnostic; vime's worker loop polls instead of waiting on an
asyncio event, so the sample-level scheduler keeps only the accounting (no
wakeup machinery — the loop re-checks capacity every ``poll_interval``).

``group`` (miles ``GroupLevelSubmission``): a slot frees when a whole group
task returns. ``sample`` (miles ``SampleBackfillSubmission``): each finished
sample frees capacity, so a replacement group is admitted once
``n_samples_per_prompt`` samples have completed — sample-level backfill.
Groups are still submitted whole; a group is never split.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vime.rollout.submission_scheduler")

__all__ = ["GroupLevelSubmission", "SampleBackfillSubmission", "make_submission_scheduler"]


class GroupLevelSubmission:
    """Capacity frees only when a whole group task returns."""

    sample_done_callback = None  # no per-sample accounting

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        return pending_groups < group_budget

    def on_submit(self, groups) -> None:
        pass


class SampleBackfillSubmission:
    """A slot is one sample; capacity frees as individual samples complete."""

    def __init__(self, group_size: int):
        self.group_size = group_size
        self.samples_in_flight = 0

    def sample_done_callback(self) -> None:
        self.samples_in_flight -= 1

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        if pending_groups == 0 and self.samples_in_flight > 0:
            # A group returned without its sample callbacks running (e.g. the
            # engine aborted before any task started). Reset rather than
            # deadlocking on a leaked count.
            logger.warning(
                "fully-async: %d leaked in-flight sample(s) with no pending group; resetting",
                self.samples_in_flight,
            )
            self.samples_in_flight = 0
        # There is always room to submit one whole group; groups are never split.
        return self.samples_in_flight + self.group_size <= group_budget * self.group_size

    def on_submit(self, groups) -> None:
        self.samples_in_flight += sum(len(group) for group in groups)


def make_submission_scheduler(args, *, default: str):
    granularity = getattr(args, "rollout_submission_granularity", None) or default
    if granularity == "group":
        return GroupLevelSubmission()
    if granularity == "sample":
        return SampleBackfillSubmission(args.n_samples_per_prompt)
    raise AssertionError(f"Unknown rollout submission granularity: {granularity}")

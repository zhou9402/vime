"""V4: deterministic async-scheduling simulation verifier.

Fake engine + fixed seed, no threads, no GPU, no Ray. The harness models the
fully-async loop (submit -> generate -> drain -> train) at step granularity
and asserts the two invariants the B5 staleness policy must satisfy:

  (a) staleness bound — with a max-staleness policy installed, every group
      handed to training has ``current_weight_version - oldest_version <=
      max_weight_staleness``;
  (b) sample conservation — at every step, ``submitted == trained + dropped +
      inflight + recycled_pending`` (no silent loss).

The policy is a pluggable hook. Here it is a reference oracle implemented in
this file; B5a plugs the real ``vime.utils.staleness.group_staleness``-based
policy into the same harness (see ``tests/test_async_staleness.py``), so the
implementation is graded against invariants the oracle already satisfies.

Run: ``python -m pytest tests/test_async_staleness_simulation.py -x -q``
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

NUM_GPUS = 0


@dataclass
class SimSample:
    prompt_index: int
    weight_version: int  # engine version when the sample finished generating


@dataclass
class SimGroup:
    prompt_index: int
    samples: list[SimSample] = field(default_factory=list)
    submit_step: int = 0
    finish_step: int = 0

    @property
    def done(self) -> bool:
        return len(self.samples) == self.expected

    expected: int = 1


class FakeEngine:
    """Fixed-seed generation latencies and scheduled weight bumps."""

    def __init__(self, seed: int, min_latency: int, max_latency: int, update_interval: int):
        self.rng = random.Random(seed)
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.update_interval = update_interval
        self.weight_version = 0

    def latency(self) -> int:
        return self.rng.randint(self.min_latency, self.max_latency)

    def maybe_bump(self, step: int) -> None:
        if self.update_interval and step and step % self.update_interval == 0:
            self.weight_version += 1


class ReferenceStalenessPolicy:
    """Oracle: drop groups whose oldest version lags by more than the limit.

    ``handler`` is ``"drop"`` (discard) or ``"retry"`` (recycle the prompt
    for regeneration), mirroring miles's ``--async-unused-samples-handler``.
    Unversioned groups are always kept, matching the miles semantics.
    """

    def __init__(self, max_staleness: int | None, handler: str = "drop"):
        assert handler in ("drop", "retry")
        self.max_staleness = max_staleness
        self.handler = handler

    def judge(self, group: SimGroup, current_weight_version: int) -> str:
        """Return ``"keep"``, ``"drop"`` or ``"retry"``."""
        if self.max_staleness is None:
            return "keep"
        oldest = min(s.weight_version for s in group.samples)
        if current_weight_version - oldest > self.max_staleness:
            return self.handler
        return "keep"


@dataclass
class SimResult:
    trained: list[SimGroup]
    dropped: list[SimGroup]
    retried_prompts: list[int]
    max_seen_staleness: int
    steps: int


def run_simulation(
    *,
    seed: int = 0,
    num_prompts: int = 24,
    group_size: int = 2,
    batch_size: int = 3,
    concurrency: int = 4,
    min_latency: int = 1,
    max_latency: int = 6,
    update_interval: int = 4,
    policy: ReferenceStalenessPolicy | None = None,
    max_steps: int = 2000,
) -> SimResult:
    engine = FakeEngine(seed, min_latency, max_latency, update_interval)
    policy = policy or ReferenceStalenessPolicy(None)

    pending_prompts = list(range(num_prompts))
    recycled: list[int] = []  # retry-handler prompts waiting for resubmission
    inflight: list[SimGroup] = []
    completed: list[SimGroup] = []  # generated, waiting to be drained
    trained: list[SimGroup] = []
    dropped: list[SimGroup] = []
    recycled_groups: list[SimGroup] = []  # group instances disposed into retry
    retried_prompts: list[int] = []

    submitted = 0
    trained_prompts: set[int] = set()
    max_seen_staleness = 0
    step = 0

    while step < max_steps:
        engine.maybe_bump(step)

        # Completions land; the engine stamps each sample with the version
        # live at finish time.
        for group in [g for g in inflight if g.finish_step <= step]:
            group.samples = [
                SimSample(group.prompt_index, engine.weight_version) for _ in range(group.expected)
            ]
            inflight.remove(group)
            completed.append(group)

        # Drain one full batch at a time; groups that finish early wait in
        # ``completed`` for their batch-mates and accrue staleness meanwhile.
        # The policy judges at commit time, against the version the trainer
        # is about to train on.
        while len(completed) >= batch_size:
            batch = [completed.pop(0) for _ in range(batch_size)]
            for group in batch:
                verdict = policy.judge(group, engine.weight_version)
                if verdict == "keep":
                    staleness = engine.weight_version - min(s.weight_version for s in group.samples)
                    max_seen_staleness = max(max_seen_staleness, staleness)
                    trained.append(group)
                    trained_prompts.add(group.prompt_index)
                elif verdict == "retry":
                    # The group instance is terminally disposed; its prompt
                    # resubmits later as a fresh instance.
                    recycled_groups.append(group)
                    recycled.append(group.prompt_index)
                    retried_prompts.append(group.prompt_index)
                else:
                    dropped.append(group)
        if len(trained_prompts) >= num_prompts:
            break
        if not (pending_prompts or recycled or inflight) and len(completed) < batch_size:
            # No fuel left to fill a whole batch: judge the partial tail.
            # Retry verdicts recycle their prompt and the loop continues.
            while completed:
                group = completed.pop(0)
                verdict = policy.judge(group, engine.weight_version)
                if verdict == "keep":
                    staleness = engine.weight_version - min(s.weight_version for s in group.samples)
                    max_seen_staleness = max(max_seen_staleness, staleness)
                    trained.append(group)
                    trained_prompts.add(group.prompt_index)
                elif verdict == "retry":
                    recycled_groups.append(group)
                    recycled.append(group.prompt_index)
                    retried_prompts.append(group.prompt_index)
                else:
                    dropped.append(group)
            if not (pending_prompts or recycled):
                break

        # Conservation must hold at every step: every submitted group instance
        # is either trained, dropped, disposed into retry, or still moving.
        accounted = len(trained) + len(dropped) + len(recycled_groups) + len(inflight) + len(completed)
        assert submitted == accounted, (
            f"step {step}: submitted={submitted} trained={len(trained)} dropped={len(dropped)} "
            f"recycled={len(recycled_groups)} inflight={len(inflight)} completed={len(completed)}"
        )

        # Submit replacements while slots are free. Recycled prompts go first.
        while len(inflight) < concurrency and (recycled or pending_prompts):
            prompt_index = recycled.pop(0) if recycled else pending_prompts.pop(0)
            group = SimGroup(
                prompt_index=prompt_index,
                expected=group_size,
                submit_step=step,
                finish_step=step + engine.latency(),
            )
            inflight.append(group)
            submitted += 1

        step += 1

    assert step < max_steps, "simulation did not terminate"
    return SimResult(trained, dropped, retried_prompts, max_seen_staleness, step)


@pytest.mark.unit
def test_conservation_without_policy():
    result = run_simulation(policy=ReferenceStalenessPolicy(None))
    assert len(result.trained) == 24
    assert result.dropped == []
    assert result.retried_prompts == []


# Schedule that actually produces staleness: slow generations (2-8 steps),
# frequent weight bumps (every 2 steps), batches of 4 filled from a pool of
# 3 in-flight groups. Without it the bound assertions below would be vacuous.
STALE_SCHEDULE = dict(update_interval=2, min_latency=2, max_latency=8, batch_size=4, concurrency=3)


@pytest.mark.unit
def test_staleness_bound_with_drop_handler():
    policy = ReferenceStalenessPolicy(max_staleness=2, handler="drop")
    result = run_simulation(policy=policy, **STALE_SCHEDULE)
    assert result.max_seen_staleness <= 2
    assert result.retried_prompts == []
    assert len(result.dropped) > 0
    # Non-vacuity: the same schedule without a policy breaches the bound.
    assert run_simulation(policy=None, **STALE_SCHEDULE).max_seen_staleness > 2


@pytest.mark.unit
def test_staleness_bound_with_retry_handler_recycles_prompts():
    policy = ReferenceStalenessPolicy(max_staleness=1, handler="retry")
    result = run_simulation(policy=policy, **STALE_SCHEDULE)
    assert result.max_seen_staleness <= 1
    assert result.dropped == []
    assert len(result.retried_prompts) > 0
    # Every prompt is eventually trained exactly once (retries included).
    assert len({g.prompt_index for g in result.trained}) == 24


@pytest.mark.unit
def test_deterministic_per_seed():
    a = run_simulation(seed=7, policy=ReferenceStalenessPolicy(2, "drop"))
    b = run_simulation(seed=7, policy=ReferenceStalenessPolicy(2, "drop"))
    assert [g.prompt_index for g in a.trained] == [g.prompt_index for g in b.trained]
    assert [g.prompt_index for g in a.dropped] == [g.prompt_index for g in b.dropped]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

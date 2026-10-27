"""Deterministic 2-policy simulation of the vime multi-policy training contract.

F09-0 verifier. Runs entirely in-process with fake producers/trainers (no Ray,
no GPU, no wall-clock dependence) and pins the behavioral invariants that the
real implementation (F09-1..4) must satisfy:

a) policy purity: every batch a trainer receives contains only samples stamped
   with that policy's ``trainer_model_id``;
b) conservation: submitted == trained + dropped + buffered, per policy — no
   sample is silently lost;
c) leader-bounded termination: when the leader finishes ``num_rollout`` rounds
   the follower is cancelled and the run ends without deadlock;
d) checkpoint state: a save at leader rollout N writes
   ``multi_policy_state/N.json`` recording every policy's position, and restore
   consistency is enforced (leader id in the file must match the filename).

Queue routing note: straw's ``Coordinator.acquire`` has no metadata predicate
(selectors are task_ids / task_prefix / control only), so per-policy routing in
the real port uses the consumer namespace ``training:{model_id}``; this sim
models that as one FIFO buffer per policy.
"""

import asyncio
import json
import random
from dataclasses import dataclass, field

import pytest

LEADER = "solver"
FOLLOWER = "verifier"
POLICIES = (LEADER, FOLLOWER)


@dataclass
class SimSample:
    group_id: int
    trainer_model_id: str


class PolicyBuffer:
    """One FIFO per policy; the real port's DefaultMultiDataBuffer analogue."""

    def __init__(self, policies):
        self._queues = {p: asyncio.Queue() for p in policies}
        self.dropped = {p: 0 for p in policies}

    def put(self, sample: SimSample):
        assert sample.trainer_model_id in self._queues, (
            f"sample with unknown/None trainer_model_id {sample.trainer_model_id!r}: "
            "the generate function must stamp every sample"
        )
        self._queues[sample.trainer_model_id].put_nowait(sample)

    async def get_batch(self, policy: str, num: int) -> list[SimSample]:
        batch = [await self._queues[policy].get()]
        while len(batch) < num:
            batch.append(await self._queues[policy].get())
        return batch

    def buffered(self, policy: str) -> int:
        return self._queues[policy].qsize()


@dataclass
class SimState:
    submitted: dict = field(default_factory=lambda: {p: 0 for p in POLICIES})
    trained: dict = field(default_factory=lambda: {p: 0 for p in POLICIES})
    positions: dict = field(default_factory=lambda: {p: 0 for p in POLICIES})
    run_ended: asyncio.Event = field(default_factory=asyncio.Event)


def _write_state_file(save_dir, leader_model_id, rollout_ids):
    state_dir = save_dir / "multi_policy_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{rollout_ids[leader_model_id]}.json"
    path.write_text(
        json.dumps({"leader_model_id": leader_model_id, "rollout_ids": rollout_ids}),
        encoding="utf-8",
    )
    return path


def _read_state_file(save_dir, leader_rollout_id):
    path = save_dir / "multi_policy_state" / f"{leader_rollout_id}.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    recorded = state["rollout_ids"][state["leader_model_id"]]
    assert recorded == leader_rollout_id, (
        f"{path} records leader {state['leader_model_id']!r} at rollout {recorded}, "
        f"but it is the record of rollout {leader_rollout_id}"
    )
    return state


async def _producer(buffer: PolicyBuffer, state: SimState, num_groups: int, seed: int):
    """Each prompt group fans out samples stamped for a random subset of policies."""
    rng = random.Random(seed)
    for group_id in range(num_groups):
        targets = [p for p in POLICIES if rng.random() < 0.8] or [LEADER]
        for policy in targets:
            for _ in range(rng.randint(1, 3)):
                buffer.put(SimSample(group_id=group_id, trainer_model_id=policy))
                state.submitted[policy] += 1
        await asyncio.sleep(0)
    # sentinel: enough no-op rounds so consumers can drain after production ends


async def _run_policy(
    policy: str,
    *,
    is_leader: bool,
    num_rollout: int,
    batch_size: int,
    buffer: PolicyBuffer,
    state: SimState,
    save_dir=None,
    save_interval: int | None = None,
    trained_batches: list | None = None,
):
    rounds = range(num_rollout) if is_leader else iter(int, 1)  # follower: unbounded
    for rollout_id in rounds:
        if state.run_ended.is_set():
            break
        batch = await buffer.get_batch(policy, batch_size)
        assert all(s.trainer_model_id == policy for s in batch)  # invariant (a)
        if trained_batches is not None:
            trained_batches.append((policy, rollout_id, batch))
        state.trained[policy] += len(batch)
        state.positions[policy] = rollout_id
        await asyncio.sleep(0)  # train step
        if is_leader and save_dir is not None and save_interval and (rollout_id + 1) % save_interval == 0:
            _write_state_file(save_dir, LEADER, dict(state.positions))
    if is_leader:
        state.run_ended.set()


def _run_sim(*, num_groups=40, num_rollout=6, batch_size=4, seed=0, save_dir=None, save_interval=None):
    async def main():
        buffer = PolicyBuffer(POLICIES)
        state = SimState()
        producer = asyncio.create_task(_producer(buffer, state, num_groups, seed))
        trained_batches = []
        tasks = [
            asyncio.create_task(
                _run_policy(
                    LEADER,
                    is_leader=True,
                    num_rollout=num_rollout,
                    batch_size=batch_size,
                    buffer=buffer,
                    state=state,
                    save_dir=save_dir,
                    save_interval=save_interval,
                    trained_batches=trained_batches,
                )
            ),
            asyncio.create_task(
                _run_policy(
                    FOLLOWER,
                    is_leader=False,
                    num_rollout=num_rollout,
                    batch_size=batch_size,
                    buffer=buffer,
                    state=state,
                    trained_batches=trained_batches,
                )
            ),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await producer
        return buffer, state, trained_batches

    return asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_policy_purity_and_conservation():
    buffer, state, trained_batches = _run_sim(num_groups=60, num_rollout=8, batch_size=4, seed=42)
    # (a) every trained batch is single-policy
    for policy, _rid, batch in trained_batches:
        assert batch and all(s.trainer_model_id == policy for s in batch)
    # both policies actually trained
    assert all(state.trained[p] > 0 for p in POLICIES)
    # (b) conservation per policy: submitted == trained + dropped + buffered
    for p in POLICIES:
        assert state.submitted[p] == state.trained[p] + buffer.dropped[p] + buffer.buffered(p)


def test_leader_termination():
    # (c) leader finishing num_rollout ends the run; sim completing at all
    # (inside wait_for timeout) proves the follower was cancelled, not hung.
    _buffer, state, _batches = _run_sim(num_groups=30, num_rollout=4, batch_size=2, seed=7)
    assert state.run_ended.is_set()
    assert state.positions[LEADER] == 3  # leader bounded at num_rollout - 1


def test_state_file_roundtrip_and_tamper_detection(tmp_path):
    _buffer, state, _batches = _run_sim(
        num_groups=60, num_rollout=6, batch_size=4, seed=3, save_dir=tmp_path, save_interval=3
    )
    # (d) save at leader rollouts 2 and 5 -> files 2.json and 5.json
    state = _read_state_file(tmp_path, leader_rollout_id=5)
    assert state["leader_model_id"] == LEADER
    assert set(state["rollout_ids"]) == set(POLICIES)
    assert state["rollout_ids"][LEADER] == 5
    # restore consistency: the file's leader id must match the filename id
    path = tmp_path / "multi_policy_state" / "5.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["rollout_ids"][LEADER] = 99
    bad_dir = tmp_path / "bad"
    _write_state_file(bad_dir, LEADER, tampered["rollout_ids"])
    # leader id 99 lands in 99.json; reading it as 99 is consistent, but a
    # hand-renamed/mismatched file must fail the assertion
    (bad_dir / "multi_policy_state" / "5.json").write_text(
        json.dumps({"leader_model_id": LEADER, "rollout_ids": {LEADER: 99, FOLLOWER: 4}}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _read_state_file(bad_dir, leader_rollout_id=5)
    # missing file -> None (fresh start), not an error
    assert _read_state_file(tmp_path, leader_rollout_id=12345) is None

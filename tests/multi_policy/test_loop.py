"""F09-0/F09-4: the multi-policy sim contract against the REAL loop.

Drives vime/utils/multi_policy/loop.py (the production train_multi_policy
loop) with fake engines: a shared sharded buffer (the queue's per-policy
analogue) and fake trainers. Asserts the F09-0 invariants end-to-end:

a) policy purity: every trained batch contains only that policy's samples;
b) conservation: submitted == trained + buffered, per policy (nothing lost);
c) leader-bounded termination: followers are cancelled when the leader ends;
d) checkpoint state: leader-driven global saves write multi_policy_state/N.json
   with every policy's position, and the record round-trips.
"""

import asyncio
import random

import pytest

from vime.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from vime.utils.multi_policy.loop import run_multi_policy_loop
from vime.utils.multi_policy.megatron_config import MultiPolicyConfig

LEADER = "solver"
FOLLOWER = "verifier"
POLICIES = (LEADER, FOLLOWER)


def _config():
    return MultiPolicyConfig(
        trainers=[
            dict(trainer_id=f"{m}-actor", model_id=m, role="actor", actor_index=i, overrides={})
            for i, m in enumerate(POLICIES)
        ]
    )


class FakeEngines:
    """Shared 'serving fleet': produces groups stamped per policy, one FIFO per policy.

    ``generate_calls`` counts production passes; a correct drain delivers each
    finished group to its policy without regenerating or re-queueing it.
    """

    def __init__(self, *, num_groups, seed, per_policy_per_group=None):
        self.queues = {p: asyncio.Queue() for p in POLICIES}
        self.submitted = {p: 0 for p in POLICIES}
        self.generate_calls = 0
        self._num_groups = num_groups
        self._seed = seed
        self._exact = per_policy_per_group

    async def produce(self):
        rng = random.Random(self._seed)
        for group_id in range(self._num_groups):
            self.generate_calls += 1
            if self._exact is not None:
                targets = {p: self._exact for p in POLICIES}
            else:
                targets = {p: rng.randint(1, 3) for p in POLICIES if rng.random() < 0.8} or {LEADER: 2}
            for policy, count in targets.items():
                for _ in range(count):
                    await self.queues[policy].put((group_id, policy))
                    self.submitted[policy] += 1
            await asyncio.sleep(0)


class FakeHooks:
    """MultiPolicyHooks over FakeEngines + fake trainers; asserts purity at train time."""

    def __init__(self, engines: FakeEngines, *, batch_size=4):
        self.engines = engines
        self.batch_size = batch_size
        self.trained = {p: 0 for p in POLICIES}
        self.weight_updates = {p: [] for p in POLICIES}
        self.saves = {p: [] for p in POLICIES}
        self.evals = []

    async def generate(self, model_id: str, rollout_id: int) -> None:
        queue = self.engines.queues[model_id]
        batch = [await queue.get()]
        while len(batch) < self.batch_size:
            batch.append(await queue.get())
        # policy purity is structural: the drain only ever sees this queue
        assert all(policy == model_id for _, policy in batch)
        self._pending = getattr(self, "_pending", {})
        self._pending[model_id] = batch

    async def train(self, model_id: str, rollout_id: int) -> None:
        batch = self._pending.pop(model_id)
        assert all(policy == model_id for _, policy in batch)  # invariant (a)
        self.trained[model_id] += len(batch)
        await asyncio.sleep(0)

    async def update_weights(self, model_id: str, rollout_id: int) -> None:
        self.weight_updates[model_id].append(rollout_id)

    async def save_model(self, model_id: str, rollout_id: int) -> None:
        self.saves[model_id].append(rollout_id)

    async def eval_global(self, leader_rollout_id: int) -> None:
        self.evals.append(leader_rollout_id)


def _run(tmp_path, *, num_groups=60, num_rollout=6, batch_size=4, seed=1, save_interval=2, eval_interval=None,
         per_policy_per_group=None):
    async def main():
        engines = FakeEngines(num_groups=num_groups, seed=seed, per_policy_per_group=per_policy_per_group)
        hooks = FakeHooks(engines, batch_size=batch_size)
        producer = asyncio.create_task(engines.produce())
        positions = await run_multi_policy_loop(
            config=_config(),
            hooks=hooks,
            num_rollout=num_rollout,
            start_rollout_ids={p: 0 for p in POLICIES},
            save_dir=str(tmp_path),
            save_interval=save_interval,
            eval_interval=eval_interval,
        )
        await producer
        return engines, hooks, positions

    return asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_real_loop_purity_conservation_termination(tmp_path):
    engines, hooks, positions = _run(tmp_path, num_groups=60, num_rollout=6)
    # (a) asserted inside FakeHooks.train for every batch
    # (b) conservation per policy: submitted == trained + still buffered
    for p in POLICIES:
        assert engines.submitted[p] == hooks.trained[p] + engines.queues[p].qsize()
        assert hooks.trained[p] > 0
    # (c) leader bounded at num_rollout; run ended (completion inside timeout)
    assert positions[LEADER] == 5
    assert positions[FOLLOWER] <= 5
    # every policy published weights after its first round
    assert hooks.weight_updates[LEADER] and hooks.weight_updates[FOLLOWER]


def test_real_loop_exact_delivery_zero_regeneration(tmp_path):
    """Each policy's drain receives exactly its own samples; nothing is regenerated.

    Mirrors the worker-side contract: finished groups are split once at
    completion into per-policy ready queues, so a drain for policy X only
    pops X's finished groups (no re-queue into the prompt source).
    """
    # exactly num_rollout * batch_size samples per policy
    engines, hooks, positions = _run(tmp_path, num_groups=6, num_rollout=6, batch_size=4, per_policy_per_group=4)
    assert engines.generate_calls == 6  # one production pass per prompt group, no regeneration
    # leader (bounded): delivered exactly its own 24 samples, nothing left
    assert engines.queues[LEADER].qsize() == 0
    assert hooks.trained[LEADER] == engines.submitted[LEADER] == 24
    assert positions[LEADER] == 5
    # follower (unbounded, cancelled at leader finish): only ever received its
    # own samples (asserted in train), conservation holds exactly
    assert engines.submitted[FOLLOWER] == 24
    assert hooks.trained[FOLLOWER] + engines.queues[FOLLOWER].qsize() == 24
    assert hooks.trained[FOLLOWER] > 0


def test_real_loop_global_save_state_file(tmp_path):
    _engines, hooks, positions = _run(tmp_path, num_groups=60, num_rollout=6, save_interval=2)
    # (d) leader-driven global saves at leader rollouts 1, 3, 5
    state = MultiPolicyCheckpointState.load(tmp_path, leader_rollout_id=5)
    assert state is not None
    assert state.leader_model_id == LEADER
    assert state.rollout_ids == positions
    # every policy's trainer saved at ITS OWN position during global saves
    assert hooks.saves[LEADER] and hooks.saves[FOLLOWER]


def test_real_loop_propagates_policy_failure(tmp_path):
    class FailingHooks(FakeHooks):
        async def train(self, model_id, rollout_id):
            if model_id == FOLLOWER and rollout_id == 2:
                raise RuntimeError("follower exploded")
            await super().train(model_id, rollout_id)

    async def main():
        engines = FakeEngines(num_groups=60, seed=3)
        hooks = FailingHooks(engines)
        producer = asyncio.create_task(engines.produce())
        with pytest.raises(RuntimeError, match="follower exploded"):
            await run_multi_policy_loop(
                config=_config(),
                hooks=hooks,
                num_rollout=6,
                start_rollout_ids={p: 0 for p in POLICIES},
                save_dir=str(tmp_path),
                save_interval=None,
                eval_interval=None,
            )
        producer.cancel()

    asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_real_loop_eval_parked_and_leader_only(tmp_path):
    _engines, hooks, _positions = _run(tmp_path, num_groups=60, num_rollout=4, eval_interval=2)
    assert hooks.evals == [1, 3]  # leader's rollout ids, followers never eval

# Multi-Policy Config: `--multi-policy-config`

`--multi-policy-config` (separate from `--megatron-config-path`, which manages PPO actor/critic role overrides) declares several **policy trainers** for multi-policy training. The value is a YAML file path or a `base64:` payload:

```yaml
trainers:
  - model_id: solver
    overrides: {lr: 1.0e-6}
  - model_id: verifier
    overrides: {lr: 2.0e-6}
```

- `model_id` is the cross-cutting join key: the custom generate function stamps it into `Sample.trainer_model_id`, rollout samples are sharded by it, each policy gets a dedicated updatable serving server, and metrics are namespaced `{model_id}/...`.
- `trainer_id` defaults to `{model_id}-{role}`; per-policy checkpoints live under `<save>/trainers/<trainer_id>`.
- Only whitelisted per-policy overrides are accepted (optimizer/lr/parallelism/RL clip knobs); batch-shaping and rollout-shared arguments (`global_batch_size`, `micro_batch_size`, `max_tokens_per_gpu`, `use_dynamic_batch_size`, `advantage_estimator`) are rejected because every policy shares the run's rollout batch semantics.
- Restrictions in v1: requires at least two policies; forbids `--colocate`, `--use-critic`, and `--debug-rollout-only`; `model_id: eval` is reserved.
- Absent the flag, the run is single-policy and nothing changes.

## Per-policy serving binding

Each policy `model_id` binds to a serving server **of the same name** declared with `update_weights: true` in the vLLM model config; server sets are disjoint across policies, so one policy's weight update never touches another's engines. The binding is validated once when the training config attaches to the serving cluster. Each policy server holds its own weight-update engine lock (`server.engine_lock`), and `ServingCluster.get_updatable_engines_and_lock(model_id=...)` / `get_weight_version(model_id=...)` address a single policy; omitting `model_id` preserves the legacy single-policy behavior (one global lock).

## Leader/follower coordination

The first trainer in `--multi-policy-config` is the **leader**; its rollout index bounds the run and drives global save/eval through a `Parker` barrier (`vime/utils/multi_policy/parker.py`) that parks all followers during the critical section. Each policy saves at its own rollout id under `<save>/trainers/<trainer_id>`; global consistency is recorded in `<save>/multi_policy_state/<leader_rollout_id>.json` (`MultiPolicyCheckpointState`), and resume asserts every policy's position matches that record exactly.

## Training entry (F09-4, stacked on the F05 fully-async staleness work)

`train_multi_policy.py` runs one asyncio task per policy over a shared `RolloutManager`: each round is `manager.generate(rollout_id, model_id=...)` (per-policy weight version sampled from that policy's server, carried via `rollout_context`; the fully-async drain splits mixed groups by `Sample.trainer_model_id` and re-queues foreign shares) followed by `async_train` on the policy's own trainer group and a per-policy weight update targeting only that policy's engines. v1 constraints: local (non-distributed) fully-async worker, no `--colocate`, no `--use-critic`; the leader's `--num-rollout` bounds the run.

## End-to-end verification (F09-5)

`tests/test_multi_policy_2policy_e2e.py` (GPU-only CI) trains two policies (solver/verifier, `examples/multi_policy/generate_solver_verifier.py`) and asserts the `multi_policy_state/<N>.json` record plus both per-policy checkpoints under `<save>/trainers/<trainer_id>`; reward/grad-norm series are tracked via the F10 metric-history gate declarations.

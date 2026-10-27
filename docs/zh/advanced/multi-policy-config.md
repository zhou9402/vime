# 多策略配置：`--multi-policy-config`

`--multi-policy-config`（与 `--megatron-config-path` 相互独立，后者管理 PPO actor/critic 角色覆盖）声明多个**策略训练器**，用于多策略训练。取值是 YAML 文件路径或 `base64:` 负载：

```yaml
trainers:
  - model_id: solver
    overrides: {lr: 1.0e-6}
  - model_id: verifier
    overrides: {lr: 2.0e-6}
```

- `model_id` 是贯穿全局的连接键：自定义 generate 函数把它写入 `Sample.trainer_model_id`，rollout 样本按它分片，每个策略绑定独立的可更新 serving server，指标以 `{model_id}/...` 为前缀。
- `trainer_id` 默认为 `{model_id}-{role}`；每个策略的 checkpoint 位于 `<save>/trainers/<trainer_id>`。
- 只接受白名单内的按策略覆盖项（optimizer/lr/并行/RL clip 参数）；批次与 rollout 共享参数（`global_batch_size`、`micro_batch_size`、`max_tokens_per_gpu`、`use_dynamic_batch_size`、`advantage_estimator`）会被拒绝，因为所有策略共享同一 rollout 批次语义。
- v1 限制：至少声明两个策略；禁止 `--colocate`、`--use-critic`、`--debug-rollout-only`；`model_id: eval` 保留。
- 不传该 flag 时行为与单策略完全一致。

## 按策略的 serving 绑定

每个策略的 `model_id` 绑定到 vLLM model config 中**同名**且声明 `update_weights: true` 的 serving server；各策略的 server 集合互不相交，一个策略的权重更新不会触碰其他策略的 engine。绑定关系在训练配置 attach 到 serving cluster 时一次性校验。每个策略 server 持有自己的权重更新 engine lock（`server.engine_lock`）；`ServingCluster.get_updatable_engines_and_lock(model_id=...)` / `get_weight_version(model_id=...)` 按策略寻址；不传 `model_id` 时保持单策略行为（一把全局 lock）不变。

## leader/follower 协调

`--multi-policy-config` 中第一个 trainer 是 **leader**：其 rollout 序号界定整个 run，并通过 `Parker` 屏障（`vime/utils/multi_policy/parker.py`）在全局 save/eval 临界区停住所有 follower。每个策略按自己的 rollout id 保存到 `<save>/trainers/<trainer_id>`；全局一致性记录在 `<save>/multi_policy_state/<leader_rollout_id>.json`（`MultiPolicyCheckpointState`），resume 时会断言每个策略的位置与该记录完全一致。

## 训练入口（F09-4，叠加在 F05 fully-async staleness 工作之上）

`train_multi_policy.py` 为每个策略起一个 asyncio task，共享一个 `RolloutManager`：每轮先 `manager.generate(rollout_id, model_id=...)`（按策略从对应 server 采样权重版本，经 `rollout_context` 传递；fully-async drain 按 `Sample.trainer_model_id` 拆分混合 group 并把其他策略的部分重新入队），再在该策略自己的 trainer group 上 `async_train`，最后只向该策略的 engine 集合发布权重。v1 限制：仅支持本地（非分布式）fully-async worker，禁止 `--colocate`、`--use-critic`；run 由 leader 的 `--num-rollout` 界定。

## 端到端验证（F09-5）

`tests/test_multi_policy_2policy_e2e.py`（仅 GPU CI）训练 solver/verifier 两个策略（`examples/multi_policy/generate_solver_verifier.py`），断言 `multi_policy_state/<N>.json` 记录和 `<save>/trainers/<trainer_id>` 下两个策略各自的 checkpoint；reward/grad-norm 曲线通过 F10 metric-history gate 声明做跟踪。

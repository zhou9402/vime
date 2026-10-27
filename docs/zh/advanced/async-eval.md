# 非阻塞 Eval 调度

移植自 miles@ba28991834(`ray/rollout/eval_dispatch.py`)。

默认情况下,vime 在共享 rollout 引擎上以阻塞调用执行评测。eval dispatcher
为基于快照的非阻塞评测提供基础设施:每个评测点导出当前 actor 权重的 HF
快照并异步执行评测,受在飞上限与可配置的溢出策略约束。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--eval-hf-dir` | `None` | 每个评测点的 HF 快照暂存目录,位于 `{eval_hf_dir}/step_{rollout_id}`(推荐 tmpfs)。快照评测(`--eval-num-gpus` 或 `CheckpointEvalFn`)必填;`--save-hf` 检查点不会被复用为 eval 快照。需要 `--eval-interval`。 |
| `--eval-max-in-flight` | `2` | 同时在飞的异步评测上限。 |
| `--eval-overflow-policy` | `backpressure` | `backpressure` 等待最老的挂起评测;`skip` 丢弃该点并记录 `eval/skipped_busy`。 |
| `--eval-keep-snapshots` | `2` | `--eval-hf-dir` 下保留的已导出快照数量,超出即回收。 |

## 行为

- dispatcher 运行在带独立事件循环的守护线程上,由同步驱动(`train.py`)驱动;
  退出前 `drain()` 会等所有挂起评测结束。
- 在专用 eval fleet(`--eval-num-gpus`)或 `CheckpointEvalFn` 落地之前,
  `eval_uses_snapshots` 为 false,调度退化为原有的阻塞式共享引擎评测。
- 每个被跳过的评测点都会带原因上报(`busy`、`export_failed`、`crashed`);
  在 `--ci-test` 下任何跳过都会使任务失败。

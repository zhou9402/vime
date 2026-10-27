# 非阻塞 Eval 调度

移植自 miles@ba28991834(`ray/rollout/eval_dispatch.py`)。

默认情况下,vime 在共享 rollout 引擎上以阻塞调用执行评测。eval dispatcher
为基于快照的非阻塞评测提供基础设施:每个评测点导出当前 actor 权重的 HF
快照并异步执行评测,受在飞上限与可配置的溢出策略约束。

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--eval-num-gpus` | `0` | 专用 eval 引擎池大小。>0 时评测运行在独立引擎(独立 router)上,通过加载导出的 HF 快照同步权重,从不加入训练权重更新。0 保持共享引擎评测。 |
| `--eval-num-gpus-per-engine` | `1` | 每个 eval 引擎的 GPU 数(TP 大小)。 |
| `--eval-hf-dir` | `None` | 每个评测点的 HF 快照暂存目录,位于 `{eval_hf_dir}/step_{rollout_id}`(推荐 tmpfs)。快照评测(`--eval-num-gpus` 或 `CheckpointEvalFn`)必填;`--save-hf` 检查点不会被复用为 eval 快照。需要 `--eval-interval`。 |
| `--eval-max-in-flight` | `2` | 同时在飞的异步评测上限。 |
| `--eval-overflow-policy` | `backpressure` | `backpressure` 等待最老的挂起评测;`skip` 丢弃该点并记录 `eval/skipped_busy`。 |
| `--eval-keep-snapshots` | `2` | `--eval-hf-dir` 下保留的已导出快照数量,超出即回收。 |

## Eval fleet

设置 `--eval-num-gpus N` 后,vLLM 配置会增加一个 `eval` 模型
(`update_weights=False`,独立 vllm-router),并在 placement group 末尾追加
N 个 GPU bundle——包括 colocate 布局在内,训练侧的 GPU 偏移都不会改变。
每个评测点会先 pin 整个 fleet:所有引擎通过 `reload_weights` 加载快照、
回读权重版本(任一引擎过期则以 `pin_violation` 跳过该点)并通过健康探测。
评测本身运行在 driver 持有的 `EvalRunner` actor 上,慢速快照加载不会占用
RolloutManager actor。

已知缺口(v1):eval 引擎**不**属于可恢复的常驻 serving 拓扑——它们随
driver 退出而销毁,容错机制不会重启它们。`--eval-num-gpus` 不支持与外部
rollout 引擎或 `--prefill-num-servers` 同时使用;使用 `--vllm-config` 时,
YAML 必须恰好定义一个名为 `eval`、覆盖 `--eval-num-gpus` 个 GPU 的模型。

## 行为

- dispatcher 运行在带独立事件循环的守护线程上,由同步驱动(`train.py`)驱动;
  退出前 `drain()` 会等所有挂起评测结束。
- 未配置 eval fleet(`--eval-num-gpus 0`)或 `CheckpointEvalFn` 时,
  `eval_uses_snapshots` 为 false,调度退化为原有的阻塞式共享引擎评测。
- 每个被跳过的评测点都会带原因上报(`busy`、`export_failed`、`crashed`);
  在 `--ci-test` 下任何跳过都会使任务失败。

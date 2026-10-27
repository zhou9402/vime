# Day-0 模型插件

vime 将 miles 的 day-0 模型覆盖移植为 Megatron 训练侧插件，位于
`vime_plugins/models/`，通过 Megatron 的 `--spec <module> <fn>` 选择。

## DeepSeek-V4（`vime_plugins/models/deepseek_v4/`）

- `--spec vime_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec`
- `--dsv4-impl megatron`（默认）使用 Megatron 原生 `dsv4_hybrid` attention variant
  （仅 TP=1）；`--dsv4-impl miles` 使用移植的自定义内核实现
  （`V4Indexer`、`DeepSeekV4Compressor`、THD packing、CP 负载均衡）。
- DSA 内核位于 `vime/kernels/attention/dsa/`（见 [DSA 内核](dsa-kernels.md)）；
  indexer top-k 通过 `vime/utils/replay_base.py`（`IndexerReplayManager`）
  记录/回放 rollout 的选取。
- FP8 QAT 模拟（`ops/qat.py`）在缺少 TileKernels / quant 内核快速路径时
  回退到纯 torch 的 UE8M0 实现。

测试：`tests/plugins/test_dsv4_ops.py`、`test_dsv4_thd.py`、`test_dsv4_cp_row_balance.py`
（CPU）；`test_dsv4_indexer_cp_balance.py`（torchrun，GPU）。

## DSA indexer top-k 回放（rollout → trainer）

使用 `--use-rollout-indexer-replay` 和 `--rollout-indexer-topk-num-streams <L>`（indexer 层数
= 隐藏层数减去全注意力层数），vLLM server（以 `--enable-return-indexer-topk` 启动，见
`docker/patch/latest/vllm-indexer-topk.patch`）会在 generate choice 上以 base64 `.npy`
返回逐 token 的 indexer top-k 索引；vime 解码到 `Sample.rollout_indexer_topk`
（`[rows, num_streams, index_topk]` int32，rows = `tokens[1:]`），供训练侧的
`IndexerReplayManager`（`vime/utils/replay_base.py`）使用。ROCm 镜像尚未包含该 vLLM patch
（需要针对 ROCm nightly base 做 rebase）。

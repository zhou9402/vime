# DSA 内核（DeepSeek 稀疏注意力）

`vime/kernels/attention/dsa/` 移植了 miles 的 Triton DSA 内核：lightning-indexer
打分内核（`indexer_logits`）、精确 top-k 选择（`get_dsa_topk_fn`）、带自动求导的
top-k 分数收集（`indexer_topk_scores`、`lighting_indexer`）。

与 miles 的差异：

- 未移植 TileLang 后端；indexer 默认 `score_backend` 为 `"triton"`，缺少 tilelang 时
  top-k 分数的反向传播回退到 torch 重算 `sum_h relu(k·q_h)·w_h`。**torch 反向目前是
  生产路径**：每次反向需要 O(T·topk·H·D) 的 fp32 gather+einsum（无融合内核），
  首次触发时记录一次性 warning。
- sglang 的 canonical top-k 内核（`topk_transform_ragged_v2`）在 vLLM 中没有对应实现；
  `canonical_dsa_topk` 会告警并回退到 torch top-k（并列顺序可能不同）。
  `VIME_DSA_TOPK_BACKEND` 可覆盖后端，`VIME_DSA_TOPK_FLASHINFER_DETERMINISTIC` /
  `VIME_DSA_TOPK_FLASHINFER_TIE_BREAK` 替代 sglang 的环境变量。

测试：`tests/kernels/test_dsa_indexer.py`（GPU）、`tests/kernels/test_dsa_torch_fallback.py`（CPU）。

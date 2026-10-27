# 模型转换器测试框架

Day-0 模型覆盖（DeepSeek-V4/V4.1、GLM5-Next、Kimi-K3、Qwen3.8-Next）会新增
HF↔Megatron 转换器。每个转换器必须通过 `tests/converters/harness.py` 中的共享框架：

1. **逐位往返一致** — 对每个 Megatron 参数，导出为 HF 名称再读回：
   `loader(exporter(w)) == w` 完全相等（`assert_roundtrip_bitwise`）。
2. **logits 保持** — 随机初始化的 2 层 HF 模型（tiny 配置位于
   `tests/converters/tiny_configs/<family>.json`）经 HF→Megatron→HF 映射后，
   重建模型的 logits 必须 allclose（fp32，rtol=atol=1e-5；
   `assert_hf_logits_preserved`）。

每个模型族提供 2 层 tiny 配置，检查在 CPU CI 上数秒内完成。

运行：

```bash
python -m pytest tests/test_hf_to_megatron.py -k tiny_harness -x -q
```

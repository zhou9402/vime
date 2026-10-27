# Model Converter Test Harness

Day-0 model coverage (DeepSeek-V4/V4.1, GLM5-Next, Kimi-K3, Qwen3.8-Next) adds new
HF↔Megatron converters. Every converter must pass the shared harness in
`tests/converters/harness.py`:

1. **Bitwise round trip** — for each Megatron parameter, export to HF names and reload:
   `loader(exporter(w)) == w` exactly (`assert_roundtrip_bitwise`).
2. **Logits preservation** — a randomly initialized 2-layer HF model (tiny config from
   `tests/converters/tiny_configs/<family>.json`) is mapped HF→Megatron→HF and the
   reconstructed model must produce allclose logits (fp32, rtol=atol=1e-5;
   `assert_hf_logits_preserved`).

Each family ships a tiny 2-layer config so the checks run on CPU CI in seconds.

Run:

```bash
python -m pytest tests/test_hf_to_megatron.py -k tiny_harness -x -q
```

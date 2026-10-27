# HF→Megatron Load Modes (builtin / mbridge)

`--hf-to-megatron-load-mode {builtin,mbridge}` selects how HuggingFace checkpoints are
loaded into the Megatron model when `--load` points at an HF directory.

- `builtin` (default): vime's hand-written converters in
  `vime/backends/megatron_utils/hf_to_megatron/`, dispatched on the checkpoint's
  `config.model_type`.
- `mbridge`: the external [mbridge](https://github.com/ISEEKYAN/mbridge) package
  (pinned to commit `89eb1088`, same as miles) with vime's bridge registrations in
  `vime_plugins/mbridge/`. Required for day-0 models without a builtin converter
  (DeepSeek-V4/V4.1, GLM5-Next, Kimi-K3, Qwen3.8-Next).

Mapping to miles: miles' `--megatron-to-hf-mode raw` corresponds to `--hf-to-megatron-load-mode
mbridge`; miles' `bridge` mode (NVIDIA Megatron-Bridge) is not supported in vime. The
Megatron→HF export path is unaffected by this flag.

mbridge mode requires the pinned package: `pip install --no-deps
git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c`
(preinstalled in the vime Docker images).

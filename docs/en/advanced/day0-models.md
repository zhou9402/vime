# Day-0 Model Plugins

vime ports miles' day-0 model coverage as Megatron training-side plugins under
`vime_plugins/models/`, selected with Megatron's `--spec <module> <fn>` flag.

## DeepSeek-V4 (`vime_plugins/models/deepseek_v4/`)

- `--spec vime_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec`
- `--dsv4-impl megatron` (default) uses Megatron's native `dsv4_hybrid` attention variant
  (TP=1 only); `--dsv4-impl miles` uses the ported custom-kernel implementation
  (`V4Indexer`, `DeepSeekV4Compressor`, THD packing, CP row balance).
- DSA kernels live in `vime/kernels/attention/dsa/` (see
  [DSA kernels](dsa-kernels.md)); the indexer top-k seam records/replays rollout picks via
  `vime/utils/replay_base.py` (`IndexerReplayManager`).
- FP8 QAT simulation (`ops/qat.py`) falls back to a pure-torch UE8M0 implementation when
  the TileKernels / quant-kernel fast path is unavailable.

Tests: `tests/plugins/test_dsv4_ops.py`, `test_dsv4_thd.py`, `test_dsv4_cp_row_balance.py`
(CPU); `test_dsv4_indexer_cp_balance.py` (torchrun, GPU).

## DSA indexer top-k replay (rollout → trainer)

With `--use-rollout-indexer-replay` plus `--rollout-indexer-topk-num-streams <L>` (indexer
layers = hidden layers minus full-attention layers), the vLLM server (started with
`--enable-return-indexer-topk`, see `docker/patch/latest/vllm-indexer-topk.patch`) returns
per-token indexer top-k indices as a base64 `.npy` blob on the generate choice; vime decodes
them onto `Sample.rollout_indexer_topk` (`[rows, num_streams, index_topk]` int32, rows =
`tokens[1:]`) for the trainer-side `IndexerReplayManager` (`vime/utils/replay_base.py`).
ROCm images do not carry the vLLM patch yet (needs a rebase against the ROCm nightly base).

Trainer-side: `vime/backends/megatron_utils/indexer_replay.py` fills the per-layer replay
queues in microbatch order (THD token-position rebase, zigzag CP, SP slicing). Current limits:
no disk-backed (TensorRef) indexer payloads and no allgather_cp yet; both raise loudly.

## Kimi-K3 (`vime_plugins/models/kimi_k3/`)

- `--spec vime_plugins.models.kimi_k3 get_kimi_k3_spec`; KDA (linear-attention) layers are the
  non-%4 layers, MLA otherwise.
- Shared helpers: `vime_plugins/models/linear_attn.py` (TP/CP-sharded GDN/KDA base),
  `vime/backends/megatron_utils/megatron_to_hf/linear_attn_layout.py` (group-major QKV layout).
- Converters: builtin `kimi_k3` loader/exporter (nested `language_model.model.*` HF names),
  plus the `kimi_k3` mbridge bridge for `--hf-to-megatron-load-mode mbridge`.

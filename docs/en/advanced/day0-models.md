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

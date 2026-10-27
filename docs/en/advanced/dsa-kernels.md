# DSA Kernels (DeepSeek Sparse Attention)

`vime/kernels/attention/dsa/` ports miles' Triton DSA kernels: the lightning-indexer
score kernel (`indexer_logits`), exact top-k selection (`get_dsa_topk_fn`), and
gathered top-k scores with autograd (`indexer_topk_scores`, `lighting_indexer`).

Notes vs miles:

- The TileLang backends are not ported; the indexer default `score_backend` is
  `"triton"` and the top-k-score backward falls back to a torch recomputation of
  `sum_h relu(k·q_h)·w_h` when tilelang is absent. **The torch backward is the
  production path for now**: it costs an O(T·topk·H·D) fp32 gather+einsum per
  backward (no fused kernel) and logs a one-time warning when first taken.
- The sglang canonical top-k kernel (`topk_transform_ragged_v2`) has no vLLM
  equivalent; `canonical_dsa_topk` falls back to torch top-k with a warning
  (tie-break order may differ). `VIME_DSA_TOPK_BACKEND` overrides the backend,
  `VIME_DSA_TOPK_FLASHINFER_DETERMINISTIC` / `VIME_DSA_TOPK_FLASHINFER_TIE_BREAK`
  replace the sglang envs.

Tests: `tests/kernels/test_dsa_indexer.py` (GPU), `tests/kernels/test_dsa_torch_fallback.py` (CPU).

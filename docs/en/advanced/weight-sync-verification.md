# Weight-Sync Verification

Port of miles@ba28991834 (`backends/training_utils/weight_update/checksum_utils.py`,
`hf_weight_iterator/{bucketing,atomic_groups}.py`, `utils/weight_version.py`).

vime verifies every weight update on all three sync paths (colocate IPC, NCCL
distributed, disk-delta) with four mechanisms:

1. **Transfer checksums** — `--check-weight-transfer-checksum` (defaults on under
   `--ci-test`): the trainer SHA-256 hashes each CPU send buffer; after the write
   lands, the engine's `check_weights` RPC returns per-tensor hashes and the
   trainer fails the update on any mismatch.
2. **Post-sync equality check** — `--check-weight-update-equal` with
   `--check-weight-update-selector {all,target,draft}` (default: auto = `target`,
   plus `draft` only when a draft model is configured via
   `--vllm-speculative-config`/dspark), `--check-weight-update-skip-list`
   (substring names downgraded to non-fatal), and
   `--check-weight-update-allow-quant-error` (quantized tensors compared in
   dequantized space, ≤1 ULP per side). Engine-side snapshots are **hashes-only**
   by default (no CPU copy); `--check-weight-update-snapshot full` stores CPU
   copies, enabling quant-tolerant value compare and the separate `restore`
   action. `reset_tensors` CLOBBERS live weights with a deterministic zero
   sentinel (never restores): the startup flow snapshot -> reset -> update ->
   compare proves the first update overwrites every tensor.
3. **Atomic update groups** — HF-name suffixes whose tensors a vLLM loader fuses
   into one engine parameter (e.g. DeepSeek `wq_a` + `wkv`) are never split
   across update buckets. fp8 weight+scale pairs are always emitted as one unit.
4. **Weight-version staleness guard** — asserts the driver publishes a weight
   version after every update and that rollout samples carry sane numeric
   versions. v1 stamps versions at sample-submission level; per-token
   engine-reported spans (like sglang's `meta_info["weight_versions"]`) are a
   known gap and not yet implemented for vLLM.

Tamper detection is verified by `tests/test_weight_sync_tamper.py`: a one-bit
flip of an engine-side tensor after sync must make `check_weights(action="compare")`
fail; no flip must pass.

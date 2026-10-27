# P2P / RDMA Weight Transfer

> Status: complete (F01, stacked on F02 `miles-port/f02-weight-sync-verification`,
> whose `check_weights` RPC and `checksum_utils` this feature reuses).
> GPU e2e: `tests/test_p2p_weight_update_e2e.py`.

For **disaggregated** (non-colocated) training at scale, NCCL broadcast weight
sync cost grows with the engine world size. P2P mode instead writes weights
directly into each engine over RDMA (Mooncake transfer engine). Miles'
measurements: slower than NCCL at ≤2 nodes (+68–72%), but 47–86% faster at ≥4
nodes (GLM-5 744B: 58.3 s → 8.5 s; Kimi-K2 1T: 53.3 s → 7.2 s per update).

## Design (replica-free, "Design A")

Unlike miles (which rebuilds the engine's model layout in the trainer process
via a model replica), vime's port is replica-free:

1. Each engine worker allocates **one pinned staging buffer** (sized 2× the max
   update bucket for double-buffering) and registers it with its local Mooncake
   transfer engine. It publishes `(session_id, staging_address, capacity)` via
   a `get_p2p_transfer_info` RPC.
2. The trainer packs an HF-layout bucket (name-tagged manifest) into a pinned
   transfer buffer and RDMA-writes it into each assigned engine's staging
   buffer.
3. The engine **reshards the staged bytes with its own vLLM loader**
   (`load_weights_from_staged`) and returns per-name SHA-256 checksums, which
   the trainer verifies before releasing the write window.

No trainer-side model replica, no trainer/engine version or GPU-architecture
lockstep.

## MTP draft sync

With `--enable-mtp-training` and an mtp `--vllm-speculative-config`, P2P runs a
second weight-update session per version: after the target buckets finish, the
same buckets are re-sent inside a `start_draft_weight_update` session, and the
vLLM worker's draft target swap routes the staged loads into the draft runner —
so trained MTP layers reach the draft (miles v0.1.3 parity). dspark speculative
decoding is NOT supported over p2p (its draft weights need
`export_dspark_model_weights`, which P2P does not iterate) and is rejected at
argument validation; use `--update-weight-transport nccl` for dspark.

## Constraints (same as miles)

- Disaggregated only: rejected with `--colocate` (colocate uses CUDA IPC).
- No LoRA over P2P. Speculative decoding: mtp only (see above); dspark rejected.
- Mooncake is an optional dependency (`pip install vime[p2p]`, i.e.
  `mooncake-transfer-engine`); nothing imports it unless P2P is enabled.
- Requires routable RDMA (or TCP fallback: `MOONCAKE_PROTOCOL=tcp`) between
  trainer and engine hosts; `MOONCAKE_DEVICE` selects the RDMA NIC.

## Failure semantics

- Weight writes are **never retried** (the weight-update session is no-retry
  by design; the miles #4402 retry hardening covers Mooncake *object-store*
  ops only). A failed write marks its engine errored and the update continues
  for healthy engines (partial-target quarantine).
- A write with unknown native completion ("uncertain write") additionally pins
  the registered transfer-buffer memory forever: it is never unregistered.
- Per-bucket SHA-256 checksums are verified on arrival (converges with the
  generic `check_weights` hardening, F02).

## Usage

```bash
# image built with --build-arg ENABLE_P2P=1 (installs mooncake-transfer-engine)
python train.py ... --update-weight-transport p2p --p2p-transfer-timeout 30 \
    --check-weight-transfer-checksum
```

Validation rejects `--colocate`, LoRA, and speculative decoding with p2p.
Each engine worker's staging buffer is sized 1x `--update-weight-buffer-size`
(512 MiB default): the per-engine write thread serializes write -> load, so a
second slot is never used. The trainer's transfer buffers are double-buffered.

## Engine backend (vLLM patch)

`docker/patch/latest/vllm-p2p-weight-transfer.patch` adds vLLM's `p2p` weight
transfer backend (`vllm/distributed/weight_transfer/p2p_engine.py`): each
worker allocates one Mooncake-registered staging buffer and exposes
`get_p2p_transfer_info` / `load_weights_from_staged` as worker RPCs (reachable
via `/collective_rpc`). `load_weights_from_staged` verifies per-name SHA-256
checksums of the staged bytes, then loads through the model's own
`load_weights`. Mooncake is installed only with `--build-arg ENABLE_P2P=1`.

## Hardening (F01-P4, with F02)

- Per-bucket SHA-256 verification uses F02's shared `checksum_utils`; the
  `--check-weight-transfer-checksum` flag (default on under `--ci-test`) lives
  in F02's `add_weight_check_arguments`.
- Control-plane queries (never data writes) retry transient errors up to
  `--p2p-transfer-timeout` (`vime/utils/retry_utils.py`, miles #4402 lineage).
- Errored engines are union-quarantined across trainer ranks (gloo all-gather),
  then retired via the engine's existing shutdown path (router unregister +
  process kill) before the update raises.

## Verifier

`python -m pytest tests/test_p2p_weight_transfer.py -x -q` covers the contract
with fakes: byte-equivalence vs direct copy, tamper detection, transient retry,
and permanent-failure quarantine.

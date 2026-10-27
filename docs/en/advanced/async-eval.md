# Non-Blocking Eval Dispatch

Port of miles@ba28991834 (`ray/rollout/eval_dispatch.py`).

By default vime evaluates on the shared rollout engines with a blocking call in
the training loop. The eval dispatcher adds the infrastructure for
snapshot-based, non-blocking evaluation: each eval point exports an HF snapshot
of the current actor weights and runs the eval asynchronously, bounded by an
in-flight limit with a configurable overflow policy.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--eval-num-gpus` | `0` | Dedicated eval engine fleet size. >0 runs eval on its own engines behind their own router, synced by loading exported HF snapshots, never by joining training weight updates. 0 keeps shared-engine eval. |
| `--eval-num-gpus-per-engine` | `1` | GPUs per eval engine (TP size). |
| `--eval-hf-dir` | `None` | Staging dir for per-eval HF snapshots at `{eval_hf_dir}/step_{rollout_id}` (tmpfs recommended). Required for snapshot eval (`--eval-num-gpus` or a `CheckpointEvalFn`); `--save-hf` checkpoints are never reused as eval snapshots. Requires `--eval-interval`. |
| `--eval-max-in-flight` | `2` | Max concurrently pending async evals. |
| `--eval-overflow-policy` | `backpressure` | `backpressure` awaits the oldest pending eval; `skip` drops the point and logs `eval/skipped_busy`. |
| `--eval-keep-snapshots` | `2` | Exported snapshot dirs retained under `--eval-hf-dir` before GC. |

## Eval fleet

With `--eval-num-gpus N`, the vLLM config gains an `eval` model
(`update_weights=False`) behind its own vllm-router, and N extra GPU bundles
are appended to the placement group past every other region — trainer offsets
never shift, including in colocate layouts. Each eval point pins the fleet:
every engine loads the snapshot via `reload_weights`, reads back the weight
version (one stale engine skips the point as `pin_violation`), and passes a
health probe. The eval itself runs on a driver-owned `EvalRunner` actor, so a
slow snapshot load never occupies the RolloutManager actor.

Known gaps (v1): eval engines are **not** part of the detached, recoverable
serving topology — they die with the driver and are not restarted by fault
tolerance. `--eval-num-gpus` is not supported with external rollout engines or
`--prefill-num-servers`; with `--vllm-config` the YAML must define exactly one
model named `eval` covering `--eval-num-gpus` GPUs.

## CheckpointEvalFn

Point `--eval-function-path` at a subclass of
`vime.rollout.checkpoint_eval.CheckpointEvalFn` to eval snapshots on a backend
you own (an external vLLM server, a scoring service, a black-box HTTP API).
Each eval point delivers a finished HF checkpoint directory to
`evaluate_checkpoint(checkpoint_dir, input)`; raise `EvalSkip(reason)` to skip
a point with attribution. Mutually exclusive with `--eval-num-gpus` (the fleet
already owns weight delivery); requires `--eval-hf-dir`.
`examples/eval/external_vllm_eval_fn.py` is a working implementation, and
`retarget_args(args, router_ip, router_port, engine_gpu_counts)` redirects the
stock eval datasets at any vLLM router.

## Behavior

- The dispatcher runs on a daemon thread with its own event loop, driven from
  the synchronous driver (`train.py`); `drain()` settles all pending points
  before teardown.
- Without the eval fleet (`--eval-num-gpus 0`) or a `CheckpointEvalFn`,
  `eval_uses_snapshots` is false and dispatch degrades to the previous
  blocking shared-engine eval.
- Every skipped point is reported with a reason (`busy`, `export_failed`,
  `crashed`); under `--ci-test` any skip fails the run.

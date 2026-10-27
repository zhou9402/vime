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
| `--eval-hf-dir` | `None` | Staging dir for per-eval HF snapshots at `{eval_hf_dir}/step_{rollout_id}` (tmpfs recommended). Required for snapshot eval (`--eval-num-gpus` or a `CheckpointEvalFn`); `--save-hf` checkpoints are never reused as eval snapshots. Requires `--eval-interval`. |
| `--eval-max-in-flight` | `2` | Max concurrently pending async evals. |
| `--eval-overflow-policy` | `backpressure` | `backpressure` awaits the oldest pending eval; `skip` drops the point and logs `eval/skipped_busy`. |
| `--eval-keep-snapshots` | `2` | Exported snapshot dirs retained under `--eval-hf-dir` before GC. |

## Behavior

- The dispatcher runs on a daemon thread with its own event loop, driven from
  the synchronous driver (`train.py`); `drain()` settles all pending points
  before teardown.
- Until the dedicated eval fleet (`--eval-num-gpus`) or a `CheckpointEvalFn`
  lands, `eval_uses_snapshots` is false and dispatch degrades to the previous
  blocking shared-engine eval.
- Every skipped point is reported with a reason (`busy`, `export_failed`,
  `crashed`); under `--ci-test` any skip fails the run.

# MFU Metric and Metric-History Gate

This page documents the performance-accounting and CI regression-gate tooling
ported from miles (source `miles@ba28991834`).

## FLOPs model and the V5 oracle

`vime/utils/flops_model.py` computes analytical forward and training FLOPs of a
decoder-only model from its Hugging Face config. It covers dense GQA, MoE, MLA,
Gated DeltaNet / KDA linear attention, DSA/QSA sparse-attention indexers,
DeepSeek-V4 compressed attention, and Nemotron-H Mamba2 hybrid blocks. MTP
layers are intentionally not counted (parity with miles); adding a vime-only
MTP term is a documented follow-up.

The vendored oracle `tests/ci/flops_oracle/flops_utils_miles_ba28991834.py`
pins miles' implementation as ground truth. The verifier compares any
implementation against it over 8 architecture configs and fails on a relative
error of 1e-6 or more:

```bash
python tests/ci/flops_oracle/verify_flops.py --impl vime/utils/flops_model.py
```

Exit code 0 means pass (or SKIP while the implementation file does not exist
yet); exit code 1 means mismatch.

## MFU metric

`log_perf_data` logs `perf/actor_train_mfu` (plus `perf/mfu_peak_tflops` and
`perf/actor_train_tflops_hf`) per rollout, alongside the slime-origin
`perf/actor_train_tflops`, which is unchanged. The MFU numerator comes from the
HF-config FLOPs model (`model_flops_from_hf_config` on `args.hf_checkpoint`,
cached per checkpoint); the denominator is `--mfu-peak-tflops` when set, else
the built-in device table in `vime/utils/device_flops.py` (NVIDIA A100/H100/
H200/GH200/B200/B300/GB200/GB300 and AMD MI300X/MI325X/MI355X). When neither is
available for the local device, no `perf/mfu*` keys are logged.

## Metric-history store

`tests/ci/metric_history/storage/` defines the metric-history contract
(`MetricHistoryStore`, `RunIdentity`, `RunProvenance`) and a SQLite backend.
A run's baseline is scoped by `(test_path, backend, suite)`; the comparison
coordinate is `(metric_key, step)`; trust is a run-level flag revocable via
`mark_untrusted` (by `run_id`, `ci_run_id`, or `commit_sha`). Provenance
columns use CI-neutral names (`ci_run_id` = Buildkite build number,
`ci_run_attempt` = retry count, `pipeline` = pipeline slug).

## Metric-history gate (capture + evaluation)

When the CI harness sets `VIME_CI_GATE_RECORD_DIR`, every `logging_utils.log()`
call captures the whitelisted metrics (`train/grad_norm`, `train/ppo_kl`,
`train/train_rollout_logprob_abs_diff`, `train/train_rollout_kl`,
`rollout/raw_reward`, the two `rollout/tito_session_mismatch_rate/*` keys, and
`perf/actor_train_mfu`) into a per-process JSONL snapshot (atomic temp+rename;
non-finite values become `"NaN"`/`"Infinity"`/`"-Infinity"` markers).

A test opts into gating with a top-level declaration, parsed from the AST:

```python
register_ci_gate(metric_key="rollout/raw_reward")  # band filled from GATE_DEFAULTS
register_ci_gate(metric_key="perf/actor_train_mfu")  # tracked, never judged
```

`tests/ci/metric_history/run_gate.py` merges the records, evaluates each spec
against the mean of up to 20 trusted baselines, and exits 1 when an enforced
spec errors or fails with at least 10 baselines. The store path comes from
`--db` / `--metric-history-path`, else `$VIME_METRIC_HISTORY_PATH`, else
`./metric_history.db`. `--write-baseline` records trusted history (scheduled
builds only); `--mark-untrusted-sha SHA` revokes a bad commit's runs. The
end-to-end check is `python tests/ci/metric_history/verify_gate_e2e.py`.

## Buildkite wiring

The CPU verifiers run in the `plugin-contracts` step of
`.buildkite/pipeline.yml`: `tests/ci/flops_oracle/verify_flops.py` (V5),
`pytest tests/ci/metric_history`, and `tests/ci/metric_history/verify_gate_e2e.py`
(V1). GPU-suite opt-in: a test exports `VIME_CI_GATE_RECORD_DIR` (per-build
directory, uploaded as a build artifact) and a follow-up step runs
`run_gate.py --write-baseline` on scheduled builds with
`VIME_METRIC_HISTORY_PATH` naming the shared SQLite store. No hostPath
persistence is assumed; the store travels as a Buildkite artifact.

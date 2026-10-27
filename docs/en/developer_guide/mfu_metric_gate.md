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

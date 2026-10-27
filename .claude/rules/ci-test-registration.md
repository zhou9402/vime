---
paths:
  - "tests/**/*.py"
  - ".buildkite/**/*.py"
  - ".buildkite/**/*.yml"
---

# CI Test Placement And Cost

Rules for adding a test file, moving one, or changing what a test needs. GPU
CI runs on Buildkite; the GPU suites and their `(test_file, num_gpus,
extra_args, env)` entries live in `.buildkite/gpu_suites.py`. What to do when a
check goes red is in `ci-failure-triage.md`.

## Placement

- A `test_*.py` lives under `tests/` (flat or in a topical package such as
  `tests/test_agent/`). Pure-CPU tests set `NUM_GPUS = 0`; GPU tests declare
  their GPU count the same way.
- A GPU test file pays for a model download, Ray startup, and an engine launch
  on every run. Add cases to an existing file with the same launch; a new file
  is for a new fixture, model, or parallel layout.
- A GPU test joins a suite in `.buildkite/gpu_suites.py` with the GPU count it
  actually requests (`--actor-num-gpus-per-node`, `torchrun --nproc-per-node`):
  an entry idling GPUs it does not use burns the shared H100 pool. Never lower
  a test's GPU count by introducing skips or reducing the parallel topology
  that the test covers.

## Host caches

GPU jobs mount the node's `/mnt/hf-cache` as HF_HOME, so a warm cache is shared
across jobs; the rest of the pod is discarded with it.

- A checkpoint a test derives (torch_dist, bf16 cast, FP8/INT4/NVFP4) follows
  "Derived checkpoints" in `launch-and-model-scripts.md`: next to the HF one,
  made once, one directory per recipe.
- Tests must not clear or delete caches that another job may be using.

## Disabling a test

- Disable a test to unblock other PRs on a maintainer's call, never to land your
  own feature PR.
- The disabling change says what must change before the test is re-enabled and
  cites the tracking issue or PR.

## For coding agents

- Never present a guessed runtime as measured. Until a timing run exists for a
  new GPU test, say in the PR description that it is unmeasured.
- Triggering Buildkite suites spends shared GPU runners: propose the exact
  action and wait for the user.

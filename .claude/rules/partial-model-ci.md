---
paths:
  - "tests/**/*.py"
  - "scripts/models/*layer*.sh"
---

# Partial Model CI

A partial model is an N-layer slice of a model above 200B total parameters,
because the full model does not fit a CI runner; the model script's name
carries the layer count (`scripts/models/deepseek-v3-5layer.sh`,
`deepseek-v3-20layer.sh`). A slice is a sanity check, not a convergence or
throughput run, so a larger batch only adds runtime on the GPU stages.

- Default to 16 samples per rollout and a global batch of 16, e.g.
  `--rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16`.
- A rollout may grow to 32 samples
  (`rollout-batch-size × n-samples-per-prompt`), never more.
- Set these values through the launcher's argument arrays (`MODEL_ARGS` /
  `ROLLOUT_ARGS` / ...), not by hardcoding them in shared code; the launcher's
  defaults stay the full-model recipe.

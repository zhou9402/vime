---
paths:
  - "scripts/**/*.sh"
  - "examples/**/*.py"
  - "examples/**/*.sh"
  - "vime/utils/external_utils/**/*.py"
---

# Launch And Model Scripts

Conventions for launchers under `scripts/` and `examples/`, and for the model
definitions under `scripts/models/`. Follow them for a new launcher or model
definition, and when substantially modifying an existing one.
`.claude/rules/general-code-style.md` still applies to the python itself.

## Shell launchers and model definitions

- Launchers are `scripts/run-*.sh` and model definitions are
  `scripts/models/*.sh` exposing a `MODEL_ARGS=(...)` array. Both are pinned by
  the snapshot suite under `tests/launch_scripts/`: any flag or model-arg
  change shows up as a golden diff there.
- **One file per model, named after the model it serves** (`glm5-744B-A40B.sh`).
- **Model args are data, not logic.** Keep `MODEL_ARGS` a flat flag list; a
  variant derives from its base model instead of copying it.
- **Any environment knob a launcher or model script reads must be frozen by the
  snapshot harness** (`tests/launch_scripts/sh_harness.py`). The snapshots pin
  expanded model args, so an unfrozen knob lets a developer's exported override
  fail the suite.

## Launcher anatomy

- **Build the argv as grouped blocks** — checkpoint, rollout, perf, algorithm,
  optimizer, vllm, misc — then concatenate them. It keeps a recipe diffable
  against its siblings.

## Nothing hardcoded about the machine

- **Directories are parameters, not literals.** Never hardcode a checkout path
  such as `/root/vime` or `/workspace/vime` — a hygiene test enforces this for
  shell scripts.
- **wandb engages only when `WANDB_API_KEY` is set.** No hardcoded project or
  group names.
- **Read the environment at run time.** A value captured into a snapshot-visible
  default leaks a stale allocation into every snapshot.
- **A cluster that is already joined is expressed with
  `VIME_SCRIPT_EXTERNAL_RAY=1`** (see `vime/utils/external_utils/command_utils.py`),
  not by deleting the `ray start` from the launcher.

## Derived checkpoints

- **A checkpoint derived from the HF one lives next to it** —
  `<model>_torch_dist`, `<model>-bf16`, `<model>-FP8`, `<model>-INT4`,
  `<model>-NVFP4` — never under a separate checkpoint directory. CI mounts the
  models directory from the host, so a conversion made there is reused by every
  later job on it.
- **Skip a finished conversion.** `convert_checkpoint` checks its tracker and
  `fp8_cast_bf16` its index.
- **One directory per conversion recipe.** Two conversions with different
  arguments write different names; never `rm -rf` a derived checkpoint to force
  a fresh one, since another job on the host may be reading it.

## One launcher per recipe family

- **Merge near-duplicate recipes behind a case table** keyed by a `--model-name`
  style selector; a copied file per variant drifts.
- **Keep separate files when the recipes are different experiments.** Differing
  algorithm, parallelism, hardware assertion and tuning blocks mean merging
  would put a recipe in active use at risk.
- **Delete rather than half-port a recipe that provably never ran.** Say so in
  the change description instead of guessing at a configuration nobody has
  executed.

## Snapshot coverage is the review artifact

- **Every launcher issuing `ray job submit` has a recording** under
  `tests/snapshots/launch_scripts/sh/`, and every `scripts/models/*.sh` has one
  under `tests/snapshots/model_args/`. Regenerate with
  `VIME_UPDATE_SNAPSHOTS=1 pytest tests/launch_scripts` and read the resulting
  diff — it is what proves a launcher change does what it claims.

## Documentation follows the launcher

- **Adding, renaming or deleting a launcher updates the pages that invoke it** —
  `docs/**`. Keep per-variant knob tables and download destinations in step with
  the directory defaults.

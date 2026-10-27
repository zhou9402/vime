# Snapshot recordings

Golden recordings for the launch-script snapshot suite (`tests/launch_scripts/`).

- `launch_scripts/sh/**` — every external command each `scripts/**`/`examples/**`
  launcher issues, including the full `ray job submit` argv, recorded by the
  shimmed-PATH harness.
- `model_args/*.txt` — the expanded `MODEL_ARGS` token list of every
  `scripts/models/*.sh`.

Regenerate after an intentional launcher or model-args change:

```bash
VIME_UPDATE_SNAPSHOTS=1 python -m pytest tests/launch_scripts -q
```

Then read the diff — it is the review artifact that proves the change does what
it claims. With the variable unset the tests run in compare mode and any drift
fails.

---
paths:
  - ".github/workflows/**/*.yml"
  - "docs/**/*.md"
---

# CI Maintenance

## Shard runtime budget

When maintaining CI scope or sharding, exclude disabled tests and preserve each
cadence's eligibility filter. Sum the estimated runtime of each shard and target
a largest sum of at most two hours for regular-cadence PR runs and less than two
hours for nightly runs. Do not add environment setup, cleanup, or queue time to
those estimates.

Try shard counts in ascending multiples of the matching runner pool capacity and
choose the first whose largest shard sum meets the budget. Adding shards cannot
shorten an indivisible test file and adds setup cost for each nonempty shard; do
not change test coverage merely to meet the runtime target.

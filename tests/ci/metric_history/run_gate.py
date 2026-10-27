#!/usr/bin/env python3
"""Run the metric-history gate for one CI test execution (V1 harness entry).

Merges the per-process JSONL records a run captured under --record-dir, parses
the test file's register_ci_gate declarations, evaluates them against trusted
history in the SQLite store, prints one verdict line per coordinate, and:

* exit 1 if any *enforced* spec is ERROR, or FAIL with baseline_n >= 10
  (MIN_ENFORCED_BASELINES); a FAIL with less history is reported but annotated
  "NOT ENFORCED: insufficient history";
* with --write-baseline, persists the run's values as a trusted run (intended
  for scheduled/nightly builds only; PR runs evaluate without writing);
* --mark-untrusted-sha SHA revokes trust for every run of a bad commit.

Store path resolution order: --db, then $VIME_METRIC_HISTORY_PATH, then
./metric_history.db.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ci on sys.path

from metric_history.gate import MIN_ENFORCED_BASELINES, GateStatus, evaluate_gate
from metric_history.register import parse_ci_gate_specs
from metric_history.storage.sqlite_store import SQLiteMetricHistoryStore
from metric_history.storage.store import MetricSample, RunIdentity, RunProvenance


def merge_records(record_dir: str, out_path: str) -> None:
    """Merge per-process JSONL snapshots into one {metric: series} record."""
    series_by_metric: dict[str, list] = {}
    for name in sorted(os.listdir(record_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(record_dir, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                series_by_metric.setdefault(rec["metric"], []).extend(rec["series"])
    with open(out_path, "w", encoding="utf-8") as f:
        for key, series in series_by_metric.items():
            series = sorted(series, key=lambda p: (p[0] is None, p[0]))
            f.write(json.dumps({"metric": key, "series": series}, allow_nan=False) + "\n")


def provenance_from_env() -> RunProvenance:
    """Buildkite first, GitHub Actions fallback, both optional."""

    def _int(value):
        try:
            return int(value) if value else None
        except ValueError:
            return None

    return RunProvenance(
        commit_sha=os.environ.get("BUILDKITE_COMMIT") or os.environ.get("GITHUB_SHA") or "unknown",
        pr_number=_int(os.environ.get("BUILDKITE_PULL_REQUEST") if os.environ.get("BUILDKITE_PULL_REQUEST") != "false" else None)
        or _int(os.environ.get("GITHUB_PR_NUMBER")),
        ci_run_id=_int(os.environ.get("BUILDKITE_BUILD_NUMBER") or os.environ.get("GITHUB_RUN_ID")),
        ci_run_attempt=_int(os.environ.get("BUILDKITE_RETRY_COUNT") or os.environ.get("GITHUB_RUN_ATTEMPT")),
        pipeline=os.environ.get("BUILDKITE_PIPELINE_SLUG") or os.environ.get("GITHUB_EVENT_NAME"),
        ref=os.environ.get("BUILDKITE_BRANCH") or os.environ.get("GITHUB_REF"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", help="repo-relative test file carrying register_ci_gate declarations")
    parser.add_argument("--record-dir", help="directory of per-process JSONL records for this run")
    parser.add_argument("--db", default=os.environ.get("VIME_METRIC_HISTORY_PATH", "metric_history.db"))
    parser.add_argument("--suite", default="default", help="executing suite (identity's suite component)")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "rocm", "cpu"])
    parser.add_argument("--write-baseline", action="store_true", help="persist this run as trusted history")
    parser.add_argument("--mark-untrusted-sha", metavar="SHA", help="revoke trust for a commit, then exit")
    args = parser.parse_args()

    store = SQLiteMetricHistoryStore(args.db)
    if args.mark_untrusted_sha:
        n = store.mark_untrusted(commit_sha=args.mark_untrusted_sha)
        print(f"marked {n} run(s) untrusted for commit {args.mark_untrusted_sha}")
        return 0

    if not args.test_file or not args.record_dir:
        parser.error("--test-file and --record-dir are required unless --mark-untrusted-sha")

    specs = parse_ci_gate_specs(args.test_file)
    merged_path = os.path.join(args.record_dir, "_merged.jsonl")
    merge_records(args.record_dir, merged_path)

    result = evaluate_gate(
        args.test_file, merged_path, store, executing_suite=args.suite, backend=args.backend, specs=specs
    )

    enforced_failed = False
    spec_by_key = {(s.metric_key, s.steps_key, s.constraint_key): s for s in specs}
    for metric in result.metrics:
        spec = spec_by_key.get((metric.metric_key, metric.steps_key, metric.constraint_key))
        enforced = bool(spec and spec.enforce)
        verdict = metric.historical_status.value
        note = ""
        if enforced and metric.historical_status == GateStatus.FAIL and metric.baseline_n < MIN_ENFORCED_BASELINES:
            note = f" (NOT ENFORCED: insufficient history n={metric.baseline_n}<{MIN_ENFORCED_BASELINES})"
        elif enforced and metric.historical_status in (GateStatus.FAIL, GateStatus.ERROR):
            enforced_failed = True
        print(
            f"[{verdict}] {metric.metric_key} step={metric.step} cur={metric.current} "
            f"baseline_n={metric.baseline_n} mean={metric.baseline_mean}: {metric.reason}{note}"
        )

    if enforced_failed:
        print("GATE FAIL")
        return 1

    if args.write_baseline and result.trusted:
        values = [
            MetricSample(m.metric_key, m.steps_key, m.constraint_key, m.step, m.current)
            for m in result.metrics
            if m.step is not None and m.current is not None
        ]
        run_id = store.write_run(
            RunIdentity(args.test_file, args.backend, args.suite),
            provenance_from_env(),
            datetime.now(timezone.utc).isoformat(),
            True,
            values,
        )
        print(f"baseline written (run_id={run_id}, {len(values)} values)")

    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

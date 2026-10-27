#!/usr/bin/env python3
"""V1 verifier: end-to-end metric-history gate check (deterministic, no GPU).

Drives tests/ci/metric_history/run_gate.py against a synthetic test file and a
throwaway SQLite store:

1. seeds 12 trusted baseline runs of rollout/raw_reward == 1.0;
2. 2nd-run check: a run right after the first baseline already reads history
   (baseline_n >= 1, verdict "pass", not "inactive");
3. steady run (value 1.03, within the lower-only band) -> exit 0;
4. tampered value (0.5) -> exit 1;
5. tampered threshold (declaration adds a tight rel_up) -> same 1.03 run -> exit 1.

Exit 0 = all stages pass.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_GATE = HERE / "run_gate.py"

DECLARATION = 'register_ci_gate(metric_key="rollout/raw_reward", steps="last", constraint={"rel_down": 0.05})\n'
TAMPERED_DECLARATION = (
    'register_ci_gate(metric_key="rollout/raw_reward", steps="last", '
    'constraint={"rel_down": 0.05, "rel_up": 0.01})\n'
)


def make_test_file(root: Path, declaration: str) -> Path:
    # All runs share one logical test path: the gate's baseline identity keys on it.
    test_file = root / "test_fake_ci.py"
    test_file.write_text(declaration)
    return test_file


def make_record_dir(root: Path, name: str, value: float) -> Path:
    record_dir = root / name / "records"
    record_dir.mkdir(parents=True)
    (record_dir / "p0.jsonl").write_text(json.dumps({"metric": "rollout/raw_reward", "series": [[0, value]]}) + "\n")
    return record_dir


def invoke(db: Path, test_file: Path, record_dir: Path, write_baseline: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(RUN_GATE),
        "--test-file",
        str(test_file),
        "--record-dir",
        str(record_dir),
        "--db",
        str(db),
        "--suite",
        "v1-verify",
        "--backend",
        "cpu",
    ]
    if write_baseline:
        cmd.append("--write-baseline")
    env = dict(os.environ, BUILDKITE_COMMIT="v1-verify-sha", BUILDKITE_BUILD_NUMBER="1")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {label} {detail}")
    return ok


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="v1-gate-"))
    db = root / "history.db"
    failures = 0

    # Stage 1: seed 12 trusted baselines at 1.0.
    test_file = make_test_file(root, DECLARATION)
    for i in range(12):
        result = invoke(db, test_file, make_record_dir(root, f"seed{i}", 1.0), write_baseline=True)
        if result.returncode != 0:
            failures += not check(f"seed{i}", False, result.stdout + result.stderr)
    failures += not check("seeded 12 baselines", True)

    # Stage 2: a run after just one baseline reads history (no cold start).
    # (Fresh single-baseline DB to isolate the check.)
    db2 = root / "history2.db"
    invoke(db2, test_file, make_record_dir(root, "first", 1.0), write_baseline=True)
    result = invoke(db2, test_file, make_record_dir(root, "second", 1.0), write_baseline=False)
    ok = result.returncode == 0 and "baseline_n=1" in result.stdout and "[pass]" in result.stdout
    failures += not check("2nd run reads history", ok, result.stdout.strip().splitlines()[-2:] if ok else result.stdout)

    # Stage 3: steady run within the lower-only band passes with full history.
    result = invoke(db, test_file, make_record_dir(root, "steady", 1.03), write_baseline=False)
    ok = result.returncode == 0 and "baseline_n=12" in result.stdout
    failures += not check("steady run passes (history n=12)", ok, result.stdout + result.stderr)

    # Stage 4: tampered value fails.
    result = invoke(db, test_file, make_record_dir(root, "badvalue", 0.5), write_baseline=False)
    failures += not check("tampered value FAILs", result.returncode == 1, result.stdout.strip().splitlines()[-1])

    # Stage 5: tampered threshold (added tight upper band) turns the same
    # steady 1.03 run into a FAIL. The identity path must stay the original
    # file (--test-file doubles as identity), so rewrite its declaration.
    test_file.write_text(TAMPERED_DECLARATION)
    result = invoke(db, test_file, make_record_dir(root, "badthreshold", 1.03), write_baseline=False)
    test_file.write_text(DECLARATION)
    ok = result.returncode == 1 and "GATE FAIL" in result.stdout
    failures += not check("tampered threshold FAILs", ok, result.stdout.strip().splitlines()[-1])

    print(f"{'FAIL' if failures else 'PASS'} v1 gate e2e ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

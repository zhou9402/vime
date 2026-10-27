"""Self-checks for the FLOPs oracle (V5): the oracle must run standalone, the
verifier must pass on the oracle itself and fail on a mutated copy."""

import json
import subprocess
import sys
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent
ORACLE = ORACLE_DIR / "flops_utils_miles_ba28991834.py"
VERIFIER = ORACLE_DIR / "verify_flops.py"

sys.path.insert(0, str(ORACLE_DIR))
import flops_utils_miles_ba28991834 as oracle  # noqa: E402


def _run_verifier(impl: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFIER), "--impl", str(impl)],
        capture_output=True,
        text=True,
    )


def test_oracle_runs_standalone_on_all_configs():
    configs = json.loads((ORACLE_DIR / "configs.json").read_text())
    assert len(configs) == 8
    for entry in configs:
        model = oracle.model_flops_from_hf_config(dict(entry["config"]))
        fwd = model.fwd_flops([128, 1024])
        train = model.train_flops([128, 1024])
        assert fwd > 0 and train > 0, entry["name"]
        assert train <= 3 * fwd, entry["name"]


def test_verifier_identity_passes():
    result = _run_verifier(ORACLE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_verifier_detects_mutation(tmp_path):
    mutated = tmp_path / "mutated.py"
    source = ORACLE.read_text()
    needle = "lm_head = 2 * config.hidden_size"
    assert needle in source
    mutated.write_text(source.replace(needle, "lm_head = 2.01 * config.hidden_size"))
    result = _run_verifier(mutated)
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_verifier_skips_missing_impl(tmp_path):
    result = _run_verifier(tmp_path / "does_not_exist.py")
    assert result.returncode == 0
    assert "SKIP" in result.stdout

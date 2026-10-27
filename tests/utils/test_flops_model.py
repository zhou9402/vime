"""Unit tests for vime/utils/flops_model.py (F10-1)."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = REPO_ROOT / "tests" / "ci" / "flops_oracle"

sys.path.insert(0, str(REPO_ROOT))
from vime.utils import flops_model  # noqa: E402


def _config(name):
    configs = json.loads((ORACLE_DIR / "configs.json").read_text())
    return next(dict(entry["config"]) for entry in configs if entry["name"] == name)


def test_v5_verifier_passes_on_flops_model():
    result = subprocess.run(
        [sys.executable, str(ORACLE_DIR / "verify_flops.py"), "--impl", str(REPO_ROOT / "vime" / "utils" / "flops_model.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_dense_train_flops_is_3x_fwd():
    model = flops_model.model_flops_from_hf_config(_config("dense_gqa_llama3_8b"))
    seq_lens = [128, 1024, 7]
    assert model.train_flops(seq_lens) == 3 * model.fwd_flops(seq_lens)


def test_frozen_indexer_train_flops_below_3x():
    model = flops_model.model_flops_from_hf_config(_config("dsa_deepseek_v32"))
    assert model.has_indexers
    seq_lens = [512, 2048]
    assert model.train_flops(seq_lens) < 3 * model.fwd_flops(seq_lens)
    assert model.train_flops(seq_lens, trains_indexers=True) == 3 * model.fwd_flops(seq_lens)


def test_model_flops_from_args_missing_checkpoint():
    assert flops_model.model_flops_from_args(Namespace(hf_checkpoint=None)) is None


def test_model_flops_from_args_bad_path_returns_none():
    assert flops_model.model_flops_from_args(Namespace(hf_checkpoint="/nonexistent/model")) is None


def test_model_flops_from_args_loads_local_config(tmp_path):
    pytest.importorskip("transformers")
    config = _config("dense_gqa_llama3_8b")
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = flops_model.model_flops_from_args(Namespace(hf_checkpoint=str(tmp_path)))
    assert model is not None
    assert model.fwd_flops([64]) > 0

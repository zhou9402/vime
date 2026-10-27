"""Recorder tests for vime/observability/ci_metric_history.py (F10-6)."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vime.observability.ci_metric_history import CiMetricHistoryRecorder  # noqa: E402


def read_records(record_dir):
    lines = []
    for path in sorted(Path(record_dir).glob("*.jsonl")):
        lines.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return {rec["metric"]: rec["series"] for rec in lines}


def test_disabled_without_record_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("VIME_CI_GATE_RECORD_DIR", raising=False)
    recorder = CiMetricHistoryRecorder()  # env unset -> disabled
    assert not recorder.enabled
    recorder.log({"rollout/raw_reward": 1.0}, step=0)  # no-op, no crash
    assert list(tmp_path.iterdir()) == [] or read_records(tmp_path) == {}


def test_policy_prefix_matching(tmp_path, monkeypatch):
    monkeypatch.setenv("VIME_CI_GATE_POLICY_PREFIXES", "policy_b,policy_c")
    recorder = CiMetricHistoryRecorder(str(tmp_path))
    recorder.log(
        {
            "policy_b/rollout/raw_reward": 0.9,  # declared policy prefix: captured
            "eval/rollout/raw_reward": 0.1,  # eval/ is not a policy: NOT captured
            "unrelated/metric": 5.0,  # not whitelisted
            "rollout/step": 3,
        },
        step=3,
    )
    records = read_records(tmp_path)
    assert records["policy_b/rollout/raw_reward"] == [[3, 0.9]]
    assert "eval/rollout/raw_reward" not in records
    assert "unrelated/metric" not in records


def test_undeclared_prefix_not_captured(tmp_path, monkeypatch):
    monkeypatch.delenv("VIME_CI_GATE_POLICY_PREFIXES", raising=False)
    recorder = CiMetricHistoryRecorder(str(tmp_path))
    recorder.log({"policy_b/rollout/raw_reward": 0.9, "rollout/raw_reward": 0.7}, step=0)
    records = read_records(tmp_path)
    assert records == {"rollout/raw_reward": [[0, 0.7]]}


def test_numpy_and_torch_scalars_coerced(tmp_path):
    import numpy as np
    import torch

    recorder = CiMetricHistoryRecorder(str(tmp_path))
    recorder.log(
        {
            "train/ppo_kl": np.float32(0.5),
            "train/grad_norm": np.int64(2),
            "rollout/raw_reward": torch.tensor(0.7),  # 0-d tensor
        },
        step=1,
    )
    records = read_records(tmp_path)
    assert records["train/ppo_kl"] == [[1, pytest.approx(0.5, rel=1e-6)]]
    assert records["train/grad_norm"] == [[1, 2.0]]
    assert records["rollout/raw_reward"] == [[1, pytest.approx(0.7, rel=1e-6)]]
    # numpy bools are still rejected
    with pytest.raises(TypeError):
        recorder.log({"rollout/raw_reward": np.True_}, step=2)


def test_non_numeric_raises_type_error(tmp_path):
    recorder = CiMetricHistoryRecorder(str(tmp_path))
    with pytest.raises(TypeError):
        recorder.log({"rollout/raw_reward": "high"}, step=0)
    with pytest.raises(TypeError):
        recorder.log({"rollout/raw_reward": True}, step=0)
    with pytest.raises(TypeError):
        recorder.log({"rollout/raw_reward": None}, step=0)
    import torch

    with pytest.raises(TypeError):
        recorder.log({"rollout/raw_reward": torch.tensor([1.0, 2.0])}, step=0)  # not 0-d


def test_atomic_snapshot_and_accumulation(tmp_path):
    recorder = CiMetricHistoryRecorder(str(tmp_path))
    recorder.log({"train/ppo_kl": 0.01}, step=0)
    recorder.log({"train/ppo_kl": 0.02}, step=1)
    [path] = Path(tmp_path).glob("*.jsonl")
    assert not list(Path(tmp_path).glob("*.tmp"))
    records = json.loads(path.read_text())
    # full series rewritten every time: single line per metric in the latest snapshot
    assert records["series"] == [[0, 0.01], [1, 0.02]]


def test_non_finite_markers_round_trip(tmp_path):
    recorder = CiMetricHistoryRecorder(str(tmp_path))
    recorder.log({"train/grad_norm": float("nan")}, step=0)
    recorder.log({"train/grad_norm": float("inf")}, step=1)
    recorder.log({"train/grad_norm": float("-inf")}, step=2)
    series = read_records(tmp_path)["train/grad_norm"]
    assert series == [[0, "NaN"], [1, "Infinity"], [2, "-Infinity"]]
    # every line stays strict JSON
    for path in Path(tmp_path).glob("*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(AssertionError(x)))


def test_per_process_files_do_not_clobber(tmp_path):
    r1 = CiMetricHistoryRecorder(str(tmp_path))
    r2 = CiMetricHistoryRecorder(str(tmp_path))
    r1.log({"rollout/raw_reward": 1.0}, step=0)
    r2.log({"rollout/raw_reward": 2.0}, step=0)
    assert len(list(Path(tmp_path).glob("*.jsonl"))) == 2

"""CI metric-history capture (port of miles@ba28991834 utils/tracking_utils/ci_history.py).

Captures the fixed TARGET_METRIC_KEYS whitelist of training/rollout metrics from
the live process into one JSONL record file per process, for the metric-history
regression gate (tests/ci/metric_history/). Unlike miles this is not a
TrackingBackend: vime funnels all metric logging through
vime.observability.logging_utils.log(), which calls record_from_log() here.

* Collection is on only when VIME_CI_GATE_RECORD_DIR names a directory.
* Every record call atomically rewrites the whole snapshot (temp + rename), so a
  process killed mid-run still leaves a complete record.
* Non-finite values are recorded as "NaN"/"Infinity"/"-Infinity" string markers
  (strict JSON); non-numeric values raise TypeError -- an authoring error fails
  loudly instead of being dropped.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Metric keys captured for the history gate. Training keys are logged with
# step_key "train/step"; rollout and perf metrics use "rollout/step". A
# whitelisted key matches either exactly or as "<policy>/<key>" where <policy>
# is a declared multi-policy id (see POLICY_PREFIXES_ENV); an arbitrary prefix
# such as "eval/" never matches.
TARGET_METRIC_KEYS: tuple[str, ...] = (
    "train/grad_norm",
    "train/ppo_kl",
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_kl",
    "rollout/raw_reward",
    "rollout/tito_session_mismatch_rate/v1/assistant_text",
    "rollout/tito_session_mismatch_rate/v2/assistant_text",
    "perf/actor_train_mfu",
)

# Env var naming the directory the harness assigns for this run's records.
RECORD_DIR_ENV = "VIME_CI_GATE_RECORD_DIR"

# Optional comma-separated list of declared multi-policy ids whose
# "<policy>/<key>" metric forms are captured alongside the exact keys.
POLICY_PREFIXES_ENV = "VIME_CI_GATE_POLICY_PREFIXES"


def _policy_prefixes() -> tuple[str, ...]:
    raw = os.environ.get(POLICY_PREFIXES_ENV, "")
    return tuple(p for p in (part.strip() for part in raw.split(",")) if p)


def _compute_logged_key(metrics: dict[str, Any], key: str) -> str | None:
    if key in metrics:
        return key
    suffix = f"/{key}"
    for prefix in _policy_prefixes():
        logged = f"{prefix}{suffix}"
        if logged in metrics:
            return logged
    return None


def _coerce_value(key: str, value: Any) -> float:
    """Coerce a logged metric to float, accepting Python and numpy numeric
    scalars and 0-d torch tensors; strings/None/bools are authoring errors."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        raise TypeError(f"CI history metric {key!r} must be numeric, got {type(value).__name__}")
    # numpy bool_ / torch bool tensors pass float() but are not metrics
    if str(getattr(value, "dtype", "")) in ("bool", "torch.bool"):
        raise TypeError(f"CI history metric {key!r} must be numeric, got {type(value).__name__}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"CI history metric {key!r} must be numeric, got {type(value).__name__}") from None


def _encode_value(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


class CiMetricHistoryRecorder:
    """Accumulate target metrics in-process and persist the raw series to disk."""

    def __init__(self, record_dir: str | None = None) -> None:
        self._series: dict[str, list[tuple[int | None, float]]] = {}
        self._lock = threading.Lock()
        self._record_dir = record_dir if record_dir is not None else os.environ.get(RECORD_DIR_ENV)
        self._record_path: str | None = None
        if self._record_dir:
            os.makedirs(self._record_dir, exist_ok=True)
            self._record_path = os.path.join(self._record_dir, f"{os.getpid()}-{uuid.uuid4().hex}.jsonl")

    @property
    def enabled(self) -> bool:
        return self._record_path is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            captured: list[tuple[str, float]] = []
            for key in TARGET_METRIC_KEYS:
                if (logged_key := _compute_logged_key(metrics, key)) is None:
                    continue
                value = _coerce_value(logged_key, metrics[logged_key])
                captured.append((logged_key, value))
            for key, value in captured:
                self._series.setdefault(key, []).append((step, value))
            if captured:
                self._write_snapshot_locked()

    def finish(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._write_snapshot_locked()

    def _write_snapshot_locked(self) -> None:
        assert self._record_path is not None
        tmp_path = f"{self._record_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for key, points in self._series.items():
                line = {
                    "metric": key,
                    "series": [[step, _encode_value(value)] for step, value in points],
                }
                f.write(json.dumps(line, allow_nan=False) + "\n")
        os.replace(tmp_path, self._record_path)


_recorder: CiMetricHistoryRecorder | None = None
_recorder_checked = False


def _get_recorder() -> CiMetricHistoryRecorder | None:
    """Process-wide recorder, created on first use; None when the env var is unset."""
    global _recorder, _recorder_checked
    if not _recorder_checked:
        _recorder_checked = True
        if os.environ.get(RECORD_DIR_ENV):
            _recorder = CiMetricHistoryRecorder()
    return _recorder


def record_from_log(metrics: dict[str, Any], step_key: str) -> None:
    """Choke-point hook called from logging_utils.log(): capture whitelisted keys."""
    recorder = _get_recorder()
    if recorder is None:
        return
    step = metrics.get(step_key)
    recorder.log(metrics, step if isinstance(step, int) and not isinstance(step, bool) else None)

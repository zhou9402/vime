"""Persisted per-policy rollout positions for multi-policy checkpoint/resume.

Port of miles@ba28991834 miles/utils/multi_policy/checkpoint_state.py; the
atomic write helper is inlined (vime has no file_utils.atomic_write_text).
"""

import logging
import os
import tempfile
from pathlib import Path

import pydantic

logger = logging.getLogger(__name__)

MULTI_POLICY_STATE_DIRNAME = "multi_policy_state"


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp) if os.path.exists(tmp) else None
        raise


class MultiPolicyCheckpointState(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    leader_model_id: str
    rollout_ids: dict[str, int]

    def save(self, save_dir: Path) -> None:
        path = _state_path(save_dir, leader_rollout_id=self.rollout_ids[self.leader_model_id])
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, self.model_dump_json())
        logger.info(f"Saved multi policy checkpoint state {self.rollout_ids} to {path}")

    @classmethod
    def load(cls, save_dir: Path, *, leader_rollout_id: int) -> "MultiPolicyCheckpointState | None":
        path = _state_path(save_dir, leader_rollout_id=leader_rollout_id)
        if not path.exists():
            return None
        ans = cls.model_validate_json(path.read_text(encoding="utf-8"))
        recorded = ans.rollout_ids[ans.leader_model_id]
        if recorded != leader_rollout_id:
            raise ValueError(
                f"{path} records its leader policy {ans.leader_model_id!r} at rollout {recorded}, but it is "
                f"the record of rollout {leader_rollout_id}"
            )
        return ans


def _state_path(save_dir: Path, *, leader_rollout_id: int) -> Path:
    return Path(save_dir) / MULTI_POLICY_STATE_DIRNAME / f"{leader_rollout_id}.json"

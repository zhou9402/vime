"""Display debug rollout CLI: whitelisted sample keys and metrics over .pt dumps and straw archives."""

import pytest
import torch

from vime.utils.types import Sample


def _samples():
    return [
        Sample(index=0, prompt="p", response="r", response_length=1, reward=1.0),
        Sample(index=1, prompt="p2", response="r2", response_length=2, reward=0.0),
    ]


def test_display_pt_dump(tmp_path, capsys):
    from vime.tools.display_debug_rollout_data import main

    torch.save({"rollout_id": 0, "samples": [s.to_dict() for s in _samples()]}, tmp_path / "rollout_0.pt")
    main(load_debug_rollout_data_path=str(tmp_path / "rollout_{rollout_id}.pt"), category=["train"])
    out = capsys.readouterr().out
    assert "rollout_id=0" in out
    assert '"reward"' in out and '"response_length"' in out
    # non-whitelisted tensor fields must not appear
    assert "rollout_log_probs" not in out


def test_display_straw_archive(tmp_path, capsys):
    from vime.data.archive import RolloutArchive
    from vime.tools.display_debug_rollout_data import main

    RolloutArchive.save(tmp_path / "rollout_0.straw.json", _samples(), rollout_id=0, evaluation=False)
    main(load_debug_rollout_data_path=str(tmp_path / "rollout_{rollout_id}.straw.json"), category=["train"])
    out = capsys.readouterr().out
    assert '"reward"' in out
    assert "rollout_id=0" in out


def test_display_missing_files_is_quiet(tmp_path, capsys):
    from vime.tools.display_debug_rollout_data import main

    main(load_debug_rollout_data_path=str(tmp_path / "rollout_{rollout_id}.pt"), category=["train"])
    assert capsys.readouterr().out == ""

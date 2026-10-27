"""Replay reward CLI: re-run a custom reward function against a saved rollout dump."""

import sys
import textwrap

import pytest
import torch

from vime.utils.types import Sample


@pytest.fixture
def dump_path(tmp_path):
    samples = [Sample(index=0, prompt="p", response="r", response_length=1), Sample(index=1, prompt="p2", response="r2", response_length=2)]
    path = tmp_path / "rollout.pt"
    torch.save({"rollout_id": 0, "samples": [s.to_dict() for s in samples]}, path)
    return path


@pytest.fixture
def rm_module(tmp_path, monkeypatch):
    (tmp_path / "replay_rm.py").write_text(
        textwrap.dedent(
            """
            async def replay_rm(args, sample):
                return 0.25
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return "replay_rm.replay_rm"


def test_replay_reward_fn(dump_path, rm_module, capsys):
    from vime.tools.replay_reward_fn import main

    main(rollout_data_path=str(dump_path), custom_rm_path=rm_module)
    out = capsys.readouterr().out
    assert "Reward:   0.25" in out
    assert out.count("Reward:   0.25") == 2
    assert "Sample 1/2" in out and "Sample 2/2" in out


def test_cli_help():
    from typer.testing import CliRunner

    from vime.tools.replay_reward_fn import main

    runner = CliRunner()
    import typer

    app = typer.Typer()
    app.command()(main)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

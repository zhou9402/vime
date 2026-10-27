"""Snapshot the expanded MODEL_ARGS of every scripts/models/*.sh (vime has no python launcher era)."""

import subprocess

import pytest

from tests.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "model_args"

_MODEL_SCRIPTS = sorted((REPO_ROOT / "scripts" / "models").glob("*.sh"))


def _expand_model_args(script) -> list[str]:
    result = subprocess.run(
        ["bash", "-c", 'source "$1" && printf "%s\\n" "${MODEL_ARGS[@]}"', "_", str(script)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/root", "LANG": "C", "LC_ALL": "C"},
    )
    assert result.returncode == 0, f"{script.name}: {result.stderr}"
    return [line for line in result.stdout.splitlines()]


@pytest.fixture(params=[p.name for p in _MODEL_SCRIPTS], scope="module")
def tokens(request):
    return request.param, _expand_model_args(REPO_ROOT / "scripts" / "models" / request.param)


class TestModelArgs:
    def test_tokens_match_snapshot(self, tokens):
        name, actual = tokens
        assert_matches_snapshot(_SNAPSHOT_DIR / f"{name}.txt", "\n".join(actual) + "\n", f"model args of {name}")

    def test_every_token_is_a_flag_or_scalar(self, tokens):
        name, actual = tokens
        assert actual[0].startswith("--"), f"{name}: first token {actual[0]!r} is not a flag"
        for token in actual:
            assert token == token.strip() and " " not in token and "\t" not in token, (
                f"{name}: token {token!r} contains whitespace and would split"
            )


class TestModelArgsDiscovery:
    def test_snapshot_stems_equal_script_stems(self):
        scripts = {f"{p.name}.txt" for p in _MODEL_SCRIPTS}
        snapshots = {p.name for p in _SNAPSHOT_DIR.glob("*.txt")}
        assert scripts == snapshots
        assert len(scripts) > 10

from dataclasses import dataclass, field

import pytest

from tests.launch_scripts.sh_harness import (
    REPO_ROOT,
    assert_matches_snapshot,
    format_invocations,
    iter_launch_scripts,
    run_launch_script,
)

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "sh"


@dataclass(frozen=True)
class LaunchScriptCase:
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


_HEAD_NODE_IP = "10.0.0.1"

# Multi-node launchers and scripts that read required inputs; add an entry with
# explicit inputs rather than weakening the returncode assertion.
_SCRIPTS_REFUSING_TO_RUN_WITHOUT_EXPLICIT_INPUTS: dict[str, LaunchScriptCase] = {
    "scripts/run-qwen3-235B-A22B.sh": LaunchScriptCase(env={"BASE_FOLDER": "/frozen/checkpoints"}),
    "scripts/run-qwen3-235B-A22B-sft.sh": LaunchScriptCase(env={"BASE_FOLDER": "/frozen/checkpoints"}),
    "scripts/run-qwen3-next-80B-A3B.sh": LaunchScriptCase(env={"BASE_FOLDER": "/frozen/checkpoints"}),
    "scripts/run-qwen3.5-27B.sh": LaunchScriptCase(env={"BASE_FOLDER": "/frozen/checkpoints"}),
    "scripts/run-qwen3.5-35B-A3B-sft.sh": LaunchScriptCase(env={"BASE_FOLDER": "/frozen/checkpoints"}),
    "scripts/run-glm5.2-744B-A40B.sh": LaunchScriptCase(env={"BASE_DIR": "/frozen/shared"}),
}

# Launchers the shimmed harness cannot run, each with the reason. The test below
# pins the reason: an entry whose script starts passing must be removed, and a
# newly failing script must get an entry with its reason — never a silent skip.
_SCRIPTS_NOT_RUNNABLE_UNDER_THE_HARNESS: dict[str, str] = {
    "examples/dspark/run-qwen3-4B-dspark-colocate.sh": "sources the absolute /root/vime checkout path",
    "examples/dspark/run-qwen3-4B-dspark-non-colocate.sh": "sources the absolute /root/vime checkout path",
    "examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh": "sources the absolute /root/vime checkout path",
    "examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh": "sources the absolute /root/vime checkout path",
    "examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh": "sources the absolute /root/vime checkout path",
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh": "starts and polls a real teacher server",
    "examples/coding_agent_rl/run_qwen38_27b_sunabako_2plus2.sh": "needs prepared dataset files and live cluster peers",
}

_SCRIPTS = [
    script.relative_to(REPO_ROOT).as_posix()
    for script in iter_launch_scripts()
    if script.relative_to(REPO_ROOT).as_posix() not in _SCRIPTS_NOT_RUNNABLE_UNDER_THE_HARNESS
]


@pytest.fixture(params=_SCRIPTS, scope="module")
def recorded(request, tmp_path_factory):
    rel = request.param
    case = _SCRIPTS_REFUSING_TO_RUN_WITHOUT_EXPLICIT_INPUTS.get(rel, LaunchScriptCase())
    tmp_path = tmp_path_factory.mktemp("launch_script")
    workdir = tmp_path / "workdir"
    run = run_launch_script(
        REPO_ROOT / rel,
        sandbox=tmp_path,
        args=case.args,
        extra_env={key: value.format(workdir=workdir) for key, value in case.env.items()},
    )
    return rel, run


class TestEveryLaunchScript:
    def test_invocations_match_snapshot(self, recorded):
        """Every launch script must issue exactly the recorded sequence of external commands."""
        rel, run = recorded
        snapshot = _SNAPSHOT_DIR / f"{rel}.txt"
        actual = f"# returncode: {run.returncode}\n\n{format_invocations(run.invocations)}"

        assert_matches_snapshot(snapshot, actual, rel)

    def test_submits_exactly_one_ray_job(self, recorded):
        """A launch script that no longer reaches `ray job submit` is broken, whatever else it does."""
        _, run = recorded
        assert run.returncode == 0
        assert len(run.ray_job_submit_argv()) > 10


class TestDiscovery:
    def test_every_discovered_script_has_a_snapshot_and_vice_versa(self):
        """A script that stops matching the discovery filter would otherwise vanish silently."""
        discovered = {f"{rel}.txt" for rel in _SCRIPTS}
        recorded = {path.relative_to(_SNAPSHOT_DIR).as_posix() for path in _SNAPSHOT_DIR.rglob("*.txt")}

        assert discovered == recorded
        assert len(discovered) > 30

    def test_every_excluded_script_still_fails_for_its_recorded_reason(self, tmp_path_factory):
        """The exclusion list pins the harness gap: a script that starts passing must leave it."""
        for rel, reason in _SCRIPTS_NOT_RUNNABLE_UNDER_THE_HARNESS.items():
            assert (REPO_ROOT / rel).exists(), f"{rel} is gone; drop its exclusion entry ({reason})"
            tmp_path = tmp_path_factory.mktemp("excluded")
            run = run_launch_script(REPO_ROOT / rel, sandbox=tmp_path, timeout=30)
            submits = [argv for argv in run.invocations_of("ray") if argv[1:3] == ["job", "submit"]]
            assert run.returncode != 0 or len(submits) != 1, (
                f"{rel} now runs under the harness; move it out of _SCRIPTS_NOT_RUNNABLE_UNDER_THE_HARNESS ({reason})"
            )

    def test_a_shell_script_outside_the_snapshots_only_delegates(self):
        """Shell launchers stay Ray only, so the snapshots are their whole protection against drift."""
        recorded = {path.stem for path in _SNAPSHOT_DIR.rglob("*.txt")}
        undiscovered = [
            path
            for path in (REPO_ROOT / "scripts").rglob("*.sh")
            if "models" not in path.relative_to(REPO_ROOT / "scripts").parts
            and not path.name.startswith("_")
            and path.name not in recorded
            and "ray job submit" not in path.read_text(errors="replace")
        ]

        for path in undiscovered:
            text = path.read_text(errors="replace")
            delegates_to = [line for line in text.splitlines() if ".py" in line and "python" in line]
            assert delegates_to, f"{path.name} launches training without a snapshot covering it"

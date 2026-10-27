from tests.launch_scripts.sh_harness import REPO_ROOT

_HARDCODED_CHECKOUTS = ("/root/miles", "/workspace/miles", "/root/vime", "/workspace/vime")

# Pre-existing debt, frozen so it cannot grow: removing the hardcoded path from
# a script means removing it here; a new offender fails this test.
_KNOWN_OFFENDERS = [
    "examples/dspark/run-qwen3-4B-dspark-colocate.sh",
    "examples/dspark/run-qwen3-4B-dspark-non-colocate.sh",
    "examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh",
    "examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh",
    "examples/on_policy_distillation/run-qwen3-8B-opd.sh",
    "examples/train_infer_mismatch_helper/run-qwen3-4b-mis.sh",
    "scripts/run-glm5.2-744B-A40B.sh",
]


class TestShellScriptHygiene:
    def test_no_shell_script_hardcodes_the_checkout_location(self):
        """A script that assumes one absolute checkout only runs inside one container image."""
        offenders = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for root in (REPO_ROOT / "scripts", REPO_ROOT / "examples")
            for path in root.rglob("*.sh")
            for text in [path.read_text(errors="replace")]
            if any(hardcoded in text for hardcoded in _HARDCODED_CHECKOUTS)
        )

        assert offenders == sorted(_KNOWN_OFFENDERS), (
            "checkout-path offenders changed: fix the script and shrink _KNOWN_OFFENDERS, "
            "never grow it"
        )

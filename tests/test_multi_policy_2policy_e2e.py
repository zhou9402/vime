"""Two-policy (solver + verifier) end-to-end test — GPU-only, run by CI.

Trains two policies (Qwen2.5-0.5B each) with train_multi_policy.py against a
shared fully-async rollout queue: one generate call emits solver and verifier
samples (examples/multi_policy/generate_solver_verifier.py), each policy
drains only its own samples, updates only its own vLLM server's weights, and
the leader drives global saves.

Verifier (deterministic): after the run, the multi-policy checkpoint state at
the leader's final rollout must record both policies, and both policies'
checkpoints must exist under <save>/trainers/<trainer_id>. Reward history is
tracked (not gated) via the F10 metric-history gate declarations below; CI
evaluates them with tests/ci/metric_history/run_gate.py.

Requires 8 GPUs; not run on CPU dev machines.
"""

import json
import os
import tempfile
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U

# F10 metric-history gate declarations (runtime no-ops; parsed via AST by the
# gate runner). constraint=None = track-only until enough trusted baselines.
try:
    from tests.ci.metric_history.register import register_ci_gate
except Exception:  # F10 not merged into this branch yet

    def register_ci_gate(**_kwargs):
        return None


register_ci_gate(metric_key="train/grad_norm", steps="last", constraint=None)
register_ci_gate(metric_key="rollout/reward", steps="last", constraint=None)

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_ROLLOUT = 4
POLICIES = ("solver", "verifier")


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _write_configs(tmpdir: str) -> tuple[str, str]:
    policy_config = {
        "trainers": [
            {"model_id": "solver", "overrides": {"lr": 1.0e-6}},
            {"model_id": "verifier", "overrides": {"lr": 2.0e-6}},
        ]
    }
    vllm_config = {
        "models": [
            {
                "name": policy,
                "update_weights": True,
                "server_groups": [{"num_gpus_per_engine": 1, "num_engines": 1}],
            }
            for policy in POLICIES
        ]
    }
    import yaml

    policy_path = os.path.join(tmpdir, "policies.yaml")
    vllm_path = os.path.join(tmpdir, "vllm_models.yaml")
    Path(policy_path).write_text(yaml.safe_dump(policy_config))
    Path(vllm_path).write_text(yaml.safe_dump(vllm_config))
    return policy_path, vllm_path


def execute(save_dir: str):
    policy_config_path, vllm_config_path = _write_configs(save_dir)

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "
    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--rollout-function-path vime.rollout.fully_async_rollout.generate_rollout_fully_async "
        "--custom-generate-function-path examples.multi_policy.generate_solver_verifier.generate "
        "--max-weight-staleness 2 "
    )
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )
    optimizer_args = "--optimizer adam --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "  # per policy; the entry allocates one placement group per policy
        "--save-interval 2 "
        "--ci-test "
        f"--save {quote(save_dir)} "
        f"--multi-policy-config {quote(policy_config_path)} "
        f"--vllm-config {quote(vllm_config_path)} "
    )

    U.execute_train(
        train_args=f"{ckpt_args} {rollout_args} {perf_args} {optimizer_args} {grpo_args} {misc_args}",
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        config=U.ExecuteTrainConfig(train_script="train_multi_policy.py"),
    )


def verify(save_dir: str):
    """Deterministic pass/fail: checkpoint state + both policies' checkpoints."""
    state_path = Path(save_dir) / "multi_policy_state" / f"{NUM_ROLLOUT - 1}.json"
    if not state_path.exists():
        raise RuntimeError(f"missing multi-policy checkpoint state {state_path}")
    state = json.loads(state_path.read_text())
    if state["leader_model_id"] != "solver":
        raise RuntimeError(f"unexpected leader {state['leader_model_id']!r}")
    if set(state["rollout_ids"]) != set(POLICIES):
        raise RuntimeError(f"state file misses policies: {state['rollout_ids']}")
    for policy in POLICIES:
        ckpt = Path(save_dir) / "trainers" / f"{policy}-actor"
        if not (ckpt / "latest_checkpointed_iteration.txt").exists():
            raise RuntimeError(f"policy {policy!r} has no checkpoint under {ckpt}")
    print(f"PASS: multi-policy state {state['rollout_ids']}, both policies checkpointed")


def main():
    with tempfile.TemporaryDirectory(prefix="vime_multi_policy_e2e_") as tmpdir:
        save_dir = os.path.join(tmpdir, "save")
        execute(save_dir)
        verify(save_dir)


if __name__ == "__main__":
    main()

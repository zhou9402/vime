"""E2E: dedicated eval fleet must not stall training rollout.

Runs the same tiny Qwen3.5-0.8B job twice in one session:

1. baseline: eval on the shared rollout engines (blocking, --eval-num-gpus 0)
2. fleet:    eval on a dedicated 1-GPU fleet (--eval-num-gpus 1) dispatched
             non-blockingly from HF snapshots

Gate (F06 decision 4): per-step rollout time with the fleet on must be
<= 1.10x the baseline, measured in the same run. We approximate step time by
total training wall time over --num-rollout steps with --eval-interval 1, so
every step carries an eval in both runs; in the baseline run that eval blocks
the loop, in the fleet run it must not.

GPU-only: CI executes this file as a script (prepare/execute), like
test_full_disk_weight_update.py.
"""

import os
import time

import vime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
NUM_GPUS = 4  # 1 trainer + 2 rollout + 1 eval
TORCH_DIST_CKPT = f"/dev/shm/{MODEL_NAME}_torch_dist"

# Fleet wall time may exceed baseline by at most 10% (plus fixed slack for
# startup noise on tiny runs).
MAX_SLOWDOWN = 1.10
SLACK_SECONDS = 60.0


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/dev/shm",
    )


def _train_args(eval_fleet: bool, eval_hf_dir: str) -> str:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load {TORCH_DIST_CKPT} "
    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 6 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 8 "
        "--dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 16 "
        "--eval-interval 1 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--eval-input-key messages "
        "--eval-label-key label "
    )
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    fleet_args = (
        f"--eval-num-gpus 1 --eval-num-gpus-per-engine 1 --eval-hf-dir {eval_hf_dir} "
        if eval_fleet
        else ""
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--loss-mask-type qwen3_5 "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus-per-engine 1 "
        "--rollout-num-gpus 2 "
        "--vllm-gpu-memory-utilization 0.7 "
        "--ci-test "
    )
    return (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} {perf_args} "
        f"{U.get_default_wandb_args(__file__)} {fleet_args} {misc_args}"
    )


def _timed_train(eval_fleet: bool, eval_hf_dir: str) -> float:
    start = time.time()
    U.execute_train(
        train_args=_train_args(eval_fleet, eval_hf_dir),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )
    return time.time() - start


def execute():
    baseline_s = _timed_train(False, "/dev/shm/eval_fleet_baseline_unused")
    fleet_s = _timed_train(True, "/dev/shm/eval_fleet_snapshots")
    print(f"eval-fleet non-stall gate: baseline={baseline_s:.1f}s fleet={fleet_s:.1f}s")
    assert fleet_s <= baseline_s * MAX_SLOWDOWN + SLACK_SECONDS, (
        f"Eval fleet stalled rollout: fleet={fleet_s:.1f}s > {MAX_SLOWDOWN}x baseline={baseline_s:.1f}s "
        f"(+{SLACK_SECONDS:.0f}s slack)"
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()

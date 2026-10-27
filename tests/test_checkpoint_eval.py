"""CheckpointEvalFn contract tests (CPU, no Ray/GPU)."""

import asyncio
from argparse import Namespace

import pytest

from vime.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
)
from vime.rollout.checkpoint_eval import (
    CheckpointEvalFn,
    EvalSkip,
    is_checkpoint_eval_fn,
    retarget_args,
)


class _RecordingEvalFn(CheckpointEvalFn):
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0], "checkpoint": checkpoint_dir}})


class _SkipEvalFn(CheckpointEvalFn):
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        raise EvalSkip("unhealthy")


def _ctor():
    return RolloutFnConstructorInput(args=Namespace(vllm_router_ip="10.0.0.1", vllm_router_port=3000), data_source=None)


def test_is_checkpoint_eval_fn():
    assert is_checkpoint_eval_fn(f"{__name__}._RecordingEvalFn")
    assert not is_checkpoint_eval_fn("vime.rollout.vllm_rollout.generate_rollout")
    assert not is_checkpoint_eval_fn(None)


def test_call_requires_eval_input():
    fn = _RecordingEvalFn(_ctor())
    with pytest.raises(AssertionError, match="only serves eval"):
        asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))


def test_call_requires_hf_dir():
    fn = _RecordingEvalFn(_ctor())
    with pytest.raises(AssertionError, match="no snapshot"):
        asyncio.run(fn(RolloutFnEvalInput(rollout_id=0)))


def test_call_delegates_with_hf_dir():
    fn = _RecordingEvalFn(_ctor())
    out = asyncio.run(fn(RolloutFnEvalInput(rollout_id=4, hf_dir="/tmp/step_4", weight_version="4")))
    assert isinstance(out, RolloutFnEvalOutput)
    assert out.data["ds"]["checkpoint"] == "/tmp/step_4"


def test_eval_skip_propagates_with_reason():
    fn = _SkipEvalFn(_ctor())
    with pytest.raises(EvalSkip) as exc:
        asyncio.run(fn(RolloutFnEvalInput(rollout_id=0, hf_dir="/tmp/x")))
    assert exc.value.reason == "unhealthy"


def test_retarget_args_swaps_router_only():
    args = Namespace(vllm_router_ip="10.0.0.1", vllm_router_port=3000, hf_checkpoint="/m", other=5)
    retargeted = retarget_args(args=args, router_ip="10.0.0.9", router_port=4000, engine_gpu_counts=[1, 1])
    assert (retargeted.vllm_router_ip, retargeted.vllm_router_port) == ("10.0.0.9", 4000)
    assert retargeted.eval_engine_gpu_counts == [1, 1]
    assert retargeted.hf_checkpoint == "/m" and retargeted.other == 5
    # original untouched
    assert (args.vllm_router_ip, args.vllm_router_port) == ("10.0.0.1", 3000)


def test_reference_impl_is_checkpoint_eval_fn():
    assert is_checkpoint_eval_fn("examples.eval.external_vllm_eval_fn.ExternalVllmEvalFn")


def test_dispose_default_noop():
    fn = _RecordingEvalFn(_ctor())
    assert asyncio.run(fn.dispose()) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

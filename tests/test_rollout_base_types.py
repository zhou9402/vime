"""Contract tests for the BaseRolloutFn / RolloutFn*Input extension of base_types.

Port of miles@ba28991834 miles/rollout/base_types.py (class-based rollout fn API).
"""

import pytest

from vime.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)


class _DummyRolloutFn(BaseRolloutFn):
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return RolloutFnEvalOutput(data={"ds": {"rewards": [1.0]}})
        return RolloutFnTrainOutput(samples=[])


def test_constructor_input_stored():
    fn = _DummyRolloutFn(RolloutFnConstructorInput(args=None, data_source=None))
    assert fn.constructor_input.args is None


def test_train_input_evaluation_flag_and_defaults():
    inp = RolloutFnTrainInput(rollout_id=3)
    assert inp.evaluation is False
    assert inp.weight_version is None
    assert inp.trainer_model_id is None


def test_eval_input_fields():
    inp = RolloutFnEvalInput(rollout_id=5, weight_version="5", hf_dir="/tmp/x")
    assert inp.evaluation is True
    assert inp.invocation_id  # uuid4 hex generated
    other = RolloutFnEvalInput(rollout_id=5)
    assert other.invocation_id != inp.invocation_id


def test_base_input_evaluation_not_implemented():
    from vime.rollout.base_types import RolloutFnBaseInput

    with pytest.raises(NotImplementedError):
        _ = RolloutFnBaseInput(rollout_id=0).evaluation


def test_abstract_call_enforced():
    class _Missing(BaseRolloutFn):
        pass

    with pytest.raises(TypeError):
        _Missing(RolloutFnConstructorInput(args=None, data_source=None))


def test_call_dispatch_by_evaluation():
    fn = _DummyRolloutFn(RolloutFnConstructorInput(args=None, data_source=None))
    train_out = fn(RolloutFnTrainInput(rollout_id=0))
    eval_out = fn(RolloutFnEvalInput(rollout_id=0, hf_dir="/tmp/x"))
    assert isinstance(train_out, RolloutFnTrainOutput)
    assert isinstance(eval_out, RolloutFnEvalOutput)


def test_optional_hooks_are_noops():
    import asyncio
    import tempfile

    fn = _DummyRolloutFn(RolloutFnConstructorInput(args=None, data_source=None))
    with tempfile.TemporaryDirectory() as d:
        assert fn.save(d) is None
        assert fn.load(d) is None
    assert asyncio.run(fn.dispose()) is None

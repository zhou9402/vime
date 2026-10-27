"""CPU tests for vime/utils/replay_base.py (IndexerReplayManager)."""

import pytest
import torch

import vime.utils.replay_base as replay_base_module
from vime.utils.replay_base import BaseReplayManager, Replay, indexer_replay_manager


@pytest.fixture(autouse=True)
def _cpu_current_device(monkeypatch):
    # CPU CI has no accelerator backend; replay buffers land on the caller's device.
    monkeypatch.setattr(replay_base_module, "current_device", lambda: "cpu")


class _FakeManager(BaseReplayManager):
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2


def _topk(scores, k):
    return torch.topk(scores, k, dim=-1).indices.int()


def test_record_and_replay_round_trip():
    manager = _FakeManager()
    manager.enabled = True
    replay = manager.create_replay(stream_idx=0)
    manager.set_current(replay)
    torch.manual_seed(0)
    scores = torch.randn(8, 64)
    wrapped = manager.get_topk_fn(_topk, return_probs=False)

    manager.stage = "record"
    recorded = wrapped(scores, 16)
    assert replay.top_indices_list[0].device.type == "cpu"
    assert torch.equal(recorded, _topk(scores, 16))

    manager.stage = "replay_forward"
    replayed = wrapped(scores, 16)
    assert torch.equal(replayed, recorded)

    manager.stage = "replay_backward"
    assert torch.equal(wrapped(scores, 16), recorded)

    with pytest.raises(IndexError, match="pop_forward out of range"):
        manager.stage = "replay_forward"
        replay.forward_index = 99
        wrapped(scores, 16)


def test_disabled_and_fallthrough_pass_through():
    manager = _FakeManager()
    scores = torch.randn(4, 32)
    wrapped = manager.get_topk_fn(_topk, return_probs=False)
    assert torch.equal(wrapped(scores, 8), _topk(scores, 8))
    manager.enabled = True
    manager.stage = "fallthrough"
    assert torch.equal(wrapped(scores, 8), _topk(scores, 8))


def test_all_invalid_rows_filled_with_arange():
    manager = _FakeManager()
    manager.enabled = True
    replay = manager.create_replay()
    manager.set_current(replay)
    manager.stage = "record"
    scores = torch.randn(3, 16)
    wrapped = manager.get_topk_fn(_topk, return_probs=False)
    wrapped(scores, 8)
    # corrupt the record to all-padding and check the arange fill on replay
    # (replace rather than fill_: mutating pinned CPU tensors hits a torch<2.14
    # dispatch bug on MPS builds)
    replay.top_indices_list[0] = torch.full_like(replay.top_indices_list[0], -1)
    manager.stage = "replay_forward"
    replayed = wrapped(scores, 8)
    assert torch.equal(replayed, torch.arange(8, dtype=torch.int32).expand(3, 8) % 16)


def test_return_probs_gathers_scores():
    manager = _FakeManager()
    manager.enabled = True
    replay = manager.create_replay()
    manager.set_current(replay)
    manager.stage = "record"
    scores = torch.randn(4, 32)
    wrapped = manager.get_topk_fn(lambda s, k: (torch.topk(s, k, -1).values, torch.topk(s, k, -1).indices.int()), True)
    wrapped(scores, 8)
    manager.stage = "replay_forward"
    probs, indices = wrapped(scores, 8)
    assert torch.equal(probs, scores.gather(1, indices.long()))


def test_register_to_module_sets_current_on_forward():
    manager = _FakeManager()
    manager.enabled = True
    module = torch.nn.Linear(4, 4)
    manager.register_to_module(module, "indexer_replay", stream_idx=2)
    assert module.indexer_replay.stream_idx == 2
    module(torch.randn(1, 4))
    assert manager.get_current() is module.indexer_replay


def test_disabled_register_is_noop():
    manager = _FakeManager()
    module = torch.nn.Linear(4, 4)
    manager.register_to_module(module, "indexer_replay")
    assert not hasattr(module, "indexer_replay")


def test_check_replay_result_overlap_threshold():
    manager = _FakeManager()
    manager.enable_check_replay_result = True
    manager.replay_check_min_overlap_ratio = 0.5
    scores = torch.randn(2, 16)
    topk = 4
    good = _topk(scores, topk)
    manager.check_replay_result(_topk, scores, topk, good)  # identical: no raise
    bad = torch.full_like(good, 0)  # zero overlap on most rows
    with pytest.raises(AssertionError, match="R3 mismatch"):
        manager.check_replay_result(_topk, scores, topk, bad)


def test_indexer_manager_config():
    assert indexer_replay_manager.name == "indexer"
    assert indexer_replay_manager.data_key == "rollout_indexer_topk"
    assert indexer_replay_manager.replay_indices_are_token_positions is True
    assert indexer_replay_manager.if_sp_region is False


def test_replay_clear():
    replay = Replay()
    replay.record(torch.zeros(2, 4, dtype=torch.int32))
    replay.pop_forward()
    replay.clear()
    assert replay.top_indices_list == [] and replay.forward_index == 0

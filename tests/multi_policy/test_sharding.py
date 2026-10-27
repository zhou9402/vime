"""Unit tests for per-policy group sharding (vime/utils/multi_policy/sharding.py)."""

import pytest

from vime.utils.multi_policy.sharding import group_policy_id, split_group_by_policy
from vime.utils.types import Sample


def _sample(model_id, text="x"):
    return Sample(prompt="p", response=text, trainer_model_id=model_id)


def test_group_policy_id_homogeneous_and_unstamped():
    assert group_policy_id([_sample("solver"), _sample("solver")]) == "solver"
    assert group_policy_id([Sample(), Sample()]) is None
    assert group_policy_id([[_sample("a")], [_sample("a")]]) == "a"  # nested trajectory


def test_group_policy_id_rejects_mixed():
    with pytest.raises(ValueError, match="homogeneous"):
        group_policy_id([_sample("solver"), _sample("verifier")])


def test_split_partition_preserves_all_samples():
    group = [_sample("solver", "s1"), _sample("verifier", "v1"), _sample("solver", "s2")]
    own, foreign = split_group_by_policy(group, "solver")
    assert [s.response for s in own] == ["s1", "s2"]
    assert foreign == [[group[1]]]
    # nothing lost: partition covers the original multiset of samples
    rejoined = sorted((id(s) for s in own + [s for fg in foreign for s in fg]))
    assert rejoined == sorted(id(s) for s in group)


def test_split_all_foreign_returns_none():
    own, foreign = split_group_by_policy([_sample("verifier")], "solver")
    assert own is None
    assert len(foreign) == 1


def test_split_groups_foreign_by_policy():
    group = [_sample("a"), _sample("b"), _sample("c"), _sample("a")]
    own, foreign = split_group_by_policy(group, "a")
    assert len(own) == 2
    assert {fg[0].trainer_model_id for fg in foreign} == {"b", "c"}


def test_split_rejects_unstamped():
    with pytest.raises(ValueError, match="trainer_model_id"):
        split_group_by_policy([Sample()], "solver")


def test_partition_group_by_policy():
    from vime.utils.multi_policy.sharding import partition_group_by_policy

    group = [_sample("solver", "s1"), _sample("verifier", "v1"), _sample("solver", "s2")]
    parts = partition_group_by_policy(group)
    assert set(parts) == {"solver", "verifier"}
    assert [s.response for s in parts["solver"]] == ["s1", "s2"]
    assert [s.response for s in parts["verifier"]] == ["v1"]


def test_partition_rejects_unstamped():
    from vime.utils.multi_policy.sharding import partition_group_by_policy

    with pytest.raises(ValueError, match="trainer_model_id"):
        partition_group_by_policy([Sample()])

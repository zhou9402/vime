"""Per-policy group sharding for multi-policy rollout drains.

Pure helpers shared by the fully-async drain (vime/rollout/fully_async_rollout.py)
and tests. The contract mirrors miles' DefaultMultiDataBuffer split-at-put:
every leaf sample must be stamped with ``Sample.trainer_model_id`` by the
custom generate function, and one top-level item (a Sample, or a list of
Samples for multi-sample trajectories) must be single-policy.
"""


def _iter_leaf_samples(group):
    for item in group:
        if isinstance(item, list):
            yield from _iter_leaf_samples(item)
        else:
            yield item


def group_policy_id(group) -> str | None:
    """The single policy a group trains; None when no sample carries a policy stamp."""
    ids = {s.trainer_model_id for s in _iter_leaf_samples(group)}
    if len(ids) > 1:
        raise ValueError(
            f"multi-policy rollout requires homogeneous groups: one group mixes trainer_model_id "
            f"{sorted(ids)}; the generate function must emit one policy per group item"
        )
    return next(iter(ids)) if ids else None


def split_group_by_policy(group, policy_id: str) -> tuple[list | None, list[list]]:
    """Split a (possibly mixed-policy) group into this policy's share and foreign sub-groups.

    Returns ``(own_items_or_None, foreign_groups)``. Raises ValueError on
    unstamped samples or items mixing policies.
    """
    own, foreign = [], {}
    for item in group:
        item_policy = group_policy_id([item])
        if item_policy is None:
            raise ValueError(
                "multi-policy rollout: the custom generate function must stamp every sample's "
                "trainer_model_id (miles contract)"
            )
        if item_policy == policy_id:
            own.append(item)
        else:
            foreign.setdefault(item_policy, []).append(item)
    return (own or None), list(foreign.values())


def partition_group_by_policy(group) -> dict[str, list]:
    """Split a finished group once into per-policy sub-groups (worker completion path).

    Keys are the policies' ``trainer_model_id``s. Raises ValueError on
    unstamped samples or items mixing policies. In single-policy runs samples
    carry no stamp — callers must not use this path then.
    """
    partitions: dict[str, list] = {}
    for item in group:
        item_policy = group_policy_id([item])
        if item_policy is None:
            raise ValueError(
                "multi-policy rollout: the custom generate function must stamp every sample's "
                "trainer_model_id (miles contract)"
            )
        partitions.setdefault(item_policy, []).append(item)
    return partitions

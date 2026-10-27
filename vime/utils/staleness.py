"""Policy age in weight-update units (fresh samples have age zero)."""

from vime.rollout.base_types import iter_samples
from vime.utils.types import Sample


def fully_async_metrics_enabled(args) -> bool:
    return "fully_async" in (getattr(args, "rollout_function_path", None) or "")


def sample_staleness(sample: Sample, current_weight_version: int) -> int | None:
    versions = sample.weight_versions or []
    if not versions or any(not str(v).isascii() or not str(v).removeprefix("-").isdigit() for v in versions):
        return None
    numeric_versions = [int(v) for v in versions]
    if max(numeric_versions) > current_weight_version:
        return None
    return current_weight_version - min(numeric_versions)


def min_weight_version_of(group) -> int | None:
    """Oldest numeric weight version across a group's samples (None if unknown)."""
    versions = [
        int(v)
        for sample in iter_samples(group)
        for v in (sample.weight_versions or [])
        if str(v).isascii() and str(v).removeprefix("-").isdigit()
    ]
    return min(versions) if versions else None


def group_staleness(group, current_weight_version: int) -> int | None:
    """Age of a whole group: the oldest sample governs. ``None`` keeps the group.

    A group's age is unknown (and the group is always kept) when any sample's
    versions are missing or unparseable, or when a sample reports a version
    newer than ``current_weight_version`` — same rule as :func:`sample_staleness`.
    Accepts nested groups, plain sample lists, and straw ``RolloutGroupRef``s
    (via their recorded ``min_weight_version``).
    """
    from vime.data.transport import RolloutGroupRef

    if isinstance(group, RolloutGroupRef):
        oldest = group.min_weight_version
        if oldest is None or oldest > current_weight_version:
            return None
        return current_weight_version - oldest
    ages = [sample_staleness(sample, current_weight_version) for sample in iter_samples(group)]
    if not ages or any(age is None for age in ages):
        return None
    return max(ages)


def group_is_stale(group, current_weight_version: int | None, max_weight_staleness: int | None) -> bool:
    """Drop verdict for ``--max-weight-staleness``; the limit is inclusive."""
    if max_weight_staleness is None or current_weight_version is None:
        return False
    age = group_staleness(group, current_weight_version)
    return age is not None and age > max_weight_staleness


def compute_staleness_metrics(samples: list[Sample], current_weight_version: int | None) -> dict[str, float | int]:
    if current_weight_version is None:
        return {}
    ages = [sample_staleness(sample, current_weight_version) for sample in samples]
    known = [age for age in ages if age is not None]
    metrics: dict[str, float | int] = {"staleness/unknown_count": len(ages) - len(known)}
    if known:
        metrics["staleness/mean"] = sum(known) / len(known)
        metrics["staleness/max"] = max(known)
    return metrics

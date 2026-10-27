"""Weight-version staleness guards.

Port of miles@ba28991834 utils/weight_version.py, adapted to vime/vLLM:

- vime ``Sample.weight_versions`` is a flat list of per-segment versions (no
  span objects); versions are stamped at sample-submission level from the
  streaming response. Per-token engine-reported spans (sglang's
  ``meta_info["weight_versions"]`` equivalent) are a known gap.
- vime has no LoRA and no ``--update-weights-interval``; the publish threshold
  falls back to the minimum when the arg is absent.
"""

import re
from argparse import Namespace
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vime.utils.types import Sample

# Placeholder for an engine whose weights were never updated.
VLLM_DEFAULT_WEIGHT_VERSION = "default"

_NUMERIC_VERSION_PATTERN = re.compile(r"-?[0-9]+")

_MIN_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION = 3


def max_rollouts_without_published_weight_version(args: Namespace) -> int:
    return max(
        _MIN_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION,
        getattr(args, "update_weights_interval", _MIN_ROLLOUTS_WITHOUT_PUBLISHED_WEIGHT_VERSION - 1) + 1,
    )


def _guard_disabled(args: Namespace) -> bool:
    return (
        getattr(args, "debug_rollout_only", False)
        or getattr(args, "debug_train_only", False)
        or getattr(args, "debug_skip_weight_update", False)
    )


def assert_weight_version_is_published(args: Namespace, *, rollouts_since_publish: int) -> None:
    if _guard_disabled(args):
        return

    assert rollouts_since_publish <= max_rollouts_without_published_weight_version(args), (
        f"the rollout side served {rollouts_since_publish} rollouts without anyone publishing "
        f"a weight version, so its notion of the served weight version is frozen while training "
        f"advances; weight staleness accounting is silently disabled. The driver must publish "
        f"the version returned by update_weights to the rollout engines after every weight update"
    )


def assert_engine_weight_versions_sane(args: Namespace, versions: Iterable[str]) -> None:
    """Every queried engine must report a numeric version once training has
    published weights; a placeholder means the engine never got an update."""
    if _guard_disabled(args):
        return
    for version in versions:
        assert version != VLLM_DEFAULT_WEIGHT_VERSION, (
            f"engine reports weight version {VLLM_DEFAULT_WEIGHT_VERSION!r}, the placeholder for "
            f"an engine whose weights were never updated; training data must only come from "
            f"engines that received a weight update"
        )
        version = str(version)
        assert _NUMERIC_VERSION_PATTERN.fullmatch(version), (
            f"engine reports weight version {version!r}, which is not the numeric version vime "
            f"stamps on weight updates; this engine got its weights from somewhere vime does not "
            f"know about"
        )


def assert_samples_weight_version_sane(args: Namespace, samples: list) -> None:
    """Reject samples carrying placeholder / non-numeric weight versions.

    Accepts flat lists of Samples or nested sample groups.
    """
    if _guard_disabled(args):
        return

    for sample in _iter_samples(samples):
        for version in getattr(sample, "weight_versions", None) or []:
            version = str(version)
            assert version != VLLM_DEFAULT_WEIGHT_VERSION, (
                f"sample index={getattr(sample, 'index', '?')} carries weight version "
                f"{VLLM_DEFAULT_WEIGHT_VERSION!r}, the placeholder for an engine whose weights "
                f"were never updated; training data must only come from engines that received "
                f"a weight update"
            )
            assert _NUMERIC_VERSION_PATTERN.fullmatch(version), (
                f"sample index={getattr(sample, 'index', '?')} carries weight version "
                f"{version!r}, which is not the numeric version vime stamps on weight updates; "
                f"the engine serving this sample got its weights from somewhere vime does not "
                f"know about"
            )


def _iter_samples(samples: list) -> Iterable["Sample"]:
    for item in samples:
        if isinstance(item, (list, tuple)):
            yield from _iter_samples(item)
        else:
            yield item

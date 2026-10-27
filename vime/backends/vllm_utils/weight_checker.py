"""Client-side evaluation of ``check_weights`` compare results.

The engine (vLLM worker, via the docker ``vllm-check-weights`` patch) reports
per-tensor status for ``action="compare"``; this module decides what is fatal.
Skip-listed name substrings downgrade a mismatch to a logged warning, mirroring
miles@ba28991834 ``--check-weight-update-skip-list`` semantics.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_compare_results(body: dict[str, Any], *, skip_list: list[str] | None = None) -> list[str]:
    """Validate a check_weights compare response.

    Returns the names whose mismatch was downgraded via the skip list.
    Raises RuntimeError on any non-skipped mismatch, naming the tensors.
    """
    if not body.get("ok", False):
        raise RuntimeError(f"check_weights engine reported failure: {body!r}")
    if not body.get("supported", False):
        raise RuntimeError(
            "check_weights compare requested but the engine does not support it "
            "(vLLM check-weights patch missing?); refusing to silently pass"
        )

    mismatched: list[str] = []
    missing: list[str] = []
    for rank_info in body.get("results", []):
        rank = rank_info["rank"]
        for name, info in rank_info.get("tensors", {}).items():
            status = info.get("status")
            if status == "mismatch":
                mismatched.append(f"{name}@rank{rank}")
            elif status == "missing":
                missing.append(f"{name}@rank{rank}")

    downgraded: list[str] = []
    fatal = sorted(mismatched + missing)
    if skip_list:
        still_fatal = []
        for entry in fatal:
            name = entry.split("@rank")[0]
            if any(pattern in name for pattern in skip_list):
                downgraded.append(entry)
            else:
                still_fatal.append(entry)
        fatal = still_fatal

    if downgraded:
        logger.warning(
            "check_weights: %d skip-listed tensors did not match the snapshot (non-fatal): %s",
            len(downgraded),
            downgraded,
        )
    if fatal:
        raise RuntimeError(f"check_weights compare failed for {len(fatal)} tensors: {fatal}")
    return downgraded

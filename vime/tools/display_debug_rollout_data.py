"""View --save-debug-rollout-data dumps (legacy .pt or .straw.json archives).

Usage: python -m vime.tools.display_debug_rollout_data --load-debug-rollout-data 'dump/rollout_{rollout_id}.pt'
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer

from vime.observability.rollout_data_utils import load_debug_rollout_data
from vime.observability.rollout_metrics import compute_perf_metrics_from_samples

_WHITELIST_KEYS = [
    "group_index",
    "index",
    "prompt",
    "response",
    "response_length",
    "label",
    "reward",
    "status",
    "metadata",
]


def main(
    # Deliberately make this name consistent with main training arguments
    load_debug_rollout_data_path: Annotated[str, typer.Option("--load-debug-rollout-data")],
    show_metrics: bool = True,
    show_samples: bool = True,
    category: list[str] = None,
):
    if category is None:
        category = ["train", "eval"]
    for rollout_id, path in _get_rollout_dump_paths(load_debug_rollout_data_path, category):
        print("-" * 80)
        print(f"rollout_id={rollout_id} path={path}")
        print("-" * 80)

        sample_objects = load_debug_rollout_data(str(path), rollout_id=rollout_id)

        if show_metrics:
            # rollout_time is unknown for offline dumps; pass 1.0 so the tool works —
            # any *_per_sec metrics it yields are meaningless for dumps.
            args = SimpleNamespace(rollout_num_gpus=None)
            metrics = compute_perf_metrics_from_samples(args, sample_objects, rollout_time=1.0)
            metrics = {k: v for k, v in metrics.items() if not k.endswith("per_sec")}
            print("metrics", metrics)

        if show_samples:
            for sample in sample_objects:
                dump = sample.to_dict()
                print(json.dumps({k: v for k, v in dump.items() if k in _WHITELIST_KEYS}, default=repr))


def _get_rollout_dump_paths(load_debug_rollout_data: str, categories: list[str]):
    # may improve later
    for rollout_id in range(1000):
        for category in categories:
            prefix = {
                "train": "",
                "eval": "eval_",
            }[category]
            path = Path(load_debug_rollout_data.format(rollout_id=f"{prefix}{rollout_id}"))
            if path.exists():
                yield rollout_id, path


if __name__ == "__main__":
    typer.run(main)

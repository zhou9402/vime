"""Replay a custom reward function against a saved rollout dump (.pt or .straw.json).

Usage: python -m vime.tools.replay_reward_fn --rollout-data-path dump.pt --custom-rm-path my_module.my_rm
"""

import asyncio
from typing import Annotated

import typer

from vime.observability.rollout_data_utils import load_debug_rollout_data
from vime.utils.misc import load_function


def _truncate(text, max_len=200):
    """Truncate text and add ellipsis if too long."""
    if text is None:
        return None
    text = str(text).replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def main(
    rollout_data_path: Annotated[str, typer.Option()],
    custom_rm_path: Annotated[str, typer.Option()],
):
    samples = load_debug_rollout_data(rollout_data_path, rollout_id=0)
    asyncio.run(_main_async(samples=samples, custom_rm_path=custom_rm_path))


async def _main_async(samples, custom_rm_path):
    rm_function = load_function(custom_rm_path)
    rewards = await asyncio.gather(*[rm_function(None, sample) for sample in samples])

    for i, (sample, reward) in enumerate(zip(samples, rewards, strict=True)):
        print("-" * 60)
        print(f"Sample {i + 1}/{len(samples)}")
        print(f"  Index:    {sample.index}")
        print(f"  Status:   {sample.status}")
        print(f"  Reward:   {reward}")
        print(f"  Prompt:   {_truncate(sample.prompt, 200)}")
        print(f"  Response: {_truncate(sample.response, 200)}")
    print("-" * 60)


if __name__ == "__main__":
    typer.run(main)

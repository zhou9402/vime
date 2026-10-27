"""Rollout limits hold across a rollout's agent processes and free a slot when its holder dies."""

import asyncio
import os

import pytest

from tests.test_agent import agent_function_stubs as stubs
from vime.agent.agent_function import call_agent_function
from vime.agent.rollout_limits import current_scope


async def _in_agent_processes(fn, count: int, **kwargs) -> list:
    return await asyncio.gather(*(call_agent_function(fn, mode="subproc", **kwargs) for _ in range(count)))


def _peak_overlap(spans: list[tuple[float, float]]) -> int:
    return max(sum(1 for start, end in spans if start <= t < end) for t, _ in spans)


def test_semaphore_caps_holders_across_agent_processes():
    spans = asyncio.run(_in_agent_processes(stubs.hold_semaphore, 6, name="test-cap", limit=2, hold_s=1.0))
    assert _peak_overlap(spans) <= 2
    assert max(end for _, end in spans) - min(start for start, _ in spans) >= 2.9, "six 1s holds under 2 slots"


def test_lock_serializes_holders_across_agent_processes():
    spans = asyncio.run(_in_agent_processes(stubs.hold_lock, 3, name="test-lock", hold_s=0.5))
    assert _peak_overlap(spans) == 1


def test_a_dead_holder_frees_its_slot():
    with pytest.raises(RuntimeError, match="before returning a result"):
        asyncio.run(call_agent_function(stubs.die_holding_semaphore, mode="subproc", name="test-dead"))
    asyncio.run(
        asyncio.wait_for(
            call_agent_function(stubs.hold_semaphore, mode="subproc", name="test-dead", limit=1, hold_s=0.0), 30
        )
    )


def test_current_scope_inherits_the_env_var(monkeypatch):
    monkeypatch.setenv("VIME_ROLLOUT_LIMITS_SCOPE", "rollout-7")
    assert current_scope() == "rollout-7"
    monkeypatch.delenv("VIME_ROLLOUT_LIMITS_SCOPE")
    monkeypatch.setenv("MILES_ROLLOUT_LIMITS_SCOPE", "legacy-3")
    assert current_scope() == "legacy-3", "the legacy alias still applies"


def test_current_scope_falls_back_to_the_pid(monkeypatch):
    monkeypatch.delenv("VIME_ROLLOUT_LIMITS_SCOPE", raising=False)
    monkeypatch.delenv("MILES_ROLLOUT_LIMITS_SCOPE", raising=False)
    assert current_scope() == f"pid{os.getpid()}"

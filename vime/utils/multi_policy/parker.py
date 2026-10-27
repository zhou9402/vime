"""Leader/follower park barrier for multi-policy training.

Port of miles@ba28991834 miles/utils/multi_policy/parker.py (verbatim, plus a
5-minute progress log before the 1-hour deadlock assert).
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

PARK_TIMEOUT_SECONDS = 3600.0
_PARK_WARN_SECONDS = 300.0


class Parker:
    """Parks every follower at its round boundary while the leader holds `with_all_parked()`."""

    def __init__(self, *, num_followers: int) -> None:
        self._num_followers = num_followers
        self._event = asyncio.Event()
        self._event.set()
        self._num_ready = 0

    @asynccontextmanager
    async def with_all_parked(self) -> AsyncIterator[None]:
        await self._wait_num_ready(0)
        self._event.clear()
        try:
            await self._wait_num_ready(self._num_followers)
            yield
        finally:
            self._event.set()

    async def maybe_park_follower(self) -> None:
        await asyncio.sleep(0)
        self._num_ready += 1
        await self._event.wait()
        self._num_ready -= 1

    async def _wait_num_ready(self, target: int) -> None:
        deadline = time.monotonic() + PARK_TIMEOUT_SECONDS
        warned = False
        while self._num_ready != target:
            now = time.monotonic()
            assert now < deadline, (
                f"waited {PARK_TIMEOUT_SECONDS}s for the followers to park; "
                f"ready {self._num_ready}, want {target}"
            )
            if not warned and now > deadline - (PARK_TIMEOUT_SECONDS - _PARK_WARN_SECONDS):
                logger.warning(
                    f"Parker still waiting after {_PARK_WARN_SECONDS}s: ready {self._num_ready}, want {target}"
                )
                warned = True
            await asyncio.sleep(1)

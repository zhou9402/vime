"""Driver-side trainer heartbeat checker.

Polls every train actor's ``get_heartbeat_status`` on an interval and flags actors whose
heartbeat stops advancing. Off by default (``--trainer-heartbeat-checker-interval`` unset).
A stale actor is logged as an error; under ``--ci-test`` the checker raises so CI fails loudly.
"""

import logging
import threading
import time
from collections.abc import Callable

import ray

logger = logging.getLogger(__name__)

_CHECKER_THREAD_NAME = "trainer-heartbeat-checker"


class TrainerHeartbeatChecker:
    def __init__(self, args, get_actors: Callable[[], list]) -> None:
        self._interval = args.trainer_heartbeat_checker_interval
        self._timeout = args.trainer_heartbeat_checker_timeout
        self._ci_test = getattr(args, "ci_test", False)
        self._get_actors = get_actors
        self._last_bump: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._interval is None:
            return False
        if self._thread is not None:
            return True
        self._thread = threading.Thread(target=self._run, name=_CHECKER_THREAD_NAME, daemon=True)
        self._thread.start()
        logger.info(
            "Trainer heartbeat checker started (interval=%ss, timeout=%ss)", self._interval, self._timeout
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._check_once()
            except Exception:
                if self._ci_test:
                    raise
                logger.exception("Trainer heartbeat check round failed")

    def _check_once(self) -> None:
        for actor_id, actor in enumerate(self._get_actors()):
            if actor is None:
                continue
            try:
                status = ray.get(actor.get_heartbeat_status.remote(), timeout=self._timeout)
            except Exception as error:
                self._on_stale(actor_id, f"heartbeat RPC failed: {error}")
                continue

            age = time.time() - status.last_active_timestamp
            previous = self._last_bump.get(actor_id)
            self._last_bump[actor_id] = status.bump_count
            if previous is not None and status.bump_count == previous and age > self._timeout:
                self._on_stale(
                    actor_id,
                    f"no training progress for {age:.1f}s (bump_count stuck at {status.bump_count})",
                )

    def _on_stale(self, actor_id: int, detail: str) -> None:
        logger.error("Trainer actor %s heartbeat stale: %s", actor_id, detail)
        if self._ci_test:
            raise RuntimeError(f"trainer actor {actor_id} heartbeat stale: {detail}")


def maybe_start_trainer_heartbeat_checker(args, actor_model) -> TrainerHeartbeatChecker | None:
    """Start the checker against an actor group's handlers; None when disabled."""
    checker = TrainerHeartbeatChecker(args, lambda: actor_model._actor_handlers)
    if not checker.start():
        return None
    return checker

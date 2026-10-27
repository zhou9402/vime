"""retry_until_deadline_sync (miles #4402 lineage, F01-P4)."""

import time

import pytest

from vime.utils.retry_utils import retry_until_deadline_sync

NUM_GPUS = 0


def test_success_first_try():
    assert retry_until_deadline_sync(lambda: 1, total_seconds=1.0, retry_if=lambda e: True) == 1


def test_retries_transient_until_success():
    attempts = iter([ConnectionError("x"), ConnectionError("x"), 42])
    result = retry_until_deadline_sync(
        lambda: _raise_or_value(attempts),
        total_seconds=5.0,
        retry_if=lambda e: isinstance(e, ConnectionError),
        interval_seconds=0.01,
    )
    assert result == 42


def test_non_transient_propagates_immediately():
    calls = 0

    def fn():
        nonlocal calls
        calls += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        retry_until_deadline_sync(fn, total_seconds=5.0, retry_if=lambda e: False, interval_seconds=0.01)
    assert calls == 1


def test_deadline_bounds_retries():
    calls = 0

    def fn():
        nonlocal calls
        calls += 1
        raise ConnectionError("down")

    started = time.monotonic()
    with pytest.raises(ConnectionError):
        retry_until_deadline_sync(fn, total_seconds=0.2, retry_if=lambda e: True, interval_seconds=0.05)
    assert time.monotonic() - started < 1.0
    assert calls >= 2


def _raise_or_value(it):
    for value in it:
        if isinstance(value, Exception):
            raise value
        return value

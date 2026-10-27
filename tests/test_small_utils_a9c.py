"""Small standalone utilities ported from miles: run_uuid, file_arg_utils, retry_utils, metric namespaces."""

import asyncio
import base64
import re
import time

import pytest

from vime.observability.metric_utils import namespace_metrics, strip_metrics_namespace
from vime.utils.file_arg_utils import resolve_file_arg
from vime.utils.retry_utils import NonRetryableError, retry, retry_until_deadline_sync
from vime.utils.run_uuid import generate_run_uuid, validate_run_uuid


def test_run_uuid_roundtrip():
    value = generate_run_uuid()
    assert validate_run_uuid(value) == value
    assert re.fullmatch(r"[0-9a-f]{16}", value)
    with pytest.raises(ValueError):
        validate_run_uuid("xyz")
    with pytest.raises(ValueError):
        validate_run_uuid("A" * 16)


def test_resolve_file_arg_path(tmp_path):
    payload = tmp_path / "config.yaml"
    payload.write_text("hello: 1\n")
    assert resolve_file_arg(str(payload)) == "hello: 1\n"


def test_resolve_file_arg_base64():
    assert resolve_file_arg("base64:" + base64.b64encode(b"hello").decode()) == "hello"


def test_retry_succeeds_after_failures():
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        if len(calls) < 3:
            raise RuntimeError("flaky")
        return 1

    async def no_sleep(_):
        pass

    assert asyncio.run(retry(fn, sleep_fn=no_sleep)) == 1
    assert calls == [0, 1, 2]


def test_retry_non_retryable_propagates():
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        raise NonRetryableError("no")

    async def no_sleep(_):
        raise AssertionError("must not sleep")

    with pytest.raises(NonRetryableError):
        asyncio.run(retry(fn, sleep_fn=no_sleep))
    assert calls == [0]


def test_retry_max_attempts():
    calls = []

    async def fn(attempt):
        calls.append(attempt)
        raise RuntimeError("always")

    async def no_sleep(_):
        pass

    with pytest.raises(RuntimeError):
        asyncio.run(retry(fn, sleep_fn=no_sleep, max_attempts=2))
    assert calls == [0, 1]


def test_retry_until_deadline_sync_gives_up():
    start = time.monotonic()
    with pytest.raises(RuntimeError):
        retry_until_deadline_sync(
            lambda: (_ for _ in ()).throw(RuntimeError("always")),
            total_seconds=0.05,
            retry_if=lambda e: True,
            interval_seconds=0.01,
        )
    assert time.monotonic() - start < 1.0


def test_namespace_metrics_roundtrip():
    namespaced, step_key = namespace_metrics({"a": 1}, trainer_model_id="m", step_name="step", step=3)
    assert namespaced == {"m/a": 1, "m/step": 3}
    assert step_key == "m/step"
    assert strip_metrics_namespace(namespaced, trainer_model_id="m") == {"a": 1, "step": 3}
    assert namespace_metrics({"a": 1}, trainer_model_id=None, step_name="step", step=3)[0] == {"a": 1, "step": 3}
    assert strip_metrics_namespace({"a": 1}, trainer_model_id=None) == {"a": 1}

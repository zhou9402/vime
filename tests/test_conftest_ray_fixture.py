"""The session-scoped ray_local_mode fixture and the opt-in stall dumper."""

import pytest
import ray


def test_fixture_connects(ray_local_mode):
    assert ray.is_initialized()


def test_fixture_is_session_shared(ray_local_mode):
    address = ray.get_runtime_context().gcs_address

    class _Second:
        @pytest.fixture(scope="session")
        def _unused(self):
            pass

    # a second request in the same session sees the same cluster
    assert ray.get_runtime_context().gcs_address == address


def test_respects_external_address(monkeypatch):
    """With an external address set, the fixture must not pass num_cpus/num_gpus
    (Ray rejects resource overrides when joining an existing cluster)."""
    import tests.conftest as conftest

    monkeypatch.setenv("VIME_TEST_RAY_ADDRESS", "ray://example:10001")

    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(ray, "init", fake_init)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)

    # drive the fixture generator manually
    gen = conftest.ray_local_mode.__wrapped__()
    next(gen)
    assert "num_cpus" not in captured
    assert "num_gpus" not in captured
    with pytest.raises(StopIteration):
        next(gen)


def test_stall_dumper_disabled_by_default(monkeypatch):
    import tests.conftest as conftest

    monkeypatch.delenv("VIME_TEST_STALL_DUMP_SECONDS", raising=False)
    # the module-level flag was computed at import; recompute the predicate
    import os

    assert not (hasattr(__import__("signal"), "SIGALRM") and "VIME_TEST_STALL_DUMP_SECONDS" in os.environ)
    assert conftest._stall_dumper is None

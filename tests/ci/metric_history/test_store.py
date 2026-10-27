"""SQLite store contract tests (port of miles test_metric_history_store.py)."""

import pytest
from metric_history.storage.store import (
    MetricSample,
    RunIdentity,
    RunProvenance,
)
from metric_history.storage.sqlite_store import SQLiteMetricHistoryStore

IDENTITY = RunIdentity(test_path="tests/test_x.py", backend="cuda", suite="megatron")
PROVENANCE = RunProvenance(
    commit_sha="abc123", pr_number=1, ci_run_id=100, ci_run_attempt=1, pipeline="vime-ci", ref="main"
)


def sample(step=-1, value=1.0, metric="rollout/raw_reward"):
    return MetricSample(metric_key=metric, steps_key='"last"', constraint_key='{"rel_down":0.05}', step=step, value=value)


@pytest.fixture()
def store():
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def write(store, identity=IDENTITY, created_at="2026-01-01T00:00:00", trusted=True, values=None, prov=PROVENANCE):
    return store.write_run(identity, prov, created_at, trusted, values if values is not None else [sample()])


def test_write_read_roundtrip_newest_first(store):
    write(store, created_at="2026-01-01T00:00:00", values=[sample(value=1.0)])
    write(store, created_at="2026-01-02T00:00:00", values=[sample(value=2.0)])
    values = store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20)
    assert values == [2.0, 1.0]


def test_write_rejects_non_finite(store):
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            write(store, values=[sample(value=bad)])
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20) == []


def test_limit_caps_and_orders(store):
    for i in range(5):
        write(store, created_at=f"2026-01-0{i + 1}T00:00:00", values=[sample(value=float(i))])
    values = store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 3)
    assert values == [4.0, 3.0, 2.0]


def test_identity_isolation(store):
    write(store, identity=RunIdentity("tests/test_x.py", "cuda", "megatron"), values=[sample(value=1.0)])
    write(store, identity=RunIdentity("tests/test_x.py", "rocm", "megatron"), values=[sample(value=9.0)])
    write(store, identity=RunIdentity("tests/test_y.py", "cuda", "megatron"), values=[sample(value=5.0)])
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20) == [1.0]
    assert store.recent_trusted_values("tests/test_x.py", "rocm", "megatron", "rollout/raw_reward", -1, 20) == [9.0]


def test_coordinate_is_metric_key_and_step_only(store):
    # different steps_key/constraint_key share history at the same coordinate
    v1 = MetricSample("m", '"last"', '{"rel_down":0.05}', -1, 1.0)
    v2 = MetricSample("m", '"last"', '{"rel_down":0.99}', -1, 2.0)
    write(store, created_at="2026-01-01T00:00:00", values=[v1])
    write(store, created_at="2026-01-02T00:00:00", values=[v2])
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "m", -1, 20) == [2.0, 1.0]
    # but different steps do not
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "m", 3, 20) == []


def test_untrusted_runs_never_serve_as_baseline(store):
    write(store, trusted=False, values=[sample(value=1.0)])
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20) == []


def test_mark_untrusted_by_run_id(store):
    run_id = write(store, values=[sample(value=1.0)])
    assert store.mark_untrusted(run_id=run_id) == 1
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20) == []
    # already untrusted: no-op
    assert store.mark_untrusted(run_id=run_id) == 0


def test_mark_untrusted_by_ci_run_id_and_commit_sha(store):
    write(store, prov=RunProvenance("sha1", None, 42, 1, None, None))
    write(store, prov=RunProvenance("sha2", None, 43, 1, None, None), created_at="2026-01-02T00:00:00")
    assert store.mark_untrusted(ci_run_id=42) == 1
    assert [round(v) for v in store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20)] == [1]
    assert store.mark_untrusted(commit_sha="sha2") == 1
    assert store.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20) == []


def test_mark_untrusted_requires_exactly_one_key(store):
    with pytest.raises(ValueError):
        store.mark_untrusted()
    with pytest.raises(ValueError):
        store.mark_untrusted(run_id="x", commit_sha="y")


def test_persistent_file_roundtrip(tmp_path):
    db = str(tmp_path / "history.db")
    s1 = SQLiteMetricHistoryStore(db)
    write(s1, values=[sample(value=3.0)])
    s1.close()
    s2 = SQLiteMetricHistoryStore(db)
    assert s2.recent_trusted_values("tests/test_x.py", "cuda", "megatron", "rollout/raw_reward", -1, 20) == [3.0]
    s2.close()

"""Gate-core tests (condensed port of miles test_metric_selection.py + test_history_gate.py)."""

import json
from types import SimpleNamespace

import pytest
from metric_history.constraints import evaluate_constraint
from metric_history.gate import GateStatus, evaluate_coordinate, evaluate_gate, parse_merged_record
from metric_history.selection import SelectionError, select
from metric_history.storage.sqlite_store import SQLiteMetricHistoryStore
from metric_history.storage.store import MetricSample, RunIdentity, RunProvenance

# ---- selection ----


def test_select_last():
    [sel] = select([[0, 1.0], [3, 2.0]], "last")
    assert (sel.step, sel.at_step, sel.value) == (-1, 3, 2.0)


def test_select_all_fans_out_and_rejects_duplicates():
    sels = select([[0, 1.0], [1, 2.0]], "all")
    assert [(s.step, s.value) for s in sels] == [(0, 1.0), (1, 2.0)]
    with pytest.raises(SelectionError):
        select([[0, 1.0], [0, 2.0]], "all")


def test_select_step_list_missing_is_error():
    [sel] = select([[0, 1.0], [5, 2.5]], [5])
    assert (sel.step, sel.value) == (5, 2.5)
    with pytest.raises(SelectionError):
        select([[0, 1.0]], [7])


def test_select_non_finite_picked_is_error_ignored_if_not_picked():
    with pytest.raises(SelectionError):
        select([[0, float("nan")]], "last")
    with pytest.raises(SelectionError):
        select([[0, float("inf")]], "all")
    # non-numeric values are ignored entirely
    [sel] = select([[0, None], [1, True], [2, 4.0]], "last")
    assert sel.value == 4.0


def test_select_empty_series_error():
    with pytest.raises(SelectionError):
        select([], "last")


# ---- constraints ----


def test_constraint_band_asymmetric_and_zero_floor():
    c = {"rel_up": 0.5, "abs_floor_up": 0.0, "rel_down": 0.1, "abs_floor_down": 0.0}
    out = evaluate_constraint(c, 1.4, 1.0)
    assert out.ok and out.lo == pytest.approx(0.9) and out.hi == pytest.approx(1.5)
    assert not evaluate_constraint(c, 1.6, 1.0).ok
    assert not evaluate_constraint(c, 0.8, 1.0).ok


def test_constraint_upper_optional_and_explicit_zero():
    lower_only = {"rel_down": 0.05, "abs_floor_down": 0.0}
    assert evaluate_constraint(lower_only, 100.0, 1.0).ok  # hi = inf
    pinned = {"rel_up": 0.0, "abs_floor_up": 0.0, "rel_down": 1.0, "abs_floor_down": 0.0}
    assert evaluate_constraint(pinned, 1.0, 1.0).ok
    assert not evaluate_constraint(pinned, 1.01, 1.0).ok


def test_constraint_abs_floor_near_zero():
    c = {"rel_down": 0.5, "abs_floor_down": 0.01, "rel_up": 0.5, "abs_floor_up": 0.01}
    assert evaluate_constraint(c, 0.005, 0.0).ok  # floor dominates near-zero ref


# ---- gate ----


def spec(metric="rollout/raw_reward", steps="last", constraint=None, enforce=True):
    return SimpleNamespace(
        filename="tests/test_fake.py",
        metric_key=metric,
        steps=steps,
        constraint=constraint,
        steps_key=json.dumps(steps, sort_keys=True, separators=(",", ":")),
        constraint_key=json.dumps(constraint, sort_keys=True, separators=(",", ":")),
        enforce=enforce and constraint is not None,
        allowlist_reason=None,
    )


def write_record(tmp_path, metrics):
    path = tmp_path / "record.jsonl"
    with open(path, "w") as f:
        for key, series in metrics.items():
            f.write(json.dumps({"metric": key, "series": series}) + "\n")
    return str(path)


def seed_history(store, values, metric="rollout/raw_reward", step=-1, test_path="tests/test_fake.py"):
    for i, value in enumerate(values):
        store.write_run(
            RunIdentity(test_path, "cuda", "stage"),
            RunProvenance("sha", None, None, None, None, None),
            f"2026-01-{i + 1:02d}T00:00:00",
            True,
            [MetricSample(metric, '"last"', '{"rel_down":0.05}', step, value)],
        )


@pytest.fixture()
def store():
    s = SQLiteMetricHistoryStore(":memory:")
    yield s
    s.close()


def test_parse_merged_record_decodes_non_finite_markers(tmp_path):
    path = write_record(tmp_path, {"m": [[0, "NaN"], [1, "Infinity"], [2, "-Infinity"], [3, 1.5]]})
    series = parse_merged_record(path)["m"]
    assert series[0][1] != series[0][1]  # NaN
    assert series[1][1] == float("inf")
    assert series[2][1] == float("-inf")
    assert series[3][1] == 1.5


def test_cold_start_inactive_and_trusted(store, tmp_path):
    path = write_record(tmp_path, {"rollout/raw_reward": [[0, 1.0]]})
    result = evaluate_gate(
        "tests/test_fake.py", path, store, executing_suite="stage", specs=[spec(constraint={"rel_down": 0.05, "abs_floor_down": 0.0})]
    )
    assert result.trusted
    assert result.metrics[0].historical_status == GateStatus.INACTIVE


def test_pass_and_fail_against_history(store, tmp_path):
    seed_history(store, [1.0] * 12)
    constraint = {"rel_down": 0.05, "abs_floor_down": 0.0}
    ok = evaluate_gate(
        "tests/test_fake.py", write_record(tmp_path, {"rollout/raw_reward": [[0, 1.02]]}), store,
        executing_suite="stage", specs=[spec(constraint=constraint)],
    )
    assert ok.trusted and ok.metrics[0].historical_status == GateStatus.PASS
    assert ok.metrics[0].baseline_n == 12
    bad = evaluate_gate(
        "tests/test_fake.py", write_record(tmp_path, {"rollout/raw_reward": [[0, 0.5]]}), store,
        executing_suite="stage", specs=[spec(constraint=constraint)],
    )
    assert not bad.trusted and bad.metrics[0].historical_status == GateStatus.FAIL


def test_per_step_baseline_isolation(store, tmp_path):
    # step 0 history is all 1.0; step 1 has no history
    for i in range(3):
        store.write_run(
            RunIdentity("tests/test_fake.py", "cuda", "stage"),
            RunProvenance("sha", None, None, None, None, None),
            f"2026-02-0{i + 1}T00:00:00",
            True,
            [MetricSample("train/ppo_kl", '"all"', '{"rel_up":0.1,"abs_floor_up":0.0,"rel_down":0.1,"abs_floor_down":0.0}', 0, 1.0)],
        )
    path = write_record(tmp_path, {"train/ppo_kl": [[0, 5.0], [1, 5.0]]})
    result = evaluate_gate(
        "tests/test_fake.py", path, store, executing_suite="stage",
        specs=[spec(metric="train/ppo_kl", steps="all", constraint={"rel_up": 0.1, "abs_floor_up": 0.0, "rel_down": 0.1, "abs_floor_down": 0.0})],
    )
    by_step = {m.step: m for m in result.metrics}
    assert by_step[0].historical_status == GateStatus.FAIL
    assert by_step[1].historical_status == GateStatus.INACTIVE
    assert not result.trusted


def test_last_never_feeds_per_step_baselines(store, tmp_path):
    # a steps="last" record writes to step -1 only; step 3 stays cold
    seed_history(store, [1.0] * 5)
    path = write_record(tmp_path, {"rollout/raw_reward": [[3, 1.0]]})
    result = evaluate_gate(
        "tests/test_fake.py", path, store, executing_suite="stage",
        specs=[spec(steps=[3], constraint={"rel_down": 0.05, "abs_floor_down": 0.0})],
    )
    assert result.metrics[0].historical_status == GateStatus.INACTIVE


def test_missing_metric_is_error_when_constrained_tracked_otherwise(store, tmp_path):
    path = write_record(tmp_path, {"other": [[0, 1.0]]})
    constrained = evaluate_gate(
        "tests/test_fake.py", path, store, executing_suite="stage",
        specs=[spec(constraint={"rel_down": 0.05, "abs_floor_down": 0.0})],
    )
    assert constrained.metrics[0].historical_status == GateStatus.ERROR
    assert not constrained.trusted
    tracked = evaluate_gate(
        "tests/test_fake.py", path, store, executing_suite="stage", specs=[spec(constraint=None)]
    )
    assert tracked.metrics[0].historical_status == GateStatus.TRACKED
    assert tracked.trusted


def test_no_specs_vacuously_trusted(store, tmp_path):
    path = write_record(tmp_path, {"m": [[0, 1.0]]})
    result = evaluate_gate("tests/test_fake.py", path, store, executing_suite="stage", specs=[])
    assert result.trusted and result.metrics == [] and result.backend == "cuda"


def test_evaluate_coordinate_reports_baseline_mean():
    s = spec(constraint={"rel_down": 0.1, "abs_floor_down": 0.0})
    out = evaluate_coordinate(s, current=1.0, step=-1, at_step=3, trusted_values=[0.9, 1.1])
    assert out.historical_status == GateStatus.PASS
    assert out.baseline_mean == pytest.approx(1.0)
    assert out.baseline_n == 2

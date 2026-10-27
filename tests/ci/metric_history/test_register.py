"""register_ci_gate parsing tests (condensed port of miles test_metric_selection.py parser parts)."""

import pytest
from metric_history.register import GATE_DEFAULTS, _canonical_key, parse_ci_gate_specs


def write(tmp_path, body):
    path = tmp_path / "test_fake_ci.py"
    path.write_text(body)
    return str(path)


def test_minimal_declaration_uses_gate_defaults(tmp_path):
    specs = parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="train/ppo_kl")\n'))
    assert len(specs) == 1
    spec = specs[0]
    assert spec.steps == "all"
    assert spec.constraint == GATE_DEFAULTS["train/ppo_kl"]["constraint"]
    assert spec.enforce is True


def test_explicit_literals_win_over_defaults(tmp_path):
    body = 'register_ci_gate(metric_key="rollout/raw_reward", steps="last", constraint={"rel_down": 0.1})\n'
    [spec] = parse_ci_gate_specs(write(tmp_path, body))
    assert spec.steps == "last"
    assert spec.constraint == {"rel_down": 0.1, "abs_floor_down": 0.0}
    assert spec.steps_key == '"last"'
    assert spec.constraint_key == '{"rel_down":0.1}'


def test_unknown_metric_requires_both_fields(tmp_path):
    with pytest.raises(ValueError, match="no GATE_DEFAULTS entry"):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="custom/x")\n'))


def test_rejects_non_literal_positional_unknown_and_duplicate_kwargs(tmp_path):
    with pytest.raises(ValueError, match="keyword arguments"):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate("train/ppo_kl")\n'))
    with pytest.raises(ValueError, match="unknown argument"):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="train/ppo_kl", bogus=1)\n'))
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="train/ppo_kl", steps=STEPS)\n'))


def test_duplicate_metric_key_rejected(tmp_path):
    body = (
        'register_ci_gate(metric_key="train/ppo_kl")\n'
        'register_ci_gate(metric_key="train/ppo_kl", steps="last", constraint={"rel_down": 0.1})\n'
    )
    with pytest.raises(ValueError, match="more than once"):
        parse_ci_gate_specs(write(tmp_path, body))


def test_tracked_metric_constraint_none(tmp_path):
    [spec] = parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="perf/actor_train_mfu")\n'))
    assert spec.constraint is None
    assert spec.enforce is False  # forced off without a constraint


def test_constraint_requires_lower_band_and_known_keys(tmp_path):
    with pytest.raises(ValueError, match="lower band"):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="m", steps="last", constraint={"rel_up": 0.5})\n'))
    with pytest.raises(ValueError, match="unknown key"):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="m", steps="last", constraint={"rel_down": 0.5, "zzz": 1})\n'))


def test_negative_and_bad_steps_rejected(tmp_path):
    with pytest.raises(ValueError):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="m", steps=[0, -1], constraint={"rel_down": 0.1})\n'))
    with pytest.raises(ValueError):
        parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="m", steps=[], constraint={"rel_down": 0.1})\n'))
    [spec] = parse_ci_gate_specs(write(tmp_path, 'register_ci_gate(metric_key="m", steps=[0, 4], constraint={"rel_down": 0.1})\n'))
    assert spec.steps == [0, 4]


def test_canonical_key_byte_stability():
    assert _canonical_key("last") == '"last"'
    assert _canonical_key({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert _canonical_key([3, 1]) == "[3,1]"
    assert _canonical_key(None) == "null"


def test_all_gate_defaults_keys_parse(tmp_path):
    body = "\n".join(f'register_ci_gate(metric_key="{key}")' for key in GATE_DEFAULTS)
    specs = parse_ci_gate_specs(write(tmp_path, body + "\n"))
    assert len(specs) == len(GATE_DEFAULTS)

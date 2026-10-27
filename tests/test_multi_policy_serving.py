"""Unit tests for per-policy weight-update routing (F09-2).

Covers the pure binding/selection logic in
vime/utils/multi_policy/serving_config.py plus the ServingCluster lock
wiring (per-policy locks are the very actors the servers hold; reattach and
cleanup cover them).
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from vime.utils.multi_policy.megatron_config import MultiPolicyConfig
from vime.utils.multi_policy.serving_config import (
    bind_policy_servers,
    consistent_weight_version,
    resolve_updatable_server,
)


def _servers(*specs):
    return {name: SimpleNamespace(update_weights=upd) for name, upd in specs}


def _config(*model_ids):
    return MultiPolicyConfig(
        trainers=[
            dict(trainer_id=f"{m}-actor", model_id=m, role="actor", actor_index=i, overrides={})
            for i, m in enumerate(model_ids)
        ]
    )


def test_bind_by_server_name():
    servers = _servers(("solver", True), ("verifier", True), ("rm", False))
    bound = bind_policy_servers(servers, _config("solver", "verifier"))
    assert bound == {"solver": "solver", "verifier": "verifier"}


def test_bind_rejects_missing_server():
    with pytest.raises(ValueError, match="no serving server"):
        bind_policy_servers(_servers(("solver", True)), _config("solver", "verifier"))


def test_bind_rejects_frozen_policy_server():
    servers = _servers(("solver", True), ("verifier", False))
    with pytest.raises(ValueError, match="frozen"):
        bind_policy_servers(servers, _config("solver", "verifier"))


def test_resolve_legacy_single_policy_unchanged():
    servers = _servers(("default", True), ("rm", False))
    # model_id=None keeps the first-updatable behavior, no binding needed
    assert resolve_updatable_server(servers, None, None) is servers["default"]
    frozen_only = _servers(("rm", False))
    assert resolve_updatable_server(frozen_only, None, None) is None


def test_resolve_indexes_precomputed_binding():
    servers = _servers(("solver", True), ("verifier", True))
    binding = bind_policy_servers(servers, _config("solver", "verifier"))
    assert resolve_updatable_server(servers, "solver", binding) is servers["solver"]
    assert resolve_updatable_server(servers, "verifier", binding) is servers["verifier"]


def test_resolve_model_id_requires_binding():
    servers = _servers(("solver", True))
    with pytest.raises(ValueError, match="binding"):
        resolve_updatable_server(servers, "solver", None)
    with pytest.raises(ValueError, match="not in the validated binding"):
        resolve_updatable_server(servers, "solver", {"verifier": "verifier"})


def test_per_policy_weight_versions_are_independent():
    # two policies' servers report their own versions; reducing one policy's
    # engines never touches the other's
    assert consistent_weight_version(["3", "3"]) == 3
    assert consistent_weight_version(["7", "7", "7"]) == 7
    assert consistent_weight_version([]) is None


def test_weight_version_consistency_enforced():
    with pytest.raises(RuntimeError, match="inconsistent"):
        consistent_weight_version(["3", "4"])
    with pytest.raises(RuntimeError, match="nonnumeric"):
        consistent_weight_version(["abc"])
    # allow_inconsistent seeds from the highest installed version
    assert consistent_weight_version(["3", "abc"], allow_inconsistent=True) == 3
    assert consistent_weight_version(["abc"], allow_inconsistent=True) == 0


# ---------------- ServingCluster lock wiring ----------------


def _bare_cluster(**fields):
    """A ServingCluster instance without the Ray actor wrapper or __init__."""
    from vime.ray.serving import ServingCluster

    inst = ServingCluster.__ray_actor_class__.__new__(ServingCluster.__ray_actor_class__)
    inst.servers = {}
    inst.lock = None
    inst._policy_locks = {}
    inst._policy_binding = None
    for key, value in fields.items():
        setattr(inst, key, value)
    return inst


def test_get_engine_lock_returns_the_lock_the_server_holds():
    global_lock, solver_lock, verifier_lock = object(), object(), object()
    servers = {
        "solver": SimpleNamespace(update_weights=True, engine_lock=solver_lock),
        "verifier": SimpleNamespace(update_weights=True, engine_lock=verifier_lock),
    }
    inst = _bare_cluster(
        servers=servers,
        lock=global_lock,
        _policy_binding={"solver": "solver", "verifier": "verifier"},
    )
    assert inst.get_engine_lock() is global_lock
    assert inst.get_engine_lock("solver") is solver_lock
    assert inst.get_engine_lock("solver") is servers["solver"].engine_lock
    assert inst.get_engine_lock("verifier") is verifier_lock


def test_recreate_locks_covers_policy_locks():
    import vime.ray.serving as serving_module

    global_lock, solver_lock = object(), object()
    inst = _bare_cluster(lock=global_lock, _policy_locks={"solver": solver_lock})
    killed = []
    new_global = object()
    with (
        mock.patch.object(serving_module.ray, "kill", side_effect=lambda a, **kw: killed.append(a)),
        mock.patch.object(serving_module.Lock, "options", return_value=SimpleNamespace(remote=lambda: new_global)),
    ):
        inst._recreate_locks()
    assert killed == [global_lock, solver_lock]
    assert inst.lock is new_global
    assert inst._policy_locks == {}


def test_dispose_cleanup_covers_policy_locks():
    import vime.ray.serving as serving_module

    global_lock, solver_lock = object(), object()
    inst = _bare_cluster(
        args=SimpleNamespace(rollout_cleanup_timeout=5),
        lock=global_lock,
        _policy_locks={"solver": solver_lock},
        router_processes=[],
        controller=None,
        _health_monitors=[],
        training_actors={},
        placements={},
    )
    killed = []
    with mock.patch.object(serving_module.ray, "kill", side_effect=lambda a, **kw: killed.append(a)):
        inst.dispose()
    assert global_lock in killed and solver_lock in killed
    assert inst._policy_locks == {}

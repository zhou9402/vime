"""Binding between multi-policy trainers (``--multi-policy-config``) and serving servers.

Each policy model_id binds to exactly one serving server (a ``RolloutServer``
named after the model_id, declared with ``update_weights: true`` in the vLLM
model config). Servers are disjoint across policies in v1: weight updates of
one policy never touch another policy's engines.

``bind_policy_servers`` validates the whole mapping ONCE (when the config is
attached to the cluster / manager) and returns it; afterwards
``resolve_updatable_server`` only indexes the stored mapping.

All functions here are pure so they can be unit-tested without Ray.
"""

from collections.abc import Mapping
from typing import Any

from vime.utils.multi_policy.megatron_config import MultiPolicyConfig


def bind_policy_servers(servers: Mapping[str, Any], config: MultiPolicyConfig) -> dict[str, str]:
    """Validate and return the ``model_id -> server name`` mapping.

    Raises ValueError on any mismatch; called once at attach time, not per update.
    """
    bound = {}
    for model_id in config.model_ids:
        server = servers.get(model_id)
        if server is None:
            raise ValueError(
                f"--multi-policy-config policy {model_id!r} has no serving server named {model_id!r}; "
                f"declare one model per policy (update_weights: true) in the vLLM model config, "
                f"known servers: {sorted(servers)}"
            )
        if not getattr(server, "update_weights", False):
            raise ValueError(
                f"serving server {model_id!r} is frozen (update_weights: false) but policy {model_id!r} "
                "trains; every policy needs its own updatable server"
            )
        bound[model_id] = model_id
    return bound


def resolve_updatable_server(
    servers: Mapping[str, Any],
    model_id: str | None,
    binding: Mapping[str, str] | None,
) -> Any | None:
    """Pick the updatable server.

    ``model_id=None`` preserves the single-policy behavior: the first server
    with ``update_weights=True``. With a model_id, indexes the precomputed
    binding (validated once at attach time).
    """
    if model_id is None:
        return next((server for server in servers.values() if getattr(server, "update_weights", False)), None)
    if binding is None:
        raise ValueError(f"model_id {model_id!r} given but no validated policy->server binding exists")
    server_name = binding.get(model_id)
    if server_name is None:
        raise ValueError(f"policy {model_id!r} is not in the validated binding {sorted(binding)}")
    return servers[server_name]


def consistent_weight_version(versions: list, *, allow_inconsistent: bool = False) -> int | None:
    """Reduce per-engine weight versions of one policy's server to one integer.

    Shared by ServingCluster and RolloutManager; semantics unchanged from the
    single-policy path (consistency required unless allow_inconsistent).
    """
    if not versions:
        return None
    if allow_inconsistent:
        # A partial update or a replacement engine may have no version yet.
        # Seed the restart from the highest installed version so publishing
        # restored checkpoint weights does not move the version backwards.
        return max((int(version) for version in versions if str(version).isdigit()), default=0)
    if any(not str(version).isdigit() for version in versions):
        raise RuntimeError(f"Cannot resume nonnumeric serving weight versions: {versions}")
    if len(set(versions)) != 1:
        raise RuntimeError(f"Cannot checkpoint inconsistent serving weight versions: {versions}")
    return int(versions[0])

"""Dedicated eval fleet: a pinned set of vLLM engines behind their own router.

Port of miles@ba28991834 miles/ray/rollout/eval_fleet.py, redesigned against
vime's ServingCluster/RolloutServer instead of miles' inference-controller RPC
surface, and against vLLM engine endpoints instead of SGLangApiClient:

- ``POST /update_weights_from_disk`` -> ``VLLMEngine.update_weights_from_disk``
  (collective_rpc ``reload_weights``) fanned out over every eval engine.
- ``GET /model_info`` weight-version read-back -> ``VLLMEngine.get_weight_version``.
- sglang-router ``POST /generate`` one-token probe -> per-engine ``health_generate``.

The fleet loads weights exclusively from exported HF snapshots and never joins
the training weight sync (``update_weights=False`` in the vLLM config). Eval
runs on the ``EvalRunner`` actor — a separate process so a slow snapshot load
or eval never occupies the RolloutManager actor and stalls training rollouts.
The runner is driver-owned (not detached): eval engines are not part of the
recoverable serving topology in v1 and die with the driver.
"""

import copy
import dataclasses
import logging
from pathlib import Path

import ray

logger = logging.getLogger(__name__)

EVAL_MODEL_NAME = "eval"
HF_EXPORT_MARKER = ".vime_hf_export_complete"


class EvalSkip(Exception):
    """Skip this eval point with attribution (eval/skipped_<reason>)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclasses.dataclass(frozen=True)
class EvalFleetInfo:
    router: tuple[str, int]
    engine_gpu_counts: list[int]


@dataclasses.dataclass(frozen=True)
class EvalFleetPin:
    # None means the fleet is pinned and ready; otherwise the skip reason.
    skip_reason: str | None = None


def get_eval_fleet_info(servers) -> EvalFleetInfo | None:
    """Extract the eval fleet view from a RolloutManager server snapshot."""
    server = servers.get(EVAL_MODEL_NAME) if servers else None
    if server is None:
        return None
    return EvalFleetInfo(router=(server.router_ip, server.router_port), engine_gpu_counts=list(server.engine_gpu_counts))


def pin_eval_fleet(engines, hf_dir: str, weight_version: str, *, require_marker: bool, gather=None) -> EvalFleetPin:
    """Load an HF snapshot onto every eval engine and verify the version.

    The router load-balances across engines, so one stale engine would mix
    weight versions within a single eval point; every engine must read back
    the target version before the fleet is considered pinned.
    """
    gather = gather or ray.get
    if require_marker and not Path(hf_dir, HF_EXPORT_MARKER).is_file():
        logger.error(f"Eval snapshot {hf_dir} is missing the {HF_EXPORT_MARKER} marker")
        return EvalFleetPin(skip_reason="export_incomplete")

    try:
        gather(
            [
                engine.update_weights_from_disk.remote(
                    model_path=hf_dir,
                    weight_version=str(weight_version),
                    flush_cache=True,
                )
                for engine in engines
            ]
        )
    except Exception:
        logger.exception(f"Eval fleet failed to load snapshot {hf_dir}")
        return EvalFleetPin(skip_reason="unhealthy")

    versions = gather([engine.get_weight_version.remote() for engine in engines])
    if any(str(v) != str(weight_version) for v in versions):
        logger.error(f"Eval fleet pin violation: versions {versions} != {weight_version}")
        return EvalFleetPin(skip_reason="pin_violation")

    try:
        gather([engine.health_generate.remote() for engine in engines])
    except Exception:
        logger.exception("Eval fleet health probe failed after pin")
        return EvalFleetPin(skip_reason="unhealthy")
    return EvalFleetPin()


def check_eval_fleet_version(engines, weight_version: str, *, gather=None) -> EvalFleetPin:
    """Verify the fleet still serves the requested version, without reloading."""
    gather = gather or ray.get
    try:
        versions = gather([engine.get_weight_version.remote() for engine in engines])
    except Exception:
        logger.exception("Eval fleet version check failed")
        return EvalFleetPin(skip_reason="unhealthy")
    if any(str(v) != str(weight_version) for v in versions):
        return EvalFleetPin(skip_reason="pin_violation")
    return EvalFleetPin()


@ray.remote(num_cpus=1, num_gpus=0)
class EvalRunner:
    """Runs snapshot evals in its own process, off the RolloutManager actor.

    Owning a separate GenerateState singleton here also isolates eval from
    training-side aborts and from the training concurrency semaphore.
    """

    def __init__(self, args, eval_engines, router: tuple[str, int]):
        self.args = args
        self._eval_engines = eval_engines
        self._router = router
        from vime.utils.misc import load_function

        self._eval_fn = load_function(args.eval_function_path)

    def run(self, rollout_id: int, hf_dir: str, weight_version: str, require_marker: bool):
        from vime.rollout.base_types import call_rollout_fn

        pin = pin_eval_fleet(self._eval_engines, hf_dir, weight_version, require_marker=require_marker)
        if pin.skip_reason is not None:
            raise EvalSkip(pin.skip_reason)

        # Retarget the stock eval fns at the eval fleet's router; the URL is
        # read from args on every request, so a shallow copy suffices.
        eval_args = copy.copy(self.args)
        eval_args.vllm_router_ip, eval_args.vllm_router_port = self._router

        result = call_rollout_fn(self._eval_fn, eval_args, rollout_id, None, evaluation=True)
        return result.data, result.metrics

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from vime.utils.misc import load_function
from vime.utils.types import Sample

_current_rollout_id: int | None = None
_task_rollout_id = ContextVar("rollout_sample_hook_id", default=None)
_task_weight_version = ContextVar("rollout_current_weight_version", default=None)


def set_current_rollout_id(rollout_id: int | None) -> None:
    global _current_rollout_id
    _current_rollout_id = rollout_id


def current_rollout_weight_version() -> int | None:
    """Serving weight version sampled by the manager for the active rollout."""
    return _task_weight_version.get()


@contextmanager
def rollout_context(rollout_id, weight_version=None):
    """Keep concurrent worker requests' hook contexts independent.

    ``weight_version`` carries the manager-sampled serving version so deeply
    wrapped/custom rollout functions (partials, decorators) see it without a
    signature change.
    """
    token = _task_rollout_id.set(rollout_id)
    version_token = _task_weight_version.set(weight_version)
    try:
        yield
    finally:
        _task_weight_version.reset(version_token)
        _task_rollout_id.reset(token)


def _accepted_kwargs(function, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


async def _apply_to_sample(args, sample: Sample, paths: list[str], **kwargs) -> Sample:
    for path in paths:
        hook = load_function(path)
        result = hook(args, sample, **_accepted_kwargs(hook, kwargs))
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            if not isinstance(result, Sample):
                raise TypeError(
                    f"Rollout sample hook {path!r} returned {type(result).__name__}, expected Sample or None."
                )
            sample = result
    return sample


async def apply_rollout_sample_hooks(args, value, **kwargs):
    """Apply user hooks; the owning queue publication persists replay tensors."""

    paths = getattr(args, "rollout_sample_hook_path", None) or []
    if not paths:
        return value
    rollout_id = _task_rollout_id.get()
    kwargs.setdefault("rollout_id", rollout_id if rollout_id is not None else _current_rollout_id)
    if isinstance(value, Sample):
        return await _apply_to_sample(args, value, paths, **kwargs)
    if isinstance(value, list):
        return [await apply_rollout_sample_hooks(args, item, **kwargs) for item in value]
    raise TypeError(f"Rollout sample hooks expected Sample or list, got {type(value).__name__}.")

"""Peak dense BF16 TFLOP/s per training GPU, used as the denominator of perf/actor_train_mfu.

Ported from miles@ba28991834 miles/utils/device_flops.py; AMD Instinct entries added.
"""

from __future__ import annotations

import re

_PEAK_BF16_TFLOPS: dict[str, float] = {
    "A100": 312.0,
    "H100": 989.0,
    "H100 PCIE": 756.0,
    "H200": 989.0,
    "GH200": 989.0,
    "B200": 2250.0,
    "B300": 2250.0,
    "GB200": 2500.0,
    "GB300": 2500.0,
    # https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html (peak BF16/FP16, dense)
    "MI300X": 1307.4,
    # https://www.amd.com/en/products/accelerators/instinct/mi325x.html (peak BF16/FP16, dense)
    "MI325X": 1307.4,
    # https://www.amd.com/en/products/accelerators/instinct/mi355x.html (peak BF16/FP16, dense)
    "MI355X": 2500.0,
}

_MATCH_ORDER: tuple[str, ...] = tuple(sorted(_PEAK_BF16_TFLOPS, key=len, reverse=True))


def _words(device_name: str) -> str:
    return f" {re.sub(r'[^A-Za-z0-9]+', ' ', device_name).upper().strip()} "


def peak_bf16_tflops(device_name: str) -> float | None:
    name = _words(device_name)
    for key in _MATCH_ORDER:
        if f" {key} " in name:
            return _PEAK_BF16_TFLOPS[key]
    return None


def local_peak_bf16_tflops() -> float | None:
    try:
        from vime.utils import accelerator

        device_name = accelerator.device_name()
    except Exception:  # noqa: BLE001 - no accelerator on CPU-only hosts
        return None
    return peak_bf16_tflops(device_name) if device_name else None

"""Build metadata-only rollout schemas from Hugging Face checkpoints."""

import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from vime.utils.hf_utils.weight_mapping import get_checkpoint_weight_map, get_param_name_remap
from vime.utils.mxfp8 import MXFP8_GROUP_SIZE, assert_mxfp8_quantizable_expert, should_use_mxfp8

_HEADER_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I8": torch.int8,
}
_WEIGHT_INDEX = "model.safetensors.index.json"


def _load_tensor_layouts(source_dir: Path, weight_map: dict[str, str]) -> dict[str, tuple[tuple[int, ...], str]]:
    """Shape and dtype string per tensor, read from headers without loading payloads."""
    layouts: dict[str, tuple[tuple[int, ...], str]] = {}
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)

    for shard, names in names_by_shard.items():
        with safe_open(source_dir / shard, framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            for name in names:
                if name not in available:
                    raise KeyError(f"{name} is listed in {_WEIGHT_INDEX} but missing from {shard}")
                tensor_slice = tensors.get_slice(name)
                layouts[name] = (tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
    return layouts


def _natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.findall(r"\d+|\D+", value)]


def build_mxfp8_quantization_config(source_dir: str | Path) -> dict:
    """Derive the MXFP8 rollout quantization config from checkpoint headers."""
    source = Path(source_dir)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected config.json under {source}")

    weight_map = get_checkpoint_weight_map(source)
    remap_name = get_param_name_remap(str(config_path), weight_map)
    tensor_layouts = _load_tensor_layouts(source, weight_map)

    ignored_modules = set()
    for source_name, (shape, dtype_name) in tensor_layouts.items():
        if not source_name.endswith(".weight"):
            continue
        hf_name = remap_name(source_name)
        dtype = _HEADER_DTYPES.get(dtype_name)
        packed_mxfp4_experts = dtype is torch.int8 and ".experts." in hf_name
        quantizable = dtype is not None and should_use_mxfp8(
            hf_name, shape, dtype, allow_source_fp8=True, packed_mxfp4_experts=packed_mxfp4_experts
        )
        if not quantizable:
            # Online expert updates do not consult the ignore list, so an
            # unquantizable expert must fail loudly instead of being ignored.
            assert_mxfp8_quantizable_expert(hf_name, shape, dtype)
            ignored_modules.add(hf_name.removesuffix(".weight"))

    ignored = sorted(ignored_modules, key=_natural_key)
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "mxfp8",
        "weight_block_size": [1, MXFP8_GROUP_SIZE],
        "scale_fmt": "ue8m0",
        "modules_to_not_convert": ignored,
        # vLLM's ModelOptMxFp8Config reads the flat-form `ignored_layers`
        # (mapped to `exclude_modules`); keep both spellings in sync.
        "ignored_layers": ignored,
    }


def create_mxfp8_rollout_schema(source_dir: str | Path, destination_dir: str | Path) -> Path:
    """Create a small MXFP8 config/tokenizer directory with no weight payloads."""
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    if source == destination:
        raise ValueError("The rollout schema destination must differ from the source checkpoint")

    quantization_config = build_mxfp8_quantization_config(source)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.safetensors")):
        raise ValueError(f"Refusing to use schema directory containing weight payloads: {destination}")
    for source_file in source.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == _WEIGHT_INDEX or source_file.suffix == ".safetensors":
            continue
        shutil.copy2(source_file, destination / source_file.name)

    with (source / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["quantization_config"] = quantization_config
    # miles also sets config["expert_dtype"] = "fp8"; vLLM only reads
    # `expert_dtype` in the DeepSeek-V4 family configs, not for MXFP8, so the
    # port drops it.
    with (destination / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    return destination

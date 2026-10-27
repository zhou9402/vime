import json
from collections.abc import Callable
from pathlib import Path

from safetensors import safe_open


def get_checkpoint_weight_map(checkpoint_dir: str | Path) -> dict[str, str]:
    """Read tensor names from an index or safetensors metadata without loading weights."""
    source = Path(checkpoint_dir)
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            return json.load(index_file)["weight_map"]

    weight_map = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in weight_map:
                    raise ValueError(f"Duplicate tensor {name} in {weight_map[name]} and {shard.name}")
                weight_map[name] = shard.name
    if not weight_map:
        raise FileNotFoundError(f"No safetensors weights or index under {source}")
    return weight_map


def get_param_name_remap(config_path: str, weight_map: dict[str, str]) -> Callable[[str], str]:
    """Return the checkpoint-to-HF name mapping for a supported checkpoint."""
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if "DeepseekV4ForCausalLM" in config.get("architectures", []) and "embed.weight" in weight_map:
        # miles maps SGLang-native DeepSeek-V4 checkpoints via
        # sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format;
        # vLLM has no equivalent public helper. Use an HF-layout checkpoint instead.
        raise NotImplementedError(
            "SGLang-native DeepSeek-V4 checkpoint layout is not supported by the vLLM port; "
            "convert from a checkpoint already in HF layout."
        )
    return lambda name: name


def get_param_name_remap_for_checkpoint(checkpoint_dir: str | Path) -> Callable[[str], str]:
    """Resolve the checkpoint-to-HF name mapping straight from a checkpoint directory."""
    checkpoint = Path(checkpoint_dir)
    return get_param_name_remap(str(checkpoint / "config.json"), get_checkpoint_weight_map(checkpoint))

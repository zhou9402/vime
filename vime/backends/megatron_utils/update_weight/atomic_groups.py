"""HF-namespace atomic update groups: HF weights the engine loader fuses into
one parameter, which therefore must arrive in the same update bucket.

Port of miles@ba28991834
backends/training_utils/weight_update/hf_weight_iterator/atomic_groups.py,
restricted to architectures vime converts today (see
vime/backends/megatron_utils/megatron_to_hf/). The miles DeepSeek-V4/V4.1
groups target sglang's deepseek_v4 loader cache-and-concat behavior and are
dropped: vime has no deepseekv4 converter. fp8 weight+scale pairing needs no
group: vime's converter emits both from one conversion call (one unit).
"""

from vime.backends.megatron_utils.update_weight.bucketing import AtomicUpdateGroup


def get_hf_atomic_update_groups(
    model_name: str,
    *,
    q_lora_rank: int | None = None,
) -> list[AtomicUpdateGroup]:
    """Atomic groups for a model. Only the DeepSeek-family q_a_proj +
    kv_a_proj_with_mqa pair (fused by the engine into one qkv_a param) is
    registered; every other vime architecture is split-safe."""
    normalized_model_name = model_name.lower().replace("-", "").replace("_", "")
    if q_lora_rank is not None and ("deepseek" in normalized_model_name or "kimi" in normalized_model_name):
        return [
            AtomicUpdateGroup(
                key="q_lora_a_proj",
                suffixes=(
                    ".self_attn.q_a_proj.weight",
                    ".self_attn.kv_a_proj_with_mqa.weight",
                ),
            )
        ]
    return []

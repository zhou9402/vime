import re

from vime.utils.mxfp8 import mxfp8_quantize

# Attention/MLP weight names (Megatron-side, after the layer prefix is stripped)
# that are quantized to MXFP8 for the rollout engine.
MXFP8_PARAM_NAMES = [
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc2.weight",
    # mla
    "self_attention.linear_q_proj.weight",
    "self_attention.linear_q_down_proj.weight",
    "self_attention.linear_q_up_proj.weight",
    "self_attention.linear_kv_down_proj.weight",
    "self_attention.linear_kv_up_proj.weight",
    "self_attention.wq_b.weight",
    "self_attention.wk.weight",
    # DeepSeek V4 attention
    "self_attention.linear_kv_proj.weight",
    "self_attention.core_attention.indexer.linear_wq_b.weight",
    "self_attention.core_attention.indexer.linear_wk.weight",
    # linear attention
    "self_attention.linear_attn.in_proj_qkv.weight",
    "self_attention.linear_attn.in_proj_z.weight",
    "self_attention.linear_attn.out_proj.weight",
]


def quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "mxfp8"

    extra_high_precision_layers = getattr(args, "extra_high_precision_layers_megatron", None) or ()
    for layer_name in extra_high_precision_layers:
        if layer_name in megatron_name:
            return converted_named_params

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "").replace("mtp_model_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Skip quantization for the BF16 head/tail of main decoder layers. The two
    # counts double as the enable switch (miles keyed this off
    # args.backend.first_last_layers_bf16, which vime does not have).
    num_layers_at_start_in_bf16 = int(getattr(args, "num_layers_at_start_in_bf16", 0) or 0)
    num_layers_at_end_in_bf16 = int(getattr(args, "num_layers_at_end_in_bf16", 0) or 0)
    if num_layers_at_start_in_bf16 > 0 or num_layers_at_end_in_bf16 > 0:
        num_layers = int(args.num_layers)
        head_end_idx = num_layers_at_start_in_bf16
        tail_start_idx = num_layers - num_layers_at_end_in_bf16
        if int(layer_idx) < head_end_idx or int(layer_idx) >= tail_start_idx:
            return converted_named_params

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_param(converted_name, param))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(_quantize_param(converted_name, param))

            return quantize_named_params

    if rest in MXFP8_PARAM_NAMES:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(converted_name, param))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_param(name, weight):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    qweight, scale = mxfp8_quantize(weight)
    # vLLM's ModelOptMxFp8Config expects the scale tensor named `weight_scale`
    # (uint8 UE8M0, [..., K // 32]); miles/SGLang used `weight_scale_inv`.
    scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]

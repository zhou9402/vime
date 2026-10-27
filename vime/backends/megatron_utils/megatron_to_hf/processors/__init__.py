from .padding_remover import remove_padding

__all__ = ["quantize_params", "remove_padding"]


def quantize_params(args, megatron_name, converted_named_params, quantization_config, transform_ue8m0=True):
    if quantization_config is None:
        return converted_named_params

    if quantization_config["quant_method"] == "fp8":
        from .quantizer_fp8 import quantize_params_fp8

        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config, transform_ue8m0)

    if quantization_config["quant_method"] == "compressed-tensors":
        from .quantizer_compressed_tensors import quantize_params_compressed_tensors

        # only int4 at the moment.
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)

    if quantization_config["quant_method"] == "mxfp8":
        from .quantizer_mxfp8 import quantize_params_mxfp8

        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)

    if quantization_config["quant_method"] in ("nvfp4", "NVFP4", "modelopt_fp4"):
        raise NotImplementedError(
            f"quant_method {quantization_config['quant_method']!r} requires the NVFP4 quantizer "
            "(F08-3/F08-4, not yet ported); refusing to silently pass BF16 weights through."
        )

    # Unknown quant method (e.g. mxfp4) — pass through BF16 params as-is
    return converted_named_params

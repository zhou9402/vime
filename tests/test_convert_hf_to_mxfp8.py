"""CPU tests for tools/convert_hf_to_mxfp8.py and the metadata-only rollout schema.

The TE-backed ``mxfp8_quantize`` is monkeypatched with the ``quant_ref`` oracle
so the conversion runs on CPU; oracle numerics are pinned in
test_low_precision_quant_ref.py.

Port of miles@ba28991834 tools/convert_hf_to_mxfp8.py coverage.
"""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
import safetensors.torch  # noqa: E402

from vime.utils import quant_ref  # noqa: E402
from vime.utils.hf_rollout_schema import create_mxfp8_rollout_schema  # noqa: E402


def _load_converter():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "convert_hf_to_mxfp8.py"
    spec = importlib.util.spec_from_file_location("convert_hf_to_mxfp8", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converter():
    return _load_converter()


def _write_source_checkpoint(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    weights = {
        "model.embed_tokens.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        "model.layers.0.mlp.gate_up_proj.weight": torch.randn(16, 64, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(64, dtype=torch.bfloat16),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(weights, src / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in weights}}
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    (src / "config.json").write_text(
        json.dumps({"architectures": ["ToyForCausalLM"], "num_hidden_layers": 2, "hidden_size": 64})
    )
    (src / "tokenizer_config.json").write_text("{}")
    return src


def test_convert_bf16_checkpoint(converter, tmp_path, monkeypatch):
    src = _write_source_checkpoint(tmp_path)
    monkeypatch.setattr(converter, "quantize_mxfp8", quant_ref.mxfp8_quantize_ref)
    out = tmp_path / "out"
    converter.convert_mxfp8(str(src), str(out), "cpu")

    cfg = json.loads((out / "config.json").read_text())
    qcfg = cfg["quantization_config"]
    assert qcfg["quant_method"] == "mxfp8"
    assert qcfg["fmt"] == "e4m3"
    assert qcfg["activation_scheme"] == "dynamic"
    assert qcfg["weight_block_size"] == [1, 32]
    assert qcfg["scale_fmt"] == "ue8m0"
    # Both spellings are kept in sync; vLLM's ModelOptMxFp8Config reads the flat form.
    assert qcfg["modules_to_not_convert"] == qcfg["ignored_layers"]
    assert "model.embed_tokens" in qcfg["modules_to_not_convert"]
    assert "model.layers.0.input_layernorm" in qcfg["modules_to_not_convert"]
    assert "expert_dtype" not in cfg

    with safetensors.safe_open(out / "model-00001-of-00001.safetensors", framework="pt", device="cpu") as f:
        keys = set(f.keys())
        # Scale tensors use the vLLM `weight_scale` naming (not weight_scale_inv).
        assert "model.layers.0.mlp.gate_up_proj.weight" in keys
        assert "model.layers.0.mlp.gate_up_proj.weight_scale" in keys
        assert not any(k.endswith(".weight_scale_inv") for k in keys)
        qweight = f.get_tensor("model.layers.0.mlp.gate_up_proj.weight")
        scale = f.get_tensor("model.layers.0.mlp.gate_up_proj.weight_scale")
        embed = f.get_tensor("model.embed_tokens.weight")

    assert qweight.dtype == torch.float8_e4m3fn and qweight.shape == (16, 64)
    assert scale.dtype == torch.uint8 and scale.shape == (16, 2)
    source = safetensors.torch.load_file(src / "model-00001-of-00001.safetensors")
    dequant = quant_ref.mxfp8_dequant_ref(qweight, scale)
    rel = (dequant - source["model.layers.0.mlp.gate_up_proj.weight"].float()).abs() / source[
        "model.layers.0.mlp.gate_up_proj.weight"
    ].float().abs().clamp_min(1e-6)
    assert (rel <= 2**-4 + 1e-6).all()
    # Skipped tensors are copied bit-exact.
    assert torch.equal(embed, source["model.embed_tokens.weight"])


def test_convert_first_last_layers_bf16(converter, tmp_path, monkeypatch):
    src = _write_source_checkpoint(tmp_path)
    monkeypatch.setattr(converter, "quantize_mxfp8", quant_ref.mxfp8_quantize_ref)
    out = tmp_path / "out"
    converter.convert_mxfp8(str(src), str(out), "cpu", num_layers_at_end_in_bf16=1)

    with safetensors.safe_open(out / "model-00001-of-00001.safetensors", framework="pt", device="cpu") as f:
        # Layer 1 (last) stays BF16: no scale tensor, bf16 dtype.
        assert "model.layers.1.self_attn.o_proj.weight_scale" not in set(f.keys())
        assert f.get_tensor("model.layers.1.self_attn.o_proj.weight").dtype == torch.bfloat16
    cfg = json.loads((out / "config.json").read_text())
    assert "model.layers.1." in cfg["quantization_config"]["modules_to_not_convert"]


def test_metadata_only_schema(tmp_path):
    src = _write_source_checkpoint(tmp_path)
    out = tmp_path / "schema"
    create_mxfp8_rollout_schema(src, out)

    assert not list(out.glob("*.safetensors"))
    assert not (out / "model.safetensors.index.json").exists()
    assert (out / "tokenizer_config.json").exists()
    cfg = json.loads((out / "config.json").read_text())
    qcfg = cfg["quantization_config"]
    assert qcfg["quant_method"] == "mxfp8"
    assert qcfg["weight_block_size"] == [1, 32]
    assert qcfg["modules_to_not_convert"] == qcfg["ignored_layers"]
    assert "model.embed_tokens" in qcfg["modules_to_not_convert"]
    assert "expert_dtype" not in cfg


def _write_moe_checkpoint(tmp_path: Path, expert_cols: int) -> Path:
    src = tmp_path / "src_moe"
    src.mkdir()
    weights = {
        "model.embed_tokens.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(16, expert_cols, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(weights, src / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in weights}}
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    (src / "config.json").write_text(
        json.dumps({"architectures": ["ToyMoeForCausalLM"], "num_hidden_layers": 1, "hidden_size": 64})
    )
    return src


def test_convert_raises_on_unquantizable_expert(converter, tmp_path, monkeypatch):
    # Expert K=48 is not divisible by the MXFP8 group size 32.
    src = _write_moe_checkpoint(tmp_path, expert_cols=48)
    monkeypatch.setattr(converter, "quantize_mxfp8", quant_ref.mxfp8_quantize_ref)
    with pytest.raises(ValueError, match="Unsupported MXFP8 expert"):
        converter.convert_mxfp8(str(src), str(tmp_path / "out"), "cpu")


def test_schema_raises_on_unquantizable_expert(tmp_path):
    src = _write_moe_checkpoint(tmp_path, expert_cols=48)
    with pytest.raises(ValueError, match="Unsupported MXFP8 expert"):
        create_mxfp8_rollout_schema(src, tmp_path / "schema")


def test_schema_accepts_quantizable_expert(tmp_path):
    src = _write_moe_checkpoint(tmp_path, expert_cols=64)
    out = tmp_path / "schema"
    create_mxfp8_rollout_schema(src, out)
    cfg = json.loads((out / "config.json").read_text())
    assert "model.layers.0.mlp.experts.0.gate_proj" not in cfg["quantization_config"]["modules_to_not_convert"]

"""CPU tests for tools/convert_hf_to_nvfp4.py.

TE-backed quantization is monkeypatched with oracle-based wrappers; oracle
numerics pinned in test_low_precision_quant_ref.py.

Port of miles@ba28991834 tools/convert_hf_to_nvfp4.py coverage.
"""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
import safetensors.torch  # noqa: E402

from vime.utils import quant_ref  # noqa: E402


def _load_converter():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "convert_hf_to_nvfp4.py"
    spec = importlib.util.spec_from_file_location("convert_hf_to_nvfp4", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converter():
    return _load_converter()


@pytest.fixture(autouse=True)
def patch_quantizers(converter, monkeypatch):
    def _pair_ref(first, second):
        combined = torch.cat((first.float(), second.float()))
        q, s, g = quant_ref.nvfp4_quantize_ref(combined)
        rows = first.shape[0]
        return (q[:rows], s[:rows], g), (q[rows:], s[rows:], g.clone())

    monkeypatch.setattr(converter, "nvfp4_quantize_1d", lambda w: quant_ref.nvfp4_quantize_ref(w.float()))
    monkeypatch.setattr(converter, "nvfp4_quantize_1d_pair", _pair_ref)


def _write_moe_checkpoint(tmp_path: Path, *, num_layers: int = 2, split_pair: bool = False) -> Path:
    src = tmp_path / "src_nvfp4"
    src.mkdir()
    weights = {
        "model.embed_tokens.weight": torch.randn(32, 64, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(16, 48, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(16, 48, dtype=torch.bfloat16),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(33, 48, dtype=torch.bfloat16),  # M % 16 != 0
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(16, 48, dtype=torch.bfloat16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
    }
    for layer in range(1, num_layers):
        weights[f"model.layers.{layer}.mlp.experts.0.down_proj.weight"] = torch.randn(16, 48, dtype=torch.bfloat16)

    if split_pair:
        first = {k: v for k, v in weights.items() if "up_proj" not in k}
        second = {k: v for k, v in weights.items() if "up_proj" in k}
        safetensors.torch.save_file(first, src / "model-00001-of-00002.safetensors", metadata={"format": "pt"})
        safetensors.torch.save_file(second, src / "model-00002-of-00002.safetensors", metadata={"format": "pt"})
        weight_map = {k: "model-00001-of-00002.safetensors" for k in first}
        weight_map.update({k: "model-00002-of-00002.safetensors" for k in second})
    else:
        safetensors.torch.save_file(weights, src / "model-00001-of-00001.safetensors", metadata={"format": "pt"})
        weight_map = {k: "model-00001-of-00001.safetensors" for k in weights}
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (src / "config.json").write_text(
        json.dumps({"architectures": ["ToyMoeForCausalLM"], "num_hidden_layers": num_layers, "hidden_size": 64})
    )
    return src


def _read(out: Path, name: str):
    index = json.loads((out / "model.safetensors.index.json").read_text())
    with safetensors.safe_open(out / index["weight_map"][name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


@pytest.mark.parametrize("split_pair", [False, True], ids=["same_shard", "across_shards"])
def test_convert_moe_checkpoint(converter, tmp_path, split_pair):
    src = _write_moe_checkpoint(tmp_path, split_pair=split_pair)
    out = tmp_path / "out"
    converter.convert_nvfp4(str(src), str(out), "cpu")

    base = "model.layers.0.mlp.experts.0"
    gq = _read(out, f"{base}.gate_proj.weight")
    gs = _read(out, f"{base}.gate_proj.weight_scale")
    g2 = _read(out, f"{base}.gate_proj.weight_scale_2")
    u2 = _read(out, f"{base}.up_proj.weight_scale_2")
    inp = _read(out, f"{base}.gate_proj.input_scale")
    # vLLM modelopt_fp4 contract.
    assert gq.dtype == torch.uint8 and gq.shape == (16, 24)
    assert gs.dtype == torch.float8_e4m3fn and gs.shape == (16, 3)
    assert g2.dtype == torch.float32 and torch.equal(g2, u2)  # shared amax pair
    assert torch.equal(inp, torch.ones_like(inp))

    # Non-expert and shared-expert weights stay BF16 and bit-exact.
    source = {}
    for shard in src.glob("*.safetensors"):
        source.update(safetensors.torch.load_file(shard))
    dense = _read(out, "model.layers.0.self_attn.o_proj.weight")
    assert dense.dtype == torch.bfloat16
    assert torch.equal(dense, source["model.layers.0.self_attn.o_proj.weight"])
    assert torch.equal(
        _read(out, "model.layers.0.mlp.shared_experts.gate_proj.weight"),
        source["model.layers.0.mlp.shared_experts.gate_proj.weight"],
    )

    cfg = json.loads((out / "config.json").read_text())
    qcfg = cfg["quantization_config"]
    assert qcfg["quant_algo"] == "NVFP4"
    assert qcfg["quant_method"] == "modelopt"
    assert qcfg["group_size"] == 16
    assert "model.embed_tokens" in qcfg["ignore"]
    assert "model.layers.0.self_attn.o_proj" in qcfg["ignore"]

    hf_quant = json.loads((out / "hf_quant_config.json").read_text())
    assert hf_quant["quantization"]["quant_algo"] == "NVFP4"
    assert hf_quant["quantization"]["exclude_modules"] == qcfg["ignore"]


def test_convert_last_layer_bf16(converter, tmp_path):
    src = _write_moe_checkpoint(tmp_path, num_layers=2)
    out = tmp_path / "out"
    converter.convert_nvfp4(str(src), str(out), "cpu", num_layers_at_end_in_bf16=1)

    # Last-layer expert stays BF16; ignore list carries the FusedMoE container.
    last = _read(out, "model.layers.1.mlp.experts.0.down_proj.weight")
    assert last.dtype == torch.bfloat16
    cfg = json.loads((out / "config.json").read_text())
    ignore = cfg["quantization_config"]["ignore"]
    assert "model.layers.1.mlp.experts" in ignore
    # Layer 0 expert is still quantized.
    assert _read(out, "model.layers.0.mlp.experts.0.down_proj.weight").dtype == torch.uint8


def test_extra_high_precision_layers_hf(converter, tmp_path):
    src = _write_moe_checkpoint(tmp_path)
    out = tmp_path / "out"
    converter.convert_nvfp4(str(src), str(out), "cpu", extra_high_precision_layers_hf=("experts.0.down_proj",))
    assert _read(out, "model.layers.0.mlp.experts.0.down_proj.weight").dtype == torch.bfloat16
    assert _read(out, "model.layers.0.mlp.experts.0.gate_proj.weight").dtype == torch.uint8

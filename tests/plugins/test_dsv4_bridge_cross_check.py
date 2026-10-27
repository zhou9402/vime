"""Name-level cross-check: hand-written DSv4 converters vs the authoritative
mbridge bridge (vime_plugins/mbridge/deepseek_v4.py, verbatim from
miles@ba28991834 miles_plugins/mbridge/deepseek_v4.py).

mbridge is not installed in CPU CI, so the bridge's DSv4-specific mapping
deltas are vendored below (BRIDGE_TABLES). If the bridge file changes, update
the vendored tables in the same PR.
"""

import pytest
import torch

from tests.test_hf_to_megatron import DictReader
from vime.backends.megatron_utils.hf_to_megatron.deepseek_v4 import deepseek_v4_hf_tensor
from vime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from vime.backends.megatron_utils.megatron_to_hf.deepseekv4 import _PENDING_ALPHAS

# Vendored from vime_plugins/mbridge/deepseek_v4.py (miles@ba28991834). Values are the
# HF keys the bridge maps each Megatron name to, with {layer_number} -> layer 1.
BRIDGE_LAYERS = {
    "self_attention.linear_q_down_proj.weight": "self_attn.wq_a.weight",
    "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
    "self_attention.linear_q_up_proj.weight": "self_attn.wq_b.weight",
    "self_attention.linear_kv_proj.weight": "self_attn.wkv.weight",
    "self_attention.kv_layernorm.weight": "self_attn.kv_norm.weight",
    "self_attention.linear_o_group_proj": "self_attn.wo_a.weight",
    "self_attention.linear_proj.weight": "self_attn.wo_b.weight",
    "self_attention.core_attention.attn_sink": "self_attn.attn_sink",
    "self_attention.core_attention.compressor.ape": "self_attn.compressor.ape",
    "self_attention.core_attention.compressor.linear_wkv.weight": "self_attn.compressor.wkv.weight",
    "self_attention.core_attention.compressor.linear_wgate.weight": "self_attn.compressor.wgate.weight",
    "self_attention.core_attention.compressor.norm.weight": "self_attn.compressor.norm.weight",
    "self_attention.core_attention.indexer.linear_wq_b.weight": "self_attn.indexer.wq_b.weight",
    # The next three come from mbridge's base DeepseekV3Bridge (V3.2 DSA), not the DSv4 delta.
    "self_attention.core_attention.indexer.linear_wk.weight": "self_attn.indexer.wk.weight",
    "self_attention.core_attention.indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "self_attention.core_attention.indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "self_attention.core_attention.indexer.linear_weights_proj.weight": "self_attn.indexer.weights_proj.weight",
    "self_attention.core_attention.indexer.compressor.ape": "self_attn.indexer.compressor.ape",
    "self_attention.core_attention.indexer.compressor.linear_wkv.weight": "self_attn.indexer.compressor.wkv.weight",
    "self_attention.core_attention.indexer.compressor.linear_wgate.weight": "self_attn.indexer.compressor.wgate.weight",
    "self_attention.core_attention.indexer.compressor.norm.weight": "self_attn.indexer.compressor.norm.weight",
    "mlp.router.tid2eid": "mlp.topk.tid2eid",
}

# The bridge pops these V3 mappings: DSv4 must not resolve them.
BRIDGE_POPPED = [
    "self_attention.linear_q_up_proj.layer_norm_weight",
    "self_attention.linear_kv_up_proj.layer_norm_weight",
]

# (mcore stem, packed HF key, alpha segments in bridge _ALPHA_SEGMENTS order)
BRIDGE_HC = [
    ("self_attention_hyper_connection", "hc_attn_fn", "hc_attn_base", "hc_attn_scale"),
    ("mlp_hyper_connection", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"),
]
BRIDGE_DIRECT = {
    "decoder.hc_head_fn": "model.hc_head_fn",
    "decoder.hc_head_base": "model.hc_head_base",
    "decoder.hc_head_scale": "model.hc_head_scale",
}

_EXPORT_ARGS = __import__("types").SimpleNamespace(
    kv_channels=2, hidden_size=8, num_attention_heads=4, num_query_groups=2, num_layers=2,
    q_lora_rank=None, vocab_size=1024,
)
_CONFIG = __import__("types").SimpleNamespace(model_type="deepseek_v4", tie_word_embeddings=False)


class RecordingReader(DictReader):
    def __init__(self, tensors):
        super().__init__(tensors)
        self.read = []

    def get_tensor(self, name):
        self.read.append(name)
        return super().get_tensor(name)


@pytest.mark.parametrize("rest,hf_suffix", sorted(BRIDGE_LAYERS.items()))
def test_bridge_layer_mapping_matches(rest, hf_suffix):
    name = f"module.module.decoder.layers.1.{rest}"
    hf_key = f"model.layers.1.{hf_suffix}"
    param = torch.randn(4, 4)

    exported = _convert_to_hf_core(_EXPORT_ARGS, "deepseekv4config", name, param)
    assert [k for k, _ in exported] == [hf_key]

    reader = RecordingReader({hf_key: param})
    loaded = deepseek_v4_hf_tensor(name, reader, _CONFIG)
    assert reader.read == [hf_key]
    assert torch.equal(loaded, param)


@pytest.mark.parametrize("rest", BRIDGE_POPPED)
def test_bridge_popped_names_unresolved(rest):
    name = f"module.module.decoder.layers.0.{rest}"
    with pytest.raises((KeyError, ValueError)):
        _convert_to_hf_core(_EXPORT_ARGS, "deepseekv4config", name, torch.randn(4))
    with pytest.raises(KeyError):
        deepseek_v4_hf_tensor(name, RecordingReader({}), _CONFIG)


@pytest.mark.parametrize("stem,fn_key,base_key,scale_key", BRIDGE_HC)
def test_bridge_hyper_connection_packing_matches(stem, fn_key, base_key, scale_key):
    w, b = torch.randn(4, 4), torch.randn(4)
    alphas = [torch.tensor([float(i)]) for i in range(3)]  # pre, post, res

    for rest, param in [
        (f"{stem}.mapping_proj.weight", w),
        (f"{stem}.bias", b),
        *[(f"{stem}.alpha_{seg}", a) for seg, a in zip(("pre", "post", "res"), alphas)],
    ]:
        name = f"module.module.decoder.layers.1.{rest}"
        exported = _convert_to_hf_core(_EXPORT_ARGS, "deepseekv4config", name, param)
        if "alpha_" in rest and "alpha_res" not in rest:
            assert exported == []  # stashed until the site is complete
        elif "alpha_res" in rest:
            assert [k for k, _ in exported] == [f"model.layers.1.{scale_key}"]
            assert torch.equal(exported[0][1], torch.cat(alphas))  # packed in [pre, post, res] order
        else:
            expected = f"model.layers.1.{fn_key if rest.endswith('weight') else base_key}"
            assert [k for k, _ in exported] == [expected]
    assert not _PENDING_ALPHAS

    # loader direction: same HF keys, scale split back into the same segment order
    packed = torch.cat(alphas)
    for seg, i in zip(("pre", "post", "res"), (0, 1, 2)):
        reader = RecordingReader({f"model.layers.1.{scale_key}": packed})
        loaded = deepseek_v4_hf_tensor(f"module.module.decoder.layers.1.{stem}.alpha_{seg}", reader, _CONFIG)
        assert reader.read == [f"model.layers.1.{scale_key}"]
        assert torch.equal(loaded, alphas[i])


@pytest.mark.parametrize("rest,hf_key", sorted(BRIDGE_DIRECT.items()))
def test_bridge_direct_mapping_matches(rest, hf_key):
    param = torch.randn(4, 4)
    name = f"module.module.{rest}"
    exported = _convert_to_hf_core(_EXPORT_ARGS, "deepseekv4config", name, param)
    assert [k for k, _ in exported] == [hf_key]
    reader = RecordingReader({hf_key: param})
    assert torch.equal(deepseek_v4_hf_tensor(name, reader, _CONFIG), param)
    assert reader.read == [hf_key]

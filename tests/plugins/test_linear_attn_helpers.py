"""CPU tests for the Kimi-K3 shared linear-attn helpers (F07-6a).

Layout tests ported from miles@ba28991834 tests/fast/models/test_linear_attn_layout.py.
"""

import torch

from vime.backends.megatron_utils.fp32_param_utils import (
    enforce_marked_param_dtypes,
    mark_param_dtype,
)
from vime.backends.megatron_utils.megatron_to_hf.linear_attn_layout import (
    LinearAttnHeads,
    gdn_heads,
    group_major,
    qkv_flat_to_group_major,
    qkv_group_major_to_flat,
    split_group_major,
)

HEADS = LinearAttnHeads(num_k_heads=4, num_v_heads=12, head_k_dim=16, head_v_dim=32)


def test_qkv_layout_roundtrip():
    for rest in ((24,), (1, 4)):
        weight = torch.randn(HEADS.qkv_dim, *rest)
        grouped = qkv_flat_to_group_major(weight, HEADS)
        assert grouped.shape == weight.shape
        assert torch.equal(qkv_group_major_to_flat(grouped, HEADS), weight)


def test_group_major_split_inverts():
    blocks = [torch.randn(HEADS.num_k_heads * n, 8) for n in (3, 5, 1)]
    restored = split_group_major(group_major(blocks, HEADS), [3, 5, 1], HEADS)
    assert all(torch.equal(a, b) for a, b in zip(restored, blocks, strict=True))


def test_tp_chunk_is_the_ranks_heads():
    """Chunk r of the group-major q/k/v equals the group-major tensor built from rank r's heads alone."""
    tp, cols = 2, 8
    weight = torch.randn(HEADS.qkv_dim, cols)
    q, k, v = weight.split([HEADS.key_dim, HEADS.key_dim, HEADS.value_dim])
    local = HEADS.local(tp)
    for rank, chunk in enumerate(qkv_flat_to_group_major(weight, HEADS).chunk(tp)):
        kr = slice(rank * local.key_dim, (rank + 1) * local.key_dim)
        vr = slice(rank * local.value_dim, (rank + 1) * local.value_dim)
        assert torch.equal(chunk, qkv_flat_to_group_major(torch.cat([q[kr], k[kr], v[vr]]), local))


def test_activation_split_matches_weight_layout():
    """A projection with group-major rows splits into the q/k/v a flat projection would give."""
    hidden = 24
    weight = torch.randn(HEADS.qkv_dim, hidden)
    x = torch.randn(2, 5, hidden)
    q_ref, k_ref, v_ref = (x @ weight.T).split([HEADS.key_dim, HEADS.key_dim, HEADS.value_dim], dim=-1)
    grouped = (x @ qkv_flat_to_group_major(weight, HEADS).T).view(2, 5, HEADS.num_k_heads, -1)
    q, k, v = grouped.split([HEADS.head_k_dim, HEADS.head_k_dim, HEADS.group_value_dim], dim=-1)
    for a, b in ((q, q_ref), (k, k_ref), (v, v_ref)):
        assert torch.allclose(a.flatten(-2), b, atol=1e-5)


def test_gdn_heads_reads_flat_and_nested_config():
    flat = type(
        "C",
        (),
        dict(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=32,
        ),
    )()
    heads = gdn_heads(flat)
    assert (heads.num_k_heads, heads.num_v_heads, heads.head_k_dim, heads.head_v_dim) == (2, 4, 16, 32)
    nested = type("C", (), dict(text_config=flat))()
    assert gdn_heads(nested) == heads


def test_mark_and_enforce_param_dtype():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    target = model[1].bias
    mark_param_dtype(target, torch.float32)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)

    updated = enforce_marked_param_dtypes([model])

    assert target.dtype == torch.float32
    assert model[0].weight.dtype == torch.bfloat16
    assert updated == ["1.bias"]


def test_enforce_without_marks_is_noop():
    model = torch.nn.Linear(4, 4)
    assert enforce_marked_param_dtypes([model]) == []

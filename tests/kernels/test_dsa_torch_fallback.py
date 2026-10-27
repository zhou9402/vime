"""CPU tests for the DSA torch fallbacks (F07-2).

Runs wherever triton is importable (linux CPU CI); no CUDA required. The
triton kernels themselves are covered by tests/kernels/test_dsa_indexer.py
on GPU CI.
"""

import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("triton")

from vime.kernels.attention.dsa.indexer import indexer_topk_scores  # noqa: E402
from vime.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402

_spec = importlib.util.spec_from_file_location("dsa_reference", pathlib.Path(__file__).with_name("dsa_reference.py"))
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def test_torch_topk_matches_canonical_reference_no_ties():
    torch.manual_seed(0)
    rows, n_kv, topk = 65, 300, 64
    logits = torch.randn(rows, n_kv)
    ks = torch.zeros(rows, dtype=torch.int32)
    ke = torch.arange(rows, dtype=torch.int32) * 4 % (n_kv + 1)
    columns = torch.arange(n_kv)
    in_range = (columns >= ks.long().unsqueeze(1)) & (columns < ke.long().unsqueeze(1))
    cleaned = logits.masked_fill(~in_range, float("-inf"))

    got = get_dsa_topk_fn("torch")(cleaned, topk)
    expected = reference.canonical_topk_ref(cleaned, topk, ks, ke)

    assert got.dtype == torch.int32
    # distinct scores: set-equal, and valid entries agree exactly after sorting
    assert torch.equal(got.sort(-1).values, expected.sort(-1).values)


def test_torch_topk_pads_short_rows():
    logits = torch.randn(4, 16)
    got = get_dsa_topk_fn("torch")(logits, 32)
    assert got.shape == (4, 32)
    assert (got[:, 16:] == -1).all()


def test_indexer_topk_scores_torch_backward_matches_reference():
    """The no-tilelang backward recomputes sum_h relu(k.q_h) * w_h with torch autograd."""
    torch.manual_seed(0)
    total, heads, dim, topk = 40, 4, 8, 16
    q = torch.randn(total, heads, dim, dtype=torch.float32)
    k = torch.randn(total, dim, dtype=torch.float32)
    weights = torch.randn(total, heads, dtype=torch.float32) * 0.01
    ks = torch.zeros(total, dtype=torch.int32)
    ke = torch.arange(total, dtype=torch.int32) + 1

    logits = reference.indexer_logits_ref(q, k, weights, ks, ke)
    topk_indices = logits.topk(topk, dim=-1).indices.int()
    topk_indices = topk_indices.masked_fill(torch.gather(logits, -1, topk_indices.long()) == -torch.inf, -1)
    grad_scores = torch.randn(total, topk)

    q_ref = q.clone().requires_grad_()
    k_ref = k.clone().requires_grad_()
    w_ref = weights.clone().requires_grad_()
    ref_scores = torch.gather(
        reference.indexer_logits_ref(q_ref, k_ref, w_ref, ks, ke), -1, topk_indices.clamp(min=0).long()
    )
    (torch.where(topk_indices != -1, ref_scores, 0.0) * grad_scores).sum().backward()

    q_t = q.clone().requires_grad_()
    k_t = k.clone().requires_grad_()
    w_t = weights.clone().requires_grad_()
    scores = indexer_topk_scores(q_t, k_t, w_t, logits, topk_indices)
    assert (scores[topk_indices == -1] == float("-inf")).all()
    (torch.where(topk_indices != -1, scores, 0.0) * grad_scores).sum().backward()

    for ref_g, got_g in ((q_ref.grad, q_t.grad), (k_ref.grad, k_t.grad), (w_ref.grad, w_t.grad)):
        assert reference.rel_diff(ref_g, got_g) < 1e-5


def test_torch_backward_warns_once(monkeypatch, caplog):
    import vime.kernels.attention.dsa.indexer as indexer_module

    monkeypatch.setattr(indexer_module, "_TORCH_BACKWARD_WARNED", [False])
    torch.manual_seed(0)
    q = torch.randn(6, 2, 8, requires_grad=True)
    k = torch.randn(6, 8)
    w = torch.randn(6, 2, requires_grad=True)
    k = k.requires_grad_()
    logits = reference.indexer_logits_ref(
        q, k, w, torch.zeros(6, dtype=torch.int32), torch.arange(6, dtype=torch.int32) + 1
    )
    topk_indices = logits.topk(4, dim=-1).indices.int()
    topk_indices[0, 1:] = -1  # force -1 padding row

    for _ in range(2):  # second backward must not warn again
        scores = indexer_topk_scores(q, k, w, logits, topk_indices)
        scores.nan_to_num(neginf=0.0).sum().backward(retain_graph=True)
    warnings = [r for r in caplog.records if "torch" in r.message and "fallback" in r.message]
    assert len(warnings) == 1

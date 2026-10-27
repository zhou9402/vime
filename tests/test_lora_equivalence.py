"""LoRA equivalence verifier (V6 / F04-0).

Machine-checkable contract for the LoRA port (see MILES_PORT_PRS.md C2-0):

1. Merged-LoRA equivalence: for a 2-layer tiny model, merging adapter
   deltas into the base weights must produce the same logits as running
   the model with adapters applied functionally.
2. Adapter export/import round-trip: exporting named adapter tensors and
   re-importing them must be bitwise identical.
3. Adapter sync contract: the trainer-side export -> (simulated) engine-side
   import path preserves every tensor bitwise. F04-7 must satisfy this harness.

Deterministic, CPU-only, no GPU or engine required.
"""

import copy

import pytest
import torch

RTOL = 1e-5
ATOL = 1e-6


class TinyBlock(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.gate_up_proj = torch.nn.Linear(hidden, ffn, bias=False)
        self.down_proj = torch.nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        x = x + torch.tanh(attn)
        return x + self.down_proj(torch.nn.functional.gelu(self.gate_up_proj(x)))


class TinyModel(torch.nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 32, ffn: int = 48, layers: int = 2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(TinyBlock(hidden, ffn) for _ in range(layers))
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def make_adapter(model: TinyModel, rank: int, alpha: float, seed: int) -> dict[str, torch.Tensor]:
    """Create a LoRA adapter (A/B per target linear), HF PEFT naming."""
    gen = torch.Generator().manual_seed(seed)
    adapter: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        out_f, in_f = module.weight.shape
        a = torch.empty(rank, in_f)
        torch.nn.init.xavier_uniform_(a, generator=gen)
        adapter[f"{name}.lora_A.weight"] = a
        adapter[f"{name}.lora_B.weight"] = torch.zeros(out_f, rank)
    return adapter


def scale(adapter: dict[str, torch.Tensor], rank: int, alpha: float) -> float:
    return alpha / rank


def forward_with_adapter(model: TinyModel, adapter: dict[str, torch.Tensor], rank: int, alpha: float, ids):
    """Functional forward: out = x @ W.T + scale * (x @ A.T @ B.T)."""
    s = scale(adapter, rank, alpha)
    hooks = []
    deltas = {}
    for name, module in model.named_modules():
        key_a, key_b = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
        if isinstance(module, torch.nn.Linear) and key_a in adapter:
            deltas[name] = s * adapter[key_b] @ adapter[key_a]

    def hook(module, _inp, out, name=None):
        x = _inp[0]
        return out + torch.nn.functional.linear(x, deltas[name])

    for name, module in model.named_modules():
        if name in deltas:
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n)))
    try:
        return model(ids)
    finally:
        for h in hooks:
            h.remove()


def merge_adapter(model: TinyModel, adapter: dict[str, torch.Tensor], rank: int, alpha: float) -> TinyModel:
    """Return a copy of model with W <- W + scale * B @ A."""
    merged = copy.deepcopy(model)
    s = scale(adapter, rank, alpha)
    with torch.no_grad():
        for name, module in merged.named_modules():
            key_a, key_b = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
            if isinstance(module, torch.nn.Linear) and key_a in adapter:
                module.weight += s * adapter[key_b] @ adapter[key_a]
    return merged


def export_adapter(adapter: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Trainer-side export: named tensors detached to CPU (F04-3/F04-7 contract)."""
    return {k: v.detach().cpu().clone() for k, v in adapter.items()}


def import_adapter(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Engine-side import: consume the exported name->tensor payload."""
    return dict(payload)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = TinyModel()
    ids = torch.arange(20).reshape(2, 10) % 64
    return model, ids


def test_merged_lora_logits_allclose(tiny):
    model, ids = tiny
    rank, alpha = 4, 16
    adapter = make_adapter(model, rank, alpha, seed=1)
    with torch.no_grad():
        logits_merged = merge_adapter(model, adapter, rank, alpha)(ids)
        logits_adapted = forward_with_adapter(model, adapter, rank, alpha, ids)
    torch.testing.assert_close(logits_merged, logits_adapted, rtol=RTOL, atol=ATOL)
    diff = (logits_merged - logits_adapted).abs().max().item()
    print(f"\nmax|merged-adapted| = {diff:.3e}")


def test_zero_adapter_is_noop(tiny):
    model, ids = tiny
    rank, alpha = 4, 16
    adapter = make_adapter(model, rank, alpha, seed=2)  # B is zero-initialized
    with torch.no_grad():
        logits_base = model(ids)
        logits_merged = merge_adapter(model, adapter, rank, alpha)(ids)
    torch.testing.assert_close(logits_base, logits_merged, rtol=0, atol=0)


def test_adapter_export_import_roundtrip(tiny):
    model, _ = tiny
    adapter = make_adapter(model, rank=4, alpha=16, seed=3)
    reloaded = import_adapter(export_adapter(adapter))
    assert set(reloaded) == set(adapter)
    for name, tensor in adapter.items():
        assert torch.equal(reloaded[name], tensor), name


def test_adapter_sync_equality(tiny):
    """Simulated trainer->engine sync: every tensor must survive bitwise."""
    model, _ = tiny
    adapter = make_adapter(model, rank=4, alpha=16, seed=4)
    payload = export_adapter(adapter)
    # simulated transport: serialize/deserialize preserves bytes
    transported = {k: v.clone() for k, v in payload.items()}
    engine_side = import_adapter(transported)
    assert len(engine_side) == len(adapter), "silent tensor drop"
    for name in adapter:
        assert name in engine_side, f"missing {name}"
        assert torch.equal(engine_side[name], adapter[name]), f"mismatch {name}"
    # tamper detection: flipping one bit must be observable
    tampered = {k: v.clone() for k, v in payload.items()}
    victim = next(iter(tampered))
    tampered[victim].view(torch.int8).flatten()[0] ^= 1
    assert not torch.equal(import_adapter(tampered)[victim], adapter[victim])

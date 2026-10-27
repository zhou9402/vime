# LoRA (work in progress)

vime is porting LoRA support from miles (`miles@ba28991834`, port F04). This page is built up per PR.

Current status:

- **F04-0**: equivalence verifier `tests/test_lora_equivalence.py` — merged-LoRA logits must equal adapter-applied logits on a 2-layer tiny model; adapter export/import and trainer→engine sync must be bitwise identical (tamper detection included). Run: `python -m pytest tests/test_lora_equivalence.py -x -q`.

Training-side LoRA (Megatron, native path) and vLLM adapter loading/sync land in subsequent PRs.

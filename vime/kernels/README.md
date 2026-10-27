# vime.kernels

```
activation/        swiglu_fp32
hyper_connection/  hc: hc_mix_inject · hc_combine    mhc: mhc_mix · mhc_aggregate
norm/              grouped_rmsnorm (device helpers shared by hc and ple)
position/          rope: apply_rotary_emb
quant/             fp8_blockwise_cast · act_quant · fake_quant_{fp8,fp4,compressed_kv}
softmax/           token_log_softmax
```

- Only kernels we write (Triton, TileLang, CuTe, CUDA). A third-party kernel we merely call stays with its caller.
- One directory per op. Names say what is computed, never the backend; a backend gets a subdirectory only when two coexist.
- Callers import the entry from the op's module or package, never an implementation file.
- No Megatron, no process groups, no env policy. Parallelism lives in `vime_plugins/models`.
- Variants are parameters, not copies.
- Every op has a torch-reference test in `tests/fast-gpu/kernels/<area>/`.

# Llama RMSNorm HSA asset RED contract

## Selector

- `tests/native_r9700/test_llama_rmsnorm_asset.py`

## Contract

A fresh, checked-in, non-symlink HIP source at
`native_r9700/kernels/llama_rmsnorm_f16.cpp` must expose the C-linkage GPU
kernel `llama_rmsnorm_f16`. It receives `(N, 2048)` fp16 hidden input,
2048-element fp16 scale, and fp16 hidden output through three pointers, with
per-dispatch fp32 `epsilon` as its fourth, scalar kernarg. Its computation is
GPU-indexed by workgroup and workitem IDs, uses fp32 accumulation, and contains
no host, fixture, archive, C0, CPU-model, or LDS machinery.

The HSA code-image generator must accept that reviewed source for `gfx1201` and
publish exactly one loadable image plus one JSON manifest. The manifest must
bind the exact `llama-rmsnorm-f16-v1` 32-byte ABI:

```text
hidden_input  uint64  offset 0
scale         uint64  offset 8
hidden_output uint64  offset 16
epsilon       float32 offset 24
```

It must also bind the canonical source path and SHA-256, image filename,
SHA-256 and size, descriptor and entry offsets, resources, and one admitted
kernel symbol. The HSA descriptor must declare zero group-segment bytes (no
LDS), zero private-segment bytes, and 32 kernarg bytes.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_llama_rmsnorm_asset.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. The focused
contract currently fails at the missing fresh RMSNorm HIP source, rather than
at runner, image-loader, hardware, fixture, archive, C0, or CPU-model paths.
No source implementation or generated asset is introduced by this contract.

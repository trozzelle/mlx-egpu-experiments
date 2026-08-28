# LQ-W1 Llama K projection source

## Source ABI

`native_r9700/kernels/llama_k_projection_f16.cpp` exports the C-linkage
`llama_k_projection_f16` GPU kernel for `gfx1201`. Its frozen 32-byte kernarg
layout is:

| Offset | Field | Source parameter |
| ---: | --- | --- |
| 0 | `normalized` `uint64` | `const unsigned short* normalized` |
| 8 | `k_projection_weight` `uint64` | `const unsigned short* k_projection_weight` |
| 16 | `fresh_k` `uint64` | `unsigned short* fresh_k` |
| 24 | `sequence_length` `uint32` | `unsigned int sequence_length` |
| 28 | ABI tail padding | — |

The kernel uses one 64-lane workgroup for each `(token, kv_head)` pair. Lane
`head_dimension` maps the 512 projection rows as `kv_head * 64 +
head_dimension` and writes the head-major fresh-K span at
`((kv_head * sequence_length) + token) * 64 + head_dimension`, yielding the
frozen fp16 `(1,8,N,64)` geometry. It bounds the token against the live
sequence length and accumulates all 2048 fp16 normalized/weight products in
fp32 before converting only the final result to fp16. It has no host,
fixture, cache, RoPE, query, or attention path.

## RED contract

`tests/native_r9700/test_llama_k_projection_source.py` contains the focused
source/ABI contract. It rejects a missing source, changed C-linkage argument
order or scalar width, missing bounded `(token, kv_head, head_dimension)`
mapping, non-fp32 accumulation, non-fp16 buffer storage, host logic, and
fixture/CPU/cache/RoPE/query/attention dependencies. The existing
`test_llama_kv_projection_asset.py` remains the later HSA-asset integration
contract and intentionally is not part of this source-only gate.

## Supervisor validation command

```sh
${PY} -m pytest tests/native_r9700/test_llama_k_projection_source.py -q
```

This command was intentionally not run in this source-only lane.

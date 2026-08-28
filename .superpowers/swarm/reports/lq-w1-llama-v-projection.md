# LQ-W1 Llama V projection source

## Changed files

- `native_r9700/kernels/llama_v_projection_f16.cpp`
- `tests/native_r9700/test_llama_v_projection_asset.py`

No HSA code image, catalog entry, loader, runtime dispatch, K source, stage descriptor, cache materialization, or RoPE implementation is included.

## Frozen source interface

```cpp
extern "C" __attribute__((global)) void llama_v_projection_f16(
    const unsigned short* normalized,
    const unsigned short* v_projection_weight,
    unsigned short* fresh_v,
    unsigned int sequence_length)
```

This is the frozen 32-byte V stage kernarg ABI: `normalized` at byte 0,
`v_projection_weight` at byte 8, `fresh_v` at byte 16, and the 32-bit
`sequence_length` scalar at byte 24 (with the descriptor's trailing padding).

The `gfx1201` source uses one 64-lane workgroup per `(token, KV head)` tile.
For each bounded token index and head-dimension lane, it reads live fp16
`normalized[token, 2048]` and one fp16 row from the live `(512,2048)` V-weight
window, accumulates their dot product in fp32, and stores one fp16 result at
`fresh_v[0, kv_head, token, head_dim]`. The 512 output channels are exactly
`8 * 64`; output is direct and unrotated. The source contains no host fallback,
fixture path, cache write, or RoPE operation.

## RED contract

The focused static source contract asserts the exact ABI parameter order, fixed
geometry, bounded token guard, fp32 accumulation loop, direct head-major
`(1,8,N,64)` indexing, and absence of cache/rotation/host-fallback terms. It
was written before the source and is not run by this worker.

## Supervisor validation

```sh
${PY} -m pytest tests/native_r9700/test_llama_v_projection_asset.py -q
```

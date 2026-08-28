# Llama V projection HSA-asset RED contract

## Selector

- `tests/native_r9700/test_llama_v_projection_asset.py`

## Contract

The future isolated boundary is `native_r9700/llama_v_projection_asset.{h,cpp}`.
It must reuse the checked-in HSA image from
`native_r9700/kernels/llama-kv-hsa-assets`, rather than introduce a V-specific
CPU, archive, C0, or fixture path. The loaded asset is exactly the
`llama_kv_projection_f16` kernel sourced from
`native_r9700/kernels/llama_kv_projection_f16.cpp`, with the existing 32-byte
four-`uint64` kernarg schema, fp32 accumulation, fp16 output, and 256-thread
workgroup.

Each V launch streams exactly one binder-validated fp16
`v_projection_weight` window of shape `(512,2048)` and byte span
`512 * 2048 * 2 = 2,097,152`, together with live fp16 hidden input `(N,2048)`.
The third existing HSA kernarg is the live fp16 `v_cache` output, materialized
*directly* in shape `(1,8,N,64)`. For the fixed projection width, `8 * 64 =
512`, so the cache byte span is exactly `1 * 8 * N * 64 * 2`; its GPU VA is
serialized at offset 16. The launch grid remains one thread per output element
`(N * 512, 1, 1)`.

The V boundary is exact no-RoPE: both configuration and asset must declare
`rope_mode == "none"`; a split-half Llama-3 rotation request or a rope-capable
asset is rejected. A successful launch reports
`project_v_and_materialize_direct`, `rope_mode == "none"`, and exactly one
resident weight window. It rejects a malformed or non-fp16 V cache, a malformed
or fixture-backed V weight, non-fp32 accumulation, a second resident weight
window, and a V-cache span crossing the half-open lower-BAR payload window
`[0x0000200000011000, 0x000020000A001000)`. Every rejection must preserve the
output launch unchanged.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_llama_v_projection_asset.py -q
```

## Intended current RED

The supervisor command is deliberately recorded but not run in this task. The
future V-projection header and implementation do not exist, so this focused
no-hardware contract fails first with the missing Llama V projection asset
capability. It cannot compile the probe, open a device, contact a driver, or
exercise hardware until that boundary is implemented.

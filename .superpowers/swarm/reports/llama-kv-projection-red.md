# Llama K/V projection HSA-asset RED contract

## Selector

- `tests/native_r9700/test_llama_kv_projection_asset.py`

## Contract

The future isolated projection-asset boundary is
`native_r9700/llama_kv_projection_asset.{h,cpp}`. It must load the checked-in
fresh native HSA code image from `native_r9700/kernels/llama-kv-hsa-assets` and
build a dispatch for exactly one projection kind at a time: `k` or `v`. A K/V
pair is two sequential dispatches, not a combined two-weight resident binding.

Each dispatch consumes one live, binder-validated fp16
`projection_weight` window with shape `(512,2048)` and exact byte span
`512 * 2048 * 2 = 2,097,152`. It consumes fp16 `hidden_input` with shape
`(N,2048)`, accumulates in fp32, and writes fp16 `projection_output` with shape
`(N,512)`. The HSA image schema is exactly:

```json
{"name":"llama-kv-projection-f16-v1","bytes":32,"fields":[{"name":"hidden_input","offset":0,"type":"uint64"},{"name":"projection_weight","offset":8,"type":"uint64"},{"name":"projection_output","offset":16,"type":"uint64"},{"name":"token_count","offset":24,"type":"uint64"}]}
```

The resulting kernargs serialize those three GPU virtual pointers and the
`uint64` token count at the stated offsets. The dispatch grid is one thread per
output element (`N * 512`, `1`, `1`) with a 256-thread workgroup. The projection
asset must declare fp32 accumulation and fp16 output before a dispatch may be
built.

All direct operand spans must be within the half-open small-BAR payload GPU-VA
window `[0x0000200000011000, 0x000020000A001000)`, derived from the observed
256 MiB BAR0 layout's `0x09FF0000` allocatable payload. The contract rejects
hidden, weight, and output spans that cross the upper bound; a wrong weight
span; wrong output shape; non-fp32 accumulation; fixture weight provenance; and
a combined `kv` projection kind. Rejection preserves the output launch
unchanged.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_llama_kv_projection_asset.py -q
```

## Intended current RED

The supervisor command is deliberately recorded but not run in this task. The
new projection-asset header and implementation do not exist, so the focused
no-hardware contract fails specifically with the missing Llama K/V projection
asset capability before it can compile a probe, contact a driver, open a device,
or depend on runner/image-loader work.

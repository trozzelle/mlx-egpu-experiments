# Qwen hybrid-state spill RED contract

## Selector

- `tests/native_r9700/test_qwen_hybrid_state_spill.py`

## Contract

The future `native_r9700.qwen_spill` boundary owns a **host-authoritative,
serialized** text-only Qwen3.5 hybrid-cache record. It must expose
`QwenStateSpillError`, `capture_qwen_hybrid_state`,
`serialize_qwen_hybrid_state`, `deserialize_qwen_hybrid_state`, and
`upload_qwen_hybrid_state`.

Capture accepts the selected MLX-VLM runtime's 64 text-cache layers plus a
model/config ABI identity and request-level `committed_position`. It records
all 64 layers in their original order, never reorders or filters the cache:

- layers `0, 1, 2, 4, 5, 6, …, 60, 61, 62` retain class `ArraysCache`, two
  leaves in order, shapes `(1,3,10240)` / `(1,48,128,128)`, and dtypes
  `bfloat16` / `float32`;
- layers `3, 7, …, 63` retain class `KVCache`, K then V in order, each
  `bfloat16` `(1,4,P,256)`, and the live per-cache offset `P`;
- `ArraysCache` has no per-layer serialized offset; the serialized
  request-level `committed_position` is the resume authority for that state;
- every leaf retains byte payload, byte count, shape, dtype, and SHA-256
  digest. Deserialization verifies integrity rather than recomputing a cache.

The state exists as host bytes for explicit spill/reload. It must not invoke
MLX/NumPy conversion or evaluation, numeric tensor methods, CPU tensor/model
math, fixtures, archived/C0 input, fallback cache reconstruction, or hardware
dispatch. The RED test supplies opaque byte-bearing leaves that fail on CPU
array/numeric conversion.

`upload_qwen_hybrid_state` may upload only explicitly selected complete layer
groups into the passed lower-BAR resident window. It preserves leaf byte order,
starts the selected group at window offset zero, reports the exact byte count,
and fails before any write when that bounded window cannot contain the group.
It cannot allocate or assume a full-cache VRAM allocation.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_qwen_hybrid_state_spill.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. Collection must
currently fail at the absent `native_r9700.qwen_spill` module. No source,
hardware path, model/fixture payload, compiler invocation, or acceptance claim
is introduced by this contract.

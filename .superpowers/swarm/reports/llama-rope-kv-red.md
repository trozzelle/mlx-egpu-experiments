# Llama-3 RoPE/KV materialization RED contract

## Selector

- `tests/native_r9700/test_llama_rope_kv_asset.py`

## Contract

The future `llama_rope_kv_asset` module provides a no-hardware planner for
fresh GPU Llama-3 KV materialization. Its public
`build_llama_rope_kv_materialization_launches` boundary accepts only live GPU
K/V projection and cache bindings with the config-selected `(1,8,N,64)` fp16
layout, plus exactly `head_dim / 2` fp32 RoPE divisors from a
`model_config_derived_device_buffer` binding.

A valid plan emits exactly two dispatches, in order: `rotate_k_split_half` for
K and `materialize_v_direct` for V. The contract rejects an invalid KV shape
or dtype, a divisor span not matching the split-half pair count,
fixture-sourced K bytes, host-computed divisor provenance, and an asset that
attempts to rotate V. The
probe contains descriptors only: it supplies no fixture tables, archive or C0
asset operands, CPU model math, model values, driver access, or hardware work.

## Supervisor RED command (do not run in this task)

```sh
${PY} -m pytest tests/native_r9700/test_llama_rope_kv_asset.py -q
```

## Intended current RED

The supervisor command is deliberately not run in this task. The new test
fails first and specifically with `Llama RoPE/KV materialization module is
missing` because neither `native_r9700/llama_rope_kv_asset.h` nor
`native_r9700/llama_rope_kv_asset.cpp` exists. Once the future module exists,
the same no-hardware probe will enforce the materialization contract.

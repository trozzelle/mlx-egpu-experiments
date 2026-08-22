# Qwen DeltaNet ArraysCache source

`qwen_deltanet_state` now owns one direct, bounded device update for an `ArraysCache` entry.

## Kernel boundary

The kernel accepts packed bfloat16 text-side hidden input and bfloat16 output, the two raw cache leaves (`convolution_state` and `recurrent_state`), their independent element capacities, and an absolute `position`. Its dimensions make the cache layout explicit at dispatch: convolution width/channels and the recurrent value-head, key-head, key-dimension, and value-dimension geometry.

The input layout is contiguous Q, K, V, decay, and beta slices. The kernel maps grouped-query heads, updates the position-selected convolution ring slot, applies the DeltaNet recurrence in the float32 recurrent cache, and writes bfloat16 output. It rejects zero, incompatible, overflowing, or undersized shapes before accessing device memory.

No Python metadata or cache-bridge code changed: cache shape and dtype authority remains at the existing boundary. The source has no runtime/state abstraction or host-side numerical path.

## Contract

`tests/native_r9700/test_qwen_deltanet_state_source.py` is a focused source/ABI contract for the raw pointers, capacities, absolute position, and device-only execution markers. Per the assignment, no validation command was run.

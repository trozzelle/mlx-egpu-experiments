# LQ-W0 Llama cache extent fix

## Finding

Resident span validation previously sized every non-scalar span from the fresh `sequence_length`. That accepted K/V cache storage sized for only the current tokens even when RoPE writes, attention reads, and score/probability key dimensions address the absolute range `[position, position + sequence_length)`.

## Fix

`LlamaStageSpanExtent` now makes the distinct address domains explicit:

- `kPerSequenceToken` remains for fresh activations.
- `kPerCacheToken` sizes K/V cache spans by checked `position + sequence_length`.
- `kPerFreshQueryByCacheToken` sizes score/probability storage by fresh queries times checked absolute cache keys.

The rule is assigned statically to RoPE K/V outputs, attention K/V cache inputs, attention-score output, softmax input/output, and attention-context probability input. `required_span_bytes` receives `position`, performs the checked addition before any relevant multiplication, retains checked element/dtype multiplication, and preserves the established overflow and undersized-span error paths.

## Regression probe

The no-hardware C++ probe uses `sequence_length = 2`, `position = 1` and verifies rejection of:

- K cache and V cache allocations of 2 fresh-token slots instead of 3 absolute slots;
- attention-score output storage sized as `32 * 2 * 2` fp32 elements instead of `32 * 2 * 3`;
- attention-probability output and context input storage with the same undersizing.

The probe also asserts the static extent assignments so later descriptor changes cannot silently restore fresh-token sizing.

## Changed files

- `native_r9700/llama_stage_layout.h`
- `native_r9700/llama_stage_layout.cpp`
- `tests/native_r9700/test_llama_stage_layout.py`

No validation commands were run, per assignment.

# LN-2 RMSNorm Epsilon Arithmetic Probe

## Diagnostic boundary

`llama_rmsnorm_epsilon_arithmetic_f16` is a trace-only, ABI-compatible gfx1201 HSA image generated from `native_r9700/kernels/llama_rmsnorm_epsilon_arithmetic_f16.cpp`. It preserves the RMSNorm stage-0 32-byte kernarg ABI:

- `hidden_input` at byte 0 (`uint64`)
- `scale` at byte 8 (`uint64`)
- `hidden_output` at byte 16 (`uint64`)
- `epsilon` at byte 24 (`float32`)

With the constrained trace's all-zero input, the probe evaluates the RMSNorm scalar factor only:

`inverse_rms = 1.0f / sqrt(0.0f + epsilon)`.

It converts that scalar to fp16 and stores the same value through all 2048 output elements. For the trace epsilon `1e-5`, the required output is repeated fp16 `0x5cf1` (`316.25`), recorded in trace metadata as `rmsnorm_expected_output:"f16_0x5cf1_316.25"`. The trace rejects a finite but nonmatching payload as `trace_expected_output`; consequently it distinguishes epsilon/sqrt/reciprocal execution from the original kernel's sum-of-squares reduction and input/scale multiply/store path.

## Isolation

`--rmsnorm-epsilon-arithmetic` is admitted only for the normalized layer-0, position-0 trace when all diagnostic prerequisites are present:

- `--rmsnorm-unit-scale`
- `--rmsnorm-zero-input`
- `--rmsnorm-output-sentinel`

It is mutually exclusive with `--rmsnorm-zero-store`. The override replaces only stage 0 in `build_llama_layer0_stage_trace_dispatch`; the persistent builder and production `llama_rmsnorm_f16` stage asset remain unchanged. It uses the existing stage-0 kernarg layout and PM4 path, including the real epsilon scalar at byte 24.

## Original-asset correlation

The original asset's source performs three separable groups of work: fp16-to-fp32 square accumulation, `1.0f / sqrt(mean + epsilon)`, then fp16 input/scale/output multiplication. The accompanying ISA decode locates mixed fp16/fp32 reduction FMAs at `.text+0x74..0x110`, the epsilon load from kernarg byte 24 at `.text+0x120`, and the unique scalar-root/division lowering from `.text+0x150`: `V_S_SQRT_F32`, `V_DIV_SCALE_F32`, `V_RCP_F32`, refinement FMAs, and `V_DIV_FIXUP_F32`. The final input/scale output path starts at `.text+0x290` with fp16 conversion/multiply and B16 stores.

Zero input makes the reduction and final products algebraically zero, so the probe intentionally executes only the scalar epsilon/root/reciprocal operation. A nonfinite or non-`0x5cf1` probe output implicates that operation; a passing probe directs repair to the reduction or later output multiply path.


## Asset and contracts

The generated image and manifest are in `native_r9700/kernels/llama-rmsnorm-epsilon-arithmetic-hsa-assets/`. The image SHA-256 is `e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037`.

Contracts pin the reviewed source ABI, source arithmetic, manifest/image digests, generator admission, trace-only substitution, strict option guards, exact expected fp16 metadata, and exclusion from prefill.

No validation was run, per assignment.

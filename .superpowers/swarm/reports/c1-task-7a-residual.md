# C1R-7a post-O residual cols0:64 hardware primitive chain

## Scope

Implemented the next layer-forward proof after `layer0_o_proj_full_inner_cols0_64_tiled_accum_chain`:
`layer0_attention_residual_cols0_64_after_o_proj_chain`.

This is a hardware primitive-chain proof only. `native_prefill_acceptance` remains `open`.

## Decisions

- The proof covers rows0:8/cols0:64, with rows5:8 padded to zero for prompt-0's five valid tokens.
- The source fixture is `tests/native_r9700/fixtures/layer_trace_attention_residual_cols0_64_fixtures.npz`.
- Inputs are packed as 64 contiguous 32-byte `{hidden_in[8 fp16], o_proj_output[8 fp16]}` chunks, one chunk per `(8-column tile, row)` stage.
- The chain uses resident VRAM input/output pages: input PTB17 at `0x0000200000011000`, output PTB33 at `0x0000200000021000`.
- The residual add kernel is the proven `fp16_residual_add_layer0_attention_slice8` primitive; 64 dispatches cover 512 fp16 elements.
- Hardware debugging found two real hazards:
  - Appending multiple dispatches into one compute ring stalled after stage0 (`rptr=59,wptr=118`). The chain now uses the proven single-dispatch path per stage: setup compute ring, load kernel, write kernargs, submit offset 0.
  - Polling the fixed timeline value without a settle delay could let the next stage reset/reload while the previous dispatch was still in flight. A 1 ms post-submit settle keeps the current fixed-timeline contract stable until a monotonic timeline value is introduced.
- `native_prefill_acceptance` stays `open`; this proves only post-O residual cols0:64, not RMSNorm/MLP/layer assembly.

## Verification

Focused host tests:

```sh
${PY} -m pytest \
  tests/native_r9700/test_ref_fixtures.py::test_layer_trace_attention_residual_cols0_64_fixtures_schema_shape_dtype \
  tests/native_r9700/test_ref_fixtures.py::test_layer0_attention_residual_cols0_64_fixture_matches_fp16_add_oracle \
  tests/native_r9700/test_runtime_contract.py::test_layer0_attention_residual_cols0_64_operands_are_sliced_rows8x64 \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_attention_residual_cols0_64_after_o_proj_chain \
  tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes \
  -q
```

Result: exit `0`; `5 passed in 4.18s`.

Compile:

```sh
mkdir -p build/native-r9700-runtime && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/c1_primitive_bridge.cpp \
    -o build/native-r9700-runtime/c1_primitive_bridge && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
    native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
    -o build/native-r9700-runtime/native_r9700_runner
```

Result: exit `0`, no output.

Hardware proof:

```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge \
  build/native-r9700-runtime/native_r9700_runner \
  --primitive-chain-proof layer0_attention_residual_cols0_64_after_o_proj_chain
```

Result: exit `0`; log path `logs/c1-runner-primitive-chain-proof-layer0_attention_residual_cols0_64_after_o_proj_chain-2026-08-20T01:35:59Z.log`.

Key pass markers:

- `primitive_chain_proof_wrapper_status: pass`
- `chain_stage_count: 64`
- `data_region_residency: two_distinct_vram_pages`
- `input_region_pte_status: pass`
- `output_region_pte_status: pass`
- `kernarg_rewrite_count: 64`
- `compute_dispatch_count: 64`
- `mismatch_count: 0`
- `byte_mismatch_count: 0`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `wrapper_exit_status: 0`
- `exit_status: 0`

## Blockers

None for this slice. Full layer-0/native prefill remains open until RMSNorm/MLP/layer assembly and the cache route are proven.

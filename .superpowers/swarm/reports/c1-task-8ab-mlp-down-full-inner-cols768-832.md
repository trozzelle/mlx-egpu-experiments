# C1 task 8ab — MLP down full-inner cols768:832

## Scope
Implemented bounded Llama layer0 MLP down projection full-inner primitive-chain block for output cols768:832 only.

Acceptance remains `hardware_primitive_chain_only_partial` with `native_prefill_acceptance: open`. This is not native prefill acceptance, not full hidden-width/full layer0 acceptance, and makes no Qwen claim. Qwen3.8-27B remains deferred for C1.

Runtime/fake-bridge drift markers for cols768:832 were initially seeded from the nearest prior observed chain, cols704:768, and repaired after supervisor hardware captured observed cols768:832 values.

## RED evidence
- `mkdir -p /tmp/c1down768-red && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/c1down768-red/native_r9700_runner && /tmp/c1down768-red/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain`
  - Result: exit 2; runner rejected `layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain` as unsupported with `failure_stage: primitive_chain_request`, `wrapper_exit_status: 2`.

## Changed files
- `native_r9700/ref_fixtures.py`
  - Added cols768:832 to layer0 MLP down full-inner fixture generation and split final/chunk fixture emission.
- `tests/native_r9700/fixtures/fixtures_schema.json`
  - Added deterministic schema/digest entries for cols768:832 final and chunk NPZs.
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols768_832_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols768_832_chunk0_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols768_832_chunk1_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols768_832_chunk2_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols768_832_chunk3_fixtures.npz`
- `native_r9700/c1_primitive_bridge.cpp`
  - Added cols768:832 metadata, dot2 row-pair/column-packed model weight bytes, expected fp32 bytes, chain spec, run function, and bridge dispatch.
- `native_r9700/runtime.h`
  - Added cols768:832 runtime constants/digests.
- `native_r9700/runtime.cpp`
  - Added wrapper support/marker validation for cols768:832; supervisor hardware later repaired cols768:832 drift markers to observed chain-specific values.
- `native_r9700/runner.cpp`
  - Added help exposure for `layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain`.
- `tests/native_r9700/test_ref_fixtures.py`
  - Added cols768:832 fixture oracle case and schema file expectations.
- `tests/native_r9700/test_runtime_contract.py`
  - Added cols768:832 constants, marker derivation, runner help coverage, embedded operand layout coverage, and primitive-chain wrapper test.
- `.superpowers/swarm/reports/c1-task-8ab-mlp-down-full-inner-cols768-832.md`

## Deterministic digests
- Final fixture: `9492bbaac58440e9495cf9d452c51f7889bc2035fca18dea8c4644001fe4178d`.
- Expected fp32 array: `efec5949b3122b9bb0e50ffd16d70619060448ad613e8808b117d4420d03d0d7`.
- Chunk0 fixture: `fd2abbcba99d13f855a840ab1cd9ef03d93a1206606d12ec091d5124a68fcec2`.
- Chunk1 fixture: `aaef8c2df30c3360d6ed4c514c0948933cd051270e457bd0dd317dc35b2db18d`.
- Chunk2 fixture: `4764bc4804d27a674992c12dfce1bced7373182a22844b6367b864aae7458be5`.
- Chunk3 fixture: `d6c17651b1059142100ad9807b1f8bcb2681b51f74977c899e445f7b11ea8eda`.

## GREEN verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`
  - Result: exit 0; wrote 81 fixture files including cols768:832 final/chunk NPZs and `fixtures_schema.json`.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_fixtures_match_fp32_oracle tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_layer0_mlp_down_cols768_832_embedded_operands_use_kernel_layouts tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols768_832_chain -q`
  - Result: exit 0; `17 passed in 21.24s`.
- `mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exit 0; no compiler output.

## Suggested supervisor hardware command
```sh
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain
```

Or from a clean focused build:
```sh
mkdir -p build/native-r9700-runtime && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && \
  xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && \
  build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain
```

## Supervisor verification
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -k 'mlp_down_full_inner_to_cols768_832_chain or layer0_mlp_down_cols768_832_embedded_operands_use_kernel_layouts' -q`
  - Result: exit 0; `2 passed, 130 deselected in 15.17s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain`
  - Result: real hardware primitive-chain proof exited 0 with `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `tolerance: fp32_abs<=2e-4_or_ulp<=64`, `max_abs_diff=2.86102294921875e-06`, `max_ulp_diff=174080`, `mismatch_count=0`, `byte_mismatch_count=455`, and log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain-2026-08-20T21:13:34Z.log`.

## Remaining blockers
- Acceptance remains `hardware_primitive_chain_only_partial`; `native_prefill_acceptance` remains open.
- No full hidden-width layer0/native prefill/full attention width/Qwen acceptance is claimed.

# C1R-7d MLP activation cols0:64

## Status
Done. Native R9700 hardware primitive chain proves layer-0 MLP activation cols0:64, `SiLU(gate) * up`, for rows0:5 with padded rows5:8.

## Decisions
- Match `native_r9700/prefill.py` precision: compute SiLU in fp32, round SiLU to fp16, multiply by fp16 `up`, then output fp16.
- Store activation arrays in `tests/native_r9700/fixtures/layer_trace_mlp_activation_cols0_64_fixtures.npz` rather than the compact trace archive.
- Add a narrow fused RDNA4/gfx1201 row kernel because the repo had `fp16_silu_8x8` but no multiply/fused activation primitive.
- Accept hardware output under `fp16_ulp<=1`; exact bytes are not required for this fp16 transcendental path.

## Artifacts
- Fixture: `tests/native_r9700/fixtures/layer_trace_mlp_activation_cols0_64_fixtures.npz`
- Fixture SHA256: `cb193cabaf06912806641fb058fd29bf0e1689e9eb90f642854366c5e5e3fe65`
- Expected fp16 SHA256: `343350605cc2f3469145c18978fc2b0942f373547e47020c10a9c3806237430c`
- Kernel source id: `c1r7d-layer0-mlp-silu-mul-slice8-v1`
- Kernel SHA256: `7b1a31ea7c2150c813d09f85eab4a35db925dd1fc38d644dbfe2ff726722afc1`
- Hardware log: `logs/c1-runner-primitive-chain-proof-layer0_mlp_activation_cols0_64_silu_mul_chain-2026-08-20T13:06:55Z.log`

## Verification
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge` exited 0.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited 0.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_activation_cols0_64_fixture_matches_silu_multiply_oracle tests/native_r9700/test_runtime_contract.py::test_primitive_kernel_sha_matches_embedded_text tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_activation_cols0_64_silu_mul_chain -q` exited 0: `3 passed in 2.88s`.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_activation_cols0_64_silu_mul_chain` exited 0 with `primitive_chain_proof_wrapper_status: pass`, `chain_stage_count: 64`, `kernarg_rewrite_count: 64`, `compute_dispatch_count: 64`, `max_abs_diff: 3.0517578125e-05`, `max_ulp_diff: 1`, `mismatch_count: 0`, `byte_mismatch_count: 91`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `exit_status: 0`.

## Remaining C1 scope
Full layer-0/native prefill acceptance remains open until down projection, layer assembly, cache routing, and final parity are proven.

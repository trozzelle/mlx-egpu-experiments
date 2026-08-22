# C1R-6a fp16-to-fp32 hardware primitive report

Status: Done; focused re-review approved

Scope:
- Advance C1 hardware execution beyond the C1R-5 fp32 scalar primitive without claiming full layer native execution.
- Add the next safe primitive proof: `fp16_to_fp32_cast`.
- Keep the full `r9700_native` model-forward route fail-closed until layer-0 tensor ops, RoPE/K/V layout, and hidden-state comparison are hardware-backed.

Decision:
- Chose `fp16_to_fp32_cast` before residual add or matmul.
- Rationale: it uses the same one input page, one output page, 8-lane dispatch, SDMA upload/download, compute-ring, and single-kernel proof topology as C1R-5, while proving a second tensor datatype path needed by fp16 layer work.
- Rejected a full layer wrapper here because current evidence still lacks hardware matmul/GEMM, RoPE, attention, MLP, RMSNorm, residual, and multi-page/tiled tensors. Claiming native model-forward would be false.

Implementation:
- Extended `native_r9700/c1_primitive_bridge.cpp` from a single hard-coded primitive into a small exact `PrimitiveSpec` table.
- Added RDNA4/gfx1201 kernel bytes for `fp16_to_fp32_cast`:
  - `kernel_source_id: c1r6-fp16-to-fp32-cast-v1`
  - `kernel_blob_sha256: d18f462a15fc21d48eedf32bcbcf24a0b6eb270a41707f8cdf95c1b27653ead0`
  - `kernel_text_byte_count: 64`
  - operation sequence: `s_load_b128`, per-lane `global_load_u16`, `v_cvt_f32_f16_e32`, per-lane `global_store_b32`, `s_endpgm`, ISA-alignment `s_code_end` padding.
- Kept the frozen 24-byte kernarg ABI `{output_va@0,input_va@8,scalar_va@16,scalar:u32@24}`; cast marks `scalar_bits: unused` and stores zero in the scalar slot.
- Generalized `RuntimeSession::primitive_proof` marker validation with exact expected values per primitive.
- Updated runner help to advertise both `fp32_add_scalar` and `fp16_to_fp32_cast`.
- Added no-hardware wrapper tests for the cast primitive and exact marker validation.
- Review fix: `test_primitive_kernel_sha_matches_embedded_text` now hashes both
  embedded kernel arrays, so `kC1Fp16ToFp32KernelSha256` cannot drift from
  `kC1Fp16ToFp32KernelText` unnoticed.

RED evidence:
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_proof_wraps_supplied_bridge_and_logs_fp16_to_fp32_cast -q` initially failed because `RuntimeSession::primitive_proof` rejected `fp16_to_fp32_cast` as unsupported.

Verification:
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge` -> exit 0.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` -> exit 0.
- Focused primitive wrapper tests -> 6 passed in 5.79s:
  - `test_primitive_proof_wraps_supplied_bridge_and_logs_fp32_add`
  - `test_primitive_proof_wraps_supplied_bridge_and_logs_fp16_to_fp32_cast`
  - `test_primitive_proof_rejects_missing_primitive_marker`
  - `test_primitive_proof_rejects_inexact_primitive_marker_value`
  - `test_primitive_proof_rejects_bad_evidence_marker_values`
  - `test_primitive_kernel_sha_matches_embedded_text`
- Real hardware cast proof:
  - command: `build/native-r9700-runtime/native_r9700_runner --primitive-proof fp16_to_fp32_cast`
  - exit 0
  - log: `logs/c1-runner-primitive-proof-fp16_to_fp32_cast-2026-08-19T12:24:45Z.log`
  - key markers: `primitive_backend: hardware`, `arch: gfx1201`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `kernel_launch_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `primitive_proof_wrapper_status: pass`, `wrapper_exit_status: 0`.
- Real hardware fp32 regression proof through the generalized bridge:
  - command: `build/native-r9700-runtime/native_r9700_runner --primitive-proof fp32_add_scalar`
  - exit 0
  - log: `logs/c1-runner-primitive-proof-fp32_add_scalar-2026-08-19T12:27:20Z.log`
  - key markers: `primitive_backend: hardware`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `kernel_launch_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `mismatch_count: 0`, `primitive_proof_wrapper_status: pass`, `wrapper_exit_status: 0`.
- Review-fix digest regression: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_kernel_sha_matches_embedded_text -q` -> 1 passed in 0.03s.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q` -> 19 passed in 18.90s.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> 138 passed, 2 warnings in 23.35s.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -q` -> 178 passed, 2 warnings in 57.04s.
- `git diff --check native_r9700/runtime.h native_r9700/runtime.cpp native_r9700/runner.cpp native_r9700/c1_primitive_bridge.cpp tests/native_r9700/test_runtime_contract.py docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/reports/c1r-task-6-fp16-cast.md .superpowers/swarm/reports/c1r-task-6-review.md` -> exit 0.
- Review gates: `agent://C1R6Review` returned one P2 finding; fix landed. `agent://C1R6ReReview` approved with no findings.

Known limits:
- This is still a proof-only primitive runner, not a native C1 layer forward pass.
- Full model-forward native producer remains intentionally disabled/fail-closed.
- Next C1R work must prove at least hardware matmul/tiled GEMM and K/V/hidden-state comparison before `r9700_native` can become a selectable producer.

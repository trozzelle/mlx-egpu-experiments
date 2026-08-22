# C1R-5 primitive fp32 add-scalar report

Status: Done; focused re-review approved

Scope:
- Added `native_r9700/c1_primitive_bridge.cpp`, a standalone C1R-5 hardware primitive bridge that includes the frozen C0 probe with `main` renamed and reuses its TinyGPU socket, BAR discovery, fixed VM mapping, SDMA queue, and compute queue helpers without mutating the C0 source.
- Implemented `fp32_add_scalar` as the first one-page, one-input/one-output primitive. It stays inside the current proven C0 compute contract: 8 lanes, 32 input bytes, 32 output bytes, scalar at `kernargs+24`, one input VRAM page, one output VRAM page, and existing PM4 dispatch shape.
- Added exact key/value wrapper validation for primitive proof logs and RED/GREEN tests that reject `element_count: 80`, missing `kernel_blob_sha256`, wrong `kernel_blob_sha256`, wrong `scalar_bits`, and approximate `tolerance`.

Decisions:
- Choose `fp32_add_scalar` before residual add or matmul. Reason: current C0 kernargs/page-table layout supports one input pointer plus scalar; two-input or tiled model primitives need a C1-owned expanded kernarg/PTE contract.
- Use a source-grounded 64-byte gfx1201 kernel that changes only the C0 arithmetic instruction from `v_add_nc_u32_e32` to `v_add_f32_e32`; keep the C0 load/store sequence, descriptor fields, and dispatch geometry.
- Compare exact output bytes for this primitive. The input values are exactly representable fp32 values and `+1.0f` results are exact for the chosen range.
- Keep the full `r9700_native` prefill/parity guard closed. This primitive is hardware evidence for C1R-5 only; it is not full model-forward acceptance.

Kernel identity:
- `primitive_name: fp32_add_scalar`
- `kernel_source_id: c1r5-fp32-add-scalar-v1`
- `kernel_blob_sha256: 697ba0c938e34d6f8db6498a803fb1d82181b111b28fe8c60acaac6a8d6011fd`
- `element_type: fp32`
- `element_count: 8`
- `scalar_bits: 0x3f800000`
- `input_byte_count: 32`
- `output_byte_count: 32`
- `tolerance: exact_bytes`


Review fix:
- C1R-5 review found that `RuntimeSession::primitive_proof` accepted the primitive bridge with only coarse identity/status markers, so a fake log could claim pass with wrong scalar bits, kernel hash, tolerance, or byte totals.
- Fixed by adding wrapper-owned exact constants in `native_r9700/runtime.h` and requiring `primitive_backend`, `kernel_blob_sha256`, `kernel_text_byte_count`, `scalar_bits`, `tolerance`, `max_abs_diff`, `max_ulp_diff`, `mismatch_count`, `upload_total_bytes`, and `download_total_bytes` before reporting `primitive_proof_wrapper_status: pass`. Re-review then found the logged hash was stale; fixed the bridge/runtime/tests/docs to the digest derived from the embedded 64-byte `kC1PrimitiveKernelText` and added a regression test for that derivation.

Files changed:
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.cpp`
- `native_r9700/runtime.h`
- `tests/native_r9700/test_runtime_contract.py`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `.superpowers/swarm/progress.md`

Verification:
- RED: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_proof_rejects_inexact_primitive_marker_value -q` failed because substring validation accepted `element_count: 80`.
- RED review fix: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_proof_rejects_bad_evidence_marker_values -q` failed because the wrapper accepted bad `kernel_blob_sha256`.
- RED hash fix: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_kernel_sha_matches_embedded_text -q` failed with observed digest `697ba0c938e34d6f8db6498a803fb1d82181b111b28fe8c60acaac6a8d6011fd` versus stale expected `bf97ba0c9308e34d6f8db6498a803fb1d8218b111b28fe8c60aca6a8d611fd`.
- GREEN hash fix: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_kernel_sha_matches_embedded_text -q` -> 1 passed in 0.03s.
- GREEN review fix: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_proof_wraps_supplied_bridge_and_logs_fp32_add tests/native_r9700/test_runtime_contract.py::test_primitive_proof_rejects_missing_primitive_marker tests/native_r9700/test_runtime_contract.py::test_primitive_proof_rejects_inexact_primitive_marker_value tests/native_r9700/test_runtime_contract.py::test_primitive_proof_rejects_bad_evidence_marker_values -q` -> 4 passed in 4.58s.
- Hardware proof after hash fix: `mkdir -p build/native-r9700-runtime logs && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner && build/native-r9700-runtime/native_r9700_runner --primitive-proof fp32_add_scalar` -> exit 0.
- Hardware log: `logs/c1-runner-primitive-proof-fp32_add_scalar-2026-08-19T12:14:49Z.log` contains `producer_kind: hardware_primitive`, `primitive_backend: hardware`, `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `kernel_blob_sha256: 697ba0c938e34d6f8db6498a803fb1d82181b111b28fe8c60acaac6a8d6011fd`, `kernel_text_byte_count: 64`, `scalar_bits: 0x3f800000`, `tolerance: exact_bytes`, `max_abs_diff: 0`, `max_ulp_diff: 0`, `upload_total_bytes: 32`, `download_total_bytes: 32`, `kernel_blob_load_status: pass`, `kernarg_write_status: pass`, `kernel_launch_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `mismatch_count: 0`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, and `primitive_proof_wrapper_status: pass`.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q` -> 18 passed in 17.40s.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` -> 137 passed, 2 warnings in 22.17s.
- `git diff --check native_r9700/runtime.h native_r9700/runtime.cpp native_r9700/runner.cpp native_r9700/c1_transfer_bridge.cpp native_r9700/c1_primitive_bridge.cpp tests/native_r9700/test_runtime_contract.py docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/progress.md .superpowers/swarm/reports/c1r-task-4-transfer-manager.md .superpowers/swarm/reports/c1r-task-5-primitive-fp32-add.md` -> exit 0.
- Focused final re-review: `agent://C1R5FinalReview` approved with no findings.

Known limits:
- This bridge is intentionally narrow. It proves hardware compute for a scalar unary fp32 primitive. C1R-6/C1R-7 still need expanded C1-owned primitive/kernarg/page-table support before a real Llama layer or full prefill can be marked native.

# C1R-6o Q RoPE split-half pair primitive

Status: proof-only hardware primitive complete; full C1R-6 layer-0 acceptance remains open.

Decision
- Add `fp16_rope_split_half_layer0_q_pairs8` as the Q-projection analogue of the proven K RoPE pair primitive.
- Reuse the proven K RoPE kernel text and dispatch packet; specialize only primitive name, source id, fixture arrays, model-forward scope, and oracle bytes.
- Keep `native_prefill_acceptance: open`; this is one real layer-0 Q RoPE pair slice, not native prefill.

Fixture evidence
- Regenerated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` with Q RoPE arrays.
- Current `layer_trace_fixtures.npz` SHA: `e138c82eab58403bb018d0c96089941ac3b382144cc81bf36b198a3c08c2a5e1`.
- Q RoPE source arrays:
  - `layer0_q_rope_pairs12_20_input_fp16`: shape `(2, 8)`, SHA `aaada4e4af47bd27e5751199ead39e3e797cb14baf0369ef5e1c029fdd12014f`.
  - `layer0_q_rope_pairs12_20_cos_fp32`: shape `(8,)`, SHA `d70fd3f0c66c7bd2cbb77aa69d862c135f5705ecba719f20825976a8db60b27b`.
  - `layer0_q_rope_pairs12_20_sin_fp32`: shape `(8,)`, SHA `afec2b1796d9ce562d0fcd5d64866d29d95c43becf9f02eba3fa4b0fe7160296`.
  - `layer0_q_rope_pairs12_20_expected_fp16`: shape `(2, 8)`, SHA `e9f0c590a6302e92c8342edb94332ce1e7f30400f2a410a984d328e4dd1bd2f5`.

RED before implementation
- Focused Q fixture/runtime tests failed before implementation because Q RoPE fixture arrays were absent and `native-r9700-runner --primitive-proof fp16_rope_split_half_layer0_q_pairs8` reported `unsupported primitive`.

Implementation
- `native_r9700/ref_fixtures.py` emits Q RoPE pair arrays from full pre/post-RoPE Q tensors using the same split-half layout as K: `left_pre_rope_fp16_then_right_pre_rope_fp16_then_cos_fp32_then_sin_fp32`.
- `tests/native_r9700/test_ref_fixtures.py` covers Q schema shape/dtype and split-half oracle equality.
- `native_r9700/c1_primitive_bridge.cpp` embeds Q input/oracle bytes, Q primitive metadata, and routes Q through the same `build_c1_matmul_dispatch_words` packet as the proven K RoPE kernel.
- `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, and `native_r9700/runner.cpp` expose and validate the Q primitive wrapper.
- `tests/native_r9700/test_runtime_contract.py` covers Q help text, wrapper markers, source arrays, regenerated fixture SHA, and a source regression that Q uses the RoPE dispatch packet.

Debug note
- First real hardware attempt failed with zeros/negative-zeros because Q was omitted from the primitive dispatch selection and used the generic compute packet. Root-cause isolation: K RoPE passed with the same kernel/runtime, Q bytes were embedded correctly, and adding Q to the K/RoPE dispatch condition made hardware pass.

Hardware proof
- Wrapper command: `build/native-r9700-runtime/native_r9700_runner --primitive-proof fp16_rope_split_half_layer0_q_pairs8`.
- Exit status: `0`.
- Log: `logs/c1-runner-primitive-proof-fp16_rope_split_half_layer0_q_pairs8-2026-08-19T18:41:49Z.log`.
- Required markers include `producer_kind: hardware_primitive`, `primitive_backend: hardware`, `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `primitive_name: fp16_rope_split_half_layer0_q_pairs8`, `kernel_source_id: c1r6o-fp16-rope-split-half-layer0-q-pairs8-v1`, `kernel_blob_sha256: 5e0f39471f8f0beadeffc5f043c94cd15fa926b873eb764282a9fed12c1693d8`, `fixture_sha256: e138c82eab58403bb018d0c96089941ac3b382144cc81bf36b198a3c08c2a5e1`, `source_arrays: layer0_q_rope_pairs12_20_input_fp16,layer0_q_rope_pairs12_20_cos_fp32,layer0_q_rope_pairs12_20_sin_fp32,layer0_q_rope_pairs12_20_expected_fp16`, and wrapper status `pass`.
- Numeric result: `full_fixture_shape: 1x32x5x64`, `full_element_count: 10240`, `tolerance: fp16_ulp<=1`, `max_abs_diff: 0`, `max_ulp_diff: 0`, `mismatch_count: 0`, `byte_mismatch_count: 0`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`.

Review fix
- Review gate `agent://C1R6oReview` found Q still reused K full-shape metadata. Fixed bridge/runtime/tests/docs to report Q shape `1x32x5x64` and full element count `10240`; hardware proof rerun at `2026-08-19T18:41:49Z` passed with corrected markers.

Regression proof after C1R-6o
- Focused Q fixture/wrapper/help/dispatch tests: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_q_rope_pair_slice_fixture_matches_split_half_oracle tests/native_r9700/test_runtime_contract.py::test_q_rope_uses_rope_dispatch_packet tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_proof_wraps_supplied_bridge_and_logs_fp16_rope_split_half_layer0_q_pairs8 -q` exited `0` with `6 passed in 2.67s`.

Final verification after review fix
- Focused Q fixture/wrapper/help/dispatch tests exited `0` with `6 passed in 3.21s`.
- Corrected Q hardware proof exited `0` and wrote `logs/c1-runner-primitive-proof-fp16_rope_split_half_layer0_q_pairs8-2026-08-19T18:41:49Z.log` with `full_fixture_shape: 1x32x5x64`, `full_element_count: 10240`, and zero mismatches.
- Native regression `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited `0` with `165 passed, 2 warnings`.
- Full regression `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -q` exited `0` with `205 passed, 2 warnings`.
- `git diff --check` exited `0`.
- Re-review gate `agent://C1R6oReReview` approved with no findings.

Remaining C1R-6 blockers
- Full-shape tiled/multi-workgroup projection coverage beyond one cols8 Q/K/V tile.
- Full-shape RoPE over Q/K.
- Attention score/softmax/context kernels.
- Residual/RMSNorm/SiLU/gated MLP composition at full shape.
- Post-layer hidden-state oracle or regenerated fixtures large enough to validate complete layer-0 output.

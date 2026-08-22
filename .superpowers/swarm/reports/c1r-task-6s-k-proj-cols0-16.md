# C1R-6s K projection cols0:16 tiled accumulation

Decision:
- Added proof-only chain `layer0_k_proj_full_inner_cols0_16_tiled_accum_chain` for layer-0 K projection rows `0:5` padded to 8, full inner `0:2048`, and output columns `0:16` as two adjacent 8-column primitive tiles.
- Kept `native_prefill_acceptance: open` and `acceptance_scope: hardware_primitive_chain_only`; this is a hardware primitive-chain proof, not native prefill acceptance.
- Reused the existing split-A/B GEMM kernel for chunk 0 of each output tile and the existing full-inner cols8 accumulator kernel for chunks 1..127 of each tile. The chain dispatch contract is 256 stages: tile 0 stages 0..127 for cols `0:8`, tile 1 stages 128..255 for cols `8:16`.
- The bridge computes the two 8x8 fp32 accumulator tiles in a single output VRAM page and assembles final readback as row-major `8x16` fp32 via final SDMA row/tile copies after all compute. There is no CPU readback between inner chunks or between output tiles.
- Root-cause fix after first hardware failure: the second tile upload was initially generated from raw `b[:,8:16]` fixture bytes, while the proven kernel expects each 16x8 chunk as row-pair/column-interleaved dot2-pair packing. Repacked `kC1FullInnerCols0_16KCols8_16ModelWeightChunkBytes` with the same transform that matches `kC1FullInnerCols8ModelWeightChunkBytes`.
- Wrapper marker decision: replace C1R-6s placeholder exact-zero tolerated-diff markers with observed hardware tolerance markers, matching the established C1R-6l pattern where fp32 accumulator bytes differ but `mismatch_count: 0` under `fp32_ulp<=64`.

Fixture evidence:
- Regenerated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` with:
  - `layer0_k_proj_full_inner_cols0_16_a_fp16` shape `(8, 2048)`, SHA256 over raw bytes `26261261843662c4475a0aca530b2f160f7875e7ac507955b456af614b116920`.
  - `layer0_k_proj_full_inner_cols0_16_b_fp16` shape `(2048, 16)`, SHA256 over raw bytes `6fcece1ca6aea28ced249a4aaadf1b294bce53deba2037f441d0c58ca742a07c`.
  - `layer0_k_proj_full_inner_cols0_16_expected_fp32` shape `(8, 16)`, SHA256 over raw bytes `5a749198fa05eca89df88c4d0754b9e45af874009b5f46c46957198b7d6ec7a3`.
  - `layer0_k_proj_full_inner_cols0_16_expected_fp16` shape `(8, 16)`, SHA256 over raw bytes `30610408ab7ae5902dcd215746675d55213a2f05b4f95680a23db1e368c53b80`.
- Updated `fixtures_schema.json`; current `layer_trace_fixtures.npz` file SHA256 is `07ff1261d98724a03ac2d96b69fcc0e8548ab1d68238da5a7bb41e1f5b6494ad`.
- The new cols0:16 fixture shares A, B cols `0:8`, and expected cols `0:8` exactly with the existing cols8 fixture.
- Embedded second-tile packed bytes SHA256: `85b5d75a534a7848dd59b2365a7510b6bcbf08e47ca1de0eadeb55b61b8ccfc7`.

Runtime/bridge contract:
- New chain name: `layer0_k_proj_full_inner_cols0_16_tiled_accum_chain`.
- `chain_stage_count: 256`, `kernarg_rewrite_count: 256`, `compute_dispatch_count: 256`.
- Data regions: activation 8 pages at `0x0000200000011000`, model weights 16 pages at `0x0000200000019000`, output accumulator tiles 1 page at `0x0000200000029000`; supplemental PTE count `25`.
- Upload/download byte contract: activation `32768`, model weights `65536`, upload total `98304`, output/readback `512`.
- Output/readback markers: `output_shape: 8x16`, `output_tile_count: 2`, `output_tile0_cols: 0:8`, `output_tile1_cols: 8:16`, `readback_layout: row_major_8x16_from_two_8x8_output_tiles`.
- Hardware tolerance markers: `tolerance: fp32_ulp<=64`, `max_abs_diff: 1.621246337890625e-05`, `max_ulp_diff: 35`, `mismatch_count: 0`, `byte_mismatch_count: 88`.

Validation commands observed:
- RED regression: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_k_cols0_16_second_weight_tile_uses_dot2_pair_packing -q` exited 1 before the fix because the embedded second tile equaled raw fixture bytes instead of packed bytes.
- `python3 -m py_compile native_r9700/ref_fixtures.py tests/native_r9700/test_ref_fixtures.py tests/native_r9700/test_runtime_contract.py` exited 0.
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_k_projection_full_inner_cols0_16_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_runtime_contract.py::test_layer0_k_cols0_16_second_weight_tile_uses_dot2_pair_packing tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_proj_full_inner_cols0_16_tiled_accum_chain -q` exited 0 with `3 passed`.
- `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o /tmp/native_r9700_primitive_bridge_c1r6s` exited 0.
- `mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner` exited 0.
- `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_proj_full_inner_cols0_16_tiled_accum_chain` exited 0 and wrote `logs/c1-runner-primitive-chain-proof-layer0_k_proj_full_inner_cols0_16_tiled_accum_chain-2026-08-19T20:02:00Z.log`.

Hardware proof markers:
- `primitive_chain_proof_wrapper_status: pass`.
- `producer_kind: hardware_primitive_chain`.
- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`.
- `pci_id: 1002:7551`.
- `arch: gfx1201`.
- `kernel_launch_status: pass`.
- `sdma_h2d_status: pass`.
- `sdma_d2h_status: pass`.
- `cpu_comparison_status: pass`.
- `host_device_transfer_status: pass`.
- `failure_stage: none`.
- `failure_text: none`.
- `wrapper_exit_status: 0`.
- `exit_status: 0`.
Review gate:
- `C1R6sReview` approved with no Critical/Important findings. Reviewer confirmed second K tile dot2-pair packing, wrapper tolerated marker validation, and hardware-primitive-only scope.

Full native regression after review dispatch:
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited 0 with `177 passed, 2 warnings in 77.86s`.


# C1R-6m full-inner V-projection cols8 accumulator

Status: proof-only hardware slice complete; full C1R-6 layer-0 acceptance remains open.

Decision:
- Add `layer0_v_proj_full_inner_cols8_accum_chain` as the V-projection analogue of the proven K full-inner cols8 accumulator. This keeps the current C1 boundary honest: one real layer-0 projection cols8 tile, not a full native layer.
- Reuse the existing full-inner activation/input-norm bytes because K and V share the same `input_norm` rows. Add distinct V projection weight bytes and the V `input_norm @ v_proj.weight[:8].T` fp32 oracle from `layer_trace_fixtures.npz`.
- Pack V model weights into the same dot2 pair layout proven by K: for each 16-row chunk, adjacent B rows are interleaved per output-column half as `[r0c0,r1c0,r0c1,r1c1,...]`. Raw V B bytes produced a hardware mismatch; the packed V bytes reduced the proof to expected fp32 accumulation-order roundoff.
- Use `tolerance: fp32_abs<=2e-6_or_ulp<=64` for V. The final proof observed `max_abs_diff: 1.3113021850585938e-06`, `max_ulp_diff: 832`, `mismatch_count: 0`, and `byte_mismatch_count: 44`; high ULP is confined to tiny-magnitude fp32 values, while absolute error stays below the explicit bound.
- Keep `native_prefill_acceptance: open`; this is hardware primitive-chain evidence only, not a native producer acceptance claim.

Implementation:
- `native_r9700/ref_fixtures.py` now generates `layer0_v_proj_full_inner_cols8_{a_fp16,b_fp16,expected_fp32}` into `tests/native_r9700/fixtures/layer_trace_fixtures.npz`.
- `tests/native_r9700/fixtures/fixtures_schema.json` records the new V arrays and the regenerated layer-trace fixture SHA `17d05bc7f27dda7f4f5748eb7c46a273ff2552fe887123f20c2af6e31a4dc5f6`.
- `native_r9700/c1_primitive_bridge.cpp` parameterizes the full-inner cols8 accumulator chain over K/V metadata and embedded model/oracle bytes, adds V routing, and supports the V absolute-or-ULP tolerance.
- `native_r9700/runtime.{h,cpp}` and `native_r9700/runner.cpp` expose/validate the V chain through `--primitive-chain-proof layer0_v_proj_full_inner_cols8_accum_chain`.

Verification observed by supervisor:
- RED before implementation: focused fixture/schema/runtime V tests failed because V arrays and the V primitive-chain wrapper contract were absent.
- Fixture generation command exited `0` and wrote all fixture files:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures`.
- V fixture digest check exited `0` with `3 passed in 0.04s`:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer0_v_projection_full_inner_cols8_fixture_matches_fp32_matmul_oracle -q`.
- Build command exited `0`:
  `mkdir -p build/native-r9700-runtime logs && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`.
- Focused V contract command exited `0` with `4 passed in 2.76s`:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer0_v_projection_full_inner_cols8_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_v_proj_full_inner_cols8_accum_chain -q`.
- Initial hardware proof with raw V B packing exited `1`; observed `max_abs_diff: 0.59915786981582642`, `mismatch_count: 40`, proving the raw B layout was wrong for the existing kernel.
- Packed V hardware wrapper proof exited `0`:
  `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_v_proj_full_inner_cols8_accum_chain`.
- Hardware log: `logs/c1-runner-primitive-chain-proof-layer0_v_proj_full_inner_cols8_accum_chain-2026-08-19T18:07:07Z.log`.
- Required markers in that log include `producer_kind: hardware_primitive_chain`, `primitive_backend: hardware`, `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `chain_stage_count: 128`, `chain_readback_between_stages: no`, `resident_data_page_count: 17`, `supplemental_pte_count: 17`, activation/model/output PTE statuses all `pass`, `kernarg_scalar_va_source: model_weight_region`, `kernel_reads_model_weight_region: yes`, `kernarg_rewrite_count: 128`, `compute_dispatch_count: 128`, `inner_chunk_count: 128`, `inner_chunk_size: 16`, `tile_inner: 2048`, `expected_fp32_sha256: 60eaa262244d3587aced4dcab0a267843ac727135dd0ef585f2a54b0f556e156`, `tolerance: fp32_abs<=2e-6_or_ulp<=64`, `max_abs_diff: 1.3113021850585938e-06`, `max_ulp_diff: 832`, `mismatch_count: 0`, `byte_mismatch_count: 44`, `kernel_launch_status: pass`, `sdma_h2d_status: pass`, `sdma_d2h_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `failure_stage: none`, `failure_text: none`, and `wrapper_exit_status: 0`.

- Native focused suite exited `0` with `159 passed, 2 warnings in 50.77s`:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q`.
- Full repository suite exited `0` with `199 passed, 2 warnings in 83.82s`:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -q`.
- Read-only review gate `agent://C1R6mReview` approved with no findings. Minor non-blocking note: `max_ulp_diff: 832` is surprising next to the ULP arm, but the contract is explicitly `fp32_abs<=2e-6_or_ulp<=64`, and the observed absolute error is below bound.

Remaining C1R-6 blockers:
- Full-shape tiled/multi-workgroup projection coverage beyond one cols8 K/V tile: K/V `(5,2048)x(2048,512)`, Q/O `(5,2048)x(2048,2048)`, MLP `(5,2048)x(2048,8192)` and `(5,8192)x(8192,2048)`.
- Full-shape RoPE over K/Q, attention score/softmax/context, residuals, RMSNorm/SiLU/gated MLP composition, and post-layer hidden-state comparison.
- Full post-layer hidden-state oracle or regenerated fixtures large enough to validate complete layer-0 output.

# C1R-6l full-inner K-projection cols8 accumulator

Status: proof-only hardware slice complete; full C1R-6 layer-0 acceptance remains open.

Decision:
- Add `layer0_k_proj_full_inner_cols8_accum_chain` instead of widening directly to a full native layer. This is the smallest honest next step after split-A/B residency: it proves the complete 2048-hidden-dimension inner accumulation for one real layer-0 K-projection output tile (`rows 0:5`, `cols 0:8`) while preserving `native_prefill_acceptance: open`.
- Use `tolerance: fp32_ulp<=64` for the fp32 accumulator comparison. The hardware dot2 accumulation is numerically close but not byte-identical to the NumPy/fixture fp32 oracle; observed `max_abs_diff: 1.621246337890625e-05`, `max_ulp_diff: 34`, `mismatch_count: 0`, `byte_mismatch_count: 44`.
- Keep the CPU/NumPy prefill path as reference only. No `r9700_native` producer route is advertised by this slice.

Implementation:
- `native_r9700/c1_primitive_bridge.cpp` embeds `kC1Fp16Matmul8x16x8FullInnerCols8AccumKernelText`, source id `c1r6l-layer0-k-proj-full-inner-cols8-accum-v1`, SHA-256 `e8aa56bb65c64da9862f2534219a0b2970dc95bf935ca871ca27f4fa79066853`.
- Stage 0 uses the reviewed split-A/B K-tile GEMM kernel to initialize the fp32 output accumulator from inner range `0:16`; stages 1..127 use the accumulator kernel to cover `16:2048` in 16-wide chunks.
- Activation chunks and model-weight chunks live in distinct resident VRAM spans; output accumulator has a distinct VRAM page. The chain uploads 32768 bytes of activation chunks plus 32768 bytes of model-weight chunks, clears the fp32 output page, dispatches 128 compute stages without intermediate readback, then downloads 256 bytes.

Verification observed by supervisor:
- Build command exited `0`:
  `mkdir -p build/native-r9700-runtime logs && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`.
- Focused contract command exited `0` with `3 passed in 2.66s`:
  `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_kernel_sha_matches_embedded_text tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_proj_full_inner_cols8_accum_chain -q`.
- Hardware wrapper proof exited `0`:
  `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_proj_full_inner_cols8_accum_chain`.
- Hardware log: `logs/c1-runner-primitive-chain-proof-layer0_k_proj_full_inner_cols8_accum_chain-2026-08-19T17:17:17Z.log`.
- Required markers in that log: `producer_kind: hardware_primitive_chain`, `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `chain_stage_count: 128`, `chain_readback_between_stages: no`, `data_region_count: 3`, `resident_data_page_count: 17`, `data_region_residency: seventeen_distinct_vram_pages`, activation/model/output PTE statuses all `pass`, `kernarg_rewrite_count: 128`, `compute_dispatch_count: 128`, `inner_chunk_count: 128`, `inner_chunk_size: 16`, `tile_inner: 2048`, `upload_total_bytes: 65536`, `download_total_bytes: 256`, `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `failure_stage: none`, `failure_text: none`, `wrapper_exit_status: 0`.

Remaining C1R-6 blockers:
- Full-shape tiled/multi-workgroup projection coverage beyond one cols8 K tile: K/V `(5,2048)x(2048,512)`, Q/O `(5,2048)x(2048,2048)`, MLP `(5,2048)x(2048,8192)` and `(5,8192)x(8192,2048)`.
- Full-shape RoPE over K/Q, attention score/softmax/context, residuals, RMSNorm/SiLU/gated MLP composition, and post-layer hidden-state comparison.
- Full post-layer hidden-state oracle or regenerated fixtures large enough to validate complete layer-0 output.

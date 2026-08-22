# C1R-6t — K projection cols0:64 full-head chain

Status: implemented; supervisor validation passed; review pending.

## Scope

Proof-only native R9700/eGPU primitive chain for one Llama layer-0 K-projection head slice before RoPE:

- rows: prompt-0 prefix rows 0:5 plus padded rows 5:8
- cols: K head0 columns 0:64
- inner: 0:2048
- output: fp32 row-major 8x64 stitched from eight 8-column output tiles

This does **not** close C1R-6 layer-0 forward acceptance. `native_prefill_acceptance` remains `open`; this is `hardware_primitive_chain_only` evidence.

## Decisions

- Reuse the proven full-inner 8-column accumulator kernel instead of introducing a new wide GEMM kernel. This preserves the reviewed RDNA4 kernel and isolates the scale-up to resident-memory staging and wrapper contract.
- Expand the K head as eight independent output-column tiles and 128 inner chunks per tile: 1024 compute dispatches total.
- Upload one 32 KiB activation stream plus one 256 KiB dot2-pair-packed model-weight stream. Avoid 1024 per-stage H2D uploads; all stage dispatches read resident VRAM.
- Use one contiguous D2H readback of the 8 tile-major output blocks, then stitch to row-major 8x64 on host before comparison. This avoids SDMA descriptor pressure while keeping the comparison contract row-major.
- Increase primitive chain stage status arrays from 256 to 1024 entries. The wrapper must require `stage1023_kernel_launch_status: pass`, so truncated chains fail closed.
- Refresh `layer_trace_fixtures.npz` and fixture schema because the C1R-6t fixture arrays add a full K head oracle. The new fixture SHA is `bbe9a8ff40eb3ef7f75fdce66bc02da5243b7cdcb0706566bfd364fed613fe40`.

## Files changed in this slice

- `native_r9700/c1_primitive_bridge.cpp`
  - embeds `kC1FullInnerCols0_64ModelWeightChunkBytes`
  - adds `layer0_k_proj_full_inner_cols0_64_tiled_accum_chain`
  - adds 1024-stage logging and row-major stitch after one tile-major D2H copy
- `native_r9700/runtime.h`
  - adds C1R-6t wrapper constants/markers
  - raises primitive-chain status expectations to cover 1024 stages
- `native_r9700/runtime.cpp`
  - accepts and validates the cols0:64 chain
  - requires full marker set including final stage, resident region counts, SHA, sizes, layout, tolerance, and zero mismatches
- `native_r9700/runner.cpp`
  - routes the new chain through `--primitive-chain-proof`
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`

## Validation

Build command:

```bash
mkdir -p build/native-r9700-runtime && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_c1_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Observed: exit 0, no compiler output.

Focused host tests:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough \
  tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests \
  tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_sha256_matches_disk \
  tests/native_r9700/test_ref_fixtures.py::test_layer0_k_proj_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle \
  tests/native_r9700/test_runtime_contract.py::test_layer0_k_cols0_64_weight_tiles_use_dot2_pair_packing \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_proj_full_inner_cols0_64_tiled_accum_chain \
  tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_k_cols0_64_last_stage_marker \
  -q
```

Observed: `7 passed in 4.45s` after the final marker cleanup.

Hardware proof:

```bash
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_proj_full_inner_cols0_64_tiled_accum_chain
```

Observed final tokens:

```text
producer_kind: hardware_primitive_chain
primitive_backend: hardware
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
chain_name: layer0_k_proj_full_inner_cols0_64_tiled_accum_chain
acceptance_scope: hardware_primitive_chain_only
model_forward_scope: layer0_k_proj_full_inner_cols0_64
native_prefill_acceptance: open
fixture_sha256: bbe9a8ff40eb3ef7f75fdce66bc02da5243b7cdcb0706566bfd364fed613fe40
chain_stage_count: 1024
chain_readback_between_stages: no
chain_readback_between_output_tiles: no
data_region_count: 3
resident_data_page_count: 73
data_region_residency: seventy_three_distinct_vram_pages
stage1023_kernel_launch_status: pass
kernarg_rewrite_count: 1024
compute_dispatch_count: 1024
activation_byte_count: 32768
model_weight_byte_count: 262144
output_byte_count: 2048
upload_total_bytes: 294912
download_total_bytes: 2048
output_shape: 8x64
expected_fp32_sha256: f1387d0c28aae9aec3450fa384ac1ab178786decb9e5158250e14071dd99b047
covered_element_count: 320
full_element_count: 20480
readback_layout: row_major_8x64_from_eight_8x8_output_tiles
tolerance: fp32_abs<=2e-5_or_ulp<=64
max_abs_diff: 1.621246337890625e-05
max_ulp_diff: 288
mismatch_count: 0
byte_mismatch_count: 339
kernel_launch_status: pass
sdma_h2d_status: pass
sdma_d2h_status: pass
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
failure_text: none
primitive_chain_proof_wrapper_status: pass
wrapper_exit_status: 0
exit_status: 0
```

Focused post-review help/K tests:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_ref_fixtures.py::test_layer0_k_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_runtime_contract.py::test_layer0_k_cols0_64_weight_tiles_use_dot2_pair_packing tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_proj_full_inner_cols0_64_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_k_cols0_64_last_stage_marker -q
```

Observed: `5 passed in 5.99s`.

Focused native regression after post-review help fix:

```bash
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q
```

Observed: `181 passed, 2 warnings in 81.22s`.

## Risk / next step

The K head pre-RoPE projection is now native hardware-backed for head0 cols0:64. Layer-0 forward remains open: next useful step is converting this fp32 K-head output through the existing full-head K RoPE chain and/or emitting the native K head cache slice as fp16 for comparison against `layer0_k_rope` fixture data. Full C1R-6 still needs full attention and post-attention/MLP hidden-state proof before `r9700_native` model-forward can stop failing closed.

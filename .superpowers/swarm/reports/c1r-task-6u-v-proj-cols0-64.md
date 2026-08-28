# C1R-6u — V projection cols0:64 full-head chain

Status: implemented; supervisor validation passed; review passed.

## Scope

Proof-only native R9700/eGPU primitive chain for one Llama layer-0 V-projection head slice:

- rows: prompt-0 prefix rows 0:5 plus padded rows 5:8
- cols: V head0 columns 0:64
- inner: 0:2048
- output: fp32 row-major 8x64 stitched from eight 8-column output tiles

This does **not** close C1R-6 layer-0 forward acceptance. `native_prefill_acceptance` remains `open`; this is `hardware_primitive_chain_only` evidence.

## Decisions

- Reuse the proven full-inner 8-column accumulator kernel and 1024-stage tiled accumulation used by K cols0:64, with V-specific source arrays and expected fp32/fp16 fixture data.
- Pack V model weights with the same dot2-pair tile-stream layout the RDNA4 kernel expects; raw row-major V weights fail the hardware comparison.
- Keep one resident 32 KiB activation stream, one resident 256 KiB model-weight stream, and one contiguous D2H readback stitched to row-major 8x64 on the host.
- Split bulky K/V cols0:64 oracle arrays into `layer_trace_full_inner_projection_fixtures.npz` instead of raising the 512 KiB committed-fixture bound. Final sizes: `layer_trace_fixtures.npz` 233,552 bytes; `layer_trace_full_inner_projection_fixtures.npz` 453,508 bytes.
- Preserve acceptance scope: `hardware_primitive_chain_only`; native full prefill/model-forward remains open.

## Files changed in this slice

- `native_r9700/c1_primitive_bridge.cpp`
  - embeds `kC1VFullInnerCols0_64ModelWeightChunkBytes`
  - adds `layer0_v_proj_full_inner_cols0_64_tiled_accum_chain`
  - routes K/V cols0:64 provenance to the split full-inner projection fixture
- `native_r9700/runtime.h`
  - adds V cols0:64 wrapper constants/markers
  - updates layer-trace SHA constants after fixture split
  - adds split-fixture path/SHA constants for K/V cols0:64
- `native_r9700/runtime.cpp`
  - accepts and validates the V cols0:64 chain
  - validates source fixture/SHA conditionally for K/V cols0:64 split provenance
- `native_r9700/runner.cpp`
  - lists and routes the V cols0:64 chain through `--primitive-chain-proof`
- `native_r9700/ref_fixtures.py`
  - writes `layer_trace_full_inner_projection_fixtures.npz` separately from the compact layer trace archive
- `tests/native_r9700/fixtures/layer_trace_fixtures.npz`
- `tests/native_r9700/fixtures/layer_trace_full_inner_projection_fixtures.npz`
- `tests/native_r9700/fixtures/fixtures_schema.json`
- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_runtime_contract.py`

## Validation

Fixture regeneration:

```bash
${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
```

Observed: `wrote 7 fixture files to tests/native_r9700/fixtures`.

Fixture sizes/digests after split:

```text
layer_trace_fixtures.npz 233552 dba3634875283bfba19d9d336f77b4786c0f9d5e82590e94b140fc8b3c2f4326
layer_trace_full_inner_projection_fixtures.npz 453508 b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c
fixtures_schema.json 18733 3832f55b13b92be7edbdbd77b596bd874f4a3319d299461c369fe32524021615
```

Build command:

```bash
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/c1_primitive_bridge.cpp -o build/native-r9700-runtime/native_r9700_c1_primitive_bridge && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

Observed: exit 0, no compiler output.

Focused host tests:

```bash
${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer_trace_full_inner_projection_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_k_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_v_projection_full_inner_cols0_64_fixture_matches_fp32_matmul_oracle tests/native_r9700/test_runtime_contract.py::test_layer0_k_cols0_64_weight_tiles_use_dot2_pair_packing tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_proj_full_inner_cols0_64_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_v_proj_full_inner_cols0_64_tiled_accum_chain tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_rejects_missing_v_cols0_64_last_stage_marker -q
```

Observed: `10 passed in 6.77s`.

Hardware proof:

```bash
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_v_proj_full_inner_cols0_64_tiled_accum_chain
```

Observed final markers from `logs/c1-runner-primitive-chain-proof-layer0_v_proj_full_inner_cols0_64_tiled_accum_chain-2026-08-19T21:06:46Z.log` combined proof output:

```text
producer_kind: hardware_primitive_chain
primitive_backend: hardware
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
chain_name: layer0_v_proj_full_inner_cols0_64_tiled_accum_chain
acceptance_scope: hardware_primitive_chain_only
model_forward_scope: layer0_v_proj_full_inner_cols0_64
native_prefill_acceptance: open
source_fixture: tests/native_r9700/fixtures/layer_trace_full_inner_projection_fixtures.npz
fixture_sha256: b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c
chain_stage_count: 1024
chain_readback_between_stages: no
chain_readback_between_output_tiles: no
resident_data_page_count: 73
stage1023_kernel_launch_status: pass
kernarg_rewrite_count: 1024
compute_dispatch_count: 1024
activation_byte_count: 32768
model_weight_byte_count: 262144
output_byte_count: 2048
upload_total_bytes: 294912
download_total_bytes: 2048
output_shape: 8x64
expected_fp32_sha256: 28496084d43f9c0e257095edf97ea77885d6e9a762657e1dfd8e431a0e938927
covered_element_count: 320
full_element_count: 20480
readback_layout: row_major_8x64_from_eight_8x8_output_tiles
tolerance: fp32_abs<=2e-5_or_ulp<=64
max_abs_diff: 1.3113021850585938e-06
max_ulp_diff: 4352
mismatch_count: 0
byte_mismatch_count: 347
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

K/V hardware provenance regression:

```bash
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_k_proj_full_inner_cols0_64_tiled_accum_chain && build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_v_proj_full_inner_cols0_64_tiled_accum_chain
```

Observed: both wrappers exited 0; both printed `primitive_chain_proof_wrapper_status: pass`, `cpu_comparison_status: pass`, and split-fixture SHA `b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c`.

Full native regression:

```bash
${PY} -m pytest tests/native_r9700 -q
```

Observed: `185 passed, 2 warnings in 86.64s`.

## Review

- `C1R6uVReview`: found one Important copy/paste issue in the K cols0:64 fake bridge stats after V marker insertion.
- Fix: restored K fake bridge stats to `max_abs_diff: 1.621246337890625e-05`, `max_ulp_diff: 288`, `byte_mismatch_count: 339`.
- Targeted regression after fix: `3 passed in 6.71s` for K wrapper, V wrapper, and V missing-marker rejection.
- `C1R6uVReReview`: no Critical/Important/Minor findings; safe to close after evidence recorded.

## Risk / next step

The V head projection is now native hardware-backed for layer0 head0 cols0:64, matching the K head pre-RoPE projection slice. Layer-0 forward remains open: next useful step is consuming K/V projected heads in the RoPE/cache path and then moving to attention/context and post-attention/MLP hidden-state proof before `r9700_native` model-forward can stop failing closed.

# C1R-8g MLP down full-inner cols0:64 hardware bridge

## Scope

Implemented the real `layer0_mlp_down_proj_full_inner_to_cols0_64_tiled_accum_chain` hardware primitive chain through the native R9700 bridge. Scope remains `hardware_primitive_chain_only_partial`: full inner dimension for output cols0:64 only, not full hidden width and not native prefill acceptance.

## Decision

Use streaming staging for the full-inner down weights:

- Keep activation resident in a 32-page VRAM region.
- Reuse one 32-page model-weight VRAM scratch for each 8-column output tile.
- Reuse one 32-page sysmem staging window; CPU-refill it with the next tile's 128 KiB model weights before SDMA H2D.
- Dispatch 4096 stages: 8 output tiles x 512 inner chunks.

Reason: one 288-page sysmem staging mapping reported valid pages but SDMA reads from the high tile6 source window returned zero. Streaming through the low 32-page staging window eliminates the high-page source cliff and matches the execution order.

## Evidence

Root-cause diagnostics before cleanup:

- Full upload/page-split still produced zero cols48:64.
- Tile6 model readback from the original high source/span observed zero where expected byte0 was `0xd8`.
- After streaming staging, tile6 scratch diagnostic readback returned `cmp=0 observed0=0xd8 expected0=0xd8`.

Final hardware proof after diagnostic removal:

```text
build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_mlp_down_proj_full_inner_to_cols0_64_tiled_accum_chain
```

Result: exit `0`; log `logs/c1-runner-primitive-chain-proof-layer0_mlp_down_proj_full_inner_to_cols0_64_tiled_accum_chain-2026-08-20T14:40:36Z.log`.

Key markers:

- `primitive_chain_proof_wrapper_status: pass`
- `wrapper_exit_status: 0`
- `cpu_comparison_status: pass`
- `host_device_transfer_status: pass`
- `failure_stage: none`
- `resident_data_page_count: 65`
- `model_weight_region_page_count: 32`
- `supplemental_pte_count: 65`
- `sysmem_staging_requested_size: 131072`
- `max_abs_diff: 1.7732381820678711e-06`
- `max_ulp_diff: 48688`
- `mismatch_count: 0`

Host contract verification:

```text
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_mlp_down_full_inner_to_cols0_64_chain -q
```

Result: `1 passed in 9.74s`.

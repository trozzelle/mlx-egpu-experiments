# Resident K/V Projection Hardware Dispatch — Impl Report

## Summary

Advanced `native_r9700.prefill --producer-kind r9700_native` past the input_norm
seam to a hardware-dispatched layer0 K/V projection slice. The layer0 resident
path now materializes model safetensors K/V projection weights, dispatches and
reads back K and V projection for `cols0:64, inner0:64` on the R9700, and fails
closed at the next precise seam.

## Blocker advance

```
layer0_resident_kv_projection_dispatch_not_implemented
  → layer0_kv_projection_remaining_inner_not_implemented
```

## Changed files

- `native_r9700/c1_primitive_bridge.cpp`
  - New symbols:
    - `kC1Layer0KvProjectionResidentInner`, `...ChunkCount`, `...TileCount`,
      `...StageCount`, `...ActivationByteCount`, `...ModelWeightByteCount`,
      `...UploadTotalBytes`, `...UploadStagingPageCount`, `...OutputByteCount`
    - `struct Layer0KvProjectionMaterialization`
    - `struct Layer0KvProjectionHardwareStatus`
    - `pack_resident_input_norm_activation_chunks`
    - `pack_projection_weight_tiles`
    - `materialize_layer0_kv_projection_sources`
    - `buffer_has_nonzero_byte`
    - `dispatch_one_layer0_kv_projection`
    - `run_layer0_kv_projection_hardware_slice`
  - Wired into `run_native_layer0_resident_input_path`:
    - materializes K/V projection sources from `model.layers.0.self_attn.{k,v}_proj.weight`
    - runs the hardware K/V projection slice
    - emits K/V pass / fail-closed markers
  - Bug fixed: staging page count/PTE span now uses 3-page K/V-specific
    constant instead of the larger proof-chain 72-page count.
  - Cleanup: removed unused `sdma_control` parameter from
    `dispatch_one_layer0_kv_projection` and updated call sites.
- `native_r9700/runtime.cpp`
  - `native_layer0_proof` wrapper forwards K/V pass state:
    - `kv_projection_weight_source` and `kv_projection_dispatch_status` become
      `require_present` instead of exact old values.
    - Added `require_present` for `kv_projection_readback_status` and the
      `layer0_kv_projection_*` marker set.
    - Extracted and emitted the new K/V markers in text + JSON payloads.
  - Preserves bridge's precise `failure_stage` on coherent output.
- `tests/native_r9700/test_runtime_contract.py`
  - `native_layer0_bridge_script()` updated to KV-dispatched format.
  - `test_native_layer0_proof_emits_fail_closed_resident_schema` asserts new
    K/V pass markers and `layer0_kv_projection_remaining_inner_not_implemented`.
  - `test_native_prefill_proof_consumes_layer0_blocker_and_remains_fail_closed`
    updated to new format.
- `tests/native_r9700/test_prefill.py` — KV intermediate-state test data.
- `native_r9700/native_worker.py`, `native_r9700/prefill.py` — forward new KV
  marker fields; acceptance stays `open` without `native_prefill_acceptance: pass`.

## Marker contract (successful seam)

```
resident_subgraph_scope: layer0_resident_kv_projection_cols0_64_inner0_64_hardware_dispatched
resident_subgraph_status: blocked
kernel_count: 200
transfer_bytes: 125952
kv_projection_input_source: model_prompt
kv_projection_weight_source: model_safetensors_k_v_proj_weight_tiles
kv_projection_activation_source: resident_input_norm_activation
kv_projection_parameterization_status: pass
kv_projection_dispatch_status: pass
kv_projection_readback_status: pass
layer0_kv_projection_status: pass
layer0_kv_projection_upload_status: pass
layer0_kv_projection_dispatch_status: pass
layer0_kv_projection_readback_status: pass
layer0_kv_projection_kernel_count: 64
layer0_kv_projection_transfer_bytes: 22528
layer0_kv_projection_inner_range: 0:64
layer0_resident_dataflow_status: blocked
failure_stage: layer0_kv_projection_remaining_inner_not_implemented
```

Native prefill acceptance remains `open`; no NPZ/cache emitted.

## Verification (supervisor)

Bridge + runner compile clean (no warnings).

Focused pytest (18 passed in 147.61s):
```sh
$PY -m pytest \
  tests/native_r9700/test_runtime_contract.py::test_native_worker_preserves_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_emits_fail_closed_resident_schema \
  tests/native_r9700/test_runtime_contract.py::test_native_prefill_proof_consumes_layer0_blocker_and_remains_fail_closed \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_missing_resident_input_norm_activation_evidence \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_missing_hardware_counters \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_fixture_sourced_stage_inputs \
  tests/native_r9700/test_runtime_contract.py::test_native_layer0_proof_rejects_fake_native_prefill_acceptance \
  tests/native_r9700/test_prefill.py::test_prefill_cli_logs_native_layer0_input_norm_pass_evidence_and_removes_unaccepted_npz -q
```

Real hardware bridge smoke (prompt-0):
```sh
build/native-r9700-runtime/c1_primitive_bridge --native-layer0 \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[128000,791,6864,315,9822,374]'
```
Result: exit 1 as expected; all input_norm and K/V projection slices `pass`;
`kernel_count: 200`, `transfer_bytes: 125952`;
`failure_stage: layer0_kv_projection_remaining_inner_not_implemented`.

Runner wrapper smoke:
```sh
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge \
  build/native-r9700-runtime/native_r9700_runner --native-layer0-proof \
  --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
  --token-ids-json '[128000,791,6864,315,9822,374]' \
  --json logs/native-layer0-kv-smoke.json --log logs/native-layer0-kv-smoke.log
```
Result: exit 1 as expected; wrapper preserved K/V pass markers and precise blocker.

Prefill CLI smoke:
```sh
NATIVE_R9700_PREFILL_RUNNER=build/native-r9700-runtime/native_r9700_runner \
  NATIVE_R9700_C1_PRIMITIVE_BRIDGE=build/native-r9700-runtime/c1_primitive_bridge \
  $PY -m native_r9700.prefill --model <model> --token-ids-json '[...]' \
  --producer-kind r9700_native --out logs/native-prefill-kv-smoke.npz \
  --log logs/native-prefill-kv-smoke.log
```
Result: exit 1 as expected; error text names K/V projection slice done with
`inner range 64:2048 not implemented`; no NPZ accepted.

## Next seam

`layer0_kv_projection_remaining_inner_not_implemented`. Next vertical step:
extend K/V projection to full inner `0:2048` (and/or move toward layer0 KV
cache write / RoPE seam), keeping native prefill acceptance open.

## PCI identity fix (CppReviewer finding, resolved)

`c1_primitive_bridge.cpp` no longer prints a hardcoded `pci_id: 1002:7551`
before device identity is observed. Both hardware-slice status structs
(`Layer0InputNormHardwareStatus`, `Layer0KvProjectionHardwareStatus`) gained an
`observed` `pci_id` field (default `unavailable`) set only after CFG_READ
confirms the target; the native layer0 path prints the most recent observed
identity, or `unavailable` if discovery never ran.

Verified:
- Hardware smoke prints observed `pci_id: 1002:7551`; K/V markers unchanged.
- Fail-closed smoke (`--model /tmp/nonexistent-model`) prints
  `pci_id: unavailable`.
- Runner rebuild clean; focused contracts `3 passed in 21.48s`;
  wrapper log shows observed identity exactly once.

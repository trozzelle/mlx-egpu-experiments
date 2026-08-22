# Input Norm Hardware Slice

## Files changed
- `native_r9700/c1_primitive_bridge.cpp`
- `.superpowers/swarm/reports/input-norm-hardware-slice.md`

## Marker contract
- Stores model/prompt-derived prefix embedding rows and `model.layers.0.input_layernorm.weight` bytes in `Layer0InputNormMaterialization`.
- Builds the existing 8-row, cols0:64 RMSNorm chain payloads from real prompt embedding rows, padded only past the available prefix rows.
- `resident_input_norm_activation_source: model_prompt_embedding_plus_layer0_input_norm_weight`
- On hardware upload/dispatch/readback success:
  - `resident_input_norm_activation_status: pass`
  - `resident_input_norm_activation_upload_status: pass`
  - `resident_input_norm_activation_dispatch_status: pass`
  - `resident_input_norm_activation_readback_status: pass`
  - `kv_projection_activation_source: resident_input_norm_activation`
  - `kv_projection_dispatch_status: blocked`
  - `failure_stage: layer0_resident_kv_projection_dispatch_not_implemented`
- On any earlier failure, the bridge keeps `native_prefill_acceptance: open`, marks the resident input_norm activation failed/unavailable, and reports the precise earlier `failure_stage` from source materialization, TinyGPU setup, SDMA upload, compute dispatch, or D2H readback.
- No full K/V projection, full layer loop, NPZ output, or prompt-cache acceptance is claimed.

## Supervisor verification commands to run
- Build/compile the native bridge target used by the existing R9700 workflow.
- Run bridge mode with real model/prompt inputs, for example:
  ```sh
  <bridge-binary> --native-layer0 --model <mlx-model-dir> --token-ids-json '[<prompt-token-ids>]'
  ```
- Confirm the output contains the pass markers above on hardware success, or a fail-closed earlier `failure_stage` if the hardware path cannot complete.
- Confirm `native_prefill_acceptance: open`, `kv_projection_dispatch_status: blocked`, and no NPZ/prompt-cache acceptance markers are emitted.

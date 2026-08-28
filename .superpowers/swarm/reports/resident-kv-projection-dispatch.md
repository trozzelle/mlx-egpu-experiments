# Resident K/V projection dispatch

## Changed files
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.cpp`
- `tests/native_r9700/test_runtime_contract.py`
- `tests/native_r9700/test_prefill.py`

## Exact behavior
- Advanced `--native-layer0` beyond the old `layer0_resident_kv_projection_dispatch_not_parameterized` marker.
- The bridge now emits a model/prompt-derived K/V projection parameterization stage:
  - `resident_subgraph_scope: layer0_resident_kv_projection_dispatch_parameterized`
  - `kv_projection_input_source: model_prompt`
  - `kv_projection_weight_source: model_safetensors_metadata`
  - `kv_projection_activation_source: pending_resident_input_norm`
  - `kv_projection_parameterization_status: pass` once prompt/config/safetensors metadata validate
  - `kv_projection_dispatch_status: blocked`
  - proven K/V full-inner tiled kernel/layout markers for `layer0_k_proj_full_inner_cols0_64_tiled_accum_chain` and `layer0_v_proj_full_inner_cols0_64_tiled_accum_chain`
- Runtime layer0 proof now requires the new parameterization markers, forwards the new fields into stdout/JSON, and keeps fixture-boundary markers rejected.
- Hardware counters remain honest: `kernel_count: 0` and `transfer_bytes: 0` because the bridge has not launched hardware from resident input_norm activation yet. Planned dispatch/transfer markers are separate `planned_*` fields and do not masquerade as hardware evidence.
- The next precise blocker is now `layer0_resident_input_norm_activation_not_materialized`.
- Native prefill remains fail-closed/open; no `native_prefill_acceptance: pass` is introduced.

## Verification
- Per assignment acceptance, no commands, tests, linters, formatters, package managers, hardware commands, project-wide suites, or git commands were run by this agent.
- Text-level check via repository search found no remaining `layer0_resident_kv_projection_dispatch_not_parameterized` references in `native_r9700` or `tests/native_r9700` after the edits.

## Suggested supervisor command
- `${PY} -m pytest tests/native_r9700/test_runtime_contract.py tests/native_r9700/test_prefill.py -q`

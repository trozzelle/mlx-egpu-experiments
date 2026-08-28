# C1R-7f layer0 sliced hardware proof

## Scope

Add an honest layer0 proof wrapper that composes the currently proven C1R hardware primitive chains in model-forward order without claiming full layer0 or native prefill acceptance.

## Work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch; no fallback worktree.

## Source grounding

- `.superpowers/swarm/progress.md`
- `.superpowers/swarm/reports/c1-task-7*.md`
- `docs/archive/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md`
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `native_r9700/c1_primitive_bridge.cpp`
- `tests/native_r9700/test_runtime_contract.py`
- Scout agents: `LayerAcceptanceScout`, `LayerFixtureScout`, `LayerRuntimeScout`

## Implemented

- Added runner mode `--layer0-slice-proof`.
- Added `RuntimeSession::layer0_slice_proof`.
- The wrapper runs 13 existing primitive-chain proofs in layer0 slice order:
  1. K projection cols0:64
  2. V projection cols0:64
  3. Q projection cols0:64
  4. K RoPE head0/tokens0:5
  5. Q RoPE head0/tokens0:5
  6. integrated scores -> softmax -> context head0/tokens0:5/cols0:64
  7. O projection cols0:64
  8. post-O residual cols0:64
  9. post-attention RMSNorm cols0:64
  10. MLP gate projection cols0:64
  11. MLP up projection cols0:64
  12. MLP activation SiLU(gate) * up cols0:64
  13. partial MLP down projection inner cols0:64 -> output cols0:64
- Added no-hardware tests for honest slice-only markers and one retry for transient child proof failures.
- Fixed stale wrapper expectation for generalized full-inner chain `data_region_count`: bridge logs emit `2`; page-count markers still prove activation/model/output residency.

## Decisions

- The command is explicitly `hardware_layer0_slice_chain`, not `r9700_native`.
- `acceptance_scope: hardware_layer0_slice_only` and `native_prefill_acceptance: open` remain mandatory.
- `full_layer0_acceptance: blocked` is recorded because current hardware proofs cover only head0/tokens0:5/cols0:64 and a partial MLP down contribution.
- `dataflow_status: fixture_oracle_boundaries_not_fused` is recorded because the component chains are still fixture/oracle-boundary proofs, not a resident fused layer dataflow.
- Each component gets one retry. This is not silent success: attempt logs/statuses are recorded. The retry handles observed TinyGPU timing/state transients (`chain_stage0_timeline_timeout`) while preserving fail-closed behavior if the retry also fails.

## Verification

- RED contract before implementation:
  - Command: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_slice_proof_wraps_existing_chains_without_claiming_full_prefill -q`
  - Result before implementation: failed with `error: unknown mode '--layer0-slice-proof'`.
- Retry RED before implementation:
  - Command: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_slice_proof_retries_one_transient_component_failure -q`
  - Result before retry implementation: failed with `component0_wrapper_status: fail` and no retry.
- Focused tests:
  - Command: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_slice_proof_wraps_existing_chains_without_claiming_full_prefill tests/native_r9700/test_runtime_contract.py::test_layer0_slice_proof_retries_one_transient_component_failure tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_k_proj_full_inner_cols0_64_tiled_accum_chain -q`
  - Result: `3 passed in 19.04s`.
- Runner compile:
  - Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exited `0`; no output.
- Hardware proof:
  - Command: `build/native-r9700-runtime/native_r9700_runner --layer0-slice-proof`
  - Result: exited `0` with `layer0_slice_proof_wrapper_status: pass`, `component_chain_count: 13`, all component wrapper statuses `pass`, `component7_attempt_count: 2`, `component7_retry_status: pass`, `failure_stage: none`, `wrapper_exit_status: 0`, and `exit_status: 0`.
  - Log: `logs/c1-runner-layer0-slice-proof-2026-08-20T13:42:31Z.log`.

## Current blocker

This does not complete C1R-6 full layer0 hardware forward pass. `LayerFixtureScout` found no full layer0 post-layer hidden fixture; current committed fixtures only contain compact/sliced `layer0_mlp_residual_out_fp16` and full K/V in `kv_state.npz`. Full layer0 acceptance needs a full post-layer hidden oracle and hardware coverage beyond the current head0/cols0:64/partial-MLP-down slice.

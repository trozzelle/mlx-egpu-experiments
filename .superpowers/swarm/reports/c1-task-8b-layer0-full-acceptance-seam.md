# C1R-8b layer0 full acceptance seam

## Scope

Add a fail-closed runtime/runner command for the full layer0 post-layer hidden oracle target. This records exactly what future hardware full-layer dataflow must produce without claiming native prefill acceptance.

## Work boundary

- Path: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch.

## Implemented

- Added runner mode `--layer0-full-hidden-proof`.
- Added `RuntimeSession::layer0_full_hidden_proof`.
- The command emits:
  - `producer_kind: hardware_layer0_full_hidden_chain`
  - `chain_name: layer0_full_post_layer_hidden_forward`
  - `model_forward_scope: layer0_full_post_layer_hidden_forward`
  - `acceptance_scope: hardware_layer0_full_hidden`
  - `native_prefill_acceptance: open`
  - `full_layer0_acceptance: blocked`
  - `blocker: hardware_full_width_layer0_dataflow_not_implemented`
  - `source_fixture: tests/native_r9700/fixtures/layer_trace_layer0_post_layer_hidden_fixtures.npz`
  - `source_array: layer0_post_layer_hidden_fp16`
  - `expected_output_shape: 8x2048`
  - `expected_output_dtype: fp16`
  - `expected_output_sha256: feb3f5f10bca2182d677f0edb5f386270b2e1f91c21275d7ed95c419d14bc7a7`
  - `layer0_full_hidden_proof_wrapper_status: blocked`
  - `failure_stage: layer0_full_width_dataflow_missing`
- The command returns exit status `1` by design until real hardware dataflow exists.

## Decision

Keep `--layer0-slice-proof` and `--layer0-full-hidden-proof` separate. The slice command proves current hardware primitive chains. The full-hidden command defines the acceptance seam against the new full-width oracle but fails closed until implementation can produce the full hidden tensor.

## Verification

- RED before implementation:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_full_hidden_proof_reports_blocked_until_full_width_dataflow_exists -q`
  - Result: failed first with unknown mode before implementation; renamed to `--layer0-full-hidden-proof` after scout review to avoid overclaiming full prefill.
- Compile:
  - Command: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runner.cpp native_r9700/runtime.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner`
  - Result: exited `0`; no output.
- Focused tests:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_layer0_full_hidden_proof_reports_blocked_until_full_width_dataflow_exists tests/native_r9700/test_runtime_contract.py::test_layer0_slice_proof_wraps_existing_chains_without_claiming_full_prefill -q`
  - Result: `2 passed in 10.29s`.
- Direct command:
  - Command: `build/native-r9700-runtime/native_r9700_runner --layer0-full-hidden-proof`
  - Result: emitted the expected blocked markers and exited `1` with `wrapper_exit_status: 1`, `exit_status: 1`.
  - Log pattern: `logs/c1-runner-layer0-full-hidden-proof-<timestamp>.log`.

## Remaining blocker

Full C1/native prefill acceptance remains blocked on hardware-side full-width layer0 dataflow.

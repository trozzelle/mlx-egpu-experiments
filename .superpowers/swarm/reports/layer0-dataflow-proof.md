# Layer0 resident dataflow proof

## Files changed
- `native_r9700/runtime.h`
- `native_r9700/runtime.cpp`
- `native_r9700/runner.cpp`
- `native_r9700/c1_primitive_bridge.cpp`
- `tests/native_r9700/test_runtime_contract.py`
- `.superpowers/swarm/reports/layer0-dataflow-proof.md`

## Implemented contract
Added runner mode:

```sh
native-r9700-runner --native-layer0-proof --model <mlx-model-dir> --token-ids-json '[...]' --json <path> --log <path>
```

The mode writes both the requested log and JSON contract artifact. Emitted schema fields:

- `producer_kind: r9700_native`
- `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`
- `hardware_log_path` / `log_path`
- `acceptance_scope: hardware_layer0_resident_dataflow`
- `native_prefill_acceptance: open`
- `model_dir`
- `token_ids_json`
- `layer_index: 0`
- `resident_boundary_count`
- `kernel_count`
- `transfer_bytes`
- `k_shape`
- `v_shape`
- `hidden_shape`
- `layer0_resident_dataflow_status`
- `failure_stage`
- `failure_text` in log
- `exit_status`

## Fail-closed behavior
The wrapper validates the native layer0 bridge output and fails closed if any required resident-dataflow marker is missing or wrong. It rejects:

- `native_prefill_acceptance: pass`
- missing `kernel_count`
- missing `transfer_bytes`
- fixture boundary markers such as `source_fixture`, `fixture_slice`, `source_arrays`, or `*_input_source: fixture...`

The in-tree `c1_primitive_bridge.cpp --native-layer0` path is a scaffold only: it emits the resident schema with `native_prefill_acceptance: open`, zero counters, and `failure_stage: layer0_resident_dataflow_not_implemented`, then exits nonzero. It does not claim accepted native prefill and does not write an NPZ.

## Runtime contract tests added
Focused tests in `tests/native_r9700/test_runtime_contract.py` cover:

- `--native-layer0-proof` is listed in help.
- Fail-closed resident schema/log/JSON emission.
- Rejection of missing `kernel_count` and missing `transfer_bytes`.
- Rejection of fixture-sourced stage inputs.
- Rejection of fake `native_prefill_acceptance: pass`.

## No-validation policy
Per assignment, I did not run tests, linters, formatters, package managers, hardware commands, project-wide suites, git commands, or compile commands.

## Supervisor commands
Run after review as appropriate:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q
```

Optional compile/smoke for this mode:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
NATIVE_R9700_C1_PRIMITIVE_BRIDGE=<bridge-or-fake> build/native-r9700-runtime/native_r9700_runner --native-layer0-proof --model <mlx-model-dir> --token-ids-json '[1,2,3]' --json /tmp/native-layer0.json --log /tmp/native-layer0.log
```

## Blocker
Resident full layer0 fusion was not completed in this wave. The shortest safe path is in place, but real acceptance remains blocked at `layer0_resident_dataflow_not_implemented` until the bridge uploads model/prompt buffers and executes layer0 K/V/hidden production without fixture boundaries.

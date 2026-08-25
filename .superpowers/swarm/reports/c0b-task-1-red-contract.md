# C0B task set 1: RED contract tests

## Status

Done. Supervisor verified the no-hardware pytest contract is RED for the expected missing-source reason, and reviewer accepted the test/report/docs contract with no findings.

## Changed files

- `tests/test_native_amdev_transfer_contract.py`
- `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- `.superpowers/swarm/reports/c0b-task-1-red-contract.md`

`docs/tasks/native-r9700-producer/validation-commands.md` was inspected and already contained the C0B contract-test command plus expected RED result, so it was not edited.

## Supervisor command

Run from `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Expected result before task set 2

FAIL with:

```text
AssertionError: native transfer probe source missing
```

The failure should come from `compile_probe(tmp_path)` asserting that `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` exists before invoking:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra
```

## Contract covered

- `test_remote_cmd_frame_self_test_passes` compiles the probe, runs `--self-test remote-cmd-frame`, and expects `self_test: remote-cmd-frame` plus `status: pass`.
- `test_log_contract_self_test_lists_required_fields` compiles the probe, runs `--self-test log-contract`, and expects every required `required_log_field: ...` line plus `status: pass`.
- Required log fields asserted: `runtime_substrate`, `pci_id`, `arch`, `transfer_byte_count`, `cpu_comparison_status`, `host_device_transfer_status`, `failure_stage`, `failure_text`, `exit_status`.

## Guardrails observed

- No production C++ source was added; `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` remains absent.
- No hardware command was run.
- No tinygrad or libusb path was imported or executed.
- No validation command was run by this agent; supervisor owns the RED verification run.

## Supervisor verification

- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v` exited `1` with both tests failing at `compile_probe(tmp_path)` on `AssertionError: native transfer probe source missing`.
- `C0BRedReviewer` verdict: accept; no Critical, Important, or Minor findings.

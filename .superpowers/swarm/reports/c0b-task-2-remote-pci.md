# C0B task set 2: RemoteCmd transport self-tests

## Status

Done. Supervisor verified the focused pytest passes and reviewer accepted the RemoteCmd no-hardware implementation with no findings.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- `.superpowers/swarm/reports/c0b-task-2-remote-pci.md`

## Source provenance

- `native_amdev_transfer_probe.cpp` carries an MIT/provenance header for the tinygrad-derived TinyGPU Remote PCI client ABI mechanics.
- The ported/rederived mechanics are limited to `RemoteCmd` ids and the request frame equivalent to `struct.pack('<BIIQQQ', cmd, dev_id, bar, arg0, arg1, arg2)` from `tinygrad/runtime/support/system.py` lines 302-303 and 367-376.
- No broad tinygrad subsystem, model code, Python runtime, or dynamic tinygrad dependency was added.

## Implemented no-hardware contracts

- `--self-test remote-cmd-frame` validates RemoteCmd ordering, MAP_SYSMEM_FD command id, dev_id `0x7551`, BAR `5`, little-endian args, exact 33-byte frame size, and frame hex before printing `status: pass`.
- `--self-test log-contract` prints every required log field as `required_log_field: <field>` and then `status: pass`.
- `--help` lists both self-test names.
- Unknown self-test names return nonzero and print `failure_text` plus `exit_status`.

## Guardrails observed

- No TinyGPU.app hardware connection was attempted.
- No socket, BAR mapping, VM mapping, SDMA queue, libusb path, tinygrad import/call, model code, or global ledger edit was added.
- No validation commands, tests, linters, formatters, package managers, or project-wide suites were run by this agent.
- `.superpowers/swarm/progress.md` was not edited.

## Supervisor command

Run from `<former-native-r9700-worktree>`:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

## Expected result after task set 2

```text
2 passed
```

## Supervisor verification

- `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `2 passed in 0.68s`.
- Forbidden runtime path check found only tinygrad provenance/license comments; no `libusb`, `USBIface`, hardware socket, or shell-out path is implemented.
- `C0BRemoteCmdReviewer` verdict: accept; no Critical, Important, or Minor findings.

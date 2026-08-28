# C0B task set 3: TinyGPU discovery smoke

## Status

Done. Implemented and verified the native TinyGPU.app discovery-smoke path; reviewer accepted the discovery-only implementation after fixes with no remaining findings.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `docs/tasks/native-r9700-producer/validation-commands.md`
- `docs/archive/tasks/native-r9700-producer/phase-c0b-native-amdev-sdma-transfer.md`
- `.superpowers/swarm/reports/c0b-task-3-discovery.md`

`test_help_lists_hardware_modes()` was added to require `--discovery-smoke` and `--transfer-proof` help coverage before the hardware mode implementation.

## Implemented behavior

- `--help` now declares `--discovery-smoke` and `--transfer-proof` alongside existing self-tests.
- `--transfer-proof` is a nonzero placeholder only:
  - `failure_stage: transfer-proof`
  - `failure_text: not_implemented_transfer_proof`
  - `exit_status: 1`
- `--discovery-smoke` uses `APL_REMOTE_SOCK` when set, otherwise `$TMPDIR/tinygpu.sock` or `/tmp/tinygpu.sock`.
- The probe attempts a UNIX socket connection first; only after connect failure does it launch `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU server <sock>` and retry with a bounded 100 x 50 ms loop.
- RemoteCmd response decoding handles the 17-byte `<BQQ` response header, nonzero status error payloads, response header hex reporting, config readout, BAR mapping, MMIO readout bytes, and `SO_NOSIGPIPE`-protected send failure reporting.
- Discovery smoke follows the local `APLRemotePCIDevice` path: connect to TinyGPU.app, validate `1002:7551` with `CFG_READ`, map BAR0/BAR2/BAR5 with `MAP_BAR`, read `RCC_CONFIG_MEMSIZE` through BAR5 MMIO, and attempt large-BAR IP discovery parsing for `arch`.
- Discovery logs include `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, config header/vendor/device, BAR0/BAR2/BAR5 sizes, `host_device_transfer_status: not_run`, `failure_stage`, `failure_text`, and `exit_status`. It does not print any transfer success field.

## Supervisor commands

Focused no-hardware contract:

```sh
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Discovery smoke build/run/log command:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-discovery-smoke.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --discovery-smoke; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Supervisor evidence

- Focused pytest after review fixes: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `3 passed in 1.59s`.
- First hardware run exposed the root cause: RemoteCmd `PROBE` on the local TinyGPU.app UNIX socket returned status `1` with no error payload. The tinygrad reference selects `APLRemotePCIDevice` through IOKit before socket use, then uses direct `CFG_READ`/`MAP_BAR` on dev id `0`; the native smoke now follows that path.
- Discovery command after review fixes wrote `logs/c0b-discovery-smoke.log` and exited `0`: `pci_id: 1002:7551`, `config_vendor_id: 0x1002`, `config_device_id: 0x7551`, BAR0 `268435456`, BAR2 `2097152`, BAR5 `524288`, `vram_size_bytes: 34208743424`, `host_device_transfer_status: not_run`, `failure_stage: none`, `exit_status: 0`.
- `arch` remains `not_discovered` with `arch_discovery_status: skipped_small_bar_requires_indirect_vram_read`; task set 4 owns sysmem/VM work needed for the indirect VRAM read path.

## Review fixes

- Required VRAM-size evidence: BAR5 `RCC_CONFIG_MEMSIZE` failure now exits nonzero with `failure_stage: vram-size`; optional IP-table arch parsing still records `arch_discovery_status` without failing the discovery gate.
- SIGPIPE suppression: sockets now set `SO_NOSIGPIPE` when available so closed TinyGPU.app connections flow through structured `failure_stage`/`failure_text` handling.
- Report changed-file list now includes `tests/test_native_amdev_transfer_contract.py` and the task-local help test.
- `C0BDiscoveryReReviewer` accepted the review fixes with no Critical, Important, or Minor findings.

## Guardrails observed

- No validation commands, tests, linters, formatters, package managers, git commands, project-wide suites, or hardware commands were run by this agent.
- No VM/sysmem mapping, SDMA transfer, kernel dispatch, model code, C1/C2/C3 work, or global ledger edit was added.
- No libusb, `USBIface`, `USB3.list_devices`, runtime tinygrad import/call, or shell-out to tinygrad was added.
- `.superpowers/swarm/progress.md` was not edited.

## Remaining scope

Task set 3 produced precise discovery evidence and did not claim transfer success. Task set 4 remains responsible for sysmem mapping, VM/page-table setup, and indirect VRAM reads needed before SDMA transfer proof.

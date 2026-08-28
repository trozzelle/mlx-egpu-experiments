# C0B-4 VM/sysmem mapping port

## Status

Done. Supervisor verified focused no-hardware tests and a hardware `--transfer-proof` smoke after review fixes; re-review accepted with no remaining findings. MAP_SYSMEM_FD staging/readback page lists are logged, CPU-visible mappings/fds stay live through the transfer-proof scaffold, and the command fails closed at `failure_stage: vm_mapping` before SDMA.

## Changed files

- `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
- `tests/test_native_amdev_transfer_contract.py`
- `.superpowers/swarm/reports/c0b-task-4-vm-sysmem.md`

## Implemented behavior

- Added no-hardware `--self-test sysmem-page-list` coverage with deterministic output.
- Added synthetic MAP_SYSMEM_FD page-list parsing for little-endian `(uint64 paddr, uint64 size)` pairs ending in `(0, 0)`.
- Expands page-list segments at 4 KiB granularity, preserves order, reports pre-truncation expansion count, and truncates to the requested page count.
- Added narrow MAP_SYSMEM_FD client support using the existing TinyGPU.app/APLRemotePCIDevice RemoteCmd framing plus `recvmsg`/SCM_RIGHTS fd receipt.
- Added fixed transfer-proof VM/sysmem scaffolding for:
  - staging sysmem buffer at fixed GPU VA `0x200000000000`,
  - VRAM role at fixed GPU VA `0x200000001000`,
  - readback sysmem buffer at fixed GPU VA `0x200000002000`.
- `--transfer-proof` now keeps the C0B-3 CFG_READ/BAR/discovery path, requests staging/readback sysmem buffers, parses their physical page lists, reports the fixed VM roles, and fails closed at `failure_stage: vm_mapping` once PTE programming/TLB flush prerequisites remain unavailable.
- Page-list parser rejects malformed terminators where size is zero but physical address is nonzero; a valid terminator is exactly `(0,0)`.

## Guardrails preserved

- No SDMA packets, compute queues, kernel dispatch, model code, or transfer-success claim were added.
- Source remains tinygrad-free at runtime: no tinygrad import/call/shell-out/vendor path was added.
- No libusb/USB3/USBIface acceptance path was added.
- No generalized allocator framework was added; the code is fixed to the transfer-proof staging/VRAM/readback roles.
- Existing TinyGPU.app/APLRemotePCIDevice/PCIIface and CFG_READ discovery path are retained.

## Known blocker / risk

The hardware path intentionally stops at VM mapping. The supervisor smoke observed MAP_SYSMEM_FD succeeding for staging/readback after fixing the RemoteCmd frame to use `bar=0`, `arg0=size`, `arg1=contiguous`, with 16 KiB mappings and one requested page each: staging `0x0000000080000000`, readback `0x0000000080008000`. C0B-4 then reports `failure_stage: vm_mapping` because AM page-table PTE programming and GC/MM TLB flush plumbing are not implemented in this task set.

## Supervisor evidence

- Focused pytest after review fixes: `${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v` -> `4 passed in 2.26s`.
- VM/sysmem smoke: `xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof` wrote `logs/c0b-native-amdev-vm-sysmem-smoke.log` and exited `1` with wrapper exit `1`, `pci_id: 1002:7551`, `transfer_byte_count: 32`, fixed staging/VRAM/readback GPU VA roles, MAP_SYSMEM_FD page lists for staging/readback, `host_device_transfer_status: fail`, `failure_stage: vm_mapping`, and no transfer success claim.
- `git diff --check ...` over the C0B-4 source/test/docs/report set -> no output.

## Review fixes

- `C0BVMSysmemReviewer` rejected the first pass because MAP_SYSMEM_FD used the wrong RemoteCmd fields and the mmap/fd lifetime ended before transfer-proof code could use CPU-visible buffers.
- Fixed MAP_SYSMEM_FD framing to match tinygrad's `_rpc(sock, dev_id, MAP_SYSMEM_FD, size, int(contiguous), has_fd=True)`: `bar=0`, `arg0=size`, `arg1=contiguous`.
- Added `SysmemMapping` RAII ownership so staging/readback mmap views and fds remain live until `run_transfer_proof_scaffold()` returns.
- `C0BVMSysmemReReviewer` accepted the MAP_SYSMEM_FD framing and fd/mmap lifetime fixes with no Critical, Important, or Minor findings.

## Supervisor validation commands to run later

Focused no-hardware contract:

```sh
cd <former-native-r9700-worktree> && ${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Focused hardware VM/sysmem smoke, to capture `failure_stage: vm_mapping` or the next precise blocker:

```sh
cd <former-native-r9700-worktree> && xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o /tmp/native_amdev_transfer_probe_c0b4 && /tmp/native_amdev_transfer_probe_c0b4 --transfer-proof | tee logs/c0b-native-amdev-vm-sysmem-smoke.log
```

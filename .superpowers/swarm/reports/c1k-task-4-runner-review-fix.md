# C1 Task Set 4 — Runner Runtime Review-Fix Report (C1RunnerFix2)

**Review findings source:** reviewer agent `C1RunnerReviewer` (task set 4 review). Report
path `.superpowers/swarm/reports/c1k-task-4-review.md` does NOT exist — findings were read
from the reviewer agent output only.

**Scope:** Lane A fix of the reviewer CHANGES_REQUIRED findings for the native runner
runtime shell in `native_r9700/runtime.cpp` + `runtime.h`, the contract tests
`tests/native_r9700/test_runtime_contract.py`, and the report file only. No edits to the
frozen C0 probe, `docs/adr/*`, `docs/ROADMAP.md`, `phase-c1-native-producer-parity.md`,
loader files, `config.py`, or `docs/tasks/native-r9700-producer/validation-commands.md`
(the latter states no dword-count numbers, so no correction was needed).

---

## Finding [Important 1] — SDMA copy+fence packet encoding diverged from C0

**Verdict:** Valid. The shell's `build_sdma_copy_words` emitted header
`(kSdmaOpCopy<<24)|kSdmaSubopCopyLinear` (= `0x01000000`), count `((byte_count/4)-1)<<1`,
reordered `[header,count,src_lo,src_hi,dst_lo,dst_hi,0]`, and fence
`[kSdmaFenceHeader,fence_value,0,0]`. This diverged from the frozen C0 encoding.

**Fix applied** (`native_r9700/runtime.cpp`):
- Added private helpers ported byte-for-byte from the C0 probe:
  - `build_sdma_linear_copy_packet` (probe L855-866): emits
    `[kSdmaOpCopy | (kSdmaSubopCopyLinear<<8), byte_count-1U, 0U, src_lo, src_hi, dst_lo, dst_hi]`
    (7 dwords). Header = `1 | (0<<8)` = **`0x000001`**.
  - `build_sdma_fence_packet` (probe L868-875): emits `[kFenceHeader, fence_va_lo,
    fence_va_hi, value]` (4 dwords), `kFenceHeader = 0x00030005U`.
- `build_sdma_copy_words` now emits **copy packet + fence together** (C0
  `build_sdma_copy_submit_words`, probe L878-886), dword count = `7 + 4 = 11`.
  The `fence_value` parameter is retained as the explicit fence value (C0 uses
  `am_sdma::kFenceValue = 1U` internally). The shell's builder takes no `fence_va`
  parameter, so the fence VA is wired to `0U` (layout byte-faithful; only the VA is
  placeholder in this harness-free shell).

## Finding [Important 2] — PM4 dispatch invented an 11-dword standalone dispatch

**Verdict:** Valid. `build_pm4_dispatch_words` emitted a single `0x40000040`
`PACKET3_DISPATCH_DIRECT` (11 dwords), replacing the proven C0 59-dword / 12-packet
sequence.

**Fix applied** (`native_r9700/runtime.cpp`): ported the full C0
`build_compute_dispatch_words` (probe L622-660):
- `pm4_packet3(opcode, count) = (3<<30) | ((opcode&0xff)<<8) | ((count&0x3fff)<<16)`.
- `append_pm4_packet3(words, opcode, payload)` pushes header then payload.
- Exact C0 packet order (`kPm4DispatchPacketOrder`): acquire_mem, set_sh_pgm,
  set_sh_rsrc, set_sh_rsrc3, set_sh_tmpring, set_sh_restart, set_sh_userdata,
  set_sh_resource_limits, set_sh_start, dispatch_direct, event_write, release_mem.
- Ported constants (probe L336-395, 571-605, 464-469):
  - Opcodes: `kPacket3AcquireMem=0x58U`, `kPacket3SetShReg=0x76U`,
    `kPacket3DispatchDirect=0x15U`, `kPacket3EventWrite=0x46U`,
    `kPacket3ReleaseMem=0x49U`, `kPacketType3=3U`.
  - SET_SH offsets: `kComputeStartXSetShOffset=0x204U`,
    `kComputePgmLoSetShOffset=0x20cU`, `kComputePgmRsrc1SetShOffset=0x212U`,
    `kComputeResourceLimitsSetShOffset=0x215U`, `kComputeTmpringSizeSetShOffset=0x218U`,
    `kComputeRestartXSetShOffset=0x21bU`, `kComputePgmRsrc3SetShOffset=0x228U`,
    `kComputeUserData0SetShOffset=0x240U`. The C0 segment base `kComputeSetShBase=0x00002c00U`
    is documented in a comment (the shell emits absolute offsets directly, so it is NOT
    retained as a file-scope symbol). `kPm4DispatchPacketCount=12U` is likewise not retained
    as an unused symbol — only the live `kPm4DispatchDwordCount=59U` is kept, to keep the
    build warning-free.
  - Rsrc granules: `kKernelReferenceRsrc1=0xc00c0040U`, `kKernelReferenceRsrc2=0x00000084U`,
    `kKernelReferenceRsrc3=0x00000010U`.
  - Dispatch dims: `kDispatchGlobalSizeX/Y/Z = 1U`, `kDispatchLocalSizeX=8U`,
    `Y=Z=1U` (already present in `runtime.h`, now consumed by the dispatch packet).
  - Encoders: `encode_dispatch_initiator() = (1<<0)|(1<<2)`,
    `encode_acquire_mem_gcr_cntl_for_dispatch() = GLM_WB|GLM_INV|GLK_WB|GLK_INV|GLV_INV|GL1_INV`
    (shifts 4..9), `encode_event_write_cs_partial_flush() = kEventTypeCsPartialFlush(7U)
    | (kEventIndexPartialFlush(4U)<<8)`,
    `encode_release_mem_event() = kEventTypeCacheFlushAndInvTs(20U) |
    (kReleaseMemEventIndexEndOfPipe(5U)<<8) | (1U<<12)|(1U<<13)|(1U<<14)|(1U<<15)|(1U<<20)|(1U<<21)|(1U<<22)`,
    `encode_release_mem_data_sel() = (kReleaseMemDataSelSend32BitLow(1U)<<29) |
    (kReleaseMemIntSelNone(0U)<<24)`.
  - `kReleaseMemTimelineValue = 1U`.
- Result = **59 dwords / 12 packets** (verified by packet-size count: 8+4+4+3+3+5+4+3+10+5+2+8 = 59).
  The `diry-run` dispatch word-count check in `dispatch_and_poll`
  (`dispatch_words.size() != build_pm4_dispatch_words(0,0,0).size()`) now correctly
  requires 59.

**First PM4 dword locked:** `pm4_packet3(kPacket3AcquireMem, 6U)` =
`(3<<30)|((0x58&0xff)<<8)|((6&0x3fff)<<16)` = `0xC0000000|0x5800|0x60000` = **`0xc0065800`**.

## Finding [Important 3] — Hardware stubs claim unported mechanics; dead RAII plumbing

**Verdict:** Valid. `socket_fd_`, `staging_map_`, `readback_map_`, `staging_size_`,
`readback_size_` were declared in `runtime.h` but never populated; `cleanup()` ran
munmap/close on them; stage comments claimed TinyGPU/RPC/SDMA/compute mechanics.

**Fix applied:**
- Removed the never-populated RAII members from `runtime.h`.
- Removed the munmap/close teardown from `RuntimeSession::cleanup` (runtime.cpp);
  `cleanup` now only advances the lifecycle state machine.
- Rewrote comments in `runtime.h` and `runtime.cpp` (file header + lifecycle doc +
  `initialize`/`allocate_buffers`/`copy_input`/`load_kernel`/`dispatch_and_poll`/
  `readback_and_compare`) to state plainly that the hardware stages are lifecycle
  state-machine advances that record their intended effect in the log, and that real
  TinyGPU socket connect, BAR mapping, SDMA submit, and compute doorbell/timeline
  wiring are **deferred gates for C1 task sets 5-8**, NOT implemented here.
- `allocate_buffers` no longer writes `staging_size_`/`readback_size_` (removed);
  it records host-device transfer intent and advances the state machine.
- Removed the now-unused `kPageSize` constant.

## Finding [Minor 4] — dead `kSdmaFenceValue = 1U`

**Verdict:** Valid (the only build warning). Removed `kSdmaFenceValue` from runtime.cpp;
`build_sdma_copy_words` keeps its explicit `fence_value` parameter (C0's fence value is
passed by the caller, defaulting to `1U` as the dry-run calls it with `1`).

---

## Test updates (`tests/native_r9700/test_runtime_contract.py`)

- `PM4_DISPATCH_DWORD_COUNT` changed `"11"` → `"59"`.
- `SDMA_COPY_DWORD_COUNT` unchanged (`"11"` — the VALUE is retained, but the byte layout
  is now the C0 form).
- `KERNARG_*` constants unchanged (frozen kernarg layout contract).
- `dispatch_global_size_x: 1` / `dispatch_local_size_x: 8` assertions kept.
- Added two new dry-run-emitted fields and assertions to lock the byte-faithful ports:
  - `SDMA_COPY_HEADER_HEX = "00000001"` — asserts `sdma_copy_header_hex` (first SDMA copy
    dword, the C0 `0x000001` header), emitted by `dry_run` as an 8-hex-digit numeric string.
  - `PM4_DISPATCH_FIRST_DWORD_HEX = "c0065800"` — asserts `pm4_dispatch_first_dword_hex`
    (first PM4 dword = `pm4_packet3(kPacket3AcquireMem, 6U)`), emitted by `dry_run`.
- `dry_run` (runtime.cpp) now emits `sdma_copy_header_hex` and
  `pm4_dispatch_first_dword_hex` in addition to the existing `sdma_copy_dword_count`,
  `pm4_dispatch_dword_count`, and dispatch-dim lines.

## Files changed

- `native_r9700/runtime.cpp` — C0-faithful SDMA copy+fence and 59-dword PM4 dispatch;
  removed dead `kSdmaFenceValue`, `kPageSize`, and the unused `kPm4DispatchPacketCount` /
  `kComputeSetShBase` symbols; honest hardware-stub comments; removed
  munmap/close teardown; `dry_run` emits the two new header-hex fields.
- `native_r9700/runtime.h` — removed dead RAII members; honest lifecycle/header comments;
  updated PM4 builder doc.
- `tests/native_r9700/test_runtime_contract.py` — PM4 count 59 + new header-hex locks.
- `docs/tasks/native-r9700-producer/validation-commands.md` — **unchanged** (no incorrect
  dword counts present).
- No source edit to the C0 probe, `docs/adr/*`, `ROADMAP`, or `phase-c1-native-producer-parity.md`.

---

## Commands the SUPERVISOR must run to verify (I did NOT run them)

```bash
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer

# 1. Build the runner (warning-free; catches the SDMA/PM4 port + removed members).
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o /tmp/native_r9700_runner

# 2. Focused contract tests (PM4=59, SDMA header 0x000001, PM4 first dword 0xc0065800).
${HOME}/.pyenv/versions/3.12.8/bin/python3 \
  -m pytest tests/native_r9700/test_runtime_contract.py -q

# 3. Dry-run smoke (should print the new header-hex fields).
/tmp/native_r9700_runner --lifecycle-dry-run | grep -E \
  'sdma_copy_dword_count|pm4_dispatch_dword_count|sdma_copy_header_hex|pm4_dispatch_first_dword_hex'

# 4. C0 regression (frozen probe untouched — confirm it still compiles/runs as before).
#    e.g. the probe's own build/test target per the C0 validation docs.

# 5. git diff --check (whitespace)
git diff --check
```

**Expected focused-test outcome:** all `tests/native_r9700/test_runtime_contract.py` tests
pass, including the extended `test_dry_run_reports_packet_encodings` asserting
`sdma_copy_dword_count: 11`, `pm4_dispatch_dword_count: 59`,
`sdma_copy_header_hex: 00000001`, `pm4_dispatch_first_dword_hex: c0065800`,
`dispatch_global_size_x: 1`, and `dispatch_local_size_x: 8`.

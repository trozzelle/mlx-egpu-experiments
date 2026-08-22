# C1 Task Set 4 — Runner Runtime Re-Review (C1RunnerRereview)

**Decision: APPROVE**

Re-review of the Lane A fix by `C1RunnerFix2` for reviewer `C1RunnerReviewer`'s
CHANGES_REQUIRED findings. All four findings are correctly and completely
resolved, with no new Critical or Important issues introduced. Byte-for-byte
fidelity to the frozen C0 probe was verified by direct comparison of the shell
builders against `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
(reference, read-only). No code or test file was edited by this review; this
report is the only write.

---

## Finding [Important 1] — SDMA copy+fence byte divergence: RESOLVED

Verified `native_r9700/runtime.cpp`:
- `build_sdma_linear_copy_packet` (L192-198) emits the C0 layout
  `[kSdmaOpCopy | (kSdmaSubopCopyLinear<<8), byte_count-1U, 0U, src_lo, src_hi, dst_lo, dst_hi]`
  (7 dwords), header = `1 | (0<<8)` = `0x000001`. Matches probe L855-866 exactly.
- The divergent `(kSdmaOpCopy<<24)` header and `((byte_count/4)-1)<<1` count are
  NOT present — field 2 is the literal `byte_count - 1U`, field 3 is `0U`, and
  the src/dst low/high order matches the probe.
- `build_sdma_fence_packet` (L203-206) emits `[kFenceHeader=0x00030005, fence_va_lo,
  fence_va_hi, value]` (4 dwords), matching probe L868-875.
- `build_sdma_copy_words` (L245-255) emits copy packet + fence together
  (11 dwords total), matching probe `build_sdma_copy_submit_words` L878-886.
- Fence VA is wired to `0U` (placeholder) in this harness-free shell; the header
  documents `build_sdma_copy_words` takes src/dst/byte_count/fence_value but no
  fence_va, and the fix report documents the placeholder explicitly. This matches
  the task's acceptance ("documented as placeholder 0U in this harness-free shell").

## Finding [Important 2] — PM4 dispatch must be C0's 59/12 sequence: RESOLVED

Verified `build_pm4_dispatch_words` (L258-293) is byte-for-byte identical to probe
`build_compute_dispatch_words` L623-660:
- Opcodes: kPacket3AcquireMem=0x58, kPacket3SetShReg=0x76, kPacket3DispatchDirect=0x15,
  kPacket3EventWrite=0x46, kPacket3ReleaseMem=0x49 — all match probe L366-374.
- SET_SH offsets: 0x204, 0x20c, 0x212, 0x228, 0x218, 0x21b, 0x240, 0x215 — all eight
  match probe L338-345 (start/pgmlo/rsrc1/rsrc3/tmpring/restart/userdata/resource_limits).
- Rsrc granules: 0xc00c0040 / 0x00000084 / 0x00000010 — match probe L171-173.
- Payload shapes and packet order for all 12 packets match exactly; summed
  packet sizes 8+4+4+3+3+5+4+3+10+5+2+8 = **59 dwords** over **12 packets**.
- Encoders (`encode_dispatch_initiator`, `encode_acquire_mem_gcr_cntl_for_dispatch`,
  `encode_event_write_cs_partial_flush`, `encode_release_mem_event`,
  `encode_release_mem_data_sel`) match probe L571-605 bit-for-bit.
- First dword: `pm4_packet3(kPacket3AcquireMem=0x58, count=6)` =
  `(3<<30)|(0x58<<8)|(6<<16)` = `0xc0065800` — confirmed.
- `dispatch_and_poll` word-count check (L483) requires
  `dispatch_words.size() == build_pm4_dispatch_words(0,0,0).size()` (=59). Confirmed.

## Finding [Important 3] — Hardware stubs honest: RESOLVED

- `native_r9700/runtime.h`: the never-populated RAII members `socket_fd_`,
  `staging_map_`, `readback_map_`, `staging_size_`, `readback_size_` are GONE; the
  `RuntimeSession` private section holds only `log_` and `stage_`.
- `native_r9700/runtime.cpp::cleanup()` (L475-482) no longer runs munmap/close
  teardown; it only advances the lifecycle state and comments the lack of host
  resources explicitly. The only `close(fd)` in the file is the legitimate
  log-file descriptor write in `write_run_log` (L368).
- Header and source comments now plainly state that the hardware stages are
  lifecycle state-machine advances recording intended effect, and that real
  TinyGPU socket connect, BAR mapping, SDMA submit, and compute doorbell/timeline
  wiring are deferred gates for C1 task sets 5-8 — no false claims of ported
  mechanics. `initialize`/`allocate_buffers`/`copy_input`/`load_kernel`/
  `dispatch_and_poll`/`readback_and_compare` each carry an honest deferred-gate
  comment.

## Finding [Minor 4] — dead `kSdmaFenceValue`: RESOLVED

- `kSdmaFenceValue = 1U` is removed with no residue; `build_sdma_copy_words` keeps
  its explicit `fence_value` parameter (dry-run passes `1`).

---

## Acceptance confirmations

- `dry_run` (L593-609) emits `sdma_copy_dword_count: 11`,
  `pm4_dispatch_dword_count: 59`, `sdma_copy_header_hex: 00000001` (from `sdma[0]`),
  `pm4_dispatch_first_dword_hex: c0065800` (from `dispatch[0]`),
  `dispatch_global_size_x: 1`, `dispatch_local_size_x: 8`. Confirmed.
- `tests/native_r9700/test_runtime_contract.py`:
  `test_dry_run_reports_packet_encodings` asserts all four encodings
  (`sdma_copy_dword_count: 11`, `pm4_dispatch_dword_count: 59`,
  `sdma_copy_header_hex: 00000001`, `pm4_dispatch_first_dword_hex: c0065800`) plus
  the dispatch dims; `PM4_DISPATCH_DWORD_COUNT = "59"`. Confirmed.
- `dispatch_and_poll` uses `build_pm4_dispatch_words(0,0,0).size()` (=59) as the
  word-count gate. Confirmed.
- No unused `constexpr` remains in the shell that would trip `-Wall -Wextra`: every
  constant in the anonymous namespace (SDMA op/subop/fence header/dword counts,
  all PM4 opcodes, event/GCR constants, all eight SET_SH offsets, rsrc granules,
  `kPm4DispatchDwordCount`, `kPacketType3`, `kReleaseMemTimelineValue`) is
  referenced by the builders/encoders. Dead `kSdmaFenceValue`, `kPageSize`, and
  `kPm4DispatchPacketCount`/`kComputeSetShBase` are not defined. Confirmed.
- No edits to the C0 probe (all reference builders verified intact), `docs/adr/*`,
  `ROADMAP`, or the phase doc. Confirmed from the fix report and current file state.

---

## New findings

None. No Critical/Important/Moderate issues introduced by the patch.

## Verification note

Per the session constraints I did not run git, builds, tests, or hardware.
Encodings were verified by direct source comparison against the frozen C0 probe.
The supervisor has separately confirmed the warning-free build, 27 focused tests,
and dry-run header values (PM4=59, SDMA header 0x000001, PM4 first dword 0xc0065800).

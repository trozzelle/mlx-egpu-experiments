# Task 3 Report — Sysmem Ring Backing: write_compute_ring_words -> sysmem

## Scope
File: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
Branch: `feature/native-r9700-producer` (base 1273bc2, Tasks 1-2 present)
Worktree: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`

## Change
Rewrote `write_compute_ring_words` to write the PM4 dispatch into the sysmem
`compute_control` ring span (at `kComputeControlRingCpuOffset`) instead of BAR0
VRAM (`kRingVramPaddr`), mirroring `write_sdma_ring_words`. Updated the single
caller in `submit_compute_dispatch`.

## Source lines edited

### 1. `write_compute_ring_words` — replaced lines 5344-5386 (old) with new body at lines 5340-5382
Old signature (pre-edit):
```cpp
bool write_compute_ring_words(const RemoteClient& client, const DiscoveryLog& log,
                              const std::vector<uint32_t>& words, std::string* error_text)
```
New signature:
```cpp
bool write_compute_ring_words(SysmemMapping* compute_control_mapping,
                              const std::vector<uint32_t>& words, std::string* error_text)
```

### 2. New function body (exact)
```cpp
bool write_compute_ring_words(SysmemMapping* compute_control_mapping,
                              const std::vector<uint32_t>& words, std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr) {
    *error_text = "compute ring mapping precondition failed: null SysmemMapping";
    return false;
  }
  if (words.empty()) {
    *error_text = "compute dispatch has no PM4 packet words";
    return false;
  }
  const uint64_t ring_bytes = static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (ring_bytes == 0 || ring_bytes > am_compute::kRingSize) {
    *error_text = "compute dispatch PM4 byte count exceeds compute ring size: byte_count=" +
                  std::to_string(ring_bytes) +
                  " ring_size=" + std::to_string(am_compute::kRingSize);
    return false;
  }
  const uint64_t start = am_compute::kComputeControlRingCpuOffset;
  if (start > compute_control_mapping->size ||
      ring_bytes > compute_control_mapping->size - start) {
    *error_text = "compute ring write exceeds sysmem ring span: mapped_size=" +
                  std::to_string(compute_control_mapping->size) + " ring_start=" +
                  std::to_string(start) + " ring_write_bytes=" + std::to_string(ring_bytes);
    return false;
  }
  const std::vector<uint8_t> bytes = u32_words_payload_le(words);
  std::memcpy(static_cast<uint8_t*>(compute_control_mapping->data) + start, bytes.data(),
              bytes.size());
  return true;
}
```

### 3. Updated caller in `submit_compute_dispatch` — line 5384
Before:
```cpp
if (!write_compute_ring_words(client, *log, words, error_text)) {
```
After:
```cpp
if (!write_compute_ring_words(compute_control_mapping, words, error_text)) {
```
All later submission steps unchanged: `flush_hdp(client, *log, error_text)` (line 5387),
probe_pre snapshot, `write_compute_control_u64(...kWptrOffset...)`, `std::atomic_thread_fence`,
and MEC doorbell `mmio_write_fire_and_forget(2, kMecDoorbellBar2ByteOffset, ...)`.

## Key facts
- `am_compute::kComputeControlRingCpuOffset = 2*kPageSize = 8192` (8-page ring span, pages 2..9).
- `am_compute::kRingSize = 0x8000 = 32768` bytes.
- `kComputeControlByteCount = 10*kPageSize = 40960` bytes => mapping size; ring span + payload fit.
- Dispatch is `kPm4DispatchDwordCount` dwords (~236 bytes).
- No `mmio_read`/`mmio_write_bar0` readback in the new path.

## Forbidden-work compliance
- No changes to BAR2/GDC/S2A/CP MEC range, PM4 packet sequence, scheduler/retry/AQL,
  fallback/allocator, runtime/C1-C3, ring VA/MQD ring addr/ring size, or VM indices.
- `flush_hdp` retained (line 5387). No other function modified.
- Kept `encode_hqd_pq_control_direct_pm4`, Task 1 constants, Task 2 PTE remap untouched.
- Did NOT run git, build, tests, linters, formatters.

## Deviations
None. Followed the spec exactly (signature, error strings, span bounds check, memcpy,
no readback).

## Supervisor verification commands (not run here)
```bash
# Build
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
./configure.sh && ninja -C build -j8 native_amdev_transfer_probe

# Focused pytest (compute ring backing)
python -m pytest tests/test_compute_ring_sysmem.py -x -q
```

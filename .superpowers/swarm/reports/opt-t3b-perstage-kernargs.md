# Task 3b — Per-stage kernargs ring (K=16)

Critical kernargs race fixed: each in-flight batched-dispatch stage now binds its
kernargs bytes to its own 4 KiB page (a 16-page kernargs ring), instead of every
stage overwriting the single fixed `kKernargsVa` page. Only *where* kernargs
bytes live and which VA each dispatch references changed; kernel source,
dispatch geometry, kernarg layout, buffer sizes, weight spans, and dispatch
order are untouched.

## Fixed VM layout after change (PDB0 index 0, first 2 MiB)

- Pages 0..9  — existing 10-page sysmem compute-control span (control + kernargs + 8 ring pages), CPU offsets 0..9.
- Pages 10..25 — NEW 16-page kernargs ring, mapped to GPU VAs at pages 17..32.
- Page 15 = rptr / page 6 = legacy fixed kernargs / pages 7..14 = ring (unchanged).
- Page 16 = EOP, page 17 = start of the kernargs ring (unchanged EOP VA).

## `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`

- **`am_compute::kComputeControlByteCount`** (line ~341): `10ULL * kPageSize` → `26ULL * kPageSize`; comment now `2 control + 8 ring + 16 kernargs ring`.
- **New constants** (after `kEopVa`, lines ~319-323):
  - `constexpr uint64_t kKernargsRingVa = am_vm::kVaBase + (17ULL * kPageSize);`
  - `constexpr uint32_t kKernargsRingPageCount = 16U;`
  - `constexpr uint64_t kComputeControlKernargsRingCpuOffset = 10ULL * kPageSize;`
- **`static_assert`** (lines ~326-328): now asserts `kKernargsRingVa + kKernargsRingPageCount * kPageSize <= am_vm::kVaBase + (512ULL * kPageSize)` (replaces the old `kEopVa + kPageSize` bound; the kernargs ring end at page 33 is now the furthest fixed mapping).
- **`write_fixed_page_tables`**:
  - sys_pages precondition (line ~3395): `< 10` → `< 26`, with matching error text.
  - After the existing 8-page ring mapping loop (~3492), added a 16-page kernargs-ring loop:
    `for (uint64_t i = 0; i < am_compute::kKernargsRingPageCount; ++i) add_ptb_pte(am_compute::kKernargsRingVa + i * kPageSize, compute_control->sys_pages[am_compute::kComputeControlKernargsRingCpuOffset / kPageSize + i], sysmem_flags);`
- **`compute-vm-layout` self-test** (lines ~1509-1515): now asserts `kComputeControlByteCount == 26ULL * kPageSize`, `kComputeControlKernargsCpuOffset == kPageSize`, `kComputeControlRingCpuOffset == 2ULL * kPageSize`, `kComputeControlRingByteCount == 8ULL * kPageSize`, `kComputeControlKernargsRingCpuOffset == 10ULL * kPageSize`, `kKernargsRingPageCount == 16U`; failure text `"compute_control 26-page CPU layout mismatch"`.

## `native_r9700/amdev_session.cpp`

- **`bind_resident_kernel_kernargs`** (lines ~583-593): added trailing default param `uint64_t kernargs_cpu_offset = am_compute::kComputeControlKernargsCpuOffset`; the `kernargs` pointer now uses `kernargs_cpu_offset` instead of the hard-coded `kComputeControlKernargsCpuOffset`.
- **`Impl`** (lines ~1890-1894): added `uint32_t next_kernargs_slot = 0;` with a comment mirroring the timeline-value reset policy.
- **`build_stage_pm4`** (lines ~2291-2326):
  - `const uint32_t slot = state.next_kernargs_slot++;`
  - `const uint64_t kernargs_cpu_offset = am_compute::kComputeControlKernargsRingCpuOffset + slot * kPageSize;`
  - passes `kernargs_cpu_offset` as the 4th arg to `bind_resident_kernel_kernargs`.
  - `Pm4DispatchConfig` kernargs VA is now `am_compute::kKernargsRingVa + slot * kPageSize` (was fixed `kKernargsVa`); `pm4.timeline_value = state.next_timeline_value++;` unchanged.
- **`dispatch_batch`** (line ~2343): `state.next_kernargs_slot = 0;` immediately after `Impl& state = *impl_;`, so every batch starts at slot 0.
- **`prepare`** (line ~2241) and **`Impl::reset_after_close`** (line ~1961): added `state.next_kernargs_slot = 0;` / `next_kernargs_slot = 0;` next to the existing `next_timeline_value = 1;` resets.

## Behavior

- Each stage in a batch consumes slot `0..N-1`: distinct kernargs page (CPU offset 10..9+N pages) and distinct VA (page 17..16+N). N <= 16 for a single batch.
- Single-stage `dispatch` path calls `dispatch_batch({stage})` and therefore uses slot 0.
- Legacy proof paths (amdev_session.cpp ~1204/1549/1791) call `bind_resident_kernel_kernargs` with 3 args → default offset `kComputeControlKernargsCpuOffset` (fixed `kKernargsVa`), and still build their PM4 with fixed `am_compute::kKernargsVa` — unchanged.

## Supervisor verification commands (NOT run here)

```
# Full runner build (native binary + probe)
<runner build command>

PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/test_native_amdev_transfer_contract.py -q
$PY -m pytest tests/native_r9700/test_layer0_executor_contract.py -q
```

Note: the probe's `compute-vm-layout` self-test printf of `compute_control_requested_size`
now emits `106496` (26 pages) rather than `40960`; if any contract test asserts the
old literal, that expectation follows from this layout change.

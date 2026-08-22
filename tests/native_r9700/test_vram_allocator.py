"""No-hardware RED contracts for bounded R9700 physical VRAM allocation."""

from pathlib import Path
import subprocess


ALLOCATOR_HEADER = Path("native_r9700/vram_allocator.h")
ALLOCATOR_SOURCE = Path("native_r9700/vram_allocator.cpp")
LAYOUT_HEADER = Path("native_r9700/vram_layout.h")
LAYOUT_SOURCE = Path("native_r9700/vram_layout.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_allocator_probe(tmp_path: Path) -> Path:
    """Compile only pure VRAM ownership code; no TinyGPU device is contacted."""
    assert (
        ALLOCATOR_HEADER.is_file() and ALLOCATOR_SOURCE.is_file()
    ), "Vram allocator implementation is missing"
    assert LAYOUT_HEADER.is_file() and LAYOUT_SOURCE.is_file(), "Vram layout implementation is missing"

    probe_source = tmp_path / "vram_allocator_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

#include "vram_allocator.h"
#include "vram_layout.h"

namespace {

constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kMiB = 1ULL << 20;
constexpr uint64_t kBootReservedBytes = 32ULL * kMiB;
constexpr uint64_t kPageTableReservedBytes = 4ULL * kMiB;
constexpr uint64_t kOwnedBase = kBootReservedBytes + kPageTableReservedBytes;
constexpr uint64_t kTailReservedBytes = 64ULL * kMiB;
constexpr uint64_t kResidentGpuVaBase = 0x0000200000011000ULL;
constexpr uint64_t kResidentGpuVaLimit = 0x0000200000200000ULL;

bool require(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "%s\n", message);
  return condition;
}

native_r9700::VramLayout layout_with_owned_bytes(uint64_t owned_bytes) {
  native_r9700::VramLayout layout{};
  layout.vram_bytes = kOwnedBase + owned_bytes + kTailReservedBytes;
  layout.discovery_reserved_bytes = 64ULL * 1024;
  layout.boot_reserved_bytes = kBootReservedBytes;
  layout.page_table_reserved_bytes = kPageTableReservedBytes;
  layout.allocatable_base = kOwnedBase;
  layout.allocatable_bytes = owned_bytes;
  layout.c0_reserved_physical_ranges = {{{0, 0}, {0, 0}, {0, 0}}};
  layout.resident_gpu_va_base = kResidentGpuVaBase;
  layout.resident_gpu_va_limit = kResidentGpuVaLimit;
  return layout;
}

bool same_allocation(const native_r9700::VramAllocation& left,
                     const native_r9700::VramAllocation& right) {
  return left.physical_offset == right.physical_offset &&
         left.size_bytes == right.size_bytes && left.name == right.name;
}

bool owns_only_layout_range(const native_r9700::VramAllocation& allocation,
                            uint64_t owned_bytes) {
  const uint64_t owned_end = kOwnedBase + owned_bytes;
  return allocation.physical_offset >= kOwnedBase &&
         allocation.physical_offset <= owned_end &&
         allocation.size_bytes <= owned_end - allocation.physical_offset;
}

bool owned_range() {
  constexpr uint64_t kOwnedBytes = 32ULL * kPageBytes;
  native_r9700::VramAllocator allocator(layout_with_owned_bytes(kOwnedBytes));
  native_r9700::VramAllocation first{};
  native_r9700::VramAllocation second{};
  std::string error_text;

  if (!require(allocator.allocate("first", 1, kPageBytes, &first, &error_text),
               "a one-byte request must receive the 4 KiB minimum allocation") ||
      !require(allocator.allocate("second", 1, 64ULL * 1024, &second, &error_text),
               "a distinct request must allocate from the remaining owned range")) {
    return false;
  }

  return require(first.physical_offset == kOwnedBase,
                 "first-fit allocation must start at layout.allocatable_base") &&
         require(first.size_bytes == kPageBytes,
                 "allocation size must round up to the 4 KiB minimum") &&
         require(first.name == "first", "allocation must retain its caller name") &&
         require(second.physical_offset == kOwnedBase + 64ULL * 1024,
                 "alignment must advance within the ordered free range") &&
         require(second.size_bytes == kPageBytes && second.name == "second",
                 "second allocation must retain rounded size and caller name") &&
         require(first.physical_offset != second.physical_offset,
                 "live allocations must have distinct physical offsets") &&
         require(first.physical_offset % kPageBytes == 0 &&
                     second.physical_offset % (64ULL * 1024) == 0,
                 "each allocation must obey its requested alignment") &&
         require(owns_only_layout_range(first, kOwnedBytes) &&
                     owns_only_layout_range(second, kOwnedBytes),
                 "allocator must never return boot, page-table, or tail-reserved bytes") &&
         require(allocator.contains(first) && allocator.contains(second),
                 "contains must recognize live allocations");
}

bool unchanged_on_failed_allocate(native_r9700::VramAllocator* allocator,
                                  const char* name, uint64_t size_bytes,
                                  uint64_t alignment) {
  native_r9700::VramAllocation allocation{0xDEADBEEF, 0xC0FFEE, "unchanged"};
  const native_r9700::VramAllocation sentinel = allocation;
  std::string error_text;
  return !allocator->allocate(name, size_bytes, alignment, &allocation, &error_text) &&
         !error_text.empty() && same_allocation(allocation, sentinel);
}

bool rejected_requests() {
  native_r9700::VramAllocator allocator(layout_with_owned_bytes(8ULL * kPageBytes));
  native_r9700::VramAllocation live{};
  std::string error_text;
  if (!require(allocator.allocate("unique", kPageBytes, kPageBytes, &live, &error_text),
               "a valid request must establish a live allocation")) {
    return false;
  }

  return require(unchanged_on_failed_allocate(&allocator, "unique", kPageBytes, kPageBytes),
                 "duplicate names must fail without modifying output") &&
         require(unchanged_on_failed_allocate(&allocator, "", kPageBytes, kPageBytes),
                 "empty allocation names must fail without modifying output") &&
         require(unchanged_on_failed_allocate(&allocator, "zero", 0, kPageBytes),
                 "zero-byte allocations must fail without modifying output") &&
         require(unchanged_on_failed_allocate(&allocator, "small-align", kPageBytes,
                                              kPageBytes / 2),
                 "alignment below 4 KiB must fail without modifying output") &&
         require(unchanged_on_failed_allocate(&allocator, "non-power-two", kPageBytes,
                                              3ULL * kPageBytes),
                 "non-power-of-two alignment must fail without modifying output") &&
         require(unchanged_on_failed_allocate(&allocator, "large-align", kPageBytes,
                                              4ULL * kMiB),
                 "alignment above 2 MiB must fail without modifying output");
}

bool exhausted_and_overflow() {
  native_r9700::VramAllocator allocator(layout_with_owned_bytes(3ULL * kPageBytes));
  native_r9700::VramAllocation first{};
  native_r9700::VramAllocation second{};
  std::string error_text;
  if (!require(allocator.allocate("first", 2ULL * kPageBytes, kPageBytes, &first, &error_text),
               "first allocation must consume two owned pages") ||
      !require(allocator.allocate("second", kPageBytes, kPageBytes, &second, &error_text),
               "second allocation must consume the final owned page")) {
    return false;
  }

  return require(unchanged_on_failed_allocate(&allocator, "exhausted", kPageBytes,
                                              kPageBytes),
                 "exhausted owned VRAM must fail without modifying output") &&
         require(unchanged_on_failed_allocate(&allocator, "overflow",
                                              std::numeric_limits<uint64_t>::max(),
                                              kPageBytes),
                 "overflowing request size must fail without modifying output");
}

bool invalid_transitions() {
  native_r9700::VramAllocator allocator(layout_with_owned_bytes(4ULL * kPageBytes));
  native_r9700::VramAllocation live{};
  std::string error_text;
  if (!require(allocator.allocate("live", kPageBytes, kPageBytes, &live, &error_text),
               "valid allocation must establish a release target")) {
    return false;
  }

  native_r9700::VramAllocation forged = live;
  forged.name = "forged";
  if (!require(!allocator.release(forged, &error_text) && !error_text.empty(),
               "metadata-mismatched release must fail") ||
      !require(allocator.contains(live) && !allocator.contains(forged),
               "forged release must not remove the live allocation")) {
    return false;
  }

  native_r9700::VramAllocation unowned = live;
  unowned.physical_offset += kPageBytes;
  unowned.name = "unowned";
  if (!require(!allocator.release(unowned, &error_text) && !error_text.empty(),
               "release of an unowned physical range must fail") ||
      !require(allocator.release(live, &error_text), "valid release must succeed") ||
      !require(!allocator.contains(live), "released allocation must no longer be live")) {
    return false;
  }

  return require(!allocator.release(live, &error_text) && !error_text.empty(),
                 "double free must fail");
}

bool adjacent_coalescing() {
  native_r9700::VramAllocator allocator(layout_with_owned_bytes(3ULL * kPageBytes));
  native_r9700::VramAllocation left{};
  native_r9700::VramAllocation middle{};
  native_r9700::VramAllocation guard{};
  native_r9700::VramAllocation merged{};
  std::string error_text;
  if (!require(allocator.allocate("left", kPageBytes, kPageBytes, &left, &error_text),
               "left page must allocate") ||
      !require(allocator.allocate("middle", kPageBytes, kPageBytes, &middle, &error_text),
               "middle page must allocate") ||
      !require(allocator.allocate("guard", kPageBytes, kPageBytes, &guard, &error_text),
               "guard page must prevent trailing free-range reuse") ||
      !require(allocator.release(left, &error_text), "left page must release") ||
      !require(allocator.release(middle, &error_text), "middle page must release") ||
      !require(allocator.allocate("merged", 2ULL * kPageBytes, kPageBytes, &merged,
                                  &error_text),
               "adjacent released pages must coalesce into a first-fit range")) {
    return false;
  }

  return require(merged.physical_offset == kOwnedBase &&
                     merged.size_bytes == 2ULL * kPageBytes,
                 "coalesced allocation must reuse the two adjacent first pages") &&
         require(allocator.contains(guard) && allocator.contains(merged),
                 "coalescing must preserve unrelated live allocations");
}

constexpr uint64_t kC0EarlyReservedEnd = 0x00003000ULL;
constexpr uint64_t kC0PageTableReservedBase = 0x02000000ULL;
constexpr uint64_t kC0PageTableReservedEnd = 0x02004000ULL;
constexpr uint64_t kC0MqdReservedBase = 0x06000000ULL;
constexpr uint64_t kC0MqdReservedEnd = 0x06010000ULL;
constexpr uint64_t kC0VramBytes = kC0MqdReservedEnd + kPageBytes;

native_r9700::VramLayout c0_layout() {
  native_r9700::VramLayout layout{};
  layout.vram_bytes = kC0VramBytes;
  layout.discovery_reserved_bytes = 0;
  layout.boot_reserved_bytes = 0;
  layout.page_table_reserved_bytes = 0;
  layout.allocatable_base = 0;
  layout.allocatable_bytes = kC0VramBytes;
  layout.c0_reserved_physical_ranges = {{
      {0, kC0EarlyReservedEnd},
      {kC0PageTableReservedBase,
       kC0PageTableReservedEnd - kC0PageTableReservedBase},
      {kC0MqdReservedBase, kC0MqdReservedEnd - kC0MqdReservedBase},
  }};
  layout.resident_gpu_va_base = kResidentGpuVaBase;
  layout.resident_gpu_va_limit = kResidentGpuVaLimit;
  return layout;
}

bool ranges_overlap(uint64_t left_base, uint64_t left_size, uint64_t right_base,
                    uint64_t right_size) {
  return left_base < right_base + right_size &&
         right_base < left_base + left_size;
}

bool avoids_c0_reservations(const native_r9700::VramAllocation& allocation) {
  return !ranges_overlap(allocation.physical_offset, allocation.size_bytes, 0,
                         kC0EarlyReservedEnd) &&
         !ranges_overlap(allocation.physical_offset, allocation.size_bytes,
                         kC0PageTableReservedBase,
                         kC0PageTableReservedEnd - kC0PageTableReservedBase) &&
         !ranges_overlap(allocation.physical_offset, allocation.size_bytes,
                         kC0MqdReservedBase, kC0MqdReservedEnd - kC0MqdReservedBase);
}

bool c0_physical_exclusion_gaps() {
  native_r9700::VramAllocator allocator(c0_layout());
  native_r9700::VramAllocation before_page_tables{};
  native_r9700::VramAllocation after_page_tables{};
  native_r9700::VramAllocation before_mqd{};
  native_r9700::VramAllocation after_mqd{};
  std::string error_text;
  const uint64_t first_gap_bytes = kC0PageTableReservedBase - kC0EarlyReservedEnd;
  const uint64_t second_gap_bytes =
      kC0MqdReservedBase - (kC0PageTableReservedEnd + kPageBytes);

  if (!require(allocator.allocate("before-page-tables", first_gap_bytes, kPageBytes,
                                  &before_page_tables, &error_text),
               "the first owned range after the early C0 reservation must allocate") ||
      !require(allocator.allocate("after-page-tables", kPageBytes, kPageBytes,
                                  &after_page_tables, &error_text),
               "first-fit must continue immediately after the C0 page-table reservation") ||
      !require(allocator.allocate("before-mqd", second_gap_bytes, kPageBytes, &before_mqd,
                                  &error_text),
               "the owned range before the C0 MQD reservation must allocate") ||
      !require(allocator.allocate("after-mqd", kPageBytes, kPageBytes, &after_mqd,
                                  &error_text),
               "first-fit must continue immediately after the C0 MQD reservation")) {
    return false;
  }

  return require(before_page_tables.physical_offset == kC0EarlyReservedEnd &&
                     before_page_tables.size_bytes == first_gap_bytes,
                 "first-fit must begin after the early C0 reservation and stop at page tables") &&
         require(after_page_tables.physical_offset == kC0PageTableReservedEnd &&
                     after_page_tables.size_bytes == kPageBytes,
                 "first-fit must skip the C0 page-table reservation") &&
         require(before_mqd.physical_offset == kC0PageTableReservedEnd + kPageBytes &&
                     before_mqd.size_bytes == second_gap_bytes,
                 "first-fit must use the range between C0 page tables and MQD") &&
         require(after_mqd.physical_offset == kC0MqdReservedEnd &&
                     after_mqd.size_bytes == kPageBytes,
                 "first-fit must skip the C0 MQD reservation") &&
         require(avoids_c0_reservations(before_page_tables) &&
                     avoids_c0_reservations(after_page_tables) &&
                     avoids_c0_reservations(before_mqd) &&
                     avoids_c0_reservations(after_mqd),
                 "allocator must never return a range crossing a cited C0 reservation");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string mode = argv[1];
  if (mode == "owned-range") return owned_range() ? 0 : 2;
  if (mode == "rejected-requests") return rejected_requests() ? 0 : 3;
  if (mode == "exhausted-and-overflow") return exhausted_and_overflow() ? 0 : 4;
  if (mode == "invalid-transitions") return invalid_transitions() ? 0 : 5;
  if (mode == "adjacent-coalescing") return adjacent_coalescing() ? 0 : 6;
  if (mode == "c0-physical-exclusion-gaps") return c0_physical_exclusion_gaps() ? 0 : 7;
  return 8;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "vram_allocator_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(LAYOUT_SOURCE),
            str(ALLOCATOR_SOURCE),
            str(probe_source),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def run_allocator_probe(tmp_path: Path, mode: str) -> None:
    completed = subprocess.run(
        [str(compile_allocator_probe(tmp_path)), mode], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_vram_allocator_returns_distinct_aligned_allocations_only_from_layout_owned_range(
    tmp_path: Path,
) -> None:
    """First-fit allocation begins at layout ownership and honors per-request alignment."""
    run_allocator_probe(tmp_path, "owned-range")


def test_vram_allocator_rejects_invalid_and_duplicate_requests_without_output_mutation(
    tmp_path: Path,
) -> None:
    """Bad request transitions preserve the caller's output allocation."""
    run_allocator_probe(tmp_path, "rejected-requests")


def test_vram_allocator_rejects_exhausted_and_overflow_requests_without_output_mutation(
    tmp_path: Path,
) -> None:
    """Finite layout ownership and arithmetic bounds reject allocation without output writes."""
    run_allocator_probe(tmp_path, "exhausted-and-overflow")


def test_vram_allocator_rejects_forged_and_double_releases(tmp_path: Path) -> None:
    """Only the exact live allocation can release its physical ownership once."""
    run_allocator_probe(tmp_path, "invalid-transitions")


def test_vram_allocator_coalesces_adjacent_free_ranges_for_first_fit_reuse(tmp_path: Path) -> None:
    """Releasing neighboring pages makes their combined first-fit range allocatable."""
    run_allocator_probe(tmp_path, "adjacent-coalescing")


def test_vram_allocator_skips_each_c0_physical_exclusion_with_disjoint_first_fit_ranges(
    tmp_path: Path,
) -> None:
    """C0 page-table and MQD pages remain unavailable while first-fit advances past each gap."""
    run_allocator_probe(tmp_path, "c0-physical-exclusion-gaps")

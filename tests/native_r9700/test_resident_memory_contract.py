"""No-hardware RED contracts for dynamic R9700 resident VRAM mappings."""

from pathlib import Path
import subprocess


RESIDENT_HEADER = Path("native_r9700/resident_memory.h")
RESIDENT_SOURCE = Path("native_r9700/resident_memory.cpp")
ALLOCATOR_HEADER = Path("native_r9700/vram_allocator.h")
ALLOCATOR_SOURCE = Path("native_r9700/vram_allocator.cpp")
LAYOUT_HEADER = Path("native_r9700/vram_layout.h")
LAYOUT_SOURCE = Path("native_r9700/vram_layout.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_resident_memory_probe(tmp_path: Path) -> Path:
    """Compile only resident planning with an injected in-memory page mapper."""
    assert RESIDENT_HEADER.is_file() and RESIDENT_SOURCE.is_file(), (
        "Resident memory implementation is missing"
    )
    assert ALLOCATOR_HEADER.is_file() and ALLOCATOR_SOURCE.is_file(), (
        "Vram allocator implementation is missing"
    )
    assert LAYOUT_HEADER.is_file() and LAYOUT_SOURCE.is_file(), (
        "Vram layout implementation is missing"
    )

    probe_source = tmp_path / "resident_memory_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "resident_memory.h"
#include "vram_allocator.h"
#include "vram_layout.h"

namespace {

constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kMiB = 1ULL << 20;
constexpr uint64_t kBootReservedBytes = 32ULL * kMiB;
constexpr uint64_t kPageTableReservedBytes = 4ULL * kMiB;
constexpr uint64_t kOwnedBase = kBootReservedBytes + kPageTableReservedBytes;
constexpr uint64_t kTailReservedBytes = 64ULL * kMiB;
constexpr uint64_t kResidentVaBase = 0x0000200000011000ULL;
constexpr uint64_t kResidentVaLimit = 0x0000200000200000ULL;
constexpr uint64_t kResidentVaWindowBytes = kResidentVaLimit - kResidentVaBase;
constexpr uint64_t kResidentVaWindowPages = kResidentVaWindowBytes / kPageBytes;


bool require(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "%s\n", message);
  return condition;
}

native_r9700::VramLayout layout_with_owned_pages(uint64_t page_count) {
  const uint64_t owned_bytes = page_count * kPageBytes;
  native_r9700::VramLayout layout{
      kOwnedBase + owned_bytes + kTailReservedBytes,
      64ULL * 1024,
      kBootReservedBytes,
      kPageTableReservedBytes,
      kOwnedBase,
      owned_bytes,
  };
  layout.resident_gpu_va_base = kResidentVaBase;
  layout.resident_gpu_va_limit = kResidentVaLimit;
  return layout;
}

bool same_buffer(const native_r9700::ResidentBuffer& left,
                 const native_r9700::ResidentBuffer& right) {
  return left.allocation.physical_offset == right.allocation.physical_offset &&
         left.allocation.size_bytes == right.allocation.size_bytes &&
         left.allocation.name == right.allocation.name && left.gpu_va == right.gpu_va &&
         left.size_bytes == right.size_bytes;
}

bool ranges_overlap(uint64_t left_base, uint64_t left_size, uint64_t right_base,
                    uint64_t right_size) {
  return left_base < right_base + right_size && right_base < left_base + left_size;
}

struct PageMappingCall {
  native_r9700::ResidentPageOperation operation;
  uint64_t gpu_va;
  uint64_t physical_offset;
};

class FakePageMapper {
 public:
  uint64_t fail_map_attempt = 0;
  uint64_t fail_unmap_attempt = 0;
  uint64_t map_attempts = 0;
  uint64_t unmap_attempts = 0;
  std::vector<PageMappingCall> calls;
  std::vector<PageMappingCall> map_attempt_log;

  bool apply(native_r9700::ResidentPageOperation operation, uint64_t gpu_va,
             uint64_t physical_offset, std::string* error_text) {
    calls.push_back({operation, gpu_va, physical_offset});
    if (operation == native_r9700::ResidentPageOperation::kMap) {
      ++map_attempts;
      map_attempt_log.push_back({operation, gpu_va, physical_offset});
      if (fail_map_attempt != 0 && map_attempts == fail_map_attempt) {
        if (error_text != nullptr) *error_text = "injected page-map failure";
        return false;
      }
      if (mapped_pages_.find(gpu_va) != mapped_pages_.end()) {
        if (error_text != nullptr) *error_text = "injected GPU VA collision";
        return false;
      }
      mapped_pages_.emplace(gpu_va, physical_offset);
      return true;
    }

    ++unmap_attempts;
    if (fail_unmap_attempt != 0 && unmap_attempts == fail_unmap_attempt) {
      if (error_text != nullptr) *error_text = "injected page-unmap failure";
      return false;
    }
    const auto found = mapped_pages_.find(gpu_va);
    if (found == mapped_pages_.end() || found->second != physical_offset) {
      if (error_text != nullptr) *error_text = "injected unmap mismatch";
      return false;
    }
    mapped_pages_.erase(found);
    return true;
  }

  bool empty() const { return mapped_pages_.empty(); }
  uint64_t mapped_page_count() const { return mapped_pages_.size(); }

 private:
  std::map<uint64_t, uint64_t> mapped_pages_;
};

native_r9700::ResidentMemory resident_memory(
    const native_r9700::VramLayout& layout, native_r9700::VramAllocator& allocator,
    FakePageMapper* mapper) {
  return native_r9700::ResidentMemory(
      layout, allocator,
      [mapper](native_r9700::ResidentPageOperation operation, uint64_t gpu_va,
               uint64_t physical_offset, std::string* error_text) {
        return mapper->apply(operation, gpu_va, physical_offset, error_text);
      });
}

bool multiple_nonoverlapping_buffers() {
  const native_r9700::VramLayout layout = layout_with_owned_pages(16);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer weights{};
  native_r9700::ResidentBuffer kv_cache{};
  std::string error_text;

  if (!require(memory.allocate("weights", 1, &weights, &error_text),
               "a one-byte resident request must map one page") ||
      !require(memory.allocate("kv-cache", kPageBytes + 1, &kv_cache, &error_text),
               "a second resident request must map its rounded page range")) {
    return false;
  }

  return require(weights.allocation.name == "weights" &&
                     kv_cache.allocation.name == "kv-cache",
                 "resident buffers must preserve their distinct caller names") &&
         require(weights.size_bytes == kPageBytes &&
                     kv_cache.size_bytes == 2ULL * kPageBytes,
                 "resident buffer ranges must round up to 4 KiB pages") &&
         require(weights.allocation.size_bytes == weights.size_bytes &&
                     kv_cache.allocation.size_bytes == kv_cache.size_bytes,
                 "resident and physical ranges must have the same rounded sizes") &&
         require(weights.gpu_va % kPageBytes == 0 && kv_cache.gpu_va % kPageBytes == 0 &&
                     weights.allocation.physical_offset % kPageBytes == 0 &&
                     kv_cache.allocation.physical_offset % kPageBytes == 0,
                 "every resident GPU VA and physical range must be 4 KiB aligned") &&
         require(!ranges_overlap(weights.gpu_va, weights.size_bytes, kv_cache.gpu_va,
                                 kv_cache.size_bytes) &&
                     !ranges_overlap(weights.allocation.physical_offset, weights.size_bytes,
                                     kv_cache.allocation.physical_offset,
                                     kv_cache.size_bytes),
                 "live resident buffers must have nonoverlapping GPU VA and physical ranges") &&
         require(mapper.mapped_page_count() == 3,
                 "the injected mapper must receive one mapping per resident page");
}

bool rejects_duplicate_and_invalid_ranges() {
  const native_r9700::VramLayout layout = layout_with_owned_pages(8);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer live{};
  std::string error_text;
  if (!require(memory.allocate("live", kPageBytes, &live, &error_text),
               "a valid resident request must establish a live mapping")) {
    return false;
  }

  const native_r9700::ResidentBuffer sentinel{
      {0xDEADBEEF, 0xC0FFEE, "unchanged"}, 0xABCD000, 0xC0FFEE};
  native_r9700::ResidentBuffer output = sentinel;
  const uint64_t mapped_before = mapper.mapped_page_count();
  const std::size_t calls_before = mapper.calls.size();

  return require(!memory.allocate("live", kPageBytes, &output, &error_text) &&
                     !error_text.empty() && same_buffer(output, sentinel),
                 "duplicate resident names must reject the mapping collision without output writes") &&
         require(!memory.allocate("zero", 0, &output, &error_text) && !error_text.empty() &&
                     same_buffer(output, sentinel),
                 "zero-byte resident ranges must fail without output writes") &&
         require(!memory.allocate("overflow", std::numeric_limits<uint64_t>::max(), &output,
                                  &error_text) &&
                     !error_text.empty() && same_buffer(output, sentinel),
                 "overflowing resident ranges must fail without output writes") &&
         require(mapper.mapped_page_count() == mapped_before &&
                     mapper.calls.size() == calls_before,
                 "rejected resident requests must not alter planned page mappings");
}

bool resident_va_starts_at_safe_window_base_and_ends_at_limit() {
  const native_r9700::VramLayout layout =
      layout_with_owned_pages(kResidentVaWindowPages);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer buffer{};
  std::string error_text;

  if (!require(memory.allocate("safe-window", kResidentVaWindowBytes, &buffer,
                               &error_text),
               "the complete current resident VA window must be mappable")) {
    return false;
  }
  return require(layout.resident_gpu_va_base == kResidentVaBase &&
                     layout.resident_gpu_va_limit == kResidentVaLimit,
                 "the fixture must provide the verified C0-safe VA window") &&
         require(buffer.gpu_va == layout.resident_gpu_va_base,
                 "the first resident buffer must begin after the active C0 PTB") &&
         require(buffer.gpu_va + buffer.size_bytes == layout.resident_gpu_va_limit,
                 "a complete resident buffer must end exactly at the current VA limit") &&
         require(mapper.map_attempt_log.size() == kResidentVaWindowPages &&
                     mapper.map_attempt_log.front().gpu_va ==
                         layout.resident_gpu_va_base &&
                     mapper.map_attempt_log.back().gpu_va ==
                         layout.resident_gpu_va_limit - kPageBytes,
                 "resident page mappings must cover exactly the current safe VA window");
}

bool rejects_resident_range_crossing_current_safe_va_window() {
  const native_r9700::VramLayout layout =
      layout_with_owned_pages(kResidentVaWindowPages + 1);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  const native_r9700::ResidentBuffer sentinel{
      {0xDEADBEEF, 0xC0FFEE, "unchanged"}, 0xABCD000, 0xC0FFEE};
  native_r9700::ResidentBuffer output = sentinel;
  std::string error_text;

  native_r9700::ResidentBuffer live{};
  if (!require(memory.allocate("within-safe-window", kPageBytes, &live, &error_text),
               "cross-window rejection requires an existing resident mapping")) {
    return false;
  }

  const uint64_t mapped_before = mapper.mapped_page_count();
  const std::size_t calls_before = mapper.calls.size();
  return require(
      !memory.allocate("crosses-safe-window", kResidentVaWindowBytes, &output,
                       &error_text) &&
          !error_text.empty() && same_buffer(output, sentinel) &&
          mapper.mapped_page_count() == mapped_before &&
          mapper.calls.size() == calls_before,
      "a resident range crossing the current VA limit must preserve output and mappings");
}

bool rollback_after_injected_map_failure() {
  const native_r9700::VramLayout layout = layout_with_owned_pages(8);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  mapper.fail_map_attempt = 2;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  const native_r9700::ResidentBuffer sentinel{
      {0xDEADBEEF, 0xC0FFEE, "unchanged"}, 0xABCD000, 0xC0FFEE};
  native_r9700::ResidentBuffer failed = sentinel;
  std::string error_text;

  if (!require(!memory.allocate("failed", 2ULL * kPageBytes, &failed, &error_text) &&
                   !error_text.empty(),
               "an injected second-page mapper failure must reject the whole allocation") ||
      !require(same_buffer(failed, sentinel),
               "a failed resident map must preserve the caller output") ||
      !require(mapper.map_attempt_log.size() == 2 &&
                   mapper.map_attempt_log[0].gpu_va % kPageBytes == 0 &&
                   mapper.map_attempt_log[0].physical_offset == kOwnedBase,
               "the failed request must have begun at the first available VA and physical page") ||
      !require(mapper.empty(),
               "a failed resident map must unmap every already-applied page-table entry")) {
    return false;
  }

  mapper.fail_map_attempt = 0;
  native_r9700::ResidentBuffer recovered{};
  if (!require(memory.allocate("recovered", 2ULL * kPageBytes, &recovered, &error_text),
               "mapping must succeed after the injected failure is removed")) {
    return false;
  }

  return require(recovered.gpu_va == mapper.map_attempt_log[0].gpu_va &&
                     recovered.allocation.physical_offset == kOwnedBase &&
                     recovered.size_bytes == 2ULL * kPageBytes,
                 "failure must roll back GPU VA, physical allocation, and page-table planning") &&
         require(mapper.mapped_page_count() == 2,
                 "recovered allocation must leave exactly its two pages mapped");
}

bool rollback_unmap_failure_quarantines_failed_range() {
  const native_r9700::VramLayout layout = layout_with_owned_pages(8);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  mapper.fail_map_attempt = 2;
  mapper.fail_unmap_attempt = 1;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer failed{};
  std::string error_text;

  if (!require(!memory.allocate("failed", 2ULL * kPageBytes, &failed, &error_text) &&
                   !error_text.empty(),
               "a rollback-unmap failure begins with an injected map failure") ||
      !require(mapper.mapped_page_count() == 1,
               "a failed rollback unmap must leave its original page-table mapping live")) {
    return false;
  }

  mapper.fail_map_attempt = 0;
  native_r9700::ResidentBuffer recovered{};
  if (!require(memory.allocate("recovered", 2ULL * kPageBytes, &recovered, &error_text),
               "a failed rollback unmap must not prevent a later independent allocation")) {
    return false;
  }

  return require(!ranges_overlap(recovered.gpu_va, recovered.size_bytes, kOwnedBase,
                                 2ULL * kPageBytes) &&
                     !ranges_overlap(recovered.allocation.physical_offset,
                                     recovered.allocation.size_bytes, kOwnedBase,
                                     2ULL * kPageBytes),
                 "a failed rollback unmap must quarantine its VA and physical range") &&
         require(mapper.mapped_page_count() == 3,
                 "the later allocation must coexist with the stale mapped page");
}

bool release_all_unmaps_and_reclaims_everything() {
  const native_r9700::VramLayout layout = layout_with_owned_pages(8);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer first{};
  native_r9700::ResidentBuffer second{};
  std::string error_text;
  if (!require(memory.allocate("first", kPageBytes, &first, &error_text) &&
                   memory.allocate("second", 2ULL * kPageBytes, &second, &error_text),
               "release-all requires multiple live resident buffers")) {
    return false;
  }

  memory.release_all();
  native_r9700::ResidentBuffer reused{};
  if (!require(memory.allocate("reused", kPageBytes, &reused, &error_text),
               "release-all must leave resident memory reusable")) {
    return false;
  }

  return require(mapper.mapped_page_count() == 1,
                 "release-all must remove every old page-table mapping before reuse") &&
         require(reused.gpu_va == first.gpu_va &&
                     reused.allocation.physical_offset == first.allocation.physical_offset,
                 "release-all must reclaim the earliest GPU VA and physical range");
}

bool release_all_unmap_failure_quarantines_failed_range() {
  const native_r9700::VramLayout layout = layout_with_owned_pages(8);
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer first{};
  std::string error_text;
  if (!require(memory.allocate("first", kPageBytes, &first, &error_text),
               "release-all unmap failure requires one live resident buffer")) {
    return false;
  }

  mapper.fail_unmap_attempt = 1;
  memory.release_all();
  mapper.fail_unmap_attempt = 0;
  native_r9700::ResidentBuffer recovered{};
  if (!require(memory.allocate("recovered", kPageBytes, &recovered, &error_text),
               "a failed release-all unmap must not prevent a later independent allocation")) {
    return false;
  }

  return require(!ranges_overlap(recovered.gpu_va, recovered.size_bytes, first.gpu_va,
                                 first.size_bytes) &&
                     !ranges_overlap(recovered.allocation.physical_offset,
                                     recovered.allocation.size_bytes,
                                     first.allocation.physical_offset,
                                     first.allocation.size_bytes),
                 "a failed release-all unmap must quarantine its VA and physical range") &&
         require(mapper.mapped_page_count() == 2,
                 "the later allocation must coexist with the stale release-all mapping");
}

bool maps_and_reclaims_four_gibibytes_of_resident_ranges() {
  constexpr uint64_t kOneGiB = 1ULL << 30;
  constexpr uint64_t kResidentBytes = 4ULL * kOneGiB;
  const native_r9700::VramLayout base_layout =
      layout_with_owned_pages(kResidentBytes / kPageBytes);
  native_r9700::VramLayout layout = base_layout;
  layout.resident_gpu_va_limit = kResidentVaBase + kResidentBytes;
  native_r9700::VramAllocator allocator(layout);
  FakePageMapper mapper;
  native_r9700::ResidentMemory memory = resident_memory(layout, allocator, &mapper);
  native_r9700::ResidentBuffer first{};
  native_r9700::ResidentBuffer second{};
  native_r9700::ResidentBuffer third{};
  std::string error_text;

  const uint64_t first_bytes = kOneGiB + kPageBytes;
  const uint64_t second_bytes = 2ULL * kOneGiB;
  const uint64_t third_bytes = kOneGiB - kPageBytes;
  if (!require(memory.allocate("first-gib", first_bytes, &first, &error_text),
               "first multi-GiB resident allocation must succeed") ||
      !require(memory.allocate("middle-gib", second_bytes, &second, &error_text),
               "middle multi-GiB resident allocation must succeed") ||
      !require(memory.allocate("last-gib", third_bytes, &third, &error_text),
               "last multi-GiB resident allocation must succeed")) {
    return false;
  }

  if (!require(first.gpu_va == kResidentVaBase,
               "multi-GiB allocation must start at the resident base") ||
      !require(second.gpu_va == first.gpu_va + first.size_bytes &&
                   third.gpu_va == second.gpu_va + second.size_bytes,
               "multi-GiB resident ranges must be contiguous and monotonic") ||
      !require(third.gpu_va + third.size_bytes == layout.resident_gpu_va_limit,
               "multi-GiB resident ranges must end at the expanded limit") ||
      !require(mapper.map_attempts == kResidentBytes / kPageBytes,
               "every multi-GiB resident page must be mapped exactly once") ||
      !require(mapper.mapped_page_count() == kResidentBytes / kPageBytes,
               "all multi-GiB resident pages must remain uniquely mapped")) {
    return false;
  }

  memory.release_all();
  native_r9700::ResidentBuffer reused{};
  return require(mapper.unmap_attempts == kResidentBytes / kPageBytes,
                 "release must unmap every multi-GiB resident page") &&
         require(mapper.mapped_page_count() == 0,
                 "release must leave no multi-GiB resident mapping") &&
         require(memory.allocate("reused", kPageBytes, &reused, &error_text),
                 "released multi-GiB allocation space must be reusable") &&
         require(reused.gpu_va == kResidentVaBase &&
                     reused.allocation.physical_offset == kOwnedBase,
                 "multi-GiB release must restore the first VA and physical page");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string mode = argv[1];
  if (mode == "multiple-buffers") return multiple_nonoverlapping_buffers() ? 0 : 2;
  if (mode == "rejected-ranges") return rejects_duplicate_and_invalid_ranges() ? 0 : 3;
  if (mode == "resident-va-window")
    return resident_va_starts_at_safe_window_base_and_ends_at_limit() ? 0 : 8;
  if (mode == "crosses-resident-va-window")
    return rejects_resident_range_crossing_current_safe_va_window() ? 0 : 9;
  if (mode == "rollback") return rollback_after_injected_map_failure() ? 0 : 4;
  if (mode == "rollback-unmap-failure")
    return rollback_unmap_failure_quarantines_failed_range() ? 0 : 5;
  if (mode == "release-all") return release_all_unmaps_and_reclaims_everything() ? 0 : 6;
  if (mode == "release-all-unmap-failure")
    return release_all_unmap_failure_quarantines_failed_range() ? 0 : 7;
  if (mode == "multi-gib-resident")
    return maps_and_reclaims_four_gibibytes_of_resident_ranges() ? 0 : 11;
  return 10;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "resident_memory_probe"
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
            str(RESIDENT_SOURCE),
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


def run_resident_memory_probe(tmp_path: Path, mode: str) -> None:
    completed = subprocess.run(
        [str(compile_resident_memory_probe(tmp_path)), mode],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_resident_memory_maps_multiple_nonoverlapping_page_aligned_buffers(
    tmp_path: Path,
) -> None:
    """Named buffers use unique, page-aligned GPU VA and physical ranges."""
    run_resident_memory_probe(tmp_path, "multiple-buffers")


def test_resident_memory_rejects_duplicate_name_collisions_and_invalid_ranges(
    tmp_path: Path,
) -> None:
    """Rejected requests leave the caller output and mapper planning unchanged."""
    run_resident_memory_probe(tmp_path, "rejected-ranges")

def test_resident_memory_starts_at_current_safe_va_base_and_ends_at_limit(
    tmp_path: Path,
) -> None:
    """The first resident range covers exactly the initially verified VA window."""
    run_resident_memory_probe(tmp_path, "resident-va-window")


def test_resident_memory_rejects_ranges_crossing_current_safe_va_window(
    tmp_path: Path,
) -> None:
    """An over-limit request preserves output and mapper state without mapping pages."""
    run_resident_memory_probe(tmp_path, "crosses-resident-va-window")


def test_resident_memory_rolls_back_va_physical_and_page_mapping_after_failure(
    tmp_path: Path,
) -> None:
    """A failed page map unmaps partial work and reclaims every planning resource."""
    run_resident_memory_probe(tmp_path, "rollback")


def test_resident_memory_release_all_unmaps_and_reclaims_all_buffers(tmp_path: Path) -> None:
    """Release-all removes mappings and makes both allocators immediately reusable."""
    run_resident_memory_probe(tmp_path, "release-all")


def test_resident_memory_quarantines_range_when_rollback_unmap_fails(
    tmp_path: Path,
) -> None:
    """A stale rollback mapping cannot alias a later VA or physical allocation."""
    run_resident_memory_probe(tmp_path, "rollback-unmap-failure")


def test_resident_memory_quarantines_range_when_release_all_unmap_fails(
    tmp_path: Path,
) -> None:
    """A stale release-all mapping cannot alias a later VA or physical allocation."""
    run_resident_memory_probe(tmp_path, "release-all-unmap-failure")

def test_resident_memory_maps_and_reclaims_four_gibibytes_without_host_payloads(
    tmp_path: Path,
) -> None:
    """Resident metadata/page callbacks span four GiB without allocating host payloads."""
    run_resident_memory_probe(tmp_path, "multi-gib-resident")

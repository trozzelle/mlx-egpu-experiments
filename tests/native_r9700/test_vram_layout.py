"""No-hardware RED contracts for source-backed R9700 VRAM ownership geometry."""

from pathlib import Path
import subprocess


LAYOUT_HEADER = Path("native_r9700/vram_layout.h")
LAYOUT_SOURCE = Path("native_r9700/vram_layout.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_layout_probe(tmp_path: Path) -> Path:
    """Compile the pure layout boundary; this does not contact TinyGPU hardware."""
    assert LAYOUT_HEADER.is_file() and LAYOUT_SOURCE.is_file(), "Vram layout implementation is missing"

    probe_source = tmp_path / "vram_layout_probe.cpp"
    probe_source.write_text(
        r'''
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <type_traits>

#include "vram_layout.h"

namespace {

constexpr uint64_t kKiB = 1ULL << 10;
constexpr uint64_t kMiB = 1ULL << 20;
constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kDiscoveryBytes = 64ULL * kKiB;
constexpr uint64_t kOneGiB = 1ULL << 30;
constexpr uint64_t kPdb2EntryBytes = 1ULL << 39;
constexpr uint64_t kBootBytes = 32ULL * kMiB;
constexpr uint64_t kGfx12TailReservationBytes = 64ULL * kMiB;
constexpr uint32_t kValidMemsizeMiB = 1024;
constexpr uint64_t kValidVramBytes = kValidMemsizeMiB * kMiB;
constexpr uint32_t kThirtyTwoGiBMemsizeMiB = 32768;
constexpr uint64_t kThirtyTwoGiBVramBytes =
    static_cast<uint64_t>(kThirtyTwoGiBMemsizeMiB) * kMiB;
constexpr uint64_t kVerifiedModelResidentBytes = 4ULL << 30;
constexpr uint32_t kPdb2EscapeMemsizeMiB = (1U << 19) + 97U;
constexpr uint64_t kPdb2EscapeVramBytes =
    static_cast<uint64_t>(kPdb2EscapeMemsizeMiB) * kMiB;
constexpr uint32_t kObservedSmallMemsizeMiB = 32624;
constexpr uint64_t kC0RootAndScratchReservedBase = 0x00000000ULL;
constexpr uint64_t kC0RootAndScratchReservedBytes = 0x00003000ULL;
constexpr uint64_t kC0PageTablesAndMqdReservedBase = 0x02000000ULL;
constexpr uint64_t kC0PageTablesAndMqdReservedBytes = 0x00004000ULL;
constexpr uint64_t kC0FixedApertureReservedBase = 0x06000000ULL;
constexpr uint64_t kC0FixedApertureReservedBytes = 0x00010000ULL;
constexpr uint64_t kObservedSmallBar0Bytes = 0x10000000ULL;
constexpr uint64_t kObservedSmallPageTablePoolBase = 0x02004000ULL;
constexpr uint64_t kObservedSmallPageTablePoolBytes = 0x03FFC000ULL;
constexpr uint64_t kObservedSmallPayloadBase = 0x06010000ULL;
constexpr uint64_t kObservedSmallPayloadBytes = 0x09FF0000ULL;
constexpr uint64_t kResidentGpuVaBase = 0x200000011000ULL;
constexpr uint64_t kResidentGpuVaLimit = 0x200000200000ULL;
constexpr uint64_t kC0Pdb1GpuVaBase = 0x200000000000ULL;
constexpr uint64_t kC0Pdb1GpuVaBytes = 1ULL << 30;
constexpr uint64_t kC0Pdb1GpuVaLimit = kC0Pdb1GpuVaBase + kC0Pdb1GpuVaBytes;
constexpr uint64_t kCurrentPdb2Base =
    kResidentGpuVaBase & ~(kPdb2EntryBytes - 1);
constexpr uint64_t kCurrentPdb2End = kCurrentPdb2Base + kPdb2EntryBytes;


bool require(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "%s\n", message);
  return condition;
}

template <typename Layout, typename = void>
struct HasSmallApertureFields : std::false_type {};

template <typename Layout>
struct HasSmallApertureFields<
    Layout, std::void_t<decltype(std::declval<Layout>().large_bar),
                        decltype(std::declval<Layout>().page_table_pool_base),
                        decltype(std::declval<Layout>().page_table_pool_bytes)>>
    : std::true_type {};

bool rejects(uint32_t memsize_mib, uint64_t bar0_bytes) {
  native_r9700::VramLayout layout{};
  std::string error_text;
  return !native_r9700::derive_vram_layout(memsize_mib, bar0_bytes, &layout, &error_text) &&
         !error_text.empty();
}

bool valid_layout() {
  native_r9700::VramLayout layout{};
  std::string error_text;
  if (!require(native_r9700::derive_vram_layout(kValidMemsizeMiB, kValidVramBytes, &layout,
                                                &error_text),
               "source-backed large-BAR layout must derive")) {
    return false;
  }

  const uint64_t expected_allocatable_bytes =
      kValidVramBytes - kGfx12TailReservationBytes - kBootBytes;
  const uint64_t allocatable_end = layout.allocatable_base + layout.allocatable_bytes;
  return require(layout.vram_bytes == kValidVramBytes,
                 "MEMSIZE must decode as MiB into the discovered VRAM size") &&
         require(layout.discovery_reserved_bytes == kDiscoveryBytes,
                 "discovery table backoff must remain 64 KiB") &&
         require(layout.boot_reserved_bytes == kBootBytes,
                 "boot arena must retain its source-backed 32 MiB size") &&
         require(layout.page_table_reserved_bytes == 0,
                 "large BAR must not add a small-BAR page-table arena") &&
         require(layout.allocatable_base == kBootBytes,
                 "allocatable ownership must begin after the boot arena") &&
         require(layout.allocatable_bytes == expected_allocatable_bytes,
                 "allocatable ownership must exclude the gfx12 64 MiB tail reservation") &&
         require(layout.allocatable_base % kPageBytes == 0 &&
                     layout.allocatable_bytes % kPageBytes == 0,
                 "allocatable interval must be page aligned") &&
         require(allocatable_end == kValidVramBytes - kGfx12TailReservationBytes,
                 "owned interval must end before the reserved tail containing discovery");
}

bool c0_reserved_physical_ranges() {
  native_r9700::VramLayout layout{};
  std::string error_text;
  if (!require(native_r9700::derive_vram_layout(kValidMemsizeMiB, kValidVramBytes, &layout,
                                                &error_text),
               "layout with C0 reservations must derive")) {
    return false;
  }

  static_assert(
      std::is_same_v<decltype(layout.c0_reserved_physical_ranges),
                     std::array<native_r9700::VramPhysicalRange, 3>>,
      "C0 physical reservations must remain a fixed, ordered three-range layout contract");

  const auto& ranges = layout.c0_reserved_physical_ranges;
  if (!require(ranges.size() == 3,
               "C0 layout must expose exactly the three active physical reservations")) {
    return false;
  }
  return require(ranges[0].base == kC0RootAndScratchReservedBase &&
                     ranges[0].size_bytes == kC0RootAndScratchReservedBytes,
                 "C0 root, memscratch, and dummy-page reservation must be [0x0, 0x3000)") &&
         require(ranges[1].base == kC0PageTablesAndMqdReservedBase &&
                     ranges[1].size_bytes == kC0PageTablesAndMqdReservedBytes,
                 "C0 page tables and MQD reservation must be [0x02000000, 0x02004000)") &&
         require(ranges[2].base == kC0FixedApertureReservedBase &&
                     ranges[2].size_bytes == kC0FixedApertureReservedBytes,
                 "C0 fixed input/output/code/EOP reservation must be [0x06000000, 0x06010000)");
}

template <typename Layout>
bool small_aperture_layout_impl() {
  if constexpr (!HasSmallApertureFields<Layout>::value) {
    return require(false,
                   "VramLayout must distinguish large BARs from the small-BAR page-table pool");
  } else {
    Layout layout{};
    std::string error_text;
    if (!require(native_r9700::derive_vram_layout(kObservedSmallMemsizeMiB,
                                                  kObservedSmallBar0Bytes, &layout, &error_text),
                 "the observed 256 MiB BAR0 aperture must derive")) {
      return false;
    }

    const uint64_t pool_end = layout.page_table_pool_base + layout.page_table_pool_bytes;
    const uint64_t payload_end = layout.allocatable_base + layout.allocatable_bytes;
    return require(!layout.large_bar, "a 256 MiB BAR0 must select the small-BAR layout") &&
           require(layout.page_table_pool_base == kObservedSmallPageTablePoolBase,
                   "small-BAR page-table pool must begin after the C0 table/MQD reservation") &&
           require(layout.page_table_pool_bytes == kObservedSmallPageTablePoolBytes,
                   "small-BAR page-table pool must end at the fixed C0 aperture") &&
           require(pool_end == kC0FixedApertureReservedBase,
                   "small-BAR page-table pool must be [0x02004000, 0x06000000)") &&
           require(layout.allocatable_base == kObservedSmallPayloadBase,
                   "small-BAR payload must begin after the fixed C0 aperture") &&
           require(layout.allocatable_bytes == kObservedSmallPayloadBytes,
                   "small-BAR payload must occupy [0x06010000, 0x10000000)") &&
           require(payload_end == kObservedSmallBar0Bytes,
                   "small-BAR payload must end at the BAR0 aperture boundary") &&
           require(layout.page_table_pool_base % kPageBytes == 0 &&
                       layout.page_table_pool_bytes % kPageBytes == 0 &&
                       layout.allocatable_base % kPageBytes == 0 &&
                       layout.allocatable_bytes % kPageBytes == 0,
                   "small-BAR pool and payload intervals must be page aligned");
  }
}

bool small_aperture_layout() {
  return small_aperture_layout_impl<native_r9700::VramLayout>();
}

bool rejects_invalid_small_aperture_boundaries() {
  const uint64_t c0_required_end =
      kC0FixedApertureReservedBase + kC0FixedApertureReservedBytes;
  return rejects(kObservedSmallMemsizeMiB, kBootBytes - kPageBytes) &&
         rejects(kObservedSmallMemsizeMiB, c0_required_end - kPageBytes) &&
         rejects(kObservedSmallMemsizeMiB, kObservedSmallBar0Bytes - 1);
}

bool resident_gpu_va_window() {
  native_r9700::VramLayout layout{};
  std::string error_text;
  if (!require(native_r9700::derive_vram_layout(kValidMemsizeMiB, kValidVramBytes, &layout,
                                                &error_text),
               "large-BAR layout with resident VA window must derive")) {
    return false;
  }

  const uint64_t expected_limit = kResidentGpuVaBase + layout.allocatable_bytes;
  return require(layout.large_bar,
                 "a BAR covering all discovered VRAM must select the large-BAR layout") &&
         require(layout.resident_gpu_va_base == kResidentGpuVaBase,
                 "large-BAR resident mappings must start after C0's active PTB entries") &&
         require(layout.resident_gpu_va_limit == expected_limit,
                 "large-BAR resident mappings must span the complete allocator interval") &&
         require(layout.resident_gpu_va_limit > kResidentGpuVaLimit,
                 "large-BAR resident mappings must exceed the old single-PTB limit") &&
         require(layout.resident_gpu_va_base % kPageBytes == 0 &&
                     layout.resident_gpu_va_limit % kPageBytes == 0,
                 "large-BAR resident GPU VA window must be page aligned") &&
         require(layout.resident_gpu_va_limit <= kCurrentPdb2End,
                 "large-BAR resident GPU VA window must remain in its current PDB2 entry");
}

bool large_bar_32_gib_resident_gpu_va_window() {
  native_r9700::VramLayout layout{};
  std::string error_text;
  if (!require(native_r9700::derive_vram_layout(kThirtyTwoGiBMemsizeMiB,
                                                kThirtyTwoGiBVramBytes, &layout, &error_text),
               "32 GiB large-BAR layout must derive")) {
    return false;
  }

  const uint64_t expected_allocatable_bytes =
      kThirtyTwoGiBVramBytes - kGfx12TailReservationBytes - kBootBytes;
  if (!require(layout.resident_gpu_va_base < layout.resident_gpu_va_limit,
               "32 GiB resident GPU VA window must be non-empty")) {
    return false;
  }
  const uint64_t resident_bytes =
      layout.resident_gpu_va_limit - layout.resident_gpu_va_base;
  return require(layout.large_bar,
                 "32 GiB BAR coverage must select the large-BAR layout") &&
         require(layout.allocatable_base == kBootBytes,
                 "32 GiB large-BAR allocation must begin after the boot reservation") &&
         require(layout.allocatable_bytes == expected_allocatable_bytes,
                 "32 GiB large-BAR allocation must exclude boot and gfx12 tail reservations") &&
         require(layout.resident_gpu_va_limit ==
                     layout.resident_gpu_va_base + layout.allocatable_bytes,
                 "32 GiB resident window must cover all allocator-visible bytes") &&
         require(resident_bytes > kVerifiedModelResidentBytes,
                 "32 GiB resident window must exceed the verified model resident bytes") &&
         require(resident_bytes > kOneGiB,
                 "32 GiB resident window must exceed one GiB") &&
         require(layout.resident_gpu_va_limit % kPageBytes == 0,
                 "32 GiB resident GPU VA limit must be page aligned") &&
         require(layout.resident_gpu_va_limit <= kCurrentPdb2End,
                 "32 GiB resident GPU VA window must remain in its current PDB2 entry");
}

bool rejects_large_bar_pdb2_escape() {
  if (!require(rejects(kPdb2EscapeMemsizeMiB, kPdb2EscapeVramBytes),
               "a large-BAR interval escaping the current PDB2 entry must reject")) {
    return false;
  }
  return require(rejects(std::numeric_limits<uint32_t>::max(),
                         std::numeric_limits<uint64_t>::max()),
                 "maximum decoded VRAM must reject instead of escaping the current PDB2 entry");
}

bool small_aperture_resident_gpu_va_window() {
  native_r9700::VramLayout layout{};
  std::string error_text;
  if (!require(native_r9700::derive_vram_layout(kObservedSmallMemsizeMiB,
                                                kObservedSmallBar0Bytes, &layout, &error_text),
               "small-BAR layout with resident VA window must derive")) {
    return false;
  }

  const uint64_t expected_limit = kResidentGpuVaBase + layout.allocatable_bytes;
  return require(!layout.large_bar,
                 "the observed aperture must retain its small-BAR classification") &&
         require(layout.resident_gpu_va_base == kResidentGpuVaBase,
                 "small-BAR resident mappings must start after C0's active PTB entries") &&
         require(layout.resident_gpu_va_limit == expected_limit,
                 "small-BAR resident mappings must span the complete allocatable payload") &&
         require(layout.resident_gpu_va_limit > kResidentGpuVaLimit,
                 "small-BAR resident mappings must extend beyond C0's fixed PTB0") &&
         require(layout.resident_gpu_va_base % kPageBytes == 0 &&
                     layout.resident_gpu_va_limit % kPageBytes == 0,
                 "small-BAR resident GPU VA window must be page aligned") &&
         require(layout.resident_gpu_va_base >= kC0Pdb1GpuVaBase &&
                     layout.resident_gpu_va_limit <= kC0Pdb1GpuVaLimit,
                 "small-BAR resident GPU VA window must remain in C0's PDB1 subtree");
}


}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string mode = argv[1];
  if (mode == "valid") return valid_layout() ? 0 : 2;
  if (mode == "c0-reserved-physical-ranges") return c0_reserved_physical_ranges() ? 0 : 6;
  if (mode == "resident-gpu-va-window") return resident_gpu_va_window() ? 0 : 7;
  if (mode == "large-bar-32-gib-resident-gpu-va-window") {
    return large_bar_32_gib_resident_gpu_va_window() ? 0 : 11;
  }
  if (mode == "large-bar-pdb2-escape") {
    return rejects_large_bar_pdb2_escape() ? 0 : 12;
  }
  if (mode == "small-aperture-resident-gpu-va-window") {
    return small_aperture_resident_gpu_va_window() ? 0 : 10;
  }
  if (mode == "small-aperture") return small_aperture_layout() ? 0 : 8;
  if (mode == "invalid-small-aperture-boundaries") {
    return rejects_invalid_small_aperture_boundaries() ? 0 : 9;
  }
  if (mode == "one-page-short-bar") {
    const uint64_t bar0_bytes = kValidVramBytes - kPageBytes;
    native_r9700::VramLayout layout{};
    std::string error_text;
    if (!require(native_r9700::derive_vram_layout(kValidMemsizeMiB, bar0_bytes, &layout,
                                                  &error_text),
                 "a BAR one page short of discovered VRAM must derive as a small BAR")) {
      return 3;
    }
    const uint64_t payload_end = layout.allocatable_base + layout.allocatable_bytes;
    return require(!layout.large_bar,
                   "a BAR one page short of discovered VRAM must select the small-BAR layout") &&
           require(layout.allocatable_bytes != 0,
                   "a page-short BAR must retain a nonempty small-BAR payload") &&
           require(payload_end <= bar0_bytes,
                   "a page-short BAR payload must not exceed the BAR0 aperture")
               ? 0
               : 3;
  }
  if (mode == "empty-owned-range") {
    // 64 MiB must not underflow after tail exclusion; 96 MiB leaves no owned bytes after boot.
    return rejects(64, 64 * kMiB) && rejects(96, 96 * kMiB) ? 0 : 4;
  }
  return 5;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "vram_layout_probe"
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


def run_layout_probe(tmp_path: Path, mode: str) -> None:
    completed = subprocess.run(
        [str(compile_layout_probe(tmp_path)), mode], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_vram_layout_uses_live_memsize_and_excludes_source_backed_reservations(
    tmp_path: Path,
) -> None:
    """A large BAR owns only the page-aligned interval after boot and before the gfx12 tail."""
    run_layout_probe(tmp_path, "valid")

def test_vram_layout_exposes_the_exact_active_c0_physical_exclusions(tmp_path: Path) -> None:
    """Catches an allocator-visible layout that can overlap active C0 VM/MQD/aperture pages."""
    run_layout_probe(tmp_path, "c0-reserved-physical-ranges")

def test_vram_layout_derives_the_exact_observed_256_mib_bar0_aperture(
    tmp_path: Path,
) -> None:
    """Catches a small BAR that is rejected or exposes C0 pages as dynamic PTE or payload memory."""
    run_layout_probe(tmp_path, "small-aperture")


def test_vram_layout_rejects_invalid_small_bar0_aperture_boundaries(tmp_path: Path) -> None:
    """Catches small BAR0 layouts that omit boot/C0 coverage or are not page aligned."""
    run_layout_probe(tmp_path, "invalid-small-aperture-boundaries")



def test_vram_layout_maps_the_full_large_bar_allocator_window(tmp_path: Path) -> None:
    """A large BAR must map every allocator-visible byte beyond the old ~2 MiB limit."""
    run_layout_probe(tmp_path, "resident-gpu-va-window")

def test_vram_layout_exposes_32_gib_large_bar_capacity(tmp_path: Path) -> None:
    """A 32 GiB R9700 layout must exceed the verified model and one-GiB resident spans."""
    run_layout_probe(tmp_path, "large-bar-32-gib-resident-gpu-va-window")


def test_vram_layout_rejects_large_bar_pdb2_escape_and_capacity_overflow(
    tmp_path: Path,
) -> None:
    """Large-BAR limits must fail closed before escaping or wrapping the current PDB2 entry."""
    run_layout_probe(tmp_path, "large-bar-pdb2-escape")

def test_vram_layout_maps_the_full_small_bar_payload_in_c0s_pdb1_subtree(
    tmp_path: Path,
) -> None:
    """A small BAR must dynamically map its complete payload beyond C0's fixed PTB0."""
    run_layout_probe(tmp_path, "small-aperture-resident-gpu-va-window")


def test_vram_layout_classifies_a_bar_one_page_short_of_discovered_vram_as_small(
    tmp_path: Path,
) -> None:
    """A page-short BAR is a valid small aperture whose payload stays within BAR0."""
    run_layout_probe(tmp_path, "one-page-short-bar")


def test_vram_layout_rejects_underflow_and_empty_owned_intervals(tmp_path: Path) -> None:
    """Reservations must not wrap or leave a zero-byte allocation interval."""
    run_layout_probe(tmp_path, "empty-owned-range")

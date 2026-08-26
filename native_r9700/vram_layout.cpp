#include "vram_layout.h"

#include <limits>

namespace native_r9700 {
namespace {

constexpr uint64_t kBytesPerMiB = 1ULL << 20;
constexpr uint64_t kDiscoveryReservedBytes = 64ULL << 10;
constexpr uint64_t kGfx12TailReservedBytes = 64ULL << 20;
constexpr uint64_t kBootReservedBytes = 32ULL << 20;
constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kBytesCoveredPerPageTableByte = 512;
constexpr uint64_t kPageTableArenaAlignment = 1ULL << 20;
constexpr VramPhysicalRange kC0RootAndScratchReservedRange = {0x00000000ULL, 0x00003000ULL};
constexpr VramPhysicalRange kC0PageTablesAndMqdReservedRange = {0x02000000ULL, 0x00004000ULL};
constexpr VramPhysicalRange kC0FixedApertureReservedRange = {0x06000000ULL, 0x00010000ULL};
constexpr uint64_t kC0PageTablePoolBase =
    kC0PageTablesAndMqdReservedRange.base + kC0PageTablesAndMqdReservedRange.size_bytes;
constexpr uint64_t kC0FixedApertureReservedBase = kC0FixedApertureReservedRange.base;
constexpr uint64_t kC0PayloadBase =
    kC0FixedApertureReservedRange.base + kC0FixedApertureReservedRange.size_bytes;
constexpr std::array<VramPhysicalRange, 3> kC0ReservedPhysicalRanges = {{
    kC0RootAndScratchReservedRange,
    kC0PageTablesAndMqdReservedRange,
    kC0FixedApertureReservedRange,
}};
constexpr uint64_t kResidentGpuVaBase = 0x0000200000011000ULL;
constexpr uint64_t kResidentGpuVaLimit = 0x0000200000200000ULL;
constexpr uint64_t kC0Pdb1GpuVaBase = 0x0000200000000000ULL;
constexpr uint64_t kC0Pdb1GpuVaLimit = 0x0000200040000000ULL;
constexpr uint64_t kPdb2EntryBytes = 1ULL << 39;


bool fail(std::string* error_text, const char* text) {
  if (error_text != nullptr) *error_text = text;
  return false;
}

}  // namespace

bool derive_vram_layout(uint32_t rcc_config_memsize, uint64_t bar0_bytes,
                        VramLayout* layout, std::string* error_text) {
  if (layout == nullptr) return fail(error_text, "VRAM layout output is required");

  const uint64_t memsize_mib = rcc_config_memsize;
  if (memsize_mib == 0) return fail(error_text, "RCC_CONFIG_MEMSIZE must be nonzero");
  if (memsize_mib > std::numeric_limits<uint64_t>::max() / kBytesPerMiB) {
    return fail(error_text, "RCC_CONFIG_MEMSIZE VRAM byte conversion overflows");
  }
  const uint64_t vram_bytes = memsize_mib << 20;
  if (vram_bytes == 0) return fail(error_text, "RCC_CONFIG_MEMSIZE decodes to zero VRAM");

  if (vram_bytes <= kGfx12TailReservedBytes + kBootReservedBytes) {
    return fail(error_text, "VRAM reservations exhaust the allocatable interval");
  }
  const uint64_t allocator_vram_bytes = vram_bytes - kGfx12TailReservedBytes;
  const bool large_bar = bar0_bytes >= vram_bytes;

  uint64_t page_table_reserved_bytes = 0;
  uint64_t page_table_pool_base = 0;
  uint64_t page_table_pool_bytes = 0;
  uint64_t allocatable_base = kBootReservedBytes;
  uint64_t allocatable_bytes = allocator_vram_bytes - kBootReservedBytes;
  uint64_t resident_gpu_va_limit = kResidentGpuVaLimit;
  if (large_bar) {
    const uint64_t current_pdb2_base =
        kResidentGpuVaBase & ~(kPdb2EntryBytes - 1);
    if (current_pdb2_base >
        std::numeric_limits<uint64_t>::max() - kPdb2EntryBytes) {
      return fail(error_text, "large BAR current PDB2 end overflows");
    }
    const uint64_t current_pdb2_end = current_pdb2_base + kPdb2EntryBytes;
    if (allocatable_bytes >
        std::numeric_limits<uint64_t>::max() - kResidentGpuVaBase) {
      return fail(error_text, "large BAR resident GPU VA limit overflows");
    }
    resident_gpu_va_limit = kResidentGpuVaBase + allocatable_bytes;
    if (resident_gpu_va_limit > current_pdb2_end) {
      return fail(error_text, "large BAR resident GPU VA window escapes current PDB2 entry");
    }
    if (kResidentGpuVaBase % kPageBytes != 0 ||
        resident_gpu_va_limit % kPageBytes != 0) {
      return fail(error_text, "large BAR resident GPU VA window must be page aligned");
    }
  }

  if (!large_bar) {
    if (bar0_bytes == 0 || bar0_bytes % kPageBytes != 0) {
      return fail(error_text, "small BAR0 aperture must be nonzero and page aligned");
    }

    // tinygrad's reserve_ptable path reserves one byte per 512 bytes of
    // allocator-visible VRAM, rounded to a MiB, immediately after boot.
    const uint64_t pte_arena_bytes =
        ((allocator_vram_bytes / kBytesCoveredPerPageTableByte +
          kPageTableArenaAlignment - 1) /
         kPageTableArenaAlignment) *
        kPageTableArenaAlignment;
    if (bar0_bytes < kBootReservedBytes ||
        pte_arena_bytes > bar0_bytes - kBootReservedBytes) {
      return fail(error_text, "small BAR0 aperture cannot contain the page-table arena");
    }
    const uint64_t pte_arena_end = kBootReservedBytes + pte_arena_bytes;
    if (pte_arena_end <= kC0PageTablePoolBase ||
        pte_arena_end > kC0FixedApertureReservedBase) {
      return fail(error_text, "small BAR0 page-table arena has no C0-safe pool");
    }
    if (bar0_bytes <= kC0PayloadBase) {
      return fail(error_text, "small BAR0 aperture has no C0-safe payload");
    }

    page_table_reserved_bytes = pte_arena_bytes;
    page_table_pool_base = kC0PageTablePoolBase;
    page_table_pool_bytes = pte_arena_end - page_table_pool_base;
    allocatable_base = kC0PayloadBase;
    allocatable_bytes = bar0_bytes - allocatable_base;

    if (allocatable_bytes > std::numeric_limits<uint64_t>::max() - kResidentGpuVaBase) {
      return fail(error_text, "small BAR0 resident GPU VA limit overflows");
    }
    resident_gpu_va_limit = kResidentGpuVaBase + allocatable_bytes;
    if (kResidentGpuVaBase % kPageBytes != 0 ||
        resident_gpu_va_limit % kPageBytes != 0) {
      return fail(error_text, "small BAR0 resident GPU VA window must be page aligned");
    }
    if (kResidentGpuVaBase < kC0Pdb1GpuVaBase ||
        resident_gpu_va_limit > kC0Pdb1GpuVaLimit) {
      return fail(error_text, "small BAR0 resident GPU VA window escapes C0 PDB1");
    }
  }

  // MIT source provenance: tinygrad (MIT License),
  // ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/amdev.py:279-296
  // defines VRAM as `rreg(mmRCC_CONFIG_MEMSIZE) << 20` and identifies large BAR
  // coverage as `vram.nbytes >= vram_size`.
  // MIT source provenance: tinygrad (MIT License),
  // ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/amdev.py:202-205
  // passes `vram_size - reserved_vram_size`, `boot_size=(32 << 20)`, and
  // `reserve_ptable=not large_bar`.
  // MIT source provenance: tinygrad (MIT License),
  // ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/am/amdev.py:320
  // sets gfx12's tail reservation to `64 << 20`.
  // MIT source provenance: tinygrad (MIT License),
  // ${HOME}/Development/ml/tools/tinygrad/tinygrad/runtime/support/memory.py:175-184
  // creates the small-BAR table arena after boot, then physical payload after it.
  *layout = VramLayout{
      vram_bytes,
      kDiscoveryReservedBytes,
      kBootReservedBytes,
      page_table_reserved_bytes,
      allocatable_base,
      allocatable_bytes,
      kC0ReservedPhysicalRanges,
      kResidentGpuVaBase,
      resident_gpu_va_limit,
      large_bar,
      page_table_pool_base,
      page_table_pool_bytes,
  };
  return true;
}

}  // namespace native_r9700

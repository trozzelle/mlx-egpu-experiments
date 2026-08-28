#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace native_r9700 {

struct VramPhysicalRange {
  uint64_t base;
  uint64_t size_bytes;
};

struct VramLayout {
  uint64_t vram_bytes;
  uint64_t discovery_reserved_bytes;
  uint64_t boot_reserved_bytes;
  uint64_t page_table_reserved_bytes;
  uint64_t allocatable_base;
  uint64_t allocatable_bytes;
  std::array<VramPhysicalRange, 3> c0_reserved_physical_ranges;
  uint64_t resident_gpu_va_base;
  uint64_t resident_gpu_va_limit;
  bool large_bar;
  uint64_t page_table_pool_base;
  uint64_t page_table_pool_bytes;
};

// Derives source-backed large- and small-BAR physical VRAM ownership geometry
// from RCC_CONFIG_MEMSIZE and the mapped BAR0 aperture.
// MIT source provenance: tinygrad revision
// d851aca9ae1faf4210cc0da4508bead7da57d7ee,
// tinygrad/runtime/support/am/amdev.py:202-205,279-320 and
// tinygrad/runtime/support/memory.py:175-184.
bool derive_vram_layout(uint32_t rcc_config_memsize, uint64_t bar0_bytes,
                        VramLayout* layout, std::string* error_text);

}  // namespace native_r9700

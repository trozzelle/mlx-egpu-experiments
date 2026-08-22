#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "vram_allocator.h"
#include "vram_layout.h"

namespace native_r9700 {

// Performs the BAR-visible effects required for a dynamic page-table update.
// Implementations own the hardware transport; DynamicPageTable owns no BAR mapping.
class DynamicPageTableBackend {
 public:
  virtual ~DynamicPageTableBackend() = default;

  virtual bool zero_page(uint64_t physical_page, std::string* error_text) = 0;
  virtual bool write_pte(uint64_t table_page, uint16_t entry_index, uint64_t pte,
                         std::string* error_text) = 0;
  virtual bool read_pte(uint64_t table_page, uint16_t entry_index, uint64_t* pte,
                        std::string* error_text) = 0;
  virtual bool flush_mmhub(std::string* error_text) = 0;
  virtual bool flush_gc(std::string* error_text) = 0;
};

struct FixedPageTablePages {
  uint64_t root_page;
  uint64_t pdb1_page;
  uint64_t pdb0_page;
  uint64_t ptb0_page;
};

// Maps resident VRAM through C0's fixed four-level tree. Root, PDB1, PDB0,
// and PTB0 are borrowed C0 pages: this class never allocates, clears, or frees
// them. It owns only PTBs allocated from VramAllocator for PDB0 entries beyond
// PTB0.
class DynamicPageTable {
 public:
  DynamicPageTable(const VramLayout& layout, VramAllocator& allocator,
                   DynamicPageTableBackend* backend, FixedPageTablePages fixed_pages);

  bool map_range(uint64_t gpu_virtual_address, uint64_t physical_address,
                 uint64_t size_bytes, std::string* error_text);
  bool unmap_range(uint64_t gpu_virtual_address, uint64_t size_bytes,
                   std::string* error_text);

  // Evidence-only view of currently allocated dynamic PTBs. A zero means no
  // live dynamic PTB; no accessor exposes mutable table ownership.
  uint64_t dynamic_ptb_count() const;
  uint64_t first_dynamic_ptb_physical_offset() const;

 private:
  struct DynamicPtb {
    VramAllocation allocation;
    uint16_t pdb0_index;
    uint32_t mapped_leaf_count;
  };

  struct Leaf {
    uint16_t pdb0_index;
    uint16_t ptb_index;
  };

  bool validate_range(uint64_t gpu_virtual_address, uint64_t physical_address,
                      uint64_t size_bytes, bool requires_physical_address,
                      std::string* error_text) const;
  bool is_c0_tree_address(uint64_t gpu_virtual_address) const;
  bool readback_equals(uint64_t table_page, uint16_t entry_index,
                       uint64_t expected_pte, std::string* error_text);
  bool allocate_ptb(uint16_t pdb0_index, uint32_t mapped_leaf_count,
                    std::string* error_text);
  uint64_t ptb_page_for(uint16_t pdb0_index) const;

  VramAllocator& allocator_;
  DynamicPageTableBackend* backend_;
  FixedPageTablePages fixed_pages_;
  uint64_t resident_gpu_va_base_;
  uint64_t resident_gpu_va_limit_;
  std::map<uint16_t, DynamicPtb> dynamic_ptbs_;
  std::map<uint64_t, Leaf> leaves_;
};

}  // namespace native_r9700

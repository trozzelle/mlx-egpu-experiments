#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "vram_allocator.h"
#include "vram_layout.h"

namespace native_r9700 {

// Identifies a PDB0 page by its position beneath the fixed PDB2 entry.
struct PageTableKey {
  uint16_t pdb1_index;
  uint16_t pdb0_index;

  bool operator<(const PageTableKey& other) const {
    return pdb1_index != other.pdb1_index ? pdb1_index < other.pdb1_index
                                          : pdb0_index < other.pdb0_index;
  }
};

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

// Maps resident VRAM through C0's fixed PDB2/PDB1 tree. Root, the base PDB0,
// and the base PTB are borrowed C0 pages: this class never allocates, clears,
// or frees them. It owns later PDB0 pages and PTBs below every non-base pair.
class DynamicPageTable {
 public:
  DynamicPageTable(const VramLayout& layout, VramAllocator& allocator,
                   DynamicPageTableBackend* backend, FixedPageTablePages fixed_pages);

  bool map_range(uint64_t gpu_virtual_address, uint64_t physical_address,
                 uint64_t size_bytes, std::string* error_text);
  bool unmap_range(uint64_t gpu_virtual_address, uint64_t size_bytes,
                   std::string* error_text);

  // Evidence-only views of currently allocated dynamic hierarchy pages. A
  // zero count means no live page; no accessor exposes mutable ownership.
  uint64_t dynamic_pdb0_count() const;
  uint64_t first_dynamic_pdb0_physical_offset() const;
  uint64_t dynamic_ptb_count() const;
  uint64_t first_dynamic_ptb_physical_offset() const;

 private:
  struct DynamicPdb0 {
    VramAllocation allocation;
    uint16_t pdb1_index;
    uint32_t linked_ptb_count;
    // parent_linked is conservative after an uncertain write/readback: cleanup
    // must clear the slot before releasing the allocation.
    bool parent_linked;
    bool parent_link_confirmed;
  };

  struct DynamicPtb {
    VramAllocation allocation;
    PageTableKey key;
    uint32_t mapped_leaf_count;
    // As with DynamicPdb0, retain uncertain links until an explicit unmap.
    bool parent_linked;
    bool parent_link_confirmed;
  };

  struct Leaf {
    uint16_t pdb1_index;
    uint16_t pdb0_index;
    uint16_t ptb_index;
  };

  bool validate_range(uint64_t gpu_virtual_address, uint64_t physical_address,
                      uint64_t size_bytes, bool requires_physical_address,
                      std::string* error_text) const;
  bool is_c0_tree_address(uint64_t gpu_virtual_address) const;
  bool readback_equals(uint64_t table_page, uint16_t entry_index,
                       uint64_t expected_pte, std::string* error_text);
  bool ensure_dynamic_pdb0(uint16_t pdb1_index, std::string* error_text);
  bool ensure_dynamic_ptb(const PageTableKey& key, uint32_t mapped_leaf_count,
                          std::string* error_text);
  void quarantine_ptb(const PageTableKey& key, uint32_t mapped_leaf_count);
  uint64_t pdb0_page_for(uint16_t pdb1_index) const;
  uint64_t ptb_page_for(const PageTableKey& key) const;

  VramAllocator& allocator_;
  DynamicPageTableBackend* backend_;
  FixedPageTablePages fixed_pages_;
  uint16_t fixed_pdb1_index_;
  uint16_t fixed_pdb0_index_;
  uint64_t resident_gpu_va_base_;
  uint64_t resident_gpu_va_limit_;
  std::map<uint16_t, DynamicPdb0> dynamic_pdb0s_;
  std::map<PageTableKey, DynamicPtb> dynamic_ptbs_;
  std::map<uint64_t, Leaf> leaves_;
};

}  // namespace native_r9700

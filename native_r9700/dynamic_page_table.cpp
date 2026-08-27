#include "dynamic_page_table.h"

#include <limits>
#include <string>
#include <utility>

namespace native_r9700 {
namespace {

constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kPhysicalAddressMask = 0x0000FFFFFFFFF000ULL;
constexpr uint64_t kPteValid = 1ULL << 0;
constexpr uint64_t kPteLeafVramCached = 0x8000000000000071ULL;
constexpr uint64_t kC0VirtualBase = 0x0000200000000000ULL;
constexpr uint64_t kPdb2Shift = 39;
constexpr uint64_t kPdb1Shift = 30;
constexpr uint64_t kPdb0Shift = 21;
constexpr uint64_t kPtbShift = 12;
constexpr uint64_t kPdb2EntryMask = 0x3FFULL;
constexpr uint64_t kEntryMask = 0x1FFULL;

bool fail(std::string* error_text, const char* message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

uint16_t entry_at(uint64_t address, uint64_t shift,
                  uint64_t entry_mask = kEntryMask) {
  return static_cast<uint16_t>((address >> shift) & entry_mask);
}

}  // namespace

DynamicPageTable::DynamicPageTable(const VramLayout& layout,
                                   VramAllocator& allocator,
                                   DynamicPageTableBackend* backend,
                                   FixedPageTablePages fixed_pages)
    : allocator_(allocator),
      backend_(backend),
      fixed_pages_(fixed_pages),
      fixed_pdb1_index_(entry_at(kC0VirtualBase, kPdb1Shift)),
      fixed_pdb0_index_(entry_at(kC0VirtualBase, kPdb0Shift)),
      resident_gpu_va_base_(layout.resident_gpu_va_base),
      resident_gpu_va_limit_(layout.resident_gpu_va_limit) {}

bool DynamicPageTable::validate_range(uint64_t gpu_virtual_address,
                                      uint64_t physical_address,
                                      uint64_t size_bytes,
                                      bool requires_physical_address,
                                      std::string* error_text) const {
  if (backend_ == nullptr) return fail(error_text, "dynamic page-table backend is required");
  if (size_bytes == 0 || size_bytes % kPageBytes != 0 ||
      gpu_virtual_address % kPageBytes != 0) {
    return fail(error_text, "page-table ranges must be nonempty and 4 KiB aligned");
  }
  if (gpu_virtual_address > std::numeric_limits<uint64_t>::max() - size_bytes) {
    return fail(error_text, "GPU virtual range overflows");
  }
  const uint64_t gpu_virtual_end = gpu_virtual_address + size_bytes;
  if (gpu_virtual_address < resident_gpu_va_base_ ||
      gpu_virtual_end > resident_gpu_va_limit_) {
    return fail(error_text, "GPU virtual range is outside the resident window");
  }
  if (!requires_physical_address) return true;
  if (physical_address % kPageBytes != 0 ||
      physical_address > kPhysicalAddressMask ||
      physical_address > std::numeric_limits<uint64_t>::max() - size_bytes) {
    return fail(error_text, "physical range is not representable by a PTE");
  }
  const uint64_t physical_last = physical_address + size_bytes - kPageBytes;
  if (physical_last > kPhysicalAddressMask) {
    return fail(error_text, "physical range is not representable by a PTE");
  }
  return true;
}

bool DynamicPageTable::is_c0_tree_address(uint64_t gpu_virtual_address) const {
  // C0's root has one fixed PDB2 entry. PDB1 children beneath that entry are
  // intentionally dynamic, while the base PDB0/PTB pair remains borrowed.
  return entry_at(gpu_virtual_address, kPdb2Shift, kPdb2EntryMask) ==
         entry_at(kC0VirtualBase, kPdb2Shift, kPdb2EntryMask);
}

bool DynamicPageTable::readback_equals(uint64_t table_page, uint16_t entry_index,
                                       uint64_t expected_pte,
                                       std::string* error_text) {
  uint64_t observed_pte = 0;
  if (!backend_->read_pte(table_page, entry_index, &observed_pte, error_text)) {
    return false;
  }
  if (observed_pte != expected_pte) {
    return fail(error_text, "page-table write readback mismatch");
  }
  return true;
}

bool DynamicPageTable::ensure_dynamic_pdb0(uint16_t pdb1_index,
                                           std::string* error_text) {
  if (pdb1_index == fixed_pdb1_index_) return true;

  const auto existing = dynamic_pdb0s_.find(pdb1_index);
  if (existing != dynamic_pdb0s_.end()) {
    if (existing->second.allocation.size_bytes == 0 ||
        !existing->second.parent_linked ||
        !existing->second.parent_link_confirmed) {
      return fail(error_text, "dynamic PDB0 ownership is pending cleanup");
    }
    return true;
  }

  VramAllocation allocation{};
  const std::string allocation_name =
      "dynamic-page-table-pdb0-" + std::to_string(pdb1_index);
  if (!allocator_.allocate(allocation_name, kPageBytes, kPageBytes, &allocation,
                           error_text)) {
    return false;
  }

  if (!backend_->zero_page(allocation.physical_offset, error_text)) {
    std::string release_error;
    const bool released = allocator_.release(allocation, &release_error);
    if (released) {
      allocation = VramAllocation{};
    } else {
      fail(error_text, "failed to release an unlinked dynamic PDB0");
    }
    dynamic_pdb0s_.emplace(
        pdb1_index,
        DynamicPdb0{std::move(allocation), pdb1_index, 0, false, false});
    return false;
  }

  const auto inserted = dynamic_pdb0s_.emplace(
      pdb1_index,
      DynamicPdb0{std::move(allocation), pdb1_index, 0, true, false});
  if (!inserted.second) return fail(error_text, "dynamic PDB0 already exists");

  DynamicPdb0& owner = inserted.first->second;
  const uint64_t parent_pte =
      (owner.allocation.physical_offset & kPhysicalAddressMask) | kPteValid;
  // The write may have taken effect even when the transport reports failure.
  owner.parent_linked = true;
  if (!backend_->write_pte(fixed_pages_.pdb1_page, pdb1_index, parent_pte,
                           error_text) ||
      !readback_equals(fixed_pages_.pdb1_page, pdb1_index, parent_pte,
                       error_text)) {
    return false;
  }
  owner.parent_link_confirmed = true;
  return true;
}

void DynamicPageTable::quarantine_ptb(const PageTableKey& key,
                                      uint32_t mapped_leaf_count) {
  dynamic_ptbs_.emplace(
      key, DynamicPtb{VramAllocation{}, key, mapped_leaf_count, false, false});
}

bool DynamicPageTable::ensure_dynamic_ptb(const PageTableKey& key,
                                          uint32_t mapped_leaf_count,
                                          std::string* error_text) {
  if (key.pdb1_index == fixed_pdb1_index_ &&
      key.pdb0_index == fixed_pdb0_index_) {
    return true;
  }

  const auto existing = dynamic_ptbs_.find(key);
  if (existing != dynamic_ptbs_.end()) {
    if (existing->second.allocation.size_bytes == 0 ||
        !existing->second.parent_linked ||
        !existing->second.parent_link_confirmed) {
      return fail(error_text, "dynamic PTB ownership is pending cleanup");
    }
    return true;
  }

  VramAllocation allocation{};
  const std::string allocation_name =
      "dynamic-page-table-ptb-" + std::to_string(key.pdb1_index) + "-" +
      std::to_string(key.pdb0_index);
  if (!allocator_.allocate(allocation_name, kPageBytes, kPageBytes, &allocation,
                           error_text)) {
    // The map's leaf identity still needs an explicit cleanup path even when
    // no physical page could be allocated.
    quarantine_ptb(key, mapped_leaf_count);
    return false;
  }

  if (!backend_->zero_page(allocation.physical_offset, error_text)) {
    std::string release_error;
    const bool released = allocator_.release(allocation, &release_error);
    if (released) {
      allocation = VramAllocation{};
    } else {
      fail(error_text, "failed to release an unlinked dynamic PTB");
    }
    dynamic_ptbs_.emplace(
        key, DynamicPtb{std::move(allocation), key, mapped_leaf_count, false, false});
    return false;
  }

  const auto inserted = dynamic_ptbs_.emplace(
      key, DynamicPtb{std::move(allocation), key, mapped_leaf_count, true, false});
  if (!inserted.second) return fail(error_text, "dynamic PTB already exists");

  DynamicPtb& owner = inserted.first->second;
  const uint64_t pdb0_page = pdb0_page_for(key.pdb1_index);
  if (pdb0_page == 0) return fail(error_text, "dynamic PDB0 page is unavailable");
  const uint64_t parent_pte =
      (owner.allocation.physical_offset & kPhysicalAddressMask) | kPteValid;
  if (key.pdb1_index != fixed_pdb1_index_) {
    const auto pdb0 = dynamic_pdb0s_.find(key.pdb1_index);
    if (pdb0 == dynamic_pdb0s_.end()) {
      return fail(error_text, "dynamic PDB0 ownership is missing");
    }
    ++pdb0->second.linked_ptb_count;
  }
  // The parent may have changed even if write/readback fails, so retain both
  // the allocation and a conservative link marker for explicit unmap.
  owner.parent_linked = true;
  if (!backend_->write_pte(pdb0_page, key.pdb0_index, parent_pte, error_text) ||
      !readback_equals(pdb0_page, key.pdb0_index, parent_pte, error_text)) {
    return false;
  }
  owner.parent_link_confirmed = true;
  return true;
}

uint64_t DynamicPageTable::pdb0_page_for(uint16_t pdb1_index) const {
  if (pdb1_index == fixed_pdb1_index_) return fixed_pages_.pdb0_page;
  const auto found = dynamic_pdb0s_.find(pdb1_index);
  if (found == dynamic_pdb0s_.end() ||
      found->second.allocation.size_bytes == 0) {
    return 0;
  }
  return found->second.allocation.physical_offset;
}

uint64_t DynamicPageTable::ptb_page_for(const PageTableKey& key) const {
  if (key.pdb1_index == fixed_pdb1_index_ &&
      key.pdb0_index == fixed_pdb0_index_) {
    return fixed_pages_.ptb0_page;
  }
  const auto found = dynamic_ptbs_.find(key);
  if (found == dynamic_ptbs_.end() ||
      found->second.allocation.size_bytes == 0) {
    return 0;
  }
  return found->second.allocation.physical_offset;
}

uint64_t DynamicPageTable::dynamic_pdb0_count() const {
  uint64_t count = 0;
  for (const auto& entry : dynamic_pdb0s_) {
    if (entry.second.allocation.size_bytes != 0) ++count;
  }
  return count;
}

uint64_t DynamicPageTable::first_dynamic_pdb0_physical_offset() const {
  for (const auto& entry : dynamic_pdb0s_) {
    if (entry.second.allocation.size_bytes != 0) {
      return entry.second.allocation.physical_offset;
    }
  }
  return 0;
}

uint64_t DynamicPageTable::dynamic_ptb_count() const {
  uint64_t count = 0;
  for (const auto& entry : dynamic_ptbs_) {
    if (entry.second.allocation.size_bytes != 0) ++count;
  }
  return count;
}

uint64_t DynamicPageTable::first_dynamic_ptb_physical_offset() const {
  for (const auto& entry : dynamic_ptbs_) {
    if (entry.second.allocation.size_bytes != 0) {
      return entry.second.allocation.physical_offset;
    }
  }
  return 0;
}

bool DynamicPageTable::map_range(uint64_t gpu_virtual_address,
                                 uint64_t physical_address,
                                 uint64_t size_bytes,
                                 std::string* error_text) {
  if (!validate_range(gpu_virtual_address, physical_address, size_bytes, true,
                      error_text)) {
    return false;
  }

  // Complete every collision and quarantine check before allocating, clearing,
  // writing, reading, or flushing. A failed later group can therefore retain
  // its exact leaf identity for explicit cleanup without touching a collision.
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    const uint64_t virtual_page = gpu_virtual_address + offset;
    if (!is_c0_tree_address(virtual_page)) {
      return fail(error_text, "GPU virtual range is outside the fixed C0 page-table tree");
    }
    if (leaves_.find(virtual_page) != leaves_.end()) {
      return fail(error_text, "GPU virtual range collides with an existing mapping");
    }
    const PageTableKey key{entry_at(virtual_page, kPdb1Shift),
                           entry_at(virtual_page, kPdb0Shift)};
    if (key.pdb1_index != fixed_pdb1_index_) {
      const auto pdb0 = dynamic_pdb0s_.find(key.pdb1_index);
      if (pdb0 != dynamic_pdb0s_.end() &&
          (pdb0->second.allocation.size_bytes == 0 ||
           !pdb0->second.parent_linked ||
           !pdb0->second.parent_link_confirmed)) {
        return fail(error_text, "dynamic PDB0 ownership is pending cleanup");
      }
    }
    if (!(key.pdb1_index == fixed_pdb1_index_ &&
          key.pdb0_index == fixed_pdb0_index_)) {
      const auto ptb = dynamic_ptbs_.find(key);
      if (ptb != dynamic_ptbs_.end() &&
          (ptb->second.allocation.size_bytes == 0 ||
           !ptb->second.parent_linked ||
           !ptb->second.parent_link_confirmed)) {
        return fail(error_text, "dynamic PTB ownership is pending cleanup");
      }
    }
  }

  for (uint64_t offset = 0; offset < size_bytes;) {
    const uint64_t group_offset = offset;
    const uint64_t group_virtual_page = gpu_virtual_address + group_offset;
    const PageTableKey key{entry_at(group_virtual_page, kPdb1Shift),
                           entry_at(group_virtual_page, kPdb0Shift)};
    uint64_t group_size = kPageBytes;
    while (group_size < size_bytes - group_offset) {
      const uint64_t next_page = group_virtual_page + group_size;
      if (entry_at(next_page, kPdb1Shift) != key.pdb1_index ||
          entry_at(next_page, kPdb0Shift) != key.pdb0_index) {
        break;
      }
      group_size += kPageBytes;
    }
    const uint32_t group_leaf_count =
        static_cast<uint32_t>(group_size / kPageBytes);

    for (uint64_t group_page_offset = 0; group_page_offset < group_size;
         group_page_offset += kPageBytes) {
      const uint64_t virtual_page = group_virtual_page + group_page_offset;
      leaves_.emplace(
          virtual_page,
          Leaf{key.pdb1_index, key.pdb0_index,
               entry_at(virtual_page, kPtbShift)});
    }

    if (key.pdb1_index != fixed_pdb1_index_ &&
        !ensure_dynamic_pdb0(key.pdb1_index, error_text)) {
      // Allocation failure creates no PDB0 identity; all other setup failures
      // retain one and receive a zero-sized PTB identity below for cleanup.
      if (dynamic_pdb0s_.find(key.pdb1_index) == dynamic_pdb0s_.end()) {
        for (uint64_t group_page_offset = 0; group_page_offset < group_size;
             group_page_offset += kPageBytes) {
          leaves_.erase(group_virtual_page + group_page_offset);
        }
      } else {
        quarantine_ptb(key, group_leaf_count);
      }
      return false;
    }

    const bool is_fixed_pair =
        key.pdb1_index == fixed_pdb1_index_ &&
        key.pdb0_index == fixed_pdb0_index_;
    if (!is_fixed_pair) {
      const auto existing_ptb = dynamic_ptbs_.find(key);
      if (!ensure_dynamic_ptb(key, group_leaf_count, error_text)) {
        if (dynamic_ptbs_.find(key) == dynamic_ptbs_.end()) {
          for (uint64_t group_page_offset = 0; group_page_offset < group_size;
               group_page_offset += kPageBytes) {
            leaves_.erase(group_virtual_page + group_page_offset);
          }
        }
        return false;
      }
      if (existing_ptb != dynamic_ptbs_.end()) {
        existing_ptb->second.mapped_leaf_count += group_leaf_count;
      }
    }

    const uint64_t ptb_page = ptb_page_for(key);
    if (ptb_page == 0) return fail(error_text, "dynamic PTB page is unavailable");
    for (uint64_t group_page_offset = 0; group_page_offset < group_size;
         group_page_offset += kPageBytes) {
      const uint64_t virtual_page = group_virtual_page + group_page_offset;
      const uint64_t leaf_pte =
          ((physical_address + group_offset + group_page_offset) &
           kPhysicalAddressMask) |
          kPteLeafVramCached;
      if (!backend_->write_pte(ptb_page, entry_at(virtual_page, kPtbShift),
                               leaf_pte, error_text) ||
          !readback_equals(ptb_page, entry_at(virtual_page, kPtbShift),
                           leaf_pte, error_text)) {
        return false;
      }
    }
    offset += group_size;
  }

  if (!backend_->flush_gc(error_text)) return false;
  return backend_->flush_mmhub(error_text);
}

bool DynamicPageTable::unmap_range(uint64_t gpu_virtual_address,
                                   uint64_t size_bytes,
                                   std::string* error_text) {
  if (!validate_range(gpu_virtual_address, 0, size_bytes, false, error_text)) {
    return false;
  }
  const uint64_t range_end = gpu_virtual_address + size_bytes;

  // Build the complete ownership set before the first PTE mutation.
  std::map<PageTableKey, uint32_t> selected_leaf_counts;
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    const uint64_t virtual_page = gpu_virtual_address + offset;
    const auto leaf = leaves_.find(virtual_page);
    const PageTableKey expected{entry_at(virtual_page, kPdb1Shift),
                                entry_at(virtual_page, kPdb0Shift)};
    if (!is_c0_tree_address(virtual_page) || leaf == leaves_.end()) {
      return fail(error_text, "GPU virtual range is not dynamically mapped");
    }
    if (leaf->second.pdb1_index != expected.pdb1_index ||
        leaf->second.pdb0_index != expected.pdb0_index ||
        leaf->second.ptb_index != entry_at(virtual_page, kPtbShift)) {
      return fail(error_text, "page-table leaf ownership is inconsistent");
    }
    ++selected_leaf_counts[expected];
  }

  for (const auto& selected : selected_leaf_counts) {
    const bool is_fixed_pair =
        selected.first.pdb1_index == fixed_pdb1_index_ &&
        selected.first.pdb0_index == fixed_pdb0_index_;
    if (is_fixed_pair) continue;
    const auto ptb = dynamic_ptbs_.find(selected.first);
    if (ptb == dynamic_ptbs_.end() ||
        ptb->second.mapped_leaf_count < selected.second) {
      return fail(error_text, "dynamic PTB ownership is inconsistent");
    }
    if (selected.first.pdb1_index != fixed_pdb1_index_) {
      const auto pdb0 = dynamic_pdb0s_.find(selected.first.pdb1_index);
      if (pdb0 == dynamic_pdb0s_.end()) {
        return fail(error_text, "dynamic PDB0 ownership is inconsistent");
      }
      if (pdb0->second.allocation.size_bytes == 0 &&
          pdb0->second.linked_ptb_count != 0) {
        return fail(error_text, "dynamic PDB0 child ownership is inconsistent");
      }
    }
  }

  bool leaf_mutation = false;
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    const uint64_t virtual_page = gpu_virtual_address + offset;
    const Leaf& leaf = leaves_.find(virtual_page)->second;
    const PageTableKey key{leaf.pdb1_index, leaf.pdb0_index};
    const bool is_fixed_pair =
        key.pdb1_index == fixed_pdb1_index_ &&
        key.pdb0_index == fixed_pdb0_index_;
    const auto ptb = dynamic_ptbs_.find(key);
    if (!is_fixed_pair &&
        (ptb == dynamic_ptbs_.end() ||
         ptb->second.allocation.size_bytes == 0 ||
         !ptb->second.parent_linked ||
         !ptb->second.parent_link_confirmed)) {
      continue;
    }
    const uint64_t ptb_page = ptb_page_for(key);
    if (ptb_page == 0 ||
        !backend_->write_pte(ptb_page, leaf.ptb_index, 0, error_text) ||
        !readback_equals(ptb_page, leaf.ptb_index, 0, error_text)) {
      return false;
    }
    leaf_mutation = true;
  }

  bool ptb_parent_mutation = false;
  for (const auto& selected : selected_leaf_counts) {
    const bool is_fixed_pair =
        selected.first.pdb1_index == fixed_pdb1_index_ &&
        selected.first.pdb0_index == fixed_pdb0_index_;
    if (is_fixed_pair) continue;
    DynamicPtb& ptb = dynamic_ptbs_.find(selected.first)->second;
    if (ptb.mapped_leaf_count != selected.second ||
        !ptb.parent_linked || ptb.allocation.size_bytes == 0) {
      continue;
    }
    const uint64_t pdb0_page = pdb0_page_for(selected.first.pdb1_index);
    if (pdb0_page == 0 ||
        !backend_->write_pte(pdb0_page, selected.first.pdb0_index, 0,
                             error_text) ||
        !readback_equals(pdb0_page, selected.first.pdb0_index, 0,
                         error_text)) {
      return false;
    }
    ptb_parent_mutation = true;
  }

  const bool first_flush_needed = leaf_mutation || ptb_parent_mutation;
  if (first_flush_needed &&
      (!backend_->flush_mmhub(error_text) || !backend_->flush_gc(error_text))) {
    return false;
  }

  // Children are now detached and flushed. Release each PTB before deciding
  // whether its dynamic PDB0 parent can be detached. Records stay in place
  // until the complete operation succeeds, making a release failure retryable.
  for (const auto& selected : selected_leaf_counts) {
    const bool is_fixed_pair =
        selected.first.pdb1_index == fixed_pdb1_index_ &&
        selected.first.pdb0_index == fixed_pdb0_index_;
    if (is_fixed_pair) continue;
    DynamicPtb& ptb = dynamic_ptbs_.find(selected.first)->second;
    if (ptb.mapped_leaf_count != selected.second ||
        ptb.allocation.size_bytes == 0) {
      continue;
    }
    const bool was_parent_linked = ptb.parent_linked;
    if (!allocator_.release(ptb.allocation, error_text)) return false;
    ptb.allocation = VramAllocation{};
    ptb.parent_linked = false;
    ptb.parent_link_confirmed = false;
    if (was_parent_linked &&
        selected.first.pdb1_index != fixed_pdb1_index_) {
      DynamicPdb0& pdb0 =
          dynamic_pdb0s_.find(selected.first.pdb1_index)->second;
      if (pdb0.linked_ptb_count == 0) {
        return fail(error_text, "dynamic PDB0 child ownership underflow");
      }
      --pdb0.linked_ptb_count;
    }
  }

  // A dynamic PDB0 is empty only after all of its leaves and all owned PTB
  // allocations have been removed. Determine that fact without erasing the
  // leaf records needed by a retry.
  std::map<uint16_t, bool> empty_dynamic_pdb0s;
  for (const auto& selected : selected_leaf_counts) {
    const uint16_t pdb1_index = selected.first.pdb1_index;
    if (pdb1_index == fixed_pdb1_index_) continue;
    if (empty_dynamic_pdb0s.find(pdb1_index) != empty_dynamic_pdb0s.end()) {
      continue;
    }
    bool has_remaining_leaf = false;
    for (const auto& remaining : leaves_) {
      if (remaining.second.pdb1_index != pdb1_index) continue;
      if (remaining.first < gpu_virtual_address ||
          remaining.first >= range_end) {
        has_remaining_leaf = true;
        break;
      }
    }
    bool has_live_child_allocation = false;
    for (const auto& child : dynamic_ptbs_) {
      if (child.first.pdb1_index == pdb1_index &&
          child.second.allocation.size_bytes != 0) {
        has_live_child_allocation = true;
        break;
      }
    }
    empty_dynamic_pdb0s.emplace(pdb1_index,
                                !has_remaining_leaf &&
                                    !has_live_child_allocation);
  }

  bool pdb0_parent_mutation = false;
  for (const auto& empty : empty_dynamic_pdb0s) {
    if (!empty.second) continue;
    DynamicPdb0& pdb0 = dynamic_pdb0s_.find(empty.first)->second;
    if (pdb0.allocation.size_bytes == 0 || !pdb0.parent_linked) continue;
    if (!backend_->write_pte(fixed_pages_.pdb1_page, empty.first, 0,
                             error_text) ||
        !readback_equals(fixed_pages_.pdb1_page, empty.first, 0,
                         error_text)) {
      return false;
    }
    pdb0_parent_mutation = true;
  }

  if (pdb0_parent_mutation &&
      (!backend_->flush_mmhub(error_text) || !backend_->flush_gc(error_text))) {
    return false;
  }

  for (const auto& empty : empty_dynamic_pdb0s) {
    if (!empty.second) continue;
    DynamicPdb0& pdb0 = dynamic_pdb0s_.find(empty.first)->second;
    if (pdb0.allocation.size_bytes == 0) continue;
    if (!allocator_.release(pdb0.allocation, error_text)) return false;
    pdb0.allocation = VramAllocation{};
    pdb0.parent_linked = false;
    pdb0.parent_link_confirmed = false;
    pdb0.linked_ptb_count = 0;
  }

  // All visible updates, flushes, and releases succeeded. Remove identities
  // only now so any failure above can retry the exact same range.
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    leaves_.erase(gpu_virtual_address + offset);
  }
  for (const auto& selected : selected_leaf_counts) {
    const bool is_fixed_pair =
        selected.first.pdb1_index == fixed_pdb1_index_ &&
        selected.first.pdb0_index == fixed_pdb0_index_;
    if (is_fixed_pair) continue;
    auto ptb = dynamic_ptbs_.find(selected.first);
    if (ptb == dynamic_ptbs_.end()) continue;
    if (ptb->second.mapped_leaf_count == selected.second) {
      dynamic_ptbs_.erase(ptb);
    } else {
      ptb->second.mapped_leaf_count -= selected.second;
    }
  }
  for (const auto& empty : empty_dynamic_pdb0s) {
    if (!empty.second) continue;
    const auto pdb0 = dynamic_pdb0s_.find(empty.first);
    if (pdb0 != dynamic_pdb0s_.end() &&
        pdb0->second.allocation.size_bytes == 0 &&
        pdb0->second.linked_ptb_count == 0) {
      dynamic_pdb0s_.erase(pdb0);
    }
  }
  return true;
}

}  // namespace native_r9700

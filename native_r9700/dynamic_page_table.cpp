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
  return entry_at(gpu_virtual_address, kPdb2Shift, kPdb2EntryMask) ==
             entry_at(kC0VirtualBase, kPdb2Shift, kPdb2EntryMask) &&
         entry_at(gpu_virtual_address, kPdb1Shift) ==
             entry_at(kC0VirtualBase, kPdb1Shift);
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

bool DynamicPageTable::allocate_ptb(uint16_t pdb0_index,
                                    uint32_t mapped_leaf_count,
                                    std::string* error_text) {
  VramAllocation allocation{};
  const std::string allocation_name =
      "dynamic-page-table-ptb-" + std::to_string(pdb0_index);
  if (!allocator_.allocate(allocation_name, kPageBytes, kPageBytes, &allocation,
                           error_text)) {
    return false;
  }
  if (!backend_->zero_page(allocation.physical_offset, error_text)) {
    std::string release_error;
    if (!allocator_.release(allocation, &release_error)) {
      return fail(error_text, "failed to release an unlinked dynamic PTB");
    }
    // The owning map range is already recorded in leaves_. Keep a zero-sized
    // entry as its pending cleanup identity: no parent PTE was linked and no
    // leaf PTE can have been written.
    dynamic_ptbs_.emplace(pdb0_index,
                          DynamicPtb{VramAllocation{}, pdb0_index,
                                     mapped_leaf_count});
    return false;
  }

  const auto inserted = dynamic_ptbs_.emplace(
      pdb0_index, DynamicPtb{std::move(allocation), pdb0_index, mapped_leaf_count});
  if (!inserted.second) return fail(error_text, "dynamic PTB already exists");

  const uint64_t parent_pte =
      (inserted.first->second.allocation.physical_offset & kPhysicalAddressMask) |
      kPteValid;
  if (!backend_->write_pte(fixed_pages_.pdb0_page, pdb0_index, parent_pte,
                           error_text) ||
      !readback_equals(fixed_pages_.pdb0_page, pdb0_index, parent_pte,
                       error_text)) {
    return false;
  }
  return true;
}

uint64_t DynamicPageTable::ptb_page_for(uint16_t pdb0_index) const {
  if (pdb0_index == 0) return fixed_pages_.ptb0_page;
  const auto found = dynamic_ptbs_.find(pdb0_index);
  return found == dynamic_ptbs_.end() ? 0 : found->second.allocation.physical_offset;
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

  // Complete the collision pass before allocating, clearing, writing, reading,
  // or flushing. Every group becomes owned before its parent or leaf PTE can
  // change, so a failed operation remains explicitly cleanable.
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    const uint64_t virtual_page = gpu_virtual_address + offset;
    if (!is_c0_tree_address(virtual_page)) {
      return fail(error_text, "GPU virtual range is outside the fixed C0 page-table tree");
    }
    if (leaves_.find(virtual_page) != leaves_.end()) {
      return fail(error_text, "GPU virtual range collides with an existing mapping");
    }
  }

  for (uint64_t offset = 0; offset < size_bytes;) {
    const uint64_t group_offset = offset;
    const uint64_t group_virtual_page = gpu_virtual_address + group_offset;
    const uint16_t pdb0_index = entry_at(group_virtual_page, kPdb0Shift);
    uint64_t group_size = kPageBytes;
    while (group_size < size_bytes - group_offset &&
           entry_at(group_virtual_page + group_size, kPdb0Shift) == pdb0_index) {
      group_size += kPageBytes;
    }
    const uint32_t group_leaf_count =
        static_cast<uint32_t>(group_size / kPageBytes);

    for (uint64_t group_page_offset = 0; group_page_offset < group_size;
         group_page_offset += kPageBytes) {
      const uint64_t virtual_page = group_virtual_page + group_page_offset;
      leaves_.emplace(virtual_page,
                      Leaf{pdb0_index, entry_at(virtual_page, kPtbShift)});
    }

    if (pdb0_index != 0) {
      const auto existing_ptb = dynamic_ptbs_.find(pdb0_index);
      if (existing_ptb == dynamic_ptbs_.end()) {
        if (!allocate_ptb(pdb0_index, group_leaf_count, error_text)) {
          // A failed PTB zero releases its allocation but leaves the
          // zero-sized dynamic_ptbs_ entry and leaves_ records pending for
          // explicit cleanup. Other allocation failures own no range.
          if (dynamic_ptbs_.find(pdb0_index) == dynamic_ptbs_.end()) {
            for (uint64_t group_page_offset = 0; group_page_offset < group_size;
                 group_page_offset += kPageBytes) {
              leaves_.erase(group_virtual_page + group_page_offset);
            }
          }
          return false;
        }
      } else {
        existing_ptb->second.mapped_leaf_count += group_leaf_count;
      }
    }

    for (uint64_t group_page_offset = 0; group_page_offset < group_size;
         group_page_offset += kPageBytes) {
      const uint64_t virtual_page = group_virtual_page + group_page_offset;
      const uint64_t leaf_pte =
          ((physical_address + group_offset + group_page_offset) &
           kPhysicalAddressMask) |
          kPteLeafVramCached;
      const uint64_t ptb_page = ptb_page_for(pdb0_index);
      if (ptb_page == 0 ||
          !backend_->write_pte(ptb_page, entry_at(virtual_page, kPtbShift),
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

  // Validate and count all ownership before the first PTE mutation. Records
  // remain intact through leaf/parent readback and both flushes so any failure
  // can be retried by this same explicit cleanup operation.
  std::map<uint16_t, uint32_t> selected_dynamic_leaf_counts;
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    const uint64_t virtual_page = gpu_virtual_address + offset;
    const auto leaf = leaves_.find(virtual_page);
    if (!is_c0_tree_address(virtual_page) || leaf == leaves_.end()) {
      return fail(error_text, "GPU virtual range is not dynamically mapped");
    }
    if (leaf->second.pdb0_index != 0) {
      ++selected_dynamic_leaf_counts[leaf->second.pdb0_index];
    }
  }

  for (const auto& selected : selected_dynamic_leaf_counts) {
    const auto ptb = dynamic_ptbs_.find(selected.first);
    if (ptb == dynamic_ptbs_.end() ||
        ptb->second.mapped_leaf_count < selected.second) {
      return fail(error_text, "dynamic PTB ownership is inconsistent");
    }
  }

  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    const Leaf& leaf = leaves_.find(gpu_virtual_address + offset)->second;
    const auto ptb = dynamic_ptbs_.find(leaf.pdb0_index);
    if (leaf.pdb0_index != 0 && ptb->second.allocation.size_bytes == 0) {
      continue;
    }
    const uint64_t ptb_page = ptb_page_for(leaf.pdb0_index);
    if (ptb_page == 0 ||
        !backend_->write_pte(ptb_page, leaf.ptb_index, 0, error_text) ||
        !readback_equals(ptb_page, leaf.ptb_index, 0, error_text)) {
      return false;
    }

  }

  for (const auto& selected : selected_dynamic_leaf_counts) {
    const DynamicPtb& ptb = dynamic_ptbs_.find(selected.first)->second;
    if (ptb.allocation.size_bytes == 0 ||
        ptb.mapped_leaf_count != selected.second) {
      continue;
    }
    if (!backend_->write_pte(fixed_pages_.pdb0_page, selected.first, 0,
                             error_text) ||
        !readback_equals(fixed_pages_.pdb0_page, selected.first, 0,
                         error_text)) {
      return false;
    }
  }

  if (!backend_->flush_mmhub(error_text) || !backend_->flush_gc(error_text)) {
    return false;
  }

  for (const auto& selected : selected_dynamic_leaf_counts) {
    const auto ptb = dynamic_ptbs_.find(selected.first);
    if (ptb->second.allocation.size_bytes == 0) {
      if (ptb->second.mapped_leaf_count == selected.second) {
        dynamic_ptbs_.erase(ptb);
      }
      continue;
    }
    if (ptb->second.mapped_leaf_count != selected.second) continue;
    if (!allocator_.release(ptb->second.allocation, error_text)) return false;
    dynamic_ptbs_.erase(ptb);
    for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
      const uint64_t virtual_page = gpu_virtual_address + offset;
      if (leaves_.find(virtual_page)->second.pdb0_index == selected.first) {
        leaves_.erase(virtual_page);
      }
    }
  }
  for (const auto& selected : selected_dynamic_leaf_counts) {
    const auto ptb = dynamic_ptbs_.find(selected.first);
    if (ptb != dynamic_ptbs_.end()) {
      ptb->second.mapped_leaf_count -= selected.second;
    }
  }
  for (uint64_t offset = 0; offset < size_bytes; offset += kPageBytes) {
    leaves_.erase(gpu_virtual_address + offset);
  }
  return true;
}

}  // namespace native_r9700

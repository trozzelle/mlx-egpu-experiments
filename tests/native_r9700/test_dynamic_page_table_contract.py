"""No-hardware RED contracts for dynamic R9700 page-table updates."""

from pathlib import Path
import subprocess


DYNAMIC_PAGE_TABLE_HEADER = Path("native_r9700/dynamic_page_table.h")
DYNAMIC_PAGE_TABLE_SOURCE = Path("native_r9700/dynamic_page_table.cpp")
ALLOCATOR_HEADER = Path("native_r9700/vram_allocator.h")
ALLOCATOR_SOURCE = Path("native_r9700/vram_allocator.cpp")
LAYOUT_HEADER = Path("native_r9700/vram_layout.h")
LAYOUT_SOURCE = Path("native_r9700/vram_layout.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_dynamic_page_table_probe(tmp_path: Path) -> Path:
    """Compile pure page-table planning against an injected in-memory BAR backend."""
    assert DYNAMIC_PAGE_TABLE_HEADER.is_file() and DYNAMIC_PAGE_TABLE_SOURCE.is_file(), (
        "Dynamic page-table implementation is missing"
    )
    assert ALLOCATOR_HEADER.is_file() and ALLOCATOR_SOURCE.is_file(), (
        "Vram allocator implementation is missing"
    )
    assert LAYOUT_HEADER.is_file() and LAYOUT_SOURCE.is_file(), "Vram layout implementation is missing"

    probe_source = tmp_path / "dynamic_page_table_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "dynamic_page_table.h"
#include "vram_allocator.h"
#include "vram_layout.h"

namespace {

constexpr uint64_t kPageBytes = 1ULL << 12;
constexpr uint64_t kPtbBytes = 1ULL << 21;
constexpr uint64_t kFixedRootPage = 0x00000000ULL;
constexpr uint64_t kFixedPdb1Page = 0x02000000ULL;
constexpr uint64_t kFixedPdb0Page = 0x02001000ULL;
constexpr uint64_t kFixedPtb0Page = 0x02002000ULL;
constexpr uint64_t kFirstDynamicPage = 0x02004000ULL;
constexpr uint64_t kResidentVaBase = 0x0000200000011000ULL;
constexpr uint64_t kCrossPtbVa = 0x0000200000200000ULL;
constexpr uint64_t kLeafPhysical = 0x04000000ULL;
constexpr uint64_t kSecondLeafPhysical = kLeafPhysical + kPageBytes;
constexpr uint64_t kTailReservedBytes = 64ULL << 20;
constexpr uint64_t kResidentVaLimit = kCrossPtbVa + kPtbBytes;
constexpr uint64_t kOneGib = 1ULL << 30;
constexpr uint64_t kBasePdb1Va = kResidentVaBase & ~(kOneGib - 1);
constexpr uint64_t kCrossPdb1Va = kBasePdb1Va + kOneGib - kPageBytes;
constexpr uint64_t kSecondPdb1Va = kBasePdb1Va + 2 * kOneGib;
constexpr uint64_t kPdb2AliasedVa = kResidentVaBase + (512ULL << 39);


bool require(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "%s\n", message);
  return condition;
}

native_r9700::VramLayout dynamic_table_layout(uint64_t dynamic_pages) {
  native_r9700::VramLayout layout{};
  layout.vram_bytes = kFirstDynamicPage + dynamic_pages * kPageBytes + kTailReservedBytes;
  layout.discovery_reserved_bytes = 0;
  layout.boot_reserved_bytes = 0;
  layout.page_table_reserved_bytes = 0;
  layout.allocatable_base = kFirstDynamicPage;
  layout.allocatable_bytes = dynamic_pages * kPageBytes;
  layout.c0_reserved_physical_ranges = {{
      {0x00000000ULL, 0x00003000ULL},
      {0x02000000ULL, 0x00004000ULL},
      {0, 0},
  }};
  layout.resident_gpu_va_base = kResidentVaBase;
  layout.resident_gpu_va_limit = kCrossPtbVa + kPtbBytes;
  return layout;
}

enum class OperationKind {
  kZero,
  kWrite,
  kRead,
  kFlushMmhub,
  kFlushGc,
};

enum class FailurePoint {
  kNone,
  kZero,
  kParentWrite,
  kParentRead,
  kLeafWrite,
  kLeafRead,
  kFlushMmhub,
  kFlushGc,
};


struct Operation {
  OperationKind kind;
  uint64_t table_page;
  uint16_t entry_index;
  uint64_t pte;
};

struct PteSlot {
  uint64_t table_page;
  uint16_t entry_index;

  bool operator<(const PteSlot& other) const {
    return table_page != other.table_page ? table_page < other.table_page
                                          : entry_index < other.entry_index;
  }
};

class FakeBarBackend final : public native_r9700::DynamicPageTableBackend {
 public:
  native_r9700::VramAllocator* allocator = nullptr;
  FailurePoint failure = FailurePoint::kNone;
  bool probe_dynamic_page_during_gc = false;
  uint64_t gc_probe_physical = 0;
  std::vector<Operation> operations;

  bool zero_page(uint64_t physical_page, std::string* error_text) override {
    operations.push_back({OperationKind::kZero, physical_page, 0, 0});
    zeroed_pages.push_back(physical_page);
    return !fails(FailurePoint::kZero, error_text);
  }

  bool write_pte(uint64_t table_page, uint16_t entry_index, uint64_t pte,
                 std::string* error_text) override {
    operations.push_back({OperationKind::kWrite, table_page, entry_index, pte});
    ptes[{table_page, entry_index}] = pte;
    const bool parent_page =
        table_page == kFixedPdb0Page || table_page == kFixedPdb1Page ||
        parent_pages.find(table_page) != parent_pages.end() ||
        (pte != 0 && (pte & (1ULL << 63)) == 0);
    if (parent_page && pte != 0) parent_pages[table_page] = true;
    return !fails(parent_page ? FailurePoint::kParentWrite
                              : FailurePoint::kLeafWrite,
                  error_text);
  }

  bool read_pte(uint64_t table_page, uint16_t entry_index, uint64_t* pte,
                std::string* error_text) override {
    const auto found = ptes.find({table_page, entry_index});
    *pte = found == ptes.end() ? 0 : found->second;
    operations.push_back({OperationKind::kRead, table_page, entry_index, *pte});
    const bool parent_page =
        table_page == kFixedPdb0Page || table_page == kFixedPdb1Page ||
        parent_pages.find(table_page) != parent_pages.end() ||
        (*pte != 0 && (*pte & (1ULL << 63)) == 0);
    if (parent_page && *pte != 0) parent_pages[table_page] = true;
    return !fails(parent_page ? FailurePoint::kParentRead
                              : FailurePoint::kLeafRead,
                  error_text);
  }

  bool flush_mmhub(std::string* error_text) override {
    operations.push_back({OperationKind::kFlushMmhub, 0, 0, 0});
    return !fails(FailurePoint::kFlushMmhub, error_text);
  }

  bool flush_gc(std::string* error_text) override {
    operations.push_back({OperationKind::kFlushGc, 0, 0, 0});
    if (fails(FailurePoint::kFlushGc, error_text)) return false;
    if (!probe_dynamic_page_during_gc) return true;

    native_r9700::VramAllocation probe{};
    std::string allocation_error;
    if (!allocator->allocate("gc-liveness-probe", kPageBytes, kPageBytes, &probe,
                             &allocation_error)) {
      return false;
    }
    gc_probe_physical = probe.physical_offset;
    return allocator->release(probe, &allocation_error);
  }

  void clear_operations() { operations.clear(); }
  std::vector<uint64_t> zeroed_pages;

 private:
  bool fails(FailurePoint point, std::string* error_text) {
    if (failure != point) return false;
    failure = FailurePoint::kNone;
    if (error_text != nullptr) *error_text = "injected fake BAR failure";
    return true;
  }

  std::map<PteSlot, uint64_t> ptes;
  std::map<uint64_t, bool> parent_pages;
};

native_r9700::DynamicPageTable dynamic_page_table(
    const native_r9700::VramLayout& layout, native_r9700::VramAllocator* allocator,
    FakeBarBackend* backend) {
  return native_r9700::DynamicPageTable(
      layout, *allocator, backend,
      {kFixedRootPage, kFixedPdb1Page, kFixedPdb0Page, kFixedPtb0Page});
}

bool is_operation(const Operation& operation, OperationKind kind,
                  uint64_t table_page = 0, uint16_t entry_index = 0) {
  return operation.kind == kind && operation.table_page == table_page &&
         operation.entry_index == entry_index;
}

bool maps_ptb0_leaf_without_touching_fixed_tree_pages() {
  const native_r9700::VramLayout layout = dynamic_table_layout(3);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kResidentVaBase, kLeafPhysical, kPageBytes, &error_text),
               "a page at PTB0 index 17 must map through the fixed C0 chain")) {
    return false;
  }

  native_r9700::VramAllocation first_free{};
  if (!require(allocator.allocate("first-free", kPageBytes, kPageBytes, &first_free,
                                  &error_text),
               "an existing PTB0 leaf must not allocate a dynamic page table")) {
    return false;
  }

  return require(first_free.physical_offset == kFirstDynamicPage,
                 "mapping PTB0 index 17 must leave the first dynamic physical page free") &&
         require(backend.zeroed_pages.empty(),
                 "a PTB0 leaf mapping must not zero a fixed root, PDB1, PDB0, or PTB0 page") &&
         require(backend.operations.size() == 4 &&
                     is_operation(backend.operations[0], OperationKind::kWrite,
                                  kFixedPtb0Page, 17) &&
                     backend.operations[0].pte != 0 &&
                     is_operation(backend.operations[1], OperationKind::kRead,
                                  kFixedPtb0Page, 17) &&
                     backend.operations[1].pte == backend.operations[0].pte &&
                     is_operation(backend.operations[2], OperationKind::kFlushGc) &&
                     is_operation(backend.operations[3], OperationKind::kFlushMmhub),
                 "a PTB0 leaf map must write/readback the leaf before GC then MMHUB flushes");
}

bool allocates_new_ptb_at_first_owned_page_across_2mib_boundary() {
  const native_r9700::VramLayout layout = dynamic_table_layout(3);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kCrossPtbVa, kLeafPhysical, kPageBytes, &error_text),
               "the first leaf beyond PTB0 must allocate a dynamic PTB")) {
    return false;
  }

  native_r9700::VramAllocation second_free{};
  if (!require(allocator.allocate("second-free", kPageBytes, kPageBytes, &second_free,
                                  &error_text),
               "a dynamic PTB must reserve its physical page while mapped")) {
    return false;
  }

  return require(backend.zeroed_pages.size() == 1 &&
                     backend.zeroed_pages.front() == kFirstDynamicPage,
                 "the first cross-2MiB dynamic PTB must zero physical page 0x02004000") &&
         require(backend.operations.size() == 7 &&
                     is_operation(backend.operations[0], OperationKind::kZero,
                                  kFirstDynamicPage) &&
                     is_operation(backend.operations[1], OperationKind::kWrite,
                                  kFixedPdb0Page, 1) &&
                     backend.operations[1].pte != 0 &&
                     is_operation(backend.operations[2], OperationKind::kRead,
                                  kFixedPdb0Page, 1) &&
                     backend.operations[2].pte == backend.operations[1].pte &&
                     is_operation(backend.operations[3], OperationKind::kWrite,
                                  kFirstDynamicPage, 0) &&
                     backend.operations[3].pte != 0 &&
                     is_operation(backend.operations[4], OperationKind::kRead,
                                  kFirstDynamicPage, 0) &&
                     backend.operations[4].pte == backend.operations[3].pte &&
                     is_operation(backend.operations[5], OperationKind::kFlushGc) &&
                     is_operation(backend.operations[6], OperationKind::kFlushMmhub),
                 "a new PTB must zero, parent-write/readback, leaf-write/readback, then flush GC and MMHUB") &&
         require(second_free.physical_offset == kFirstDynamicPage + kPageBytes,
                 "the dynamic PTB must hold the first owned physical page");
}

bool collision_prescan_has_no_bar_or_allocator_mutation() {
  const native_r9700::VramLayout layout = dynamic_table_layout(3);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kResidentVaBase, kLeafPhysical, kPageBytes, &error_text),
               "the collision fixture must first map PTB0 index 17")) {
    return false;
  }
  backend.clear_operations();
  const uint64_t collision_size = kCrossPtbVa - kResidentVaBase + kPageBytes;

  if (!require(!table.map_range(kResidentVaBase, kSecondLeafPhysical, collision_size,
                                &error_text) &&
                   !error_text.empty(),
               "a range containing an occupied PTB0 leaf must reject before mutation")) {
    return false;
  }

  native_r9700::VramAllocation first_free{};
  if (!require(allocator.allocate("first-free", kPageBytes, kPageBytes, &first_free,
                                  &error_text),
               "the untouched allocator must still serve its first dynamic page")) {
    return false;
  }

  return require(backend.operations.empty(),
                 "a collision pre-scan must not zero, write, read, or flush the fake BAR") &&
         require(first_free.physical_offset == kFirstDynamicPage,
                 "a collision pre-scan must not reserve a dynamic table page");
}

bool unmap_prunes_dynamic_ptb_after_safe_flushes_only() {
  const native_r9700::VramLayout layout = dynamic_table_layout(3);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kCrossPtbVa, kLeafPhysical, kPageBytes, &error_text),
               "the unmap fixture must establish a dynamic PTB leaf")) {
    return false;
  }
  backend.clear_operations();
  backend.probe_dynamic_page_during_gc = true;

  if (!require(table.unmap_range(kCrossPtbVa, kPageBytes, &error_text),
               "an owned dynamic PTB leaf must unmap")) {
    return false;
  }

  native_r9700::VramAllocation released_page{};
  if (!require(allocator.allocate("released-page", kPageBytes, kPageBytes, &released_page,
                                  &error_text),
               "the dynamic PTB page must become reusable after unmap")) {
    return false;
  }

  return require(backend.operations.size() == 6 &&
                     is_operation(backend.operations[0], OperationKind::kWrite,
                                  kFirstDynamicPage, 0) &&
                     backend.operations[0].pte == 0 &&
                     is_operation(backend.operations[1], OperationKind::kRead,
                                  kFirstDynamicPage, 0) &&
                     backend.operations[1].pte == 0 &&
                     is_operation(backend.operations[2], OperationKind::kWrite,
                                  kFixedPdb0Page, 1) &&
                     backend.operations[2].pte == 0 &&
                     is_operation(backend.operations[3], OperationKind::kRead,
                                  kFixedPdb0Page, 1) &&
                     backend.operations[3].pte == 0 &&
                     is_operation(backend.operations[4], OperationKind::kFlushMmhub) &&
                     is_operation(backend.operations[5], OperationKind::kFlushGc),
                 "unmap must clear/readback leaf then empty dynamic parent before MMHUB then GC") &&
         require(backend.gc_probe_physical == kFirstDynamicPage + kPageBytes,
                 "the GC flush must occur before releasing the dynamic PTB physical page") &&
         require(released_page.physical_offset == kFirstDynamicPage,
                 "only the dynamic PTB page must release after its safe flush sequence") &&
         require(backend.zeroed_pages.size() == 1 &&
                     backend.zeroed_pages.front() == kFirstDynamicPage,
                 "unmap must never zero fixed root, PDB1, PDB0, or PTB0 pages");
}

bool maps_full_resident_window_ending_at_exclusive_limit() {
  const native_r9700::VramLayout layout = dynamic_table_layout(2);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;
  const uint64_t full_window_bytes = kResidentVaLimit - kResidentVaBase;

  if (!require(table.map_range(kResidentVaBase, kLeafPhysical, full_window_bytes,
                               &error_text),
               "a full resident map ending exactly at the exclusive limit must succeed")) {
    return false;
  }

  native_r9700::VramAllocation next_free{};
  return require(allocator.allocate("next-free", kPageBytes, kPageBytes, &next_free,
                                    &error_text),
                 "a full resident mapping must leave an allocator page for the probe") &&
         require(next_free.physical_offset == kFirstDynamicPage + kPageBytes,
                 "the full resident window must reserve only PTB1's dynamic page");
}

bool rejects_requests_outside_the_resident_window_before_mutation() {
  const uint64_t invalid_virtual_addresses[] = {
      kResidentVaBase - kPageBytes,
      kResidentVaLimit,
      kResidentVaLimit + kPageBytes,
      kPdb2AliasedVa,
  };
  for (const uint64_t invalid_va : invalid_virtual_addresses) {
    const native_r9700::VramLayout layout = dynamic_table_layout(2);
    native_r9700::VramAllocator allocator(layout);
    FakeBarBackend backend;
    backend.allocator = &allocator;
    native_r9700::DynamicPageTable table =
        dynamic_page_table(layout, &allocator, &backend);
    std::string error_text;

    if (!require(!table.map_range(invalid_va, kLeafPhysical, kPageBytes, &error_text) &&
                     !error_text.empty(),
                 "a request outside the resident PDB2 window must fail before mutation")) {
      return false;
    }
    if (!require(backend.operations.empty(),
                 "a rejected resident-window request must not touch the fake BAR")) {
      return false;
    }

    native_r9700::VramAllocation first_free{};
    if (!require(allocator.allocate("first-free", kPageBytes, kPageBytes, &first_free,
                                   &error_text),
                 "a rejected resident-window request must leave allocator capacity intact") ||
        !require(first_free.physical_offset == kFirstDynamicPage,
                 "a rejected resident-window request must not reserve a dynamic PTB")) {
      return false;
    }
  }
  return true;
}

bool failed_map_retains_an_owned_dynamic_ptb_until_explicit_unmap(FailurePoint failure) {
  const native_r9700::VramLayout layout = dynamic_table_layout(2);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  backend.failure = failure;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(!table.map_range(kCrossPtbVa, kLeafPhysical, kPageBytes, &error_text) &&
                   !error_text.empty(),
               "an injected leaf or flush failure must fail map_range")) {
    return false;
  }

  native_r9700::VramAllocation second_free{};
  if (!require(allocator.allocate("second-free", kPageBytes, kPageBytes, &second_free,
                                  &error_text),
               "a failed dynamic map must leave an allocator page for the liveness probe") ||
      !require(second_free.physical_offset == kFirstDynamicPage + kPageBytes,
               "a failed dynamic map must quarantine its dynamic PTB page")) {
    return false;
  }

  if (!require(table.unmap_range(kCrossPtbVa, kPageBytes, &error_text),
               "a failed uncertain map must retain an owned mapping for explicit cleanup")) {
    return false;
  }

  native_r9700::VramAllocation released_page{};
  return require(allocator.allocate("released-page", kPageBytes, kPageBytes,
                                    &released_page, &error_text),
                 "cleanup of an owned failed map must release its dynamic PTB") &&
         require(released_page.physical_offset == kFirstDynamicPage,
                 "cleanup of an owned failed map must release the quarantined PTB");
}

bool cross_ptb_zero_failure_retains_prior_leaf_for_explicit_cleanup() {
  const native_r9700::VramLayout layout = dynamic_table_layout(2);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  backend.failure = FailurePoint::kZero;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;
  const uint64_t ptb0_last_leaf_va = kCrossPtbVa - kPageBytes;

  if (!require(!table.map_range(ptb0_last_leaf_va, kLeafPhysical,
                                2 * kPageBytes, &error_text) &&
                   !error_text.empty(),
               "a PTB1 zero failure after a PTB0 leaf map must fail the range")) {
    return false;
  }

  backend.clear_operations();
  if (!require(table.unmap_range(ptb0_last_leaf_va, 2 * kPageBytes, &error_text),
               "a failed cross-PTB map must retain its prior leaf for explicit cleanup")) {
    return false;
  }

  for (const Operation& operation : backend.operations) {
    if (!require(operation.table_page != kFirstDynamicPage,
                 "cleanup must not clear or write the unlinked PTB1 page")) {
      return false;
    }
  }
  backend.clear_operations();

  return require(table.map_range(ptb0_last_leaf_va, kSecondLeafPhysical,
                                 kPageBytes, &error_text),
                 "explicit cleanup must make the prior PTB0 leaf reusable");
}

bool failed_ptb_setup_has_explicit_ownership(FailurePoint failure,
                                             uint64_t expected_first_free) {
  const native_r9700::VramLayout layout = dynamic_table_layout(2);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  backend.failure = failure;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(!table.map_range(kCrossPtbVa, kLeafPhysical, kPageBytes, &error_text) &&
                   !error_text.empty(),
               "an injected PTB setup failure must fail map_range")) {
    return false;
  }

  native_r9700::VramAllocation first_free{};
  return require(allocator.allocate("first-free", kPageBytes, kPageBytes, &first_free,
                                    &error_text),
                 "the setup-failure ownership probe must allocate") &&
         require(first_free.physical_offset == expected_first_free,
                 "zero failures must release an unlinked PTB; parent uncertainty must quarantine it");
}

bool failed_unmap_retains_an_owned_dynamic_ptb_until_retry(FailurePoint failure) {
  const native_r9700::VramLayout layout = dynamic_table_layout(2);
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kCrossPtbVa, kLeafPhysical, kPageBytes, &error_text),
               "the failed-unmap fixture must establish a dynamic PTB leaf")) {
    return false;
  }
  backend.clear_operations();
  backend.failure = failure;
  if (!require(!table.unmap_range(kCrossPtbVa, kPageBytes, &error_text) &&
                   !error_text.empty(),
               "an injected clear or flush failure must fail unmap_range")) {
    return false;
  }

  native_r9700::VramAllocation second_free{};
  if (!require(allocator.allocate("second-free", kPageBytes, kPageBytes, &second_free,
                                  &error_text),
               "a failed unmap must leave an allocator page for the liveness probe") ||
      !require(second_free.physical_offset == kFirstDynamicPage + kPageBytes,
               "a failed unmap must quarantine its dynamic PTB page")) {
    return false;
  }

  if (!require(table.unmap_range(kCrossPtbVa, kPageBytes, &error_text),
               "a failed uncertain unmap must retain an owned mapping for retry")) {
    return false;
  }

  native_r9700::VramAllocation released_page{};
  return require(allocator.allocate("released-page", kPageBytes, kPageBytes,
                                    &released_page, &error_text),
                 "a retried owned unmap must release its dynamic PTB") &&
         require(released_page.physical_offset == kFirstDynamicPage,
                 "a retried owned unmap must release the quarantined PTB");
}

bool maps_across_pdb1_boundary_and_releases_hierarchy() {
  native_r9700::VramLayout layout = dynamic_table_layout(8);
  layout.resident_gpu_va_limit = kCrossPdb1Va + 2 * kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kCrossPdb1Va, kLeafPhysical, 2 * kPageBytes,
                               &error_text),
               "a range crossing 1 GiB must map across two PDB1 slots")) {
    return false;
  }
  if (!require(table.dynamic_pdb0_count() == 1,
               "crossing 1 GiB must own one dynamic PDB0 page") ||
      !require(table.dynamic_ptb_count() == 2,
               "crossing 1 GiB must own one PTB per PDB0 pair") ||
      !require(table.first_dynamic_pdb0_physical_offset() ==
                   kFirstDynamicPage + kPageBytes,
               "the later PDB1 slot must own its PDB0 page") ||
      !require(backend.zeroed_pages.size() == 3,
               "crossing 1 GiB must zero the two PTBs and one PDB0 page")) {
    return false;
  }
  for (const uint64_t page : backend.zeroed_pages) {
    if (!require(page != kFixedRootPage && page != kFixedPdb1Page &&
                     page != kFixedPdb0Page && page != kFixedPtb0Page,
                 "fixed C0 pages must never be zeroed")) {
      return false;
    }
  }

  if (!require(table.unmap_range(kCrossPdb1Va, 2 * kPageBytes, &error_text),
               "a cross-PDB1 range must unmap")) {
    return false;
  }
  native_r9700::VramAllocation released{};
  return require(table.dynamic_pdb0_count() == 0 &&
                     table.dynamic_ptb_count() == 0,
                 "cross-PDB1 unmap must release all dynamic table ownership") &&
         require(allocator.allocate("released", kPageBytes, kPageBytes,
                                    &released, &error_text),
                 "released hierarchy pages must become reusable") &&
         require(released.physical_offset == kFirstDynamicPage,
                 "released hierarchy pages must return to the first dynamic page");
}

bool maps_multiple_ptbs_under_one_dynamic_pdb0() {
  const uint64_t start = kResidentVaBase + kOneGib;
  const uint64_t size = kPtbBytes + kPageBytes;
  native_r9700::VramLayout layout = dynamic_table_layout(8);
  layout.resident_gpu_va_limit = start + size;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(start, kLeafPhysical, size, &error_text),
               "one dynamic PDB0 must support multiple PTB children")) {
    return false;
  }
  if (!require(table.dynamic_pdb0_count() == 1,
               "two PTBs in one PDB1 slot must share one PDB0") ||
      !require(table.dynamic_ptb_count() == 2,
               "crossing a PDB0 boundary must allocate a second PTB")) {
    return false;
  }
  if (!require(table.unmap_range(start, size, &error_text),
               "multiple PTBs beneath one PDB0 must unmap")) {
    return false;
  }
  native_r9700::VramAllocation released{};
  return require(table.dynamic_pdb0_count() == 0 &&
                     table.dynamic_ptb_count() == 0,
                 "unmapping multiple PTBs must release the PDB0 after children") &&
         require(allocator.allocate("released", kPageBytes, kPageBytes,
                                    &released, &error_text),
                 "all PTBs and their PDB0 must be reusable") &&
         require(released.physical_offset == kFirstDynamicPage,
                 "hierarchy release must restore the first dynamic page");
}

bool maps_two_later_pdb1_slots() {
  const uint64_t first = kResidentVaBase + kOneGib;
  native_r9700::VramLayout layout = dynamic_table_layout(8);
  layout.resident_gpu_va_limit = kSecondPdb1Va + kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(first, kLeafPhysical, kPageBytes, &error_text),
               "the first later PDB1 slot must map") ||
      !require(table.map_range(kSecondPdb1Va, kSecondLeafPhysical, kPageBytes,
                               &error_text),
               "the second later PDB1 slot must map")) {
    return false;
  }
  if (!require(table.dynamic_pdb0_count() == 2,
               "two later PDB1 slots must own two PDB0 pages") ||
      !require(table.dynamic_ptb_count() == 2,
               "two later PDB1 slots must own two PTBs")) {
    return false;
  }
  if (!require(table.unmap_range(first, kPageBytes, &error_text),
               "the first later PDB1 slot must unmap") ||
      !require(table.unmap_range(kSecondPdb1Va, kPageBytes, &error_text),
               "the second later PDB1 slot must unmap")) {
    return false;
  }
  native_r9700::VramAllocation released{};
  return require(table.dynamic_pdb0_count() == 0 &&
                     table.dynamic_ptb_count() == 0,
                 "both later PDB1 slots must release their hierarchy") &&
         require(allocator.allocate("released", kPageBytes, kPageBytes,
                                    &released, &error_text),
                 "two later PDB1 cleanups must release every owned page") &&
         require(released.physical_offset == kFirstDynamicPage,
                 "two later PDB1 cleanups must restore allocation order");
}

bool failed_dynamic_pdb0_setup_retains_ownership(FailurePoint failure) {
  const uint64_t start = kResidentVaBase + kOneGib;
  native_r9700::VramLayout layout = dynamic_table_layout(4);
  layout.resident_gpu_va_limit = start + kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  backend.failure = failure;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(!table.map_range(start, kLeafPhysical, kPageBytes, &error_text) &&
                   !error_text.empty(),
               "a dynamic PDB0 parent failure must fail map_range") ||
      !require(table.dynamic_pdb0_count() == 1,
               "a failed PDB0 parent link must quarantine its allocation")) {
    return false;
  }

  native_r9700::VramAllocation still_owned{};
  if (!require(allocator.allocate("still-owned", kPageBytes, kPageBytes,
                                  &still_owned, &error_text),
               "a failed PDB0 parent link must leave a liveness probe page") ||
      !require(still_owned.physical_offset == kFirstDynamicPage + kPageBytes,
               "a failed PDB0 parent link must retain its owned page")) {
    return false;
  }
  if (!require(table.unmap_range(start, kPageBytes, &error_text),
               "a failed PDB0 parent link must be explicitly cleanable")) {
    return false;
  }

  native_r9700::VramAllocation released{};
  return require(table.dynamic_pdb0_count() == 0,
                 "PDB0 cleanup must remove the quarantine record") &&
         require(allocator.allocate("released", kPageBytes, kPageBytes,
                                    &released, &error_text),
                 "PDB0 cleanup must release its physical page") &&
         require(released.physical_offset == kFirstDynamicPage,
                 "PDB0 cleanup must restore the first dynamic page");
}

bool failed_dynamic_leaf_map_retains_hierarchy_until_cleanup(FailurePoint failure) {
  const uint64_t start = kResidentVaBase + kOneGib;
  native_r9700::VramLayout layout = dynamic_table_layout(5);
  layout.resident_gpu_va_limit = start + kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  backend.failure = failure;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(!table.map_range(start, kLeafPhysical, kPageBytes, &error_text) &&
                   !error_text.empty(),
               "a failed dynamic leaf operation must fail map_range") ||
      !require(table.dynamic_pdb0_count() == 1 &&
                   table.dynamic_ptb_count() == 1,
               "a failed dynamic leaf operation must retain PDB0 and PTB")) {
    return false;
  }
  native_r9700::VramAllocation still_owned{};
  if (!require(allocator.allocate("still-owned", kPageBytes, kPageBytes,
                                  &still_owned, &error_text),
               "a failed dynamic leaf operation must leave a liveness probe") ||
      !require(still_owned.physical_offset == kFirstDynamicPage + 2 * kPageBytes,
               "a failed dynamic leaf operation must retain both table pages")) {
    return false;
  }
  if (!require(table.unmap_range(start, kPageBytes, &error_text),
               "a failed dynamic leaf operation must be explicitly cleanable")) {
    return false;
  }
  native_r9700::VramAllocation released{};
  return require(table.dynamic_pdb0_count() == 0 &&
                     table.dynamic_ptb_count() == 0,
                 "leaf cleanup must release PDB0 and PTB ownership") &&
         require(allocator.allocate("released", kPageBytes, kPageBytes,
                                    &released, &error_text),
                 "leaf cleanup must release all hierarchy pages") &&
         require(released.physical_offset == kFirstDynamicPage,
                 "leaf cleanup must restore the first dynamic page");
}

bool collision_prescan_crosses_pdb1_without_mutation() {
  native_r9700::VramLayout layout = dynamic_table_layout(8);
  layout.resident_gpu_va_limit = kCrossPdb1Va + 2 * kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;

  if (!require(table.map_range(kCrossPdb1Va, kLeafPhysical, kPageBytes,
                               &error_text),
               "the collision fixture must map its first cross-PDB1 leaf")) {
    return false;
  }
  backend.clear_operations();
  if (!require(!table.map_range(kCrossPdb1Va - kPageBytes, kSecondLeafPhysical,
                                3 * kPageBytes, &error_text) &&
                   !error_text.empty(),
               "a cross-PDB1 collision must fail")) {
    return false;
  }
  native_r9700::VramAllocation first_free{};
  return require(backend.operations.empty(),
                 "a cross-PDB1 collision must not mutate the fake BAR") &&
         require(allocator.allocate("first-free", kPageBytes, kPageBytes,
                                    &first_free, &error_text),
                 "a cross-PDB1 collision must leave allocator capacity intact") &&
         require(first_free.physical_offset == kFirstDynamicPage + kPageBytes,
                 "a cross-PDB1 collision must not reserve another table page");
}

bool failed_hierarchical_unmap_flush_retries() {
  const uint64_t start = kResidentVaBase + kOneGib;
  native_r9700::VramLayout layout = dynamic_table_layout(4);
  layout.resident_gpu_va_limit = start + kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;
  if (!require(table.map_range(start, kLeafPhysical, kPageBytes, &error_text),
               "the hierarchical unmap retry fixture must map")) {
    return false;
  }
  backend.clear_operations();
  backend.failure = FailurePoint::kFlushMmhub;
  if (!require(!table.unmap_range(start, kPageBytes, &error_text) &&
                   !error_text.empty(),
               "a hierarchical MMHUB flush failure must fail unmap") ||
      !require(table.dynamic_pdb0_count() == 1 &&
                   table.dynamic_ptb_count() == 1,
               "a failed hierarchical flush must retain exact ownership")) {
    return false;
  }
  if (!require(table.unmap_range(start, kPageBytes, &error_text),
               "a hierarchical unmap must retry after flush failure")) {
    return false;
  }
  native_r9700::VramAllocation released{};
  return require(table.dynamic_pdb0_count() == 0 &&
                     table.dynamic_ptb_count() == 0,
                 "a retried hierarchical unmap must release all pages") &&
         require(allocator.allocate("released", kPageBytes, kPageBytes,
                                    &released, &error_text),
                 "a retried hierarchical unmap must restore allocator capacity") &&
         require(released.physical_offset == kFirstDynamicPage,
                 "a retried hierarchical unmap must release PDB0 before reuse");
}

bool hierarchical_unmap_is_child_before_parent() {
  const uint64_t start = kResidentVaBase + kOneGib;
  native_r9700::VramLayout layout = dynamic_table_layout(4);
  layout.resident_gpu_va_limit = start + kPageBytes;
  native_r9700::VramAllocator allocator(layout);
  FakeBarBackend backend;
  backend.allocator = &allocator;
  native_r9700::DynamicPageTable table =
      dynamic_page_table(layout, &allocator, &backend);
  std::string error_text;
  if (!require(table.map_range(start, kLeafPhysical, kPageBytes, &error_text),
               "the hierarchy ordering fixture must map")) {
    return false;
  }
  const uint64_t pdb0_page = table.first_dynamic_pdb0_physical_offset();
  const uint64_t ptb_page = kFirstDynamicPage + kPageBytes;
  backend.clear_operations();
  if (!require(table.unmap_range(start, kPageBytes, &error_text),
               "the hierarchy ordering fixture must unmap")) {
    return false;
  }

  return require(backend.operations.size() == 10,
                 "hierarchical cleanup must flush each ownership level") &&
         require(is_operation(backend.operations[0], OperationKind::kWrite,
                              ptb_page, 17) &&
                     backend.operations[0].pte == 0 &&
                     is_operation(backend.operations[1], OperationKind::kRead,
                                  ptb_page, 17) &&
                     is_operation(backend.operations[2], OperationKind::kWrite,
                                  pdb0_page, 0) &&
                     backend.operations[2].pte == 0 &&
                     is_operation(backend.operations[3], OperationKind::kRead,
                                  pdb0_page, 0) &&
                     is_operation(backend.operations[4],
                                  OperationKind::kFlushMmhub) &&
                     is_operation(backend.operations[5], OperationKind::kFlushGc) &&
                     is_operation(backend.operations[6], OperationKind::kWrite,
                                  kFixedPdb1Page, 1) &&
                     backend.operations[6].pte == 0 &&
                     is_operation(backend.operations[7], OperationKind::kRead,
                                  kFixedPdb1Page, 1) &&
                     is_operation(backend.operations[8],
                                  OperationKind::kFlushMmhub) &&
                     is_operation(backend.operations[9], OperationKind::kFlushGc),
                 "hierarchical cleanup must clear child parents before PDB1");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const std::string mode = argv[1];
  if (mode == "ptb0-leaf")
    return maps_ptb0_leaf_without_touching_fixed_tree_pages() ? 0 : 2;
  if (mode == "crosses-2mib")
    return allocates_new_ptb_at_first_owned_page_across_2mib_boundary() ? 0 : 3;
  if (mode == "collision-prescan")
    return collision_prescan_has_no_bar_or_allocator_mutation() ? 0 : 4;
  if (mode == "safe-unmap")
    return unmap_prunes_dynamic_ptb_after_safe_flushes_only() ? 0 : 5;
  if (mode == "resident-window")
    return rejects_requests_outside_the_resident_window_before_mutation() ? 0 : 6;
  if (mode == "map-leaf-write")
    return failed_map_retains_an_owned_dynamic_ptb_until_explicit_unmap(
               FailurePoint::kLeafWrite)
               ? 0
               : 7;
  if (mode == "map-leaf-readback")
    return failed_map_retains_an_owned_dynamic_ptb_until_explicit_unmap(
               FailurePoint::kLeafRead)
               ? 0
               : 8;
  if (mode == "map-flush-gc")
    return failed_map_retains_an_owned_dynamic_ptb_until_explicit_unmap(
               FailurePoint::kFlushGc)
               ? 0
               : 9;
  if (mode == "map-flush-mmhub")
    return failed_map_retains_an_owned_dynamic_ptb_until_explicit_unmap(
               FailurePoint::kFlushMmhub)
               ? 0
               : 10;
  if (mode == "setup-zero")
    return failed_ptb_setup_has_explicit_ownership(FailurePoint::kZero,
                                                    kFirstDynamicPage)
               ? 0
               : 11;
  if (mode == "setup-parent-write")
    return failed_ptb_setup_has_explicit_ownership(
               FailurePoint::kParentWrite, kFirstDynamicPage + kPageBytes)
               ? 0
               : 12;
  if (mode == "setup-parent-readback")
    return failed_ptb_setup_has_explicit_ownership(
               FailurePoint::kParentRead, kFirstDynamicPage + kPageBytes)
               ? 0
               : 13;
  if (mode == "unmap-clear")
    return failed_unmap_retains_an_owned_dynamic_ptb_until_retry(
               FailurePoint::kLeafWrite)
               ? 0
               : 14;
  if (mode == "unmap-flush-mmhub")
    return failed_unmap_retains_an_owned_dynamic_ptb_until_retry(
               FailurePoint::kFlushMmhub)
               ? 0
               : 15;
  if (mode == "unmap-flush-gc")
    return failed_unmap_retains_an_owned_dynamic_ptb_until_retry(FailurePoint::kFlushGc)
               ? 0
               : 16;
  if (mode == "resident-window-limit")
    return maps_full_resident_window_ending_at_exclusive_limit() ? 0 : 17;
  if (mode == "cross-ptb-zero-cleanup")
    return cross_ptb_zero_failure_retains_prior_leaf_for_explicit_cleanup() ? 0 : 18;

  if (mode == "crosses-pdb1")
    return maps_across_pdb1_boundary_and_releases_hierarchy() ? 0 : 20;
  if (mode == "multiple-ptbs-one-pdb0")
    return maps_multiple_ptbs_under_one_dynamic_pdb0() ? 0 : 21;
  if (mode == "two-later-pdb1")
    return maps_two_later_pdb1_slots() ? 0 : 22;
  if (mode == "pdb0-parent-write")
    return failed_dynamic_pdb0_setup_retains_ownership(FailurePoint::kParentWrite)
               ? 0
               : 23;
  if (mode == "pdb0-parent-read")
    return failed_dynamic_pdb0_setup_retains_ownership(FailurePoint::kParentRead)
               ? 0
               : 24;
  if (mode == "pdb0-leaf-write")
    return failed_dynamic_leaf_map_retains_hierarchy_until_cleanup(
               FailurePoint::kLeafWrite)
               ? 0
               : 25;
  if (mode == "pdb0-leaf-read")
    return failed_dynamic_leaf_map_retains_hierarchy_until_cleanup(
               FailurePoint::kLeafRead)
               ? 0
               : 26;
  if (mode == "cross-pdb1-collision")
    return collision_prescan_crosses_pdb1_without_mutation() ? 0 : 27;
  if (mode == "hierarchical-unmap-retry")
    return failed_hierarchical_unmap_flush_retries() ? 0 : 28;
  if (mode == "hierarchical-unmap-order")
    return hierarchical_unmap_is_child_before_parent() ? 0 : 29;

  return 19;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "dynamic_page_table_probe"
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
            str(DYNAMIC_PAGE_TABLE_SOURCE),
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


def run_dynamic_page_table_probe(tmp_path: Path, mode: str) -> None:
    completed = subprocess.run(
        [str(compile_dynamic_page_table_probe(tmp_path)), mode],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_dynamic_page_table_maps_ptb0_leaf_without_mutating_c0_tree(tmp_path: Path) -> None:
    """The resident base is fixed PTB0 index 17, not a dynamic-table allocation."""
    run_dynamic_page_table_probe(tmp_path, "ptb0-leaf")


def test_dynamic_page_table_allocates_ptb_at_02004000_across_2mib_boundary(
    tmp_path: Path,
) -> None:
    """The first leaf after PTB0 gets a zeroed dynamic PTB at the first owned page."""
    run_dynamic_page_table_probe(tmp_path, "crosses-2mib")


def test_dynamic_page_table_prescans_collisions_before_any_mutation(tmp_path: Path) -> None:
    """A collision that would otherwise cross PTB0 leaves BAR and allocator untouched."""
    run_dynamic_page_table_probe(tmp_path, "collision-prescan")


def test_dynamic_page_table_unmap_flushes_before_releasing_dynamic_ptb(tmp_path: Path) -> None:
    """Unmap clears only owned entries and flushes MMHUB then GC before release."""
    run_dynamic_page_table_probe(tmp_path, "safe-unmap")


def test_dynamic_page_table_rejects_resident_window_aliases_before_mutation(
    tmp_path: Path,
) -> None:
    """DynamicPageTable requires VramLayout, which authoritatively defines resident VA bounds."""
    run_dynamic_page_table_probe(tmp_path, "resident-window")


def test_dynamic_page_table_maps_full_resident_window_ending_at_exclusive_limit(
    tmp_path: Path,
) -> None:
    """The resident limit is exclusive, so a range whose end equals it is valid."""
    run_dynamic_page_table_probe(tmp_path, "resident-window-limit")


def test_dynamic_page_table_cleans_up_prior_ptb0_leaf_after_ptb1_zero_failure(
    tmp_path: Path,
) -> None:
    """Explicit cleanup retains and clears the mapped PTB0 leaf, not the unlinked PTB1."""
    run_dynamic_page_table_probe(tmp_path, "cross-ptb-zero-cleanup")


def test_dynamic_page_table_quarantines_leaf_write_map_failure_until_cleanup(
    tmp_path: Path,
) -> None:
    """A leaf write failure can be uncertain, so the owned dynamic PTB remains cleanable."""
    run_dynamic_page_table_probe(tmp_path, "map-leaf-write")


def test_dynamic_page_table_quarantines_leaf_readback_map_failure_until_cleanup(
    tmp_path: Path,
) -> None:
    """A leaf readback failure preserves the owned mapping for an explicit safe unmap."""
    run_dynamic_page_table_probe(tmp_path, "map-leaf-readback")


def test_dynamic_page_table_quarantines_gc_flush_map_failure_until_cleanup(
    tmp_path: Path,
) -> None:
    """A GC flush failure cannot make the dynamically owned PTB reusable."""
    run_dynamic_page_table_probe(tmp_path, "map-flush-gc")


def test_dynamic_page_table_quarantines_mmhub_flush_map_failure_until_cleanup(
    tmp_path: Path,
) -> None:
    """An MMHUB flush failure cannot make the dynamically owned PTB reusable."""
    run_dynamic_page_table_probe(tmp_path, "map-flush-mmhub")


def test_dynamic_page_table_releases_unlinked_ptb_after_zero_failure(tmp_path: Path) -> None:
    """A failed zero has no visible parent link, so its allocation is safely reusable."""
    run_dynamic_page_table_probe(tmp_path, "setup-zero")


def test_dynamic_page_table_quarantines_parent_write_setup_failure(tmp_path: Path) -> None:
    """An uncertain parent write leaves the allocated PTB unavailable for reuse."""
    run_dynamic_page_table_probe(tmp_path, "setup-parent-write")


def test_dynamic_page_table_quarantines_parent_readback_setup_failure(
    tmp_path: Path,
) -> None:
    """An uncertain parent readback leaves the allocated PTB unavailable for reuse."""
    run_dynamic_page_table_probe(tmp_path, "setup-parent-readback")


def test_dynamic_page_table_quarantines_unmap_clear_failure_until_retry(
    tmp_path: Path,
) -> None:
    """A leaf clear failure preserves an owned mapping that can be safely retried."""
    run_dynamic_page_table_probe(tmp_path, "unmap-clear")


def test_dynamic_page_table_quarantines_mmhub_flush_unmap_failure_until_retry(
    tmp_path: Path,
) -> None:
    """An MMHUB flush failure preserves the owned mapping for an explicit unmap retry."""
    run_dynamic_page_table_probe(tmp_path, "unmap-flush-mmhub")


def test_dynamic_page_table_quarantines_gc_flush_unmap_failure_until_retry(
    tmp_path: Path,
) -> None:
    """A GC flush failure preserves the owned mapping for an explicit unmap retry."""
    run_dynamic_page_table_probe(tmp_path, "unmap-flush-gc")


def test_dynamic_page_table_maps_across_pdb1_boundary_and_releases_hierarchy(
    tmp_path: Path,
) -> None:
    """A single mapping may cross 1 GiB and own the later PDB0/PTB chain."""
    run_dynamic_page_table_probe(tmp_path, "crosses-pdb1")


def test_dynamic_page_table_supports_multiple_ptbs_under_one_dynamic_pdb0(
    tmp_path: Path,
) -> None:
    """A dynamic PDB0 owns one PTB for each selected 2 MiB region."""
    run_dynamic_page_table_probe(tmp_path, "multiple-ptbs-one-pdb0")


def test_dynamic_page_table_supports_two_later_pdb1_slots(tmp_path: Path) -> None:
    """Mappings in distinct later 1 GiB slots own distinct PDB0 pages."""
    run_dynamic_page_table_probe(tmp_path, "two-later-pdb1")


def test_dynamic_page_table_quarantines_dynamic_pdb0_parent_write_failure(
    tmp_path: Path,
) -> None:
    """An uncertain PDB1 parent write retains the dynamic PDB0 for cleanup."""
    run_dynamic_page_table_probe(tmp_path, "pdb0-parent-write")


def test_dynamic_page_table_quarantines_dynamic_pdb0_parent_readback_failure(
    tmp_path: Path,
) -> None:
    """An uncertain PDB1 parent readback retains the dynamic PDB0 for cleanup."""
    run_dynamic_page_table_probe(tmp_path, "pdb0-parent-read")


def test_dynamic_page_table_quarantines_dynamic_pdb0_leaf_write_failure(
    tmp_path: Path,
) -> None:
    """A leaf failure below a dynamic PDB0 retains both owned table pages."""
    run_dynamic_page_table_probe(tmp_path, "pdb0-leaf-write")


def test_dynamic_page_table_quarantines_dynamic_pdb0_leaf_readback_failure(
    tmp_path: Path,
) -> None:
    """A leaf readback failure below a dynamic PDB0 is explicitly cleanable."""
    run_dynamic_page_table_probe(tmp_path, "pdb0-leaf-read")


def test_dynamic_page_table_prescans_cross_pdb1_collisions_before_mutation(
    tmp_path: Path,
) -> None:
    """Cross-PDB1 collision rejection must leave BAR and allocator untouched."""
    run_dynamic_page_table_probe(tmp_path, "cross-pdb1-collision")


def test_dynamic_page_table_retries_hierarchical_unmap_after_flush_failure(
    tmp_path: Path,
) -> None:
    """A failed flush retains exact PDB0/PTB ownership for the retry."""
    run_dynamic_page_table_probe(tmp_path, "hierarchical-unmap-retry")


def test_dynamic_page_table_unmaps_hierarchy_child_before_parent(tmp_path: Path) -> None:
    """PTB and PDB0 parents clear before the PDB1 parent and release."""
    run_dynamic_page_table_probe(tmp_path, "hierarchical-unmap-order")

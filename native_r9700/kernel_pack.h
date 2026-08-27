#ifndef NATIVE_R9700_KERNEL_PACK_H_
#define NATIVE_R9700_KERNEL_PACK_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "kernel_catalog.h"

namespace native_r9700 {

template <typename T>
struct KernelPackSpan {
  const T* data;
  std::size_t size;
};

template <typename T>
struct KernelPackOptional {
  bool present;
  T value;
};

struct EvidenceRef {
  std::string_view record_path;
  std::string_view record_kind;
  std::string_view evidence_slot;
  std::string_view record_id;
  std::string_view record_sha256;
  std::string_view subject_target;
  std::string_view image_sha256;
  std::string_view pack_sha256;
  std::string_view producer_kind;
  std::string_view tool_digest;
  std::string_view input_digest;
  std::string_view output_digest;
};

struct KernelPackIdentity {
  std::uint32_t schema_version;
  std::string_view name;
  std::string_view version;
  std::string_view target;
  KernelPackSpan<std::string_view> required_features;
};

struct KernelPackSource {
  std::string_view path;
  std::string_view sha256;
};

struct KernelPackLicenseReview {
  std::string_view component;
  std::string_view spdx_expression;
  std::string_view review_id;
  std::string_view status;
};

struct KernelPackModification {
  std::string_view component;
  std::string_view summary;
};

struct KernelPackProvenance {
  std::string_view upstream_repository;
  std::string_view upstream_revision;
  KernelPackSpan<std::string_view> upstream_paths;
  KernelPackSpan<KernelPackSource> local_sources;
  KernelPackSpan<KernelPackLicenseReview> license_reviews;
  KernelPackSpan<KernelPackModification> modifications;
};

struct KernelPackBuildIdentity {
  std::string_view toolchain_id;
  std::string_view toolchain_revision;
  std::string_view generator_id;
  std::string_view generator_revision;
  std::string_view command_sha256;
};

struct KernelPackImage {
  std::string_view image_path;
  std::string_view image_sha256;
  std::uint64_t image_size;
  std::string_view code_object_version;
  KernelPackBuildIdentity build;
};

struct KernelPackKernargField {
  std::string_view name;
  std::string_view type;
  std::uint32_t offset;
  std::uint32_t size;
  std::uint32_t alignment;
};

struct KernelPackKernargs {
  std::uint32_t bytes;
  KernelPackSpan<KernelPackKernargField> fields;
  std::uint32_t tail_padding_bytes;
};

struct KernelPackResources {
  std::uint32_t rsrc1;
  std::uint32_t rsrc2;
  std::uint32_t rsrc3;
  std::uint32_t wave_size;
  std::uint32_t sgpr_count;
  std::uint32_t vgpr_count;
  std::uint64_t lds_bytes;
  std::uint64_t private_segment_bytes;
  std::string_view metadata_provenance;
};

struct KernelPackGeometryCase {
  std::string_view shape_family;
  std::string_view geometry_rule;
  std::uint32_t workgroup_x;
  std::uint32_t workgroup_y;
  std::uint32_t workgroup_z;
  std::uint32_t global_x;
  std::uint32_t global_y;
  std::uint32_t global_z;
  std::uint32_t grid_tile_m;
  std::uint32_t grid_tile_n;
  bool dynamic_lds_allowed;
  std::uint64_t dynamic_lds_max_bytes;
};

struct KernelPackGeometry {
  KernelPackSpan<KernelPackGeometryCase> cases;
};

struct KernelPackEntry {
  std::string_view symbol;
  std::uint64_t descriptor_offset;
  std::uint64_t entry_offset;
  KernelPackKernargs kernargs;
  KernelPackResources resources;
  KernelPackGeometry geometry;
};

struct KernelPackShapeDimension {
  std::string_view name;
  std::uint32_t value;
};

struct KernelPackRuntimeDimension {
  std::string_view name;
  std::uint32_t min_value;
  std::uint32_t max_value;
  std::uint32_t full_value;
};

struct KernelPackShapeFamily {
  std::string_view name;
  KernelPackSpan<KernelPackShapeDimension> fixed_dimensions;
  KernelPackOptional<KernelPackRuntimeDimension> runtime_dimension;
  std::string_view tail_policy;
  std::string_view geometry_rule;
};

struct KernelPackCompatibility {
  std::string_view input_dtype;
  std::string_view weight_dtype;
  std::string_view output_dtype;
  std::string_view source_tensor_layout_version;
  KernelPackSpan<KernelPackShapeFamily> shape_families;
  std::string_view weight_packing_version;
};

struct KernelPackCastPoint {
  std::string_view stage;
  std::string_view from_dtype;
  std::string_view to_dtype;
};

struct KernelPackNumerics {
  std::string_view input_dtype;
  std::string_view accumulation_dtype;
  std::string_view output_dtype;
  KernelPackSpan<KernelPackCastPoint> cast_points;
  std::string_view finite_value_rule;
  std::string_view tolerance_policy;
  std::string_view reference_set_kind;
  KernelPackOptional<EvidenceRef> retained_reference;
  KernelPackOptional<EvidenceRef> numpy_oracle;
  KernelPackOptional<EvidenceRef> scalar_native_projection;
};

struct KernelPackEvidence {
  EvidenceRef conformance;
  EvidenceRef native_run;
  EvidenceRef source_review;
  EvidenceRef resource_review;
  EvidenceRef isa_review;
  KernelPackOptional<EvidenceRef> layout_proof;
  KernelPackOptional<EvidenceRef> benchmark_record;
  std::string_view benchmark_not_applicable_reason;
};

struct KernelPackRecord {
  KernelPackIdentity identity;
  KernelPackProvenance provenance;
  KernelPackImage image;
  KernelPackSpan<KernelPackEntry> entries;
  KernelPackCompatibility compatibility;
  KernelPackNumerics numerics;
  KernelPackEvidence evidence;
};

struct KernelPackCompatibilityKey {
  std::string_view target;
  KernelPackSpan<std::string_view> required_features;
  std::string_view input_dtype;
  std::string_view weight_dtype;
  std::string_view output_dtype;
  std::string_view source_tensor_layout_version;
  std::string_view shape_family_name;
  KernelPackSpan<KernelPackShapeDimension> fixed_dimensions;
  KernelPackOptional<KernelPackShapeDimension> runtime_value;
  std::string_view weight_packing_version;
  std::string_view tolerance_policy;
};

struct KernelPackErrorBuffer {
  char* data;
  std::size_t size;
};

bool validate_kernel_pack(const KernelPackRecord& record,
                          KernelPackErrorBuffer error_text);

bool kernel_pack_matches_key(const KernelPackRecord& record,
                             const KernelPackCompatibilityKey& key);

const KernelPackRecord* find_kernel_pack(KernelPackSpan<KernelPackRecord> records,
                                         std::string_view name,
                                         std::string_view version,
                                         KernelPackErrorBuffer error_text);

const KernelPackRecord* find_kernel_pack_for_key(
    KernelPackSpan<KernelPackRecord> records,
    const KernelPackCompatibilityKey& key,
    KernelPackErrorBuffer error_text);

bool admit_kernel_pack(const KernelPackRecord& record,
                       const KernelPackCompatibilityKey& selected_key,
                       std::string_view entry_symbol,
                       std::string_view asset_root,
                       KernelDescriptor* out_descriptor,
                       KernelPackErrorBuffer error_text);
// Returns the complete generated scalar-record span in legacy manifest order.
// Records whose native evidence is pending remain visible for audit but are
// excluded by the production lookup below.
KernelPackSpan<KernelPackRecord> llama_kernel_pack_records();

// Exact scalar selection over the generated, evidence-admitted subset.  The
// version is mandatory; there is no implicit upgrade, downgrade, or fallback.
const KernelPackRecord* find_llama_kernel_pack(std::string_view name,
                                               std::string_view version,
                                               KernelPackErrorBuffer error_text);

// Select and admit one generated scalar pack through the same loader used by
// legacy Llama callers.  asset_root is the repository root for generated
// records whose image path carries its canonical asset directory.
bool admit_llama_kernel_pack(const KernelPackRecord& record,
                             const KernelPackCompatibilityKey& selected_key,
                             std::string_view entry_symbol,
                             std::string_view asset_root,
                             KernelDescriptor* out_descriptor,
                             KernelPackErrorBuffer error_text);

}  // namespace native_r9700

#endif  // NATIVE_R9700_KERNEL_PACK_H_

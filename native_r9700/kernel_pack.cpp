#include "kernel_pack.h"

#include "kernel_assets.h"

#include <filesystem>
#include <utility>

namespace native_r9700 {
namespace {

constexpr char kTarget[] = "gfx1201";
constexpr char kMetadataProvenance[] = "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst";
constexpr char kDigestHex[] = "0123456789abcdef";
constexpr char kPackDomain[] = "r9700-kernel-pack-identity-v1";
constexpr char kSourceEquivalentPacking[] = "source-equivalent-v1";
constexpr char kPinnedUpstreamRepository[] =
    "https://github.com/llvm/llvm-project";
constexpr char kPinnedUpstreamRevision[] =
    "8dba93818258d95c46fa2c17e902a8256e4d91b5";
constexpr char kPinnedUpstreamPath[] = "llvm/docs/AMDGPUUsage.rst";
constexpr std::uint64_t kJcsExactIntegerMax = 9007199254740991ULL;

void clear_error(KernelPackErrorBuffer error) {
  if (error.data != nullptr && error.size != 0) error.data[0] = '\0';
}

bool fail(KernelPackErrorBuffer error, std::string_view message) {
  if (error.data != nullptr && error.size != 0) {
    std::size_t count = message.size();
    if (count >= error.size) count = error.size - 1;
    for (std::size_t i = 0; i < count; ++i) error.data[i] = message[i];
    error.data[count] = '\0';
  }
  return false;
}


bool valid_span_size(const void* data, std::size_t size) {
  return size == 0 || data != nullptr;
}

template <typename T>
bool valid_span(KernelPackSpan<T> span) {
  return valid_span_size(span.data, span.size);
}

bool is_lower_hex(std::string_view value, std::size_t expected_size) {
  if (value.size() != expected_size) return false;
  for (char character : value) {
    const bool digit = character >= '0' && character <= '9';
    const bool letter = character >= 'a' && character <= 'f';
    if (!digit && !letter) return false;
  }
  return true;
}

bool is_safe_path(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.front() == '\\') return false;
  if (path.size() >= 2 &&
      ((path[0] >= 'A' && path[0] <= 'Z') ||
       (path[0] >= 'a' && path[0] <= 'z')) &&
      path[1] == ':') {
    return false;
  }
  for (char character : path) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (byte <= 0x1fU || byte == 0x7fU) return false;
  }
  std::size_t component_start = 0;
  while (component_start < path.size()) {
    std::size_t component_end = component_start;
    while (component_end < path.size() && path[component_end] != '/') {
      if (path[component_end] == '\0' || path[component_end] == '\\') return false;
      ++component_end;
    }
    if (component_end == component_start) return false;
    const std::string_view component = path.substr(component_start,
                                                   component_end - component_start);
    if (component == "." || component == "..") return false;
    if (component_end == path.size()) break;
    component_start = component_end + 1;
  }
  return path.back() != '/';
}
bool is_unresolved_spdx(std::string_view expression) {
  std::size_t begin = 0;
  std::size_t end = expression.size();
  while (begin < end &&
         (expression[begin] == ' ' || expression[begin] == '\t' ||
          expression[begin] == '\n' || expression[begin] == '\r')) {
    ++begin;
  }
  while (end > begin &&
         (expression[end - 1] == ' ' || expression[end - 1] == '\t' ||
          expression[end - 1] == '\n' || expression[end - 1] == '\r')) {
    --end;
  }
  const std::string_view trimmed = expression.substr(begin, end - begin);
  const auto equals_ignoring_case = [trimmed](std::string_view expected) {
    if (trimmed.size() != expected.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      char actual = trimmed[i];
      if (actual >= 'A' && actual <= 'Z') actual = static_cast<char>(actual - 'A' + 'a');
      if (actual != expected[i]) return false;
    }
    return true;
  };
  return trimmed.empty() || equals_ignoring_case("unknown") || equals_ignoring_case("pending");
}


bool is_canonical_version(std::string_view version) {
  std::size_t component_start = 0;
  unsigned component_count = 0;
  while (component_start <= version.size()) {
    std::size_t component_end = component_start;
    while (component_end < version.size() && version[component_end] != '.') {
      if (version[component_end] < '0' || version[component_end] > '9') return false;
      ++component_end;
    }
    if (component_end == component_start) return false;
    if (component_end - component_start > 1 && version[component_start] == '0') return false;
    ++component_count;
    if (component_count == 3 && component_end == version.size()) return true;
    if (component_end == version.size()) return false;
    component_start = component_end + 1;
  }
  return false;
}

bool is_known_dtype(std::string_view dtype) {
  return dtype == "fp16" || dtype == "bf16" || dtype == "fp32" || dtype == "int8" ||
         dtype == "int4";
}

bool is_power_of_two(std::uint32_t value) {
  return value != 0 && (value & (value - 1U)) == 0;
}

struct KernargTypeRule {
  std::string_view type;
  std::uint32_t size;
  std::uint32_t alignment;
};

constexpr KernargTypeRule kKernargTypeRules[] = {
    {"bool", 1U, 1U},       {"int8", 1U, 1U},    {"uint8", 1U, 1U},
    {"int16", 2U, 2U},      {"uint16", 2U, 2U},  {"int32", 4U, 4U},
    {"uint32", 4U, 4U},     {"float", 4U, 4U},   {"float32", 4U, 4U},
    {"int64", 8U, 8U},      {"uint64", 8U, 8U},  {"double", 8U, 8U},
    {"float64", 8U, 8U},    {"pointer", 8U, 8U}, {"ptr", 8U, 8U},
    {"size_t", 8U, 8U},
};

bool find_kernarg_type_rule(std::string_view type, KernargTypeRule* rule_out) {
  for (const KernargTypeRule& rule : kKernargTypeRules) {
    if (rule.type != type) continue;
    if (rule_out != nullptr) *rule_out = rule;
    return true;
  }
  return false;
}


bool has_license(KernelPackSpan<KernelPackLicenseReview> reviews,
                 std::string_view component) {
  for (std::size_t i = 0; i < reviews.size; ++i) {
    if (reviews.data[i].component == component) return true;
  }
  return false;
}

bool same_string_span(KernelPackSpan<std::string_view> left,
                      KernelPackSpan<std::string_view> right) {
  if (left.size != right.size || !valid_span(left) || !valid_span(right)) return false;
  for (std::size_t i = 0; i < left.size; ++i) {
    if (left.data[i] != right.data[i]) return false;
  }
  return true;
}

bool same_dimensions(KernelPackSpan<KernelPackShapeDimension> left,
                     KernelPackSpan<KernelPackShapeDimension> right) {
  if (left.size != right.size || !valid_span(left) || !valid_span(right)) return false;
  for (std::size_t i = 0; i < left.size; ++i) {
    if (left.data[i].name != right.data[i].name || left.data[i].value != right.data[i].value) {
      return false;
    }
  }
  return true;
}

bool same_runtime(const KernelPackOptional<KernelPackRuntimeDimension>& left,
                  const KernelPackOptional<KernelPackShapeDimension>& right) {
  if (left.present != right.present) return false;
  if (!left.present) return true;
  return left.value.name == right.value.name && right.value.value >= left.value.min_value &&
         right.value.value <= left.value.max_value;
}

bool find_family(const KernelPackRecord& record,
                 std::string_view family_name,
                 const KernelPackShapeFamily** family_out) {
  const KernelPackShapeFamily* found = nullptr;
  for (std::size_t i = 0; i < record.compatibility.shape_families.size; ++i) {
    const KernelPackShapeFamily& family = record.compatibility.shape_families.data[i];
    if (family.name != family_name) continue;
    if (found != nullptr) return false;
    found = &family;
  }
  if (family_out != nullptr) *family_out = found;
  return found != nullptr;
}

bool validate_digest_read_spans(const KernelPackRecord& record, KernelPackErrorBuffer error) {
  if (!valid_span(record.identity.required_features)) {
    return fail(error, "required feature span is invalid");
  }
  const KernelPackProvenance& provenance = record.provenance;
  if (!valid_span(provenance.upstream_paths) || !valid_span(provenance.local_sources) ||
      !valid_span(provenance.license_reviews) || !valid_span(provenance.modifications)) {
    return fail(error, "provenance span is invalid");
  }
  if (!valid_span(record.entries)) return fail(error, "entry span is invalid");
  for (std::size_t i = 0; i < record.entries.size; ++i) {
    if (!valid_span(record.entries.data[i].kernargs.fields) ||
        !valid_span(record.entries.data[i].geometry.cases)) {
      return fail(error, "entry nested span is invalid");
    }
  }
  if (!valid_span(record.compatibility.shape_families)) {
    return fail(error, "shape family span is invalid");
  }
  for (std::size_t i = 0; i < record.compatibility.shape_families.size; ++i) {
    if (!valid_span(record.compatibility.shape_families.data[i].fixed_dimensions)) {
      return fail(error, "shape dimension span is invalid");
    }
  }
  if (!valid_span(record.numerics.cast_points)) {
    return fail(error, "numeric cast-point span is invalid");
  }
  return true;
}

bool find_entry(const KernelPackRecord& record,
                std::string_view symbol,
                const KernelPackEntry** entry_out) {
  const KernelPackEntry* found = nullptr;
  for (std::size_t i = 0; i < record.entries.size; ++i) {
    const KernelPackEntry& entry = record.entries.data[i];
    if (entry.symbol != symbol) continue;
    if (found != nullptr) return false;
    found = &entry;
  }
  if (entry_out != nullptr) *entry_out = found;
  return found != nullptr;
}

bool validate_required_features(const KernelPackIdentity& identity, KernelPackErrorBuffer error) {
  if (!valid_span(identity.required_features)) return fail(error, "required feature span is invalid");
  for (std::size_t i = 0; i < identity.required_features.size; ++i) {
    if (identity.required_features.data[i].empty()) {
      return fail(error, "required feature is empty");
    }
    if (i != 0 && identity.required_features.data[i - 1] >= identity.required_features.data[i]) {
      return fail(error, "required features must be sorted and unique");
    }
  }
  return true;
}

bool validate_provenance(const KernelPackRecord& record, KernelPackErrorBuffer error) {
  const KernelPackProvenance& provenance = record.provenance;
  const bool local_source =
      provenance.upstream_repository == "local" &&
      provenance.upstream_revision == "local";
  const bool pinned_source =
      provenance.upstream_repository == kPinnedUpstreamRepository &&
      provenance.upstream_revision == kPinnedUpstreamRevision &&
      provenance.upstream_paths.size == 1 &&
      provenance.upstream_paths.data != nullptr &&
      provenance.upstream_paths.data[0] == kPinnedUpstreamPath;
  if (!local_source && !pinned_source) {
    return fail(error, "upstream provenance is not immutable");
  }
  if (!valid_span(provenance.upstream_paths) || !valid_span(provenance.local_sources) ||
      !valid_span(provenance.license_reviews) || !valid_span(provenance.modifications)) {
    return fail(error, "provenance span is invalid");
  }
  for (std::size_t i = 0; i < provenance.upstream_paths.size; ++i) {
    if (!is_safe_path(provenance.upstream_paths.data[i])) {
      return fail(error, "upstream path is not canonical");
    }
  }
  if (provenance.local_sources.size == 0) {
    return fail(error, "at least one local source is required");
  }
  for (std::size_t i = 0; i < provenance.local_sources.size; ++i) {
    const KernelPackSource& source = provenance.local_sources.data[i];
    if (!is_safe_path(source.path) || !is_lower_hex(source.sha256, 64)) {
      return fail(error, "path is not canonical or source digest is invalid");
    }
  }
  if (provenance.license_reviews.size == 0) {
    return fail(error, "license review coverage is required");
  }
  for (std::size_t i = 0; i < provenance.license_reviews.size; ++i) {
    const KernelPackLicenseReview& review = provenance.license_reviews.data[i];
    if (!is_safe_path(review.component) || review.spdx_expression.empty() ||
        is_unresolved_spdx(review.spdx_expression) || review.review_id.empty() ||
        review.status != "accepted") {
      return fail(error, "license review is not accepted");
    }
  }
  for (std::size_t i = 0; i < provenance.modifications.size; ++i) {
    const KernelPackModification& modification = provenance.modifications.data[i];
    if (!is_safe_path(modification.component) || modification.summary.empty()) {
      return fail(error, "modification record is incomplete");
    }
  }
  for (std::size_t i = 0; i < provenance.upstream_paths.size; ++i) {
    if (!has_license(provenance.license_reviews, provenance.upstream_paths.data[i])) {
      return fail(error, "upstream path lacks a license review");
    }
  }
  for (std::size_t i = 0; i < provenance.local_sources.size; ++i) {
    if (!has_license(provenance.license_reviews, provenance.local_sources.data[i].path)) {
      return fail(error, "local source lacks a license review");
    }
  }
  for (std::size_t i = 0; i < provenance.modifications.size; ++i) {
    if (!has_license(
            provenance.license_reviews,
            provenance.modifications.data[i].component)) {
      return fail(error, "modification license coverage is incomplete");
    }
  }
  if (!has_license(provenance.license_reviews, record.image.image_path)) {
    return fail(error, "image lacks a license review");
  }
  return true;
}

bool validate_image(const KernelPackImage& image, KernelPackErrorBuffer error) {
  if (!is_safe_path(image.image_path) || !is_lower_hex(image.image_sha256, 64) ||
      image.image_size == 0 || image.code_object_version.empty()) {
    return fail(error, "image identity is incomplete");
  }
  const KernelPackBuildIdentity& build = image.build;
  if (build.toolchain_id.empty() || build.generator_id.empty() ||
      build.toolchain_revision.empty() || build.generator_revision.empty() ||
      !is_lower_hex(build.command_sha256, 64)) {
    return fail(error, "image build identity is incomplete");
  }
  return true;
}

bool validate_geometry_case(const KernelPackGeometryCase& geometry,
                            std::uint64_t attested_lds_bytes,
                            KernelPackErrorBuffer error) {
  if (geometry.shape_family.empty()) return fail(error, "geometry shape family is empty");
  if (geometry.workgroup_x == 0 || geometry.workgroup_y == 0 || geometry.workgroup_z == 0) {
    return fail(error, "geometry workgroup is not positive");
  }
  if (!geometry.dynamic_lds_allowed && geometry.dynamic_lds_max_bytes != 0) {
    return fail(error, "dynamic LDS limit is set while dynamic LDS is disabled");
  }
  if (geometry.dynamic_lds_allowed &&
      (geometry.dynamic_lds_max_bytes == 0 ||
       geometry.dynamic_lds_max_bytes > attested_lds_bytes)) {
    return fail(error, "dynamic LDS limit exceeds entry resource LDS");
  }
  if (geometry.geometry_rule == "exact-global-v1") {
    if (geometry.global_x == 0 || geometry.global_y == 0 || geometry.global_z == 0 ||
        geometry.grid_tile_m != 0 || geometry.grid_tile_n != 0 ||
        geometry.global_x % geometry.workgroup_x != 0 ||
        geometry.global_y % geometry.workgroup_y != 0 ||
        geometry.global_z % geometry.workgroup_z != 0) {
      return fail(error, "exact geometry rule has invalid or nondivisible dimensions");
    }
    return true;
  }
  if (geometry.geometry_rule == "f2-wmma-64x64-m-tail-v1") {
    if (geometry.workgroup_x != 128 || geometry.workgroup_y != 4 || geometry.workgroup_z != 1 ||
        geometry.global_x != 0 || geometry.global_y != 0 || geometry.global_z != 0 ||
        geometry.grid_tile_m != 64 || geometry.grid_tile_n != 64) {
      return fail(error, "F2 geometry rule has invalid dimensions");
    }
    return true;
  }
  return fail(error, "geometry rule is not a closed v1 rule");
}

bool validate_entry(const KernelPackEntry& entry,
                    const KernelPackCompatibility& compatibility,
                    std::uint64_t image_size,
                    KernelPackErrorBuffer error) {
  if (entry.symbol.empty()) return fail(error, "entry symbol is empty");
  if (entry.descriptor_offset >= image_size || entry.entry_offset >= image_size) {
    return fail(error, "entry offsets are outside the image");
  }
  if (!valid_span(entry.kernargs.fields) || entry.kernargs.bytes == 0 ||
      entry.kernargs.fields.size == 0) {
    return fail(error, "kernarg schema is empty or invalid");
  }
  std::uint64_t previous_end = 0;
  for (std::size_t i = 0; i < entry.kernargs.fields.size; ++i) {
    const KernelPackKernargField& field = entry.kernargs.fields.data[i];
    KernargTypeRule type_rule{};
    if (field.name.empty() || field.type.empty() || field.size == 0 ||
        !find_kernarg_type_rule(field.type, &type_rule) ||
        field.size != type_rule.size || field.alignment != type_rule.alignment ||
        field.offset % type_rule.alignment != 0) {
      return fail(error, "kernarg field type, size, or alignment is not canonical");
    }
    for (std::size_t prior = 0; prior < i; ++prior) {
      if (entry.kernargs.fields.data[prior].name == field.name) {
        return fail(error, "kernarg field names are duplicated");
      }
    }
    if (i != 0 && field.offset <= entry.kernargs.fields.data[i - 1].offset) {
      return fail(error, "kernarg field offsets are not ordered");
    }
    const std::uint64_t end = static_cast<std::uint64_t>(field.offset) + field.size;
    if (end > entry.kernargs.bytes || field.offset != previous_end) {
      return fail(error, "kernarg fields overlap or contain implicit padding");
    }
    previous_end = end;
  }
  if (entry.kernargs.tail_padding_bytes != entry.kernargs.bytes - previous_end) {
    return fail(error, "kernarg tail padding is not explicit");
  }
  const KernelPackResources& resources = entry.resources;
  if (resources.rsrc1 == 0 || resources.rsrc2 == 0 || resources.rsrc3 == 0 ||
      resources.wave_size != 32 || resources.sgpr_count == 0 || resources.vgpr_count == 0 ||
      resources.metadata_provenance != kMetadataProvenance) {
    return fail(error, "entry resources are not admitted source metadata");
  }
  if (!valid_span(entry.geometry.cases) || entry.geometry.cases.size == 0 ||
      entry.geometry.cases.size != compatibility.shape_families.size) {
    return fail(error, "entry geometry does not cover every shape family");
  }
  for (std::size_t i = 0; i < entry.geometry.cases.size; ++i) {
    if (!validate_geometry_case(entry.geometry.cases.data[i], resources.lds_bytes, error)) return false;
    bool found = false;
    for (std::size_t family_index = 0; family_index < compatibility.shape_families.size;
         ++family_index) {
      const KernelPackShapeFamily& family = compatibility.shape_families.data[family_index];
      if (family.name == entry.geometry.cases.data[i].shape_family &&
          family.geometry_rule == entry.geometry.cases.data[i].geometry_rule) {
        if (found) return fail(error, "entry geometry contains a duplicate family");
        found = true;
      }
    }
    if (!found) return fail(error, "entry geometry names an unknown family");
  }
  return true;
}

bool validate_compatibility(const KernelPackCompatibility& compatibility,
                            KernelPackErrorBuffer error) {
  if (!is_known_dtype(compatibility.input_dtype) || !is_known_dtype(compatibility.weight_dtype) ||
      !is_known_dtype(compatibility.output_dtype) ||
      compatibility.source_tensor_layout_version.empty() ||
      compatibility.weight_packing_version.empty() || !valid_span(compatibility.shape_families) ||
      compatibility.shape_families.size == 0) {
    return fail(error, "compatibility contract is incomplete");
  }
  for (std::size_t i = 0; i < compatibility.shape_families.size; ++i) {
    const KernelPackShapeFamily& family = compatibility.shape_families.data[i];
    if (family.name.empty() || family.tail_policy.empty() || family.geometry_rule.empty() ||
        (family.tail_policy != "none" && family.tail_policy != "masked/padded")) {
      return fail(error, "shape family policy is not a closed value");
    }
    if (!valid_span(family.fixed_dimensions) || family.fixed_dimensions.size == 0) {
      return fail(error, "shape family fixed dimensions are missing");
    }
    for (std::size_t dimension_index = 0; dimension_index < family.fixed_dimensions.size;
         ++dimension_index) {
      const KernelPackShapeDimension& dimension = family.fixed_dimensions.data[dimension_index];
      if (dimension.name.empty() || dimension.value == 0) {
        return fail(error, "shape dimension is malformed");
      }
      for (std::size_t prior = 0; prior < dimension_index; ++prior) {
        if (family.fixed_dimensions.data[prior].name == dimension.name) {
          return fail(error, "shape dimensions are duplicated");
        }
      }
    }
    if (family.runtime_dimension.present) {
      const KernelPackRuntimeDimension& runtime = family.runtime_dimension.value;
      if (runtime.name.empty() || runtime.min_value == 0 || runtime.min_value > runtime.max_value ||
          runtime.full_value < runtime.min_value || runtime.full_value > runtime.max_value) {
        return fail(error, "runtime shape dimension is not bounded");
      }
    } else if (family.geometry_rule == "f2-wmma-64x64-m-tail-v1") {
      return fail(error, "F2 geometry requires a runtime shape dimension");
    }
    if (family.geometry_rule == "exact-global-v1" &&
        (family.runtime_dimension.present || family.tail_policy != "none")) {
      return fail(error, "exact geometry family cannot carry a runtime tail");
    }
    for (std::size_t prior = 0; prior < i; ++prior) {
      if (compatibility.shape_families.data[prior].name == family.name) {
        return fail(error, "shape families are duplicated");
      }
    }
    if (family.geometry_rule != "exact-global-v1" &&
        family.geometry_rule != "f2-wmma-64x64-m-tail-v1") {
      return fail(error, "shape family geometry rule is not closed");
    }
    if (family.geometry_rule == "f2-wmma-64x64-m-tail-v1" &&
        (family.name != "f2-linear-gate-up-f16-v1" || family.tail_policy != "masked/padded" ||
         family.runtime_dimension.value.name != "M")) {
      return fail(error, "F2 shape family identity is malformed");
    }
  }
  return true;
}

// Closed EvidenceRef slots include source_review, isa_review, resource_review,
// layout_proof, scalar_native_projection, conformance, native_run, benchmark,
// and numpy_oracle.
bool validate_ref_shape(const EvidenceRef& reference,
                        std::string_view expected_kind,
                        std::string_view expected_slot,
                        const KernelPackRecord& record,
                        std::string_view expected_pack_digest,
                        KernelPackErrorBuffer error) {
  if (!is_safe_path(reference.record_path) || reference.record_id.empty() ||
      !is_lower_hex(reference.record_sha256, 64) ||
      reference.record_kind != expected_kind ||
      reference.evidence_slot != expected_slot) {
    return fail(error, "evidence path is not canonical or identity is malformed");
  }
  if (reference.record_kind == "offline_oracle") {
    if (reference.producer_kind != "cpu_reference" || !reference.subject_target.empty() ||
        !reference.image_sha256.empty() || !reference.pack_sha256.empty() ||
        !reference.tool_digest.empty() || !is_lower_hex(reference.input_digest, 64) ||
        !is_lower_hex(reference.output_digest, 64)) {
      return fail(error, "offline oracle evidence does not match its matrix row");
    }
    return true;
  }
  if (reference.record_kind == "offline_review") {
    if (!reference.producer_kind.empty() || reference.subject_target != record.identity.target ||
        reference.image_sha256 != record.image.image_sha256 ||
        reference.pack_sha256 != expected_pack_digest ||
        !is_lower_hex(reference.tool_digest, 64) || !is_lower_hex(reference.input_digest, 64) ||
        !is_lower_hex(reference.output_digest, 64)) {
      return fail(error, "offline review evidence does not match its matrix row");
    }
    return true;
  }
  if (reference.record_kind == "target_conformance" || reference.record_kind == "native_run") {
    if (reference.producer_kind != "r9700_native" ||
        reference.subject_target != record.identity.target ||
        reference.image_sha256 != record.image.image_sha256 ||
        reference.pack_sha256 != expected_pack_digest || !reference.tool_digest.empty() ||
        !is_lower_hex(reference.input_digest, 64) || !is_lower_hex(reference.output_digest, 64)) {
      return fail(error, "native evidence does not match its matrix row");
    }
    return true;
  }
  if (reference.record_kind == "benchmark") {
    if (reference.producer_kind != "r9700_native" ||
        reference.subject_target != record.identity.target ||
        reference.image_sha256 != record.image.image_sha256 ||
        reference.pack_sha256 != expected_pack_digest || !is_lower_hex(reference.tool_digest, 64) ||
        !is_lower_hex(reference.input_digest, 64) || !is_lower_hex(reference.output_digest, 64)) {
      return fail(error, "benchmark evidence does not match its matrix row");
    }
    return true;
  }
  return fail(error, "evidence record kind is unknown");
}

// The executable lifecycle is unseen -> validating -> admitted|rejected -> loaded -> retired.
class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667U,
               0xbb67ae85U,
               0x3c6ef372U,
               0xa54ff53aU,
               0x510e527fU,
               0x9b05688cU,
               0x1f83d9abU,
               0x5be0cd19U},
        block_{},
        block_size_(0),
        total_size_(0) {}

  void update(const char* data, std::size_t size) {
    total_size_ += size;
    while (size != 0) {
      std::size_t count = 64U - block_size_;
      if (count > size) count = size;
      for (std::size_t i = 0; i < count; ++i) block_[block_size_ + i] = data[i];
      block_size_ += count;
      data += count;
      size -= count;
      if (block_size_ == 64U) {
        compress(block_);
        block_size_ = 0;
      }
    }
  }

  void finish(std::uint8_t digest[32]) const {
    Sha256 copy = *this;
    const std::uint64_t bit_size = copy.total_size_ * 8U;
    const char marker = static_cast<char>(0x80);
    copy.update(&marker, 1);
    const char zero = 0;
    while (copy.block_size_ != 56U) copy.update(&zero, 1);
    char length[8];
    for (unsigned i = 0; i < 8; ++i) {
      length[7U - i] = static_cast<char>((bit_size >> (i * 8U)) & 0xffU);
    }
    copy.update(length, sizeof(length));
    for (unsigned i = 0; i < 8; ++i) {
      digest[i * 4U] = static_cast<std::uint8_t>(copy.state_[i] >> 24U);
      digest[i * 4U + 1U] = static_cast<std::uint8_t>(copy.state_[i] >> 16U);
      digest[i * 4U + 2U] = static_cast<std::uint8_t>(copy.state_[i] >> 8U);
      digest[i * 4U + 3U] = static_cast<std::uint8_t>(copy.state_[i]);
    }
  }

 private:
  static std::uint32_t rotate_right(std::uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32U - bits));
  }

  void compress(const char block[64]) {
    constexpr std::uint32_t round_constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::uint32_t words[64];
    for (unsigned i = 0; i < 16; ++i) {
      words[i] = (static_cast<std::uint32_t>(static_cast<unsigned char>(block[i * 4U])) << 24U) |
                 (static_cast<std::uint32_t>(static_cast<unsigned char>(block[i * 4U + 1U]))
                  << 16U) |
                 (static_cast<std::uint32_t>(static_cast<unsigned char>(block[i * 4U + 2U]))
                  << 8U) |
                 static_cast<std::uint32_t>(static_cast<unsigned char>(block[i * 4U + 3U]));
    }
    for (unsigned i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15U], 7U) ^
                               rotate_right(words[i - 15U], 18U) ^ (words[i - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(words[i - 2U], 17U) ^
                               rotate_right(words[i - 2U], 19U) ^ (words[i - 2U] >> 10U);
      words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (unsigned i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 = h + s1 + choice + round_constants[i] + words[i];
      const std::uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::uint32_t state_[8];
  char block_[64];
  std::size_t block_size_;
  std::uint64_t total_size_;
};

class JsonWriter {
 public:
  void character(char value) { hash_.update(&value, 1); }

  void literal(const char* value) {
    std::size_t size = 0;
    while (value[size] != '\0') ++size;
    hash_.update(value, size);
  }

  void quoted(std::string_view value) {
    character('"');
    for (char item : value) {
      switch (item) {
        case '"':
          literal("\\\"");
          break;
        case '\\':
          literal("\\\\");
          break;
        case '\b':
          literal("\\b");
          break;
        case '\f':
          literal("\\f");
          break;
        case '\n':
          literal("\\n");
          break;
        case '\r':
          literal("\\r");
          break;
        case '\t':
          literal("\\t");
          break;
        default:
          if (static_cast<unsigned char>(item) < 0x20U) {
            const char escaped[] = {'\\', 'u', '0', '0', kDigestHex[(item >> 4) & 0x0f],
                                    kDigestHex[item & 0x0f]};
            hash_.update(escaped, sizeof(escaped));
          } else {
            character(item);
          }
          break;
      }
    }
    character('"');
  }

  void unsigned_number(std::uint64_t value) {
    char digits[24];
    std::size_t size = 0;
    do {
      digits[size++] = static_cast<char>('0' + (value % 10U));
      value /= 10U;
    } while (value != 0);
    while (size != 0) character(digits[--size]);
  }

  void boolean(bool value) { literal(value ? "true" : "false"); }

  void finish(std::uint8_t digest[32]) const { hash_.finish(digest); }

 private:
  Sha256 hash_;
};

void key(JsonWriter& writer, std::string_view value, bool& first) {
  if (!first) writer.character(',');
  first = false;
  writer.quoted(value);
  writer.character(':');
}

void write_strings(JsonWriter& writer, KernelPackSpan<std::string_view> values) {
  writer.character('[');
  for (std::size_t i = 0; i < values.size; ++i) {
    if (i != 0) writer.character(',');
    writer.quoted(values.data[i]);
  }
  writer.character(']');
}

void write_ref_without_pack(JsonWriter& writer, const EvidenceRef& reference) {
  writer.character('{');
  bool first = true;
  key(writer, "evidence_slot", first);
  writer.quoted(reference.evidence_slot);
  key(writer, "image_sha256", first);
  writer.quoted(reference.image_sha256);
  key(writer, "input_digest", first);
  writer.quoted(reference.input_digest);
  key(writer, "output_digest", first);
  writer.quoted(reference.output_digest);
  key(writer, "producer_kind", first);
  writer.quoted(reference.producer_kind);
  key(writer, "record_id", first);
  writer.quoted(reference.record_id);
  key(writer, "record_kind", first);
  writer.quoted(reference.record_kind);
  key(writer, "record_path", first);
  writer.quoted(reference.record_path);
  key(writer, "subject_target", first);
  writer.quoted(reference.subject_target);
  key(writer, "tool_digest", first);
  writer.quoted(reference.tool_digest);
  writer.character('}');
}

void write_dimension(JsonWriter& writer, const KernelPackShapeDimension& dimension) {
  writer.character('{');
  bool first = true;
  key(writer, "name", first);
  writer.quoted(dimension.name);
  key(writer, "value", first);
  writer.unsigned_number(dimension.value);
  writer.character('}');
}

void write_shape_family(JsonWriter& writer, const KernelPackShapeFamily& family) {
  writer.character('{');
  bool first = true;
  key(writer, "fixed_dimensions", first);
  writer.character('[');
  for (std::size_t i = 0; i < family.fixed_dimensions.size; ++i) {
    if (i != 0) writer.character(',');
    write_dimension(writer, family.fixed_dimensions.data[i]);
  }
  writer.character(']');
  key(writer, "geometry_rule", first);
  writer.quoted(family.geometry_rule);
  key(writer, "name", first);
  writer.quoted(family.name);
  key(writer, "runtime_dimension", first);
  if (!family.runtime_dimension.present) {
    writer.literal("null");
  } else {
    const KernelPackRuntimeDimension& runtime = family.runtime_dimension.value;
    writer.character('{');
    bool runtime_first = true;
    key(writer, "full_value", runtime_first);
    writer.unsigned_number(runtime.full_value);
    key(writer, "max_value", runtime_first);
    writer.unsigned_number(runtime.max_value);
    key(writer, "min_value", runtime_first);
    writer.unsigned_number(runtime.min_value);
    key(writer, "name", runtime_first);
    writer.quoted(runtime.name);
    writer.character('}');
  }
  key(writer, "tail_policy", first);
  writer.quoted(family.tail_policy);
  writer.character('}');
}

void write_geometry_case(JsonWriter& writer, const KernelPackGeometryCase& geometry) {
  writer.character('{');
  bool first = true;
  key(writer, "dynamic_lds_allowed", first);
  writer.boolean(geometry.dynamic_lds_allowed);
  key(writer, "dynamic_lds_max_bytes", first);
  writer.unsigned_number(geometry.dynamic_lds_max_bytes);
  key(writer, "geometry_rule", first);
  writer.quoted(geometry.geometry_rule);
  key(writer, "global_x", first);
  writer.unsigned_number(geometry.global_x);
  key(writer, "global_y", first);
  writer.unsigned_number(geometry.global_y);
  key(writer, "global_z", first);
  writer.unsigned_number(geometry.global_z);
  key(writer, "grid_tile_m", first);
  writer.unsigned_number(geometry.grid_tile_m);
  key(writer, "grid_tile_n", first);
  writer.unsigned_number(geometry.grid_tile_n);
  key(writer, "shape_family", first);
  writer.quoted(geometry.shape_family);
  key(writer, "workgroup_x", first);
  writer.unsigned_number(geometry.workgroup_x);
  key(writer, "workgroup_y", first);
  writer.unsigned_number(geometry.workgroup_y);
  key(writer, "workgroup_z", first);
  writer.unsigned_number(geometry.workgroup_z);
  writer.character('}');
}

void write_entry(JsonWriter& writer, const KernelPackEntry& entry) {
  writer.character('{');
  bool first = true;
  key(writer, "descriptor_offset", first);
  writer.unsigned_number(entry.descriptor_offset);
  key(writer, "entry_offset", first);
  writer.unsigned_number(entry.entry_offset);
  key(writer, "geometry", first);
  writer.character('{');
  bool geometry_first = true;
  key(writer, "cases", geometry_first);
  writer.character('[');
  for (std::size_t i = 0; i < entry.geometry.cases.size; ++i) {
    if (i != 0) writer.character(',');
    write_geometry_case(writer, entry.geometry.cases.data[i]);
  }
  writer.character(']');
  writer.character('}');
  key(writer, "kernargs", first);
  writer.character('{');
  bool kernargs_first = true;
  key(writer, "bytes", kernargs_first);
  writer.unsigned_number(entry.kernargs.bytes);
  key(writer, "fields", kernargs_first);
  writer.character('[');
  for (std::size_t i = 0; i < entry.kernargs.fields.size; ++i) {
    if (i != 0) writer.character(',');
    const KernelPackKernargField& field = entry.kernargs.fields.data[i];
    writer.character('{');
    bool field_first = true;
    key(writer, "alignment", field_first);
    writer.unsigned_number(field.alignment);
    key(writer, "name", field_first);
    writer.quoted(field.name);
    key(writer, "offset", field_first);
    writer.unsigned_number(field.offset);
    key(writer, "size", field_first);
    writer.unsigned_number(field.size);
    key(writer, "type", field_first);
    writer.quoted(field.type);
    writer.character('}');
  }
  writer.character(']');
  key(writer, "tail_padding_bytes", kernargs_first);
  writer.unsigned_number(entry.kernargs.tail_padding_bytes);
  writer.character('}');
  key(writer, "resources", first);
  writer.character('{');
  bool resources_first = true;
  key(writer, "lds_bytes", resources_first);
  writer.unsigned_number(entry.resources.lds_bytes);
  key(writer, "metadata_provenance", resources_first);
  writer.quoted(entry.resources.metadata_provenance);
  key(writer, "private_segment_bytes", resources_first);
  writer.unsigned_number(entry.resources.private_segment_bytes);
  key(writer, "rsrc1", resources_first);
  writer.unsigned_number(entry.resources.rsrc1);
  key(writer, "rsrc2", resources_first);
  writer.unsigned_number(entry.resources.rsrc2);
  key(writer, "rsrc3", resources_first);
  writer.unsigned_number(entry.resources.rsrc3);
  key(writer, "sgpr_count", resources_first);
  writer.unsigned_number(entry.resources.sgpr_count);
  key(writer, "vgpr_count", resources_first);
  writer.unsigned_number(entry.resources.vgpr_count);
  key(writer, "wave_size", resources_first);
  writer.unsigned_number(entry.resources.wave_size);
  writer.character('}');
  key(writer, "symbol", first);
  writer.quoted(entry.symbol);
  writer.character('}');
}

void write_pack_without_top_level_evidence(JsonWriter& writer, const KernelPackRecord& record) {
  writer.character('{');
  bool first = true;
  key(writer, "compatibility", first);
  writer.character('{');
  bool compatibility_first = true;
  key(writer, "input_dtype", compatibility_first);
  writer.quoted(record.compatibility.input_dtype);
  key(writer, "output_dtype", compatibility_first);
  writer.quoted(record.compatibility.output_dtype);
  key(writer, "shape_families", compatibility_first);
  writer.character('[');
  for (std::size_t i = 0; i < record.compatibility.shape_families.size; ++i) {
    if (i != 0) writer.character(',');
    write_shape_family(writer, record.compatibility.shape_families.data[i]);
  }
  writer.character(']');
  key(writer, "source_tensor_layout_version", compatibility_first);
  writer.quoted(record.compatibility.source_tensor_layout_version);
  key(writer, "weight_dtype", compatibility_first);
  writer.quoted(record.compatibility.weight_dtype);
  key(writer, "weight_packing_version", compatibility_first);
  writer.quoted(record.compatibility.weight_packing_version);
  writer.character('}');
  key(writer, "entries", first);
  writer.character('[');
  for (std::size_t i = 0; i < record.entries.size; ++i) {
    if (i != 0) writer.character(',');
    write_entry(writer, record.entries.data[i]);
  }
  writer.character(']');
  key(writer, "image", first);
  writer.character('{');
  bool image_first = true;
  key(writer, "build", image_first);
  writer.character('{');
  bool build_first = true;
  key(writer, "command_sha256", build_first);
  writer.quoted(record.image.build.command_sha256);
  key(writer, "generator_id", build_first);
  writer.quoted(record.image.build.generator_id);
  key(writer, "generator_revision", build_first);
  writer.quoted(record.image.build.generator_revision);
  key(writer, "toolchain_id", build_first);
  writer.quoted(record.image.build.toolchain_id);
  key(writer, "toolchain_revision", build_first);
  writer.quoted(record.image.build.toolchain_revision);
  writer.character('}');
  key(writer, "code_object_version", image_first);
  writer.quoted(record.image.code_object_version);
  key(writer, "image_path", image_first);
  writer.quoted(record.image.image_path);
  key(writer, "image_sha256", image_first);
  writer.quoted(record.image.image_sha256);
  key(writer, "image_size", image_first);
  writer.unsigned_number(record.image.image_size);
  writer.character('}');
  key(writer, "name", first);
  writer.quoted(record.identity.name);
  key(writer, "numerics", first);
  writer.character('{');
  bool numerics_first = true;
  key(writer, "accumulation_dtype", numerics_first);
  writer.quoted(record.numerics.accumulation_dtype);
  key(writer, "cast_points", numerics_first);
  writer.character('[');
  for (std::size_t i = 0; i < record.numerics.cast_points.size; ++i) {
    if (i != 0) writer.character(',');
    const KernelPackCastPoint& point = record.numerics.cast_points.data[i];
    writer.character('{');
    bool point_first = true;
    key(writer, "from_dtype", point_first);
    writer.quoted(point.from_dtype);
    key(writer, "stage", point_first);
    writer.quoted(point.stage);
    key(writer, "to_dtype", point_first);
    writer.quoted(point.to_dtype);
    writer.character('}');
  }
  writer.character(']');
  key(writer, "finite_value_rule", numerics_first);
  writer.quoted(record.numerics.finite_value_rule);
  key(writer, "input_dtype", numerics_first);
  writer.quoted(record.numerics.input_dtype);
  key(writer, "numpy_oracle", numerics_first);
  if (record.numerics.numpy_oracle.present) {
    write_ref_without_pack(writer, record.numerics.numpy_oracle.value);
  } else {
    writer.literal("null");
  }
  key(writer, "output_dtype", numerics_first);
  writer.quoted(record.numerics.output_dtype);
  key(writer, "reference_set_kind", numerics_first);
  writer.quoted(record.numerics.reference_set_kind);
  key(writer, "retained_reference", numerics_first);
  if (record.numerics.retained_reference.present) {
    write_ref_without_pack(writer, record.numerics.retained_reference.value);
  } else {
    writer.literal("null");
  }
  key(writer, "scalar_native_projection", numerics_first);
  if (record.numerics.scalar_native_projection.present) {
    write_ref_without_pack(writer, record.numerics.scalar_native_projection.value);
  } else {
    writer.literal("null");
  }
  key(writer, "tolerance_policy", numerics_first);
  writer.quoted(record.numerics.tolerance_policy);
  writer.character('}');
  key(writer, "provenance", first);
  writer.character('{');
  bool provenance_first = true;
  key(writer, "license_reviews", provenance_first);
  writer.character('[');
  for (std::size_t i = 0; i < record.provenance.license_reviews.size; ++i) {
    if (i != 0) writer.character(',');
    const KernelPackLicenseReview& review = record.provenance.license_reviews.data[i];
    writer.character('{');
    bool review_first = true;
    key(writer, "component", review_first);
    writer.quoted(review.component);
    key(writer, "review_id", review_first);
    writer.quoted(review.review_id);
    key(writer, "spdx_expression", review_first);
    writer.quoted(review.spdx_expression);
    key(writer, "status", review_first);
    writer.quoted(review.status);
    writer.character('}');
  }
  writer.character(']');
  key(writer, "local_sources", provenance_first);
  writer.character('[');
  for (std::size_t i = 0; i < record.provenance.local_sources.size; ++i) {
    if (i != 0) writer.character(',');
    const KernelPackSource& source = record.provenance.local_sources.data[i];
    writer.character('{');
    bool source_first = true;
    key(writer, "path", source_first);
    writer.quoted(source.path);
    key(writer, "sha256", source_first);
    writer.quoted(source.sha256);
    writer.character('}');
  }
  writer.character(']');
  key(writer, "modifications", provenance_first);
  writer.character('[');
  for (std::size_t i = 0; i < record.provenance.modifications.size; ++i) {
    if (i != 0) writer.character(',');
    const KernelPackModification& modification = record.provenance.modifications.data[i];
    writer.character('{');
    bool modification_first = true;
    key(writer, "component", modification_first);
    writer.quoted(modification.component);
    key(writer, "summary", modification_first);
    writer.quoted(modification.summary);
    writer.character('}');
  }
  writer.character(']');
  key(writer, "upstream_paths", provenance_first);
  write_strings(writer, record.provenance.upstream_paths);
  key(writer, "upstream_repository", provenance_first);
  writer.quoted(record.provenance.upstream_repository);
  key(writer, "upstream_revision", provenance_first);
  writer.quoted(record.provenance.upstream_revision);
  writer.character('}');
  key(writer, "required_features", first);
  write_strings(writer, record.identity.required_features);
  key(writer, "schema_version", first);
  writer.unsigned_number(record.identity.schema_version);
  key(writer, "target", first);
  writer.quoted(record.identity.target);
  key(writer, "version", first);
  writer.quoted(record.identity.version);
  writer.character('}');
}

void compute_pack_digest(const KernelPackRecord& record, char output[65]) {
  JsonWriter writer;
  writer.character('{');
  bool first = true;
  key(writer, "domain", first);
  writer.quoted(kPackDomain);
  key(writer, "pack", first);
  write_pack_without_top_level_evidence(writer, record);
  writer.character('}');
  std::uint8_t digest[32];
  writer.finish(digest);
  for (unsigned i = 0; i < 32; ++i) {
    output[i * 2U] = kDigestHex[digest[i] >> 4U];
    output[i * 2U + 1U] = kDigestHex[digest[i] & 0x0fU];
  }
  output[64] = '\0';
}

bool validate_numerics(const KernelPackRecord& record,
                       std::string_view expected_pack_digest,
                       KernelPackErrorBuffer error) {
  const KernelPackNumerics& numerics = record.numerics;
  if (numerics.input_dtype != record.compatibility.input_dtype ||
      numerics.output_dtype != record.compatibility.output_dtype ||
      !is_known_dtype(numerics.accumulation_dtype) ||
      numerics.finite_value_rule != "finite-input-output-v1" ||
      numerics.tolerance_policy.empty() || !valid_span(numerics.cast_points) ||
      numerics.cast_points.size == 0) {
    return fail(error, "finite-value rule or numerical contract is incomplete");
  }
  for (std::size_t i = 0; i < numerics.cast_points.size; ++i) {
    const KernelPackCastPoint& point = numerics.cast_points.data[i];
    if (point.stage.empty() || !is_known_dtype(point.from_dtype) ||
        !is_known_dtype(point.to_dtype) ||
        point.to_dtype != numerics.accumulation_dtype) {
      return fail(error, "numeric cast point is malformed");
    }
  }
  if (numerics.reference_set_kind == "b0_scalar_control") {
    if (!numerics.retained_reference.present || numerics.numpy_oracle.present ||
        numerics.scalar_native_projection.present ||
        !validate_ref_shape(numerics.retained_reference.value, "offline_oracle", "numpy_oracle",
                            record, expected_pack_digest, error)) {
      return fail(error, "B0 numerical references are malformed");
    }
  } else if (numerics.reference_set_kind == "f2_wmma_dual") {
    if (numerics.retained_reference.present || !numerics.numpy_oracle.present ||
        !numerics.scalar_native_projection.present ||
        !validate_ref_shape(numerics.numpy_oracle.value, "offline_oracle", "numpy_oracle",
                            record, expected_pack_digest, error) ||
        !validate_ref_shape(numerics.scalar_native_projection.value, "target_conformance",
                            "scalar_native_projection", record, expected_pack_digest, error) ||
        numerics.numpy_oracle.value.input_digest !=
            numerics.scalar_native_projection.value.input_digest) {
      return fail(error, "F2 numerical references are malformed");
    }
  } else {
    return fail(error, "reference set kind is unknown");
  }
  return true;
}

bool validate_evidence(const KernelPackRecord& record,
                       std::string_view expected_pack_digest,
                       KernelPackErrorBuffer error) {
  const KernelPackEvidence& evidence = record.evidence;
  if (!validate_ref_shape(evidence.conformance, "target_conformance", "conformance", record,
                          expected_pack_digest, error) ||
      !validate_ref_shape(evidence.native_run, "native_run", "native_run", record,
                          expected_pack_digest, error) ||
      !validate_ref_shape(evidence.source_review, "offline_review", "source_review", record,
                          expected_pack_digest, error) ||
      !validate_ref_shape(evidence.resource_review, "offline_review", "resource_review", record,
                          expected_pack_digest, error) ||
      !validate_ref_shape(evidence.isa_review, "offline_review", "isa_review", record,
                          expected_pack_digest, error)) {
    return false;
  }
  const bool physical_layout =
      record.compatibility.weight_packing_version != kSourceEquivalentPacking;
  if (evidence.layout_proof.present != physical_layout) {
    return fail(error, "layout proof presence does not match physical packing");
  }
  if (evidence.layout_proof.present &&
      !validate_ref_shape(evidence.layout_proof.value, "offline_review", "layout_proof", record,
                          expected_pack_digest, error)) {
    return false;
  }
  if (evidence.benchmark_record.present) {
    if (!validate_ref_shape(evidence.benchmark_record.value, "benchmark", "benchmark", record,
                            expected_pack_digest, error) ||
        !evidence.benchmark_not_applicable_reason.empty()) {
      return fail(error, "benchmark evidence has an inapplicable reason");
    }
  } else if (evidence.benchmark_not_applicable_reason.empty()) {
    return fail(error, "correctness-control pack lacks benchmark reason");
  }
  return true;
}

bool validate_digest_integer_ranges(const KernelPackRecord& record,
                                    KernelPackErrorBuffer error) {
  const auto within_jcs = [error](std::uint64_t value) {
    return value <= kJcsExactIntegerMax ||
           fail(error, "digest integer exceeds JCS exact ceiling");
  };
  if (!within_jcs(record.image.image_size)) return false;
  for (std::size_t i = 0; i < record.entries.size; ++i) {
    const KernelPackEntry& entry = record.entries.data[i];
    if (!within_jcs(entry.descriptor_offset) ||
        !within_jcs(entry.entry_offset) ||
        !within_jcs(entry.resources.lds_bytes) ||
        !within_jcs(entry.resources.private_segment_bytes)) {
      return false;
    }
    for (std::size_t j = 0; j < entry.geometry.cases.size; ++j) {
      if (!within_jcs(entry.geometry.cases.data[j].dynamic_lds_max_bytes)) {
        return false;
      }
    }
  }
  return true;
}

bool validate_record_shape(const KernelPackRecord& record, KernelPackErrorBuffer error) {
  if (record.identity.schema_version != 1 || record.identity.name.empty() ||
      !is_canonical_version(record.identity.version) || record.identity.target != kTarget) {
    return fail(error, "pack identity is not canonical");
  }
  if (!validate_required_features(record.identity, error)) return false;
  if (!validate_provenance(record, error) || !validate_image(record.image, error) ||
      !validate_compatibility(record.compatibility, error)) {
    return false;
  }
  if (!valid_span(record.entries) || record.entries.size != 1) {
    return fail(error, "Kernel Pack schema v1 requires exactly one entry");
  }
  for (std::size_t i = 0; i < record.entries.size; ++i) {
    for (std::size_t prior = 0; prior < i; ++prior) {
      if (record.entries.data[prior].symbol == record.entries.data[i].symbol) {
        return fail(error, "entry symbols are duplicated");
      }
    }
    if (!validate_entry(record.entries.data[i], record.compatibility,
                        record.image.image_size, error)) {
      return false;
    }
  }
  return true;
}

bool matches_key_after_validation(const KernelPackRecord& record,
                                  const KernelPackCompatibilityKey& key) {
  if (record.identity.target != key.target ||
      !same_string_span(record.identity.required_features, key.required_features) ||
      record.compatibility.input_dtype != key.input_dtype ||
      record.compatibility.weight_dtype != key.weight_dtype ||
      record.compatibility.output_dtype != key.output_dtype ||
      record.compatibility.source_tensor_layout_version != key.source_tensor_layout_version ||
      record.compatibility.weight_packing_version != key.weight_packing_version ||
      record.numerics.tolerance_policy != key.tolerance_policy) {
    return false;
  }
  const KernelPackShapeFamily* family = nullptr;
  if (!find_family(record, key.shape_family_name, &family) ||
      !same_dimensions(family->fixed_dimensions, key.fixed_dimensions)) {
    return false;
  }
  if (family->runtime_dimension.present != key.runtime_value.present) return false;
  if (!family->runtime_dimension.present) return true;
  return same_runtime(family->runtime_dimension, key.runtime_value);
}

}  // namespace

bool validate_kernel_pack(const KernelPackRecord& record, KernelPackErrorBuffer error_text) {
  clear_error(error_text);
  if (!validate_digest_read_spans(record, error_text)) return false;
  if (!validate_record_shape(record, error_text)) return false;
  if (!validate_digest_integer_ranges(record, error_text)) return false;
  char pack_digest[65];
  compute_pack_digest(record, pack_digest);
  if (!validate_numerics(record, std::string_view(pack_digest, 64), error_text)) return false;
  if (!validate_evidence(record, std::string_view(pack_digest, 64), error_text)) return false;
  return true;
}

bool kernel_pack_matches_key(const KernelPackRecord& record,
                             const KernelPackCompatibilityKey& key) {
  char ignored_error[2] = {};
  if (!validate_kernel_pack(record, {ignored_error, sizeof(ignored_error)})) return false;
  return matches_key_after_validation(record, key);
}

const KernelPackRecord* find_kernel_pack(KernelPackSpan<KernelPackRecord> records,
                                         std::string_view name,
                                         std::string_view version,
                                         KernelPackErrorBuffer error_text) {
  clear_error(error_text);
  if (!valid_span(records) || name.empty() || version.empty()) {
    fail(error_text, "pack lookup requires an explicit name and version");
    return nullptr;
  }
  const KernelPackRecord* found = nullptr;
  for (std::size_t i = 0; i < records.size; ++i) {
    const KernelPackRecord& record = records.data[i];
    if (!validate_kernel_pack(record, error_text)) return nullptr;
    if (record.identity.name != name || record.identity.version != version) continue;
    if (found != nullptr) {
      fail(error_text, "pack identity lookup is ambiguous");
      return nullptr;
    }
    found = &record;
  }
  if (found == nullptr) fail(error_text, "pack identity was not found");
  return found;
}

const KernelPackRecord* find_kernel_pack_for_key(KernelPackSpan<KernelPackRecord> records,
                                                  const KernelPackCompatibilityKey& key,
                                                  KernelPackErrorBuffer error_text) {
  clear_error(error_text);
  if (!valid_span(records)) {
    fail(error_text, "pack key lookup span is invalid");
    return nullptr;
  }
  const KernelPackRecord* found = nullptr;
  for (std::size_t i = 0; i < records.size; ++i) {
    if (!validate_kernel_pack(records.data[i], error_text)) return nullptr;
    if (!matches_key_after_validation(records.data[i], key)) continue;
    if (found != nullptr) {
      fail(error_text, "pack key lookup is ambiguous");
      return nullptr;
    }
    found = &records.data[i];
  }
  if (found == nullptr) fail(error_text, "pack compatibility key was not found");
  return found;
}

bool admit_kernel_pack(const KernelPackRecord& record,
                       const KernelPackCompatibilityKey& selected_key,
                       std::string_view entry_symbol,
                       std::string_view asset_root,
                       KernelDescriptor* out_descriptor,
                       KernelPackErrorBuffer error_text) {
  clear_error(error_text);
  if (out_descriptor == nullptr) return fail(error_text, "output descriptor is required");
  if (!validate_kernel_pack(record, error_text)) return false;
  if (!matches_key_after_validation(record, selected_key)) {
    return fail(error_text, "selected compatibility key does not match the pack");
  }

  const KernelPackEntry* entry = nullptr;
  if (!find_entry(record, entry_symbol, &entry)) {
    return fail(error_text, "pack entry symbol was not found");
  }
  const KernelPackGeometryCase* geometry = nullptr;
  for (std::size_t i = 0; i < entry->geometry.cases.size; ++i) {
    const KernelPackGeometryCase& candidate = entry->geometry.cases.data[i];
    if (candidate.shape_family != selected_key.shape_family_name) continue;
    if (geometry != nullptr) {
      return fail(error_text, "selected geometry family is ambiguous");
    }
    geometry = &candidate;
  }
  if (geometry == nullptr) {
    return fail(error_text, "selected geometry family is not present in the entry");
  }

  const LlamaKernelAsset* asset = find_llama_kernel_asset(entry_symbol);
  const KernelAssetPackAttestation* attestation =
      find_kernel_pack_attestation(entry_symbol);
  if (asset == nullptr || attestation == nullptr) {
    return fail(error_text, "kernel asset lacks reviewed pack attestation");
  }
  KernelDescriptor loaded{};
  const std::filesystem::path root(asset_root.begin(), asset_root.end());
  if (!load_verified_kernel_code(
          *asset, root, attestation->kernarg_schema, &loaded, nullptr)) {
    return fail(error_text, "existing kernel asset/HSA admission rejected the image");
  }
  if (asset->descriptor.name != entry->symbol ||
      asset->kernarg_schema != attestation->kernarg_schema ||
      std::string_view(asset->location.target) != attestation->target ||
      std::string_view(asset->location.code_path) != attestation->image_path ||
      std::string_view(asset->location.sha256) != attestation->image_sha256 ||
      record.identity.target != attestation->target ||
      record.image.image_path != attestation->image_path ||
      record.image.image_sha256 != attestation->image_sha256 ||
      record.image.image_size != attestation->image_size ||
      record.image.code_object_version != attestation->code_object_version ||
      entry->descriptor_offset != attestation->descriptor_offset ||
      entry->entry_offset != attestation->entry_offset ||
      entry->kernargs.bytes != attestation->kernarg_bytes ||
      entry->kernargs.tail_padding_bytes !=
          attestation->kernarg_tail_padding_bytes ||
      entry->kernargs.fields.size != attestation->kernarg_field_count) {
    return fail(error_text, "pack image or ABI metadata is not asset-attested");
  }
  for (std::size_t i = 0; i < entry->kernargs.fields.size; ++i) {
    const KernelPackKernargField& field = entry->kernargs.fields.data[i];
    const KernelAssetKernargField& expected = attestation->kernarg_fields[i];
    if (field.name != expected.name || field.type != expected.type ||
        field.offset != expected.offset || field.size != expected.size ||
        field.alignment != expected.alignment) {
      return fail(error_text, "pack kernarg field is not asset-attested");
    }
  }
  const KernelPackResources& resources = entry->resources;
  if (resources.rsrc1 != attestation->rsrc1 ||
      resources.rsrc2 != attestation->rsrc2 ||
      resources.rsrc3 != attestation->rsrc3 ||
      resources.wave_size != attestation->wave_size ||
      resources.sgpr_count != attestation->sgpr_count ||
      resources.vgpr_count != attestation->vgpr_count ||
      resources.lds_bytes != attestation->lds_bytes ||
      resources.private_segment_bytes != attestation->private_segment_bytes ||
      resources.metadata_provenance != attestation->metadata_provenance ||
      geometry->workgroup_x != attestation->workgroup_x ||
      geometry->workgroup_y != attestation->workgroup_y ||
      geometry->workgroup_z != attestation->workgroup_z ||
      geometry->global_x != attestation->global_x ||
      geometry->global_y != attestation->global_y ||
      geometry->global_z != attestation->global_z ||
      geometry->grid_tile_m != attestation->grid_tile_m ||
      geometry->grid_tile_n != attestation->grid_tile_n ||
      geometry->dynamic_lds_allowed != attestation->dynamic_lds_allowed ||
      geometry->dynamic_lds_max_bytes !=
          attestation->dynamic_lds_max_bytes) {
    return fail(error_text, "pack resources or geometry are not asset-attested");
  }
  if (loaded.name != entry->symbol ||
      loaded.sha256 != attestation->image_sha256 ||
      loaded.code.size() != attestation->image_size ||
      loaded.rsrc1 != attestation->rsrc1 ||
      loaded.rsrc2 != attestation->rsrc2 ||
      loaded.rsrc3 != attestation->rsrc3 ||
      loaded.workgroup_x != attestation->workgroup_x ||
      loaded.workgroup_y != attestation->workgroup_y ||
      loaded.workgroup_z != attestation->workgroup_z ||
      loaded.global_x != attestation->global_x ||
      loaded.global_y != attestation->global_y ||
      loaded.global_z != attestation->global_z ||
      loaded.kernarg_bytes != attestation->kernarg_bytes) {
    return fail(error_text, "loaded kernel does not match asset attestation");
  }
  *out_descriptor = std::move(loaded);
  return true;
}

}  // namespace native_r9700

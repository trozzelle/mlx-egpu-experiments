#ifndef NATIVE_R9700_KERNEL_ASSETS_H_
#define NATIVE_R9700_KERNEL_ASSETS_H_
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "kernel_catalog.h"

namespace native_r9700 {

// Location and reviewed source metadata for a code file stored beneath an
// explicit asset root. These values accompany, but do not duplicate, the
// dispatch descriptor's resource and geometry contract.
struct KernelAssetLocation {
  std::string code_path;
  std::string sha256;
  std::string target;
  int32_t sgpr_count = 0;
  int32_t vgpr_count = 0;
  int32_t lds_bytes = 0;
  std::string resource_metadata_provenance;
};
// Allocation-free ABI metadata owned by the reviewed asset boundary. Only
// assets explicitly returned by find_kernel_pack_attestation() may cross the
// Kernel Pack admission boundary.
struct KernelAssetKernargField {
  std::string_view name;
  std::string_view type;
  std::uint32_t offset;
  std::uint32_t size;
  std::uint32_t alignment;
};

struct KernelAssetPackAttestation {
  std::string_view target;
  std::string_view image_path;
  std::string_view image_sha256;
  std::uint64_t image_size;
  std::string_view code_object_version;
  std::uint64_t descriptor_offset;
  std::uint64_t entry_offset;
  std::string_view kernarg_schema;
  std::uint32_t kernarg_bytes;
  std::uint32_t kernarg_tail_padding_bytes;
  const KernelAssetKernargField* kernarg_fields;
  std::size_t kernarg_field_count;
  std::uint32_t rsrc1;
  std::uint32_t rsrc2;
  std::uint32_t rsrc3;
  std::uint32_t wave_size;
  std::uint32_t sgpr_count;
  std::uint32_t vgpr_count;
  std::uint64_t lds_bytes;
  std::uint64_t private_segment_bytes;
  std::string_view metadata_provenance;
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


// A manifest entry for a reviewed Llama stage. Its descriptor intentionally
// carries no code until load_verified_kernel_code verifies its file asset.
struct LlamaKernelAsset {
  KernelDescriptor descriptor;
  KernelAssetLocation location;
  std::string kernarg_schema;
};

// Returns the reviewed Llama manifest entry for name, or nullptr if none is
// available. The manifest order is the stable scalar stage/diagnostic order.
const LlamaKernelAsset* find_llama_kernel_asset(std::string_view name);

// Returns exact ABI metadata only for assets admitted through the reviewed
// Kernel Pack boundary. Unattested assets, including Qwen/G0 assets, return
// nullptr rather than receiving synthesized metadata.
const KernelAssetPackAttestation* find_kernel_pack_attestation(std::string_view name);

// Returns the reviewed Qwen manifest entry for name, or nullptr if none is
// available. Reuses the LlamaKernelAsset record shape (descriptor + location +
// kernarg schema) so the shared loader path stays unchanged.
const LlamaKernelAsset* find_qwen_kernel_asset(std::string_view name);

// Materializes a descriptor from its verified, manifest-relative code file.
// On failure, out_descriptor is not modified and error_text receives a reason
// when it is non-null.
bool load_verified_kernel_code(const LlamaKernelAsset& asset,
                               const std::filesystem::path& asset_root,
                               std::string_view expected_kernarg_schema,
                               KernelDescriptor* out_descriptor,
                               std::string* error_text);

}  // namespace native_r9700

#endif  // NATIVE_R9700_KERNEL_ASSETS_H_

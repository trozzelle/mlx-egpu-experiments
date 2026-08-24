#ifndef NATIVE_R9700_KERNEL_ASSETS_H_
#define NATIVE_R9700_KERNEL_ASSETS_H_

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

// A manifest entry for a reviewed Llama stage. Its descriptor intentionally
// carries no code until load_verified_kernel_code verifies its file asset.
struct LlamaKernelAsset {
  KernelDescriptor descriptor;
  KernelAssetLocation location;
  std::string kernarg_schema;
};

// Returns the reviewed Llama manifest entry for name, or nullptr if none is
// available. The manifest remains empty until stage assets are reviewed.
const LlamaKernelAsset* find_llama_kernel_asset(std::string_view name);

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

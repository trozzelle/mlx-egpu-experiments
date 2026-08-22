#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace native_r9700 {

// A fully attested copy of the one generated code image accepted by the
// runtime boundary. Loading this object performs no driver or device work.
struct HsaCodeImageAsset {
  std::vector<std::uint8_t> image;
  std::string image_sha256;
  std::uint64_t descriptor_offset = 0;
  std::uint64_t entry_offset = 0;
  std::uint32_t rsrc1 = 0;
  std::uint32_t rsrc2 = 0;
  std::uint32_t rsrc3 = 0;
  std::string schema;
  std::string source_path;
  std::string source_sha256;
};

// Loads only the generated V1 llama_embed_row_f16 HSA image from root. The
// output is changed only after every filesystem and manifest check succeeds.
bool load_llama_embed_hsa_image(const std::filesystem::path& root,
                                HsaCodeImageAsset* out,
                                std::string* error);

}  // namespace native_r9700

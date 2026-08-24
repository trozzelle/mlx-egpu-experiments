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
  bool wave32 = false;  // ENABLE_WAVEFRONT_SIZE32 from the AMDHSA kernel descriptor.
  std::string schema;
  std::string source_path;
  std::string source_sha256;
};

// Computes the lowercase SHA-256 digest of raw bytes. Asset loading and native
// trace artifacts share this implementation so their digest semantics match.

// Decodes ENABLE_WAVEFRONT_SIZE32 from the AMDHSA kernel descriptor's
// kernel_code_properties (uint16 at descriptor offset +56, bit 10 = 0x400).
inline bool image_is_wave32(const std::vector<std::uint8_t>& image, std::uint64_t descriptor_offset) {
  if (descriptor_offset + 58U > image.size()) return false;
  const std::uint16_t kernel_code_properties =
      static_cast<std::uint16_t>(image[descriptor_offset + 56U]) |
      (static_cast<std::uint16_t>(image[descriptor_offset + 57U]) << 8);
  return (kernel_code_properties & 0x400U) != 0U;
}
std::string sha256_hex(const std::vector<std::uint8_t>& bytes);

// Loads only the generated V1 llama_embed_row_f16 HSA image from root. The
// output is changed only after every filesystem and manifest check succeeds.
bool load_llama_embed_hsa_image(const std::filesystem::path& root,
                                HsaCodeImageAsset* out,
                                std::string* error);

}  // namespace native_r9700

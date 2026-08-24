#include "hsa_code_image_asset.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace native_r9700 {
namespace {

constexpr std::size_t kMaximumImageBytes = 4U * 1024U * 1024U;
constexpr std::size_t kMaximumManifestBytes = 64U * 1024U;
constexpr char kImageName[] = "llama_embed_row_f16.image";
constexpr char kManifestName[] = "llama_embed_row_f16.json";
constexpr char kImageSha256[] =
    "389d8726a5a3e0d827f05680fb73b0d08e2dade34d8ae2ef79d5010c0bfdb53e";
constexpr char kSourcePath[] = "native_r9700/kernels/llama_embed_row_f16.cpp";
constexpr char kSourceSha256[] =
    "a4c6be25193895d54549530beb9c3224addda22562999c8bc949a6d87153043f";
constexpr char kSchema[] =
    R"({"name":"llama-embed-row-f16-v1","bytes":24,"fields":[{"name":"embedding_rows","offset":0,"type":"uint64"},{"name":"hidden_output","offset":8,"type":"uint64"},{"name":"selected_row","offset":16,"type":"uint64"}]})";
constexpr std::uint64_t kImageSize = 14833;
constexpr std::uint64_t kDescriptorOffset = 1536;
constexpr std::uint64_t kEntryOffset = 5888;
constexpr std::uint32_t kRsrc1 = 3222208512U;
constexpr std::uint32_t kRsrc2 = 132U;
constexpr std::uint32_t kRsrc3 = 32U;

void set_error(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
}

class Sha256 {
 public:
  Sha256() : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(const std::uint8_t* data, std::size_t size) {
    bit_count_ += static_cast<std::uint64_t>(size) * 8U;
    while (size != 0U) {
      const std::size_t available = block_.size() - block_size_;
      const std::size_t count = size < available ? size : available;
      std::memcpy(block_.data() + block_size_, data, count);
      block_size_ += count;
      data += count;
      size -= count;
      if (block_size_ == block_.size()) {
        transform(block_.data());
        block_size_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finish() {
    const std::uint64_t original_bit_count = bit_count_;
    const std::uint8_t one = 0x80U;
    update(&one, 1U);
    const std::uint8_t zero = 0;
    while (block_size_ != 56U) update(&zero, 1U);
    std::array<std::uint8_t, 8> length{};
    for (std::size_t index = 0; index < length.size(); ++index) {
      length[length.size() - 1U - index] =
          static_cast<std::uint8_t>(original_bit_count >> (index * 8U));
    }
    update(length.data(), length.size());

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
      digest[index * 4U] = static_cast<std::uint8_t>(state_[index] >> 24U);
      digest[index * 4U + 1U] = static_cast<std::uint8_t>(state_[index] >> 16U);
      digest[index * 4U + 2U] = static_cast<std::uint8_t>(state_[index] >> 8U);
      digest[index * 4U + 3U] = static_cast<std::uint8_t>(state_[index]);
    }
    return digest;
  }

 private:
  static std::uint32_t rotate_right(std::uint32_t value, unsigned amount) {
    return (value >> amount) | (value << (32U - amount));
  }

  void transform(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> kRoundConstants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
        0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
        0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
        0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
        0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
        0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
        0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
        0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      words[index] = (static_cast<std::uint32_t>(block[index * 4U]) << 24U) |
                     (static_cast<std::uint32_t>(block[index * 4U + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(block[index * 4U + 2U]) << 8U) |
                     static_cast<std::uint32_t>(block[index * 4U + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                               rotate_right(words[index - 15U], 18U) ^
                               (words[index - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                               rotate_right(words[index - 2U], 19U) ^
                               (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                               rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 = h + s1 + choose + kRoundConstants[index] + words[index];
      const std::uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                               rotate_right(a, 22U);
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

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

std::string hex_digest(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  Sha256 hash;
  hash.update(bytes.data(), bytes.size());
  const std::array<std::uint8_t, 32> digest = hash.finish();
  std::string result(digest.size() * 2U, '\0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[index * 2U] = kHex[digest[index] >> 4U];
    result[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return result;
}

bool read_regular_child(int directory_fd, const char* name, std::size_t maximum_size,
                        std::vector<std::uint8_t>* contents) {
  const int fd = openat(directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return false;

  struct stat status {};
  if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uintmax_t>(status.st_size) > maximum_size) {
    close(fd);
    return false;
  }

  contents->assign(static_cast<std::size_t>(status.st_size), 0U);
  std::size_t offset = 0;
  while (offset < contents->size()) {
    const ssize_t count = read(fd, contents->data() + offset, contents->size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    close(fd);
    return false;
  }
  return close(fd) == 0;
}

class ManifestReader {
 public:
  explicit ManifestReader(const std::vector<std::uint8_t>& data)
      : current_(reinterpret_cast<const char*>(data.data())),
        end_(current_ + data.size()) {}

  bool validate() {
    skip_space();
    if (!parse_top_level()) return false;
    skip_space();
    return current_ == end_;
  }

 private:
  struct LayoutEntry {
    const char* name;
    std::uint64_t address;
    std::uint64_t image_offset;
    std::uint64_t size;
  };

  static constexpr std::array<LayoutEntry, 10> kLayout{{
      {".note", 568U, 568U, 684U},
      {".dynsym", 1256U, 1256U, 96U},
      {".gnu.hash", 1352U, 1352U, 40U},
      {".hash", 1392U, 1392U, 40U},
      {".dynstr", 1432U, 1432U, 72U},
      {".rodata", 1536U, 1536U, 64U},
      {".text", 5888U, 5888U, 640U},
      {".dynamic", 10624U, 10624U, 112U},
      {".relro_padding", 10736U, 10736U, 1552U},
      {".bss", 14832U, 14832U, 1U},
  }};

  void skip_space() {
    while (current_ != end_ && (*current_ == ' ' || *current_ == '\n' || *current_ == '\r' ||
                               *current_ == '\t')) {
      ++current_;
    }
  }

  bool consume(char expected) {
    skip_space();
    if (current_ == end_ || *current_ != expected) return false;
    ++current_;
    return true;
  }

  bool parse_string(std::string* value) {
    skip_space();
    if (current_ == end_ || *current_ != '"') return false;
    ++current_;
    const char* start = current_;
    while (current_ != end_ && *current_ != '"') {
      const unsigned char character = static_cast<unsigned char>(*current_);
      if (character < 0x20U || *current_ == '\\') return false;
      ++current_;
    }
    if (current_ == end_) return false;
    value->assign(start, static_cast<std::size_t>(current_ - start));
    ++current_;
    return true;
  }

  bool parse_unsigned(std::uint64_t* value) {
    skip_space();
    if (current_ == end_ || *current_ < '0' || *current_ > '9') return false;
    if (*current_ == '0') {
      ++current_;
      if (current_ != end_ && *current_ >= '0' && *current_ <= '9') return false;
      *value = 0;
      return true;
    }
    std::uint64_t parsed = 0;
    while (current_ != end_ && *current_ >= '0' && *current_ <= '9') {
      const std::uint64_t digit = static_cast<std::uint64_t>(*current_ - '0');
      if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) return false;
      parsed = parsed * 10U + digit;
      ++current_;
    }
    *value = parsed;
    return true;
  }

  bool parse_exact_string(const char* expected) {
    std::string parsed;
    return parse_string(&parsed) && parsed == expected;
  }

  bool parse_exact_unsigned(std::uint64_t expected) {
    std::uint64_t parsed = 0;
    return parse_unsigned(&parsed) && parsed == expected;
  }

  bool separator_or_end(bool* first, char end) {
    skip_space();
    if (*first) {
      *first = false;
      return current_ != end_ && *current_ != end;
    }
    if (current_ == end_) return false;
    if (*current_ == end) return false;
    if (*current_ != ',') return false;
    ++current_;
    return true;
  }

  bool parse_top_level() {
    if (!consume('{')) return false;
    unsigned int seen = 0;
    bool first = true;
    while (separator_or_end(&first, '}')) {
      std::string key;
      if (!parse_string(&key) || !consume(':')) return false;
      unsigned int bit = 0;
      bool valid = false;
      if (key == "descriptor_offset") {
        bit = 1U << 0U;
        valid = parse_exact_unsigned(kDescriptorOffset);
      } else if (key == "elf_admission") {
        bit = 1U << 1U;
        valid = parse_elf_admission();
      } else if (key == "entry_offset") {
        bit = 1U << 2U;
        valid = parse_exact_unsigned(kEntryOffset);
      } else if (key == "image_layout") {
        bit = 1U << 3U;
        valid = parse_image_layout();
      } else if (key == "image_path") {
        bit = 1U << 4U;
        valid = parse_exact_string(kImageName);
      } else if (key == "image_sha256") {
        bit = 1U << 5U;
        valid = parse_exact_string(kImageSha256);
      } else if (key == "image_size") {
        bit = 1U << 6U;
        valid = parse_exact_unsigned(kImageSize);
      } else if (key == "kernarg_schema") {
        bit = 1U << 7U;
        valid = parse_schema();
      } else if (key == "name") {
        bit = 1U << 8U;
        valid = parse_exact_string("llama_embed_row_f16");
      } else if (key == "rsrc1") {
        bit = 1U << 9U;
        valid = parse_exact_unsigned(kRsrc1);
      } else if (key == "rsrc2") {
        bit = 1U << 10U;
        valid = parse_exact_unsigned(kRsrc2);
      } else if (key == "rsrc3") {
        bit = 1U << 11U;
        valid = parse_exact_unsigned(kRsrc3);
      } else if (key == "source_path") {
        bit = 1U << 12U;
        valid = parse_exact_string(kSourcePath);
      } else if (key == "source_sha256") {
        bit = 1U << 13U;
        valid = parse_exact_string(kSourceSha256);
      } else if (key == "target") {
        bit = 1U << 14U;
        valid = parse_exact_string("gfx1201");
      } else {
        return false;
      }
      if (!valid || (seen & bit) != 0U) return false;
      seen |= bit;
    }
    return consume('}') && seen == ((1U << 15U) - 1U);
  }

  bool parse_elf_admission() {
    if (!consume('{')) return false;
    unsigned int seen = 0;
    bool first = true;
    while (separator_or_end(&first, '}')) {
      std::string key;
      if (!parse_string(&key) || !consume(':')) return false;
      unsigned int bit = 0;
      bool valid = false;
      if (key == "admitted_allocated_sections") {
        bit = 1U << 0U;
        valid = parse_admitted_sections();
      } else if (key == "relocation_count") {
        bit = 1U << 1U;
        valid = parse_exact_unsigned(0U);
      } else if (key == "section_count") {
        bit = 1U << 2U;
        valid = parse_exact_unsigned(10U);
      } else if (key == "symbol_record_count") {
        bit = 1U << 3U;
        valid = parse_exact_unsigned(2U);
      } else if (key == "symbol_target_count") {
        bit = 1U << 4U;
        valid = parse_exact_unsigned(1U);
      } else {
        return false;
      }
      if (!valid || (seen & bit) != 0U) return false;
      seen |= bit;
    }
    return consume('}') && seen == ((1U << 5U) - 1U);
  }

  bool parse_admitted_sections() {
    static constexpr std::array<const char*, 10> kSections{{
        ".note", ".dynsym", ".gnu.hash", ".hash", ".dynstr", ".rodata", ".text",
        ".dynamic", ".relro_padding", ".bss"}};
    if (!consume('[')) return false;
    for (std::size_t index = 0; index < kSections.size(); ++index) {
      if (index != 0U && !consume(',')) return false;
      if (!parse_exact_string(kSections[index])) return false;
    }
    return consume(']');
  }

  bool parse_image_layout() {
    if (!consume('[')) return false;
    for (std::size_t index = 0; index < kLayout.size(); ++index) {
      if (index != 0U && !consume(',')) return false;
      if (!parse_layout_entry(kLayout[index])) return false;
    }
    return consume(']');
  }

  bool parse_layout_entry(const LayoutEntry& expected) {
    if (!consume('{')) return false;
    unsigned int seen = 0;
    bool first = true;
    while (separator_or_end(&first, '}')) {
      std::string key;
      if (!parse_string(&key) || !consume(':')) return false;
      unsigned int bit = 0;
      bool valid = false;
      if (key == "address") {
        bit = 1U << 0U;
        valid = parse_exact_unsigned(expected.address);
      } else if (key == "image_offset") {
        bit = 1U << 1U;
        valid = parse_exact_unsigned(expected.image_offset);
      } else if (key == "name") {
        bit = 1U << 2U;
        valid = parse_exact_string(expected.name);
      } else if (key == "size") {
        bit = 1U << 3U;
        valid = parse_exact_unsigned(expected.size);
      } else {
        return false;
      }
      if (!valid || (seen & bit) != 0U) return false;
      seen |= bit;
    }
    return consume('}') && seen == ((1U << 4U) - 1U);
  }

  bool parse_schema() {
    if (!consume('{')) return false;
    unsigned int seen = 0;
    bool first = true;
    while (separator_or_end(&first, '}')) {
      std::string key;
      if (!parse_string(&key) || !consume(':')) return false;
      unsigned int bit = 0;
      bool valid = false;
      if (key == "bytes") {
        bit = 1U << 0U;
        valid = parse_exact_unsigned(24U);
      } else if (key == "fields") {
        bit = 1U << 1U;
        valid = parse_schema_fields();
      } else if (key == "name") {
        bit = 1U << 2U;
        valid = parse_exact_string("llama-embed-row-f16-v1");
      } else {
        return false;
      }
      if (!valid || (seen & bit) != 0U) return false;
      seen |= bit;
    }
    return consume('}') && seen == ((1U << 3U) - 1U);
  }

  bool parse_schema_fields() {
    static constexpr std::array<const char*, 3> kNames{{"embedding_rows", "hidden_output",
                                                          "selected_row"}};
    static constexpr std::array<std::uint64_t, 3> kOffsets{{0U, 8U, 16U}};
    if (!consume('[')) return false;
    for (std::size_t index = 0; index < kNames.size(); ++index) {
      if (index != 0U && !consume(',')) return false;
      if (!parse_schema_field(kNames[index], kOffsets[index])) return false;
    }
    return consume(']');
  }

  bool parse_schema_field(const char* expected_name, std::uint64_t expected_offset) {
    if (!consume('{')) return false;
    unsigned int seen = 0;
    bool first = true;
    while (separator_or_end(&first, '}')) {
      std::string key;
      if (!parse_string(&key) || !consume(':')) return false;
      unsigned int bit = 0;
      bool valid = false;
      if (key == "name") {
        bit = 1U << 0U;
        valid = parse_exact_string(expected_name);
      } else if (key == "offset") {
        bit = 1U << 1U;
        valid = parse_exact_unsigned(expected_offset);
      } else if (key == "type") {
        bit = 1U << 2U;
        valid = parse_exact_string("uint64");
      } else {
        return false;
      }
      if (!valid || (seen & bit) != 0U) return false;
      seen |= bit;
    }
    return consume('}') && seen == ((1U << 3U) - 1U);
  }

  const char* current_;
  const char* end_;
};

bool load_asset(const std::filesystem::path& root, HsaCodeImageAsset* candidate) {
  char canonical_root[PATH_MAX];
  if (root.empty() || realpath(root.c_str(), canonical_root) == nullptr) return false;

  const int directory_fd = open(canonical_root, O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  if (directory_fd < 0) return false;
  struct stat directory_status {};
  if (fstat(directory_fd, &directory_status) != 0 || !S_ISDIR(directory_status.st_mode)) {
    close(directory_fd);
    return false;
  }

  std::vector<std::uint8_t> manifest;
  const bool manifest_read =
      read_regular_child(directory_fd, kManifestName, kMaximumManifestBytes, &manifest);
  if (!manifest_read || manifest.empty()) {
    close(directory_fd);
    return false;
  }
  ManifestReader reader(manifest);
  if (!reader.validate()) {
    close(directory_fd);
    return false;
  }

  std::vector<std::uint8_t> image;
  const bool image_read = read_regular_child(directory_fd, kImageName, kMaximumImageBytes, &image);
  if (close(directory_fd) != 0 || !image_read || image.size() != kImageSize ||
      kDescriptorOffset >= image.size() || kEntryOffset == 0U || kEntryOffset >= image.size() ||
      hex_digest(image) != kImageSha256) {
    return false;
  }
  // ENABLE_WAVEFRONT_SIZE32 from the AMDHSA kernel descriptor (tinygrad
  // ops_amd.py:591 derives wave32 the same way).
  candidate->wave32 = image_is_wave32(image, kDescriptorOffset);

  candidate->image = std::move(image);
  candidate->image_sha256 = kImageSha256;
  candidate->descriptor_offset = kDescriptorOffset;
  candidate->entry_offset = kEntryOffset;
  candidate->rsrc1 = kRsrc1;
  candidate->rsrc2 = kRsrc2;
  candidate->rsrc3 = kRsrc3;
  candidate->schema = kSchema;
  candidate->source_path = kSourcePath;
  candidate->source_sha256 = kSourceSha256;
  return true;
}

}  // namespace

std::string sha256_hex(const std::vector<std::uint8_t>& bytes) {
  return hex_digest(bytes);
}

bool load_llama_embed_hsa_image(const std::filesystem::path& root, HsaCodeImageAsset* out,
                                std::string* error) {
  if (out == nullptr) {
    set_error(error, "HSA code-image output is required");
    return false;
  }
  try {
    HsaCodeImageAsset candidate;
    if (!load_asset(root, &candidate)) {
      set_error(error, "rejected llama_embed_row_f16 HSA code image");
      return false;
    }
    *out = std::move(candidate);
    if (error != nullptr) error->clear();
    return true;
  } catch (const std::exception&) {
    set_error(error, "failed to load llama_embed_row_f16 HSA code image");
    return false;
  }
}

}  // namespace native_r9700

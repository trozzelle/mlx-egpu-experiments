#include "kernel_catalog.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace native_r9700 {
namespace {

// No Llama asset is catalogued until its code, resource registers, geometry,
// kernarg layout, and digest have been reviewed together. The C0 add-one blob
// remains probe provenance, not a product stage asset.
const std::array<KernelDescriptor, 0> kCatalog = {};

constexpr std::array<uint32_t, 64> kSha256RoundConstants = {
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

uint32_t rotate_right(uint32_t value, uint32_t amount) {
  return (value >> amount) | (value << (32U - amount));
}

void sha256_compress(std::array<uint32_t, 8>* state, const uint8_t* block) {
  std::array<uint32_t, 64> words = {};
  for (uint32_t index = 0; index < 16; ++index) {
    const uint32_t byte_index = index * 4;
    words[index] = (static_cast<uint32_t>(block[byte_index]) << 24U) |
                   (static_cast<uint32_t>(block[byte_index + 1]) << 16U) |
                   (static_cast<uint32_t>(block[byte_index + 2]) << 8U) |
                   static_cast<uint32_t>(block[byte_index + 3]);
  }
  for (uint32_t index = 16; index < words.size(); ++index) {
    const uint32_t sigma0 =
        rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^
        (words[index - 15] >> 3U);
    const uint32_t sigma1 =
        rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^
        (words[index - 2] >> 10U);
    words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
  }

  uint32_t a = (*state)[0];
  uint32_t b = (*state)[1];
  uint32_t c = (*state)[2];
  uint32_t d = (*state)[3];
  uint32_t e = (*state)[4];
  uint32_t f = (*state)[5];
  uint32_t g = (*state)[6];
  uint32_t h = (*state)[7];
  for (uint32_t index = 0; index < words.size(); ++index) {
    const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    const uint32_t choice = (e & f) ^ (~e & g);
    const uint32_t temporary1 = h + sum1 + choice + kSha256RoundConstants[index] + words[index];
    const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  (*state)[0] += a;
  (*state)[1] += b;
  (*state)[2] += c;
  (*state)[3] += d;
  (*state)[4] += e;
  (*state)[5] += f;
  (*state)[6] += g;
  (*state)[7] += h;
}

bool sha256_matches(const std::vector<uint8_t>& code, std::string_view digest) {
  std::array<uint32_t, 8> state = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::size_t offset = 0;
  while (code.size() - offset >= 64) {
    sha256_compress(&state, code.data() + offset);
    offset += 64;
  }

  std::array<uint8_t, 64> final_block = {};
  const std::size_t remaining = code.size() - offset;
  for (std::size_t index = 0; index < remaining; ++index) {
    final_block[index] = code[offset + index];
  }
  final_block[remaining] = 0x80U;
  if (remaining >= 56) {
    sha256_compress(&state, final_block.data());
    final_block = {};
  }

  const uint64_t bit_length = static_cast<uint64_t>(code.size()) * 8U;
  for (uint32_t index = 0; index < 8; ++index) {
    final_block[63U - index] = static_cast<uint8_t>(bit_length >> (index * 8U));
  }
  sha256_compress(&state, final_block.data());

  constexpr char kHexDigits[] = "0123456789abcdef";
  for (uint32_t word = 0; word < state.size(); ++word) {
    for (uint32_t nibble = 0; nibble < 8; ++nibble) {
      const uint32_t shift = 28U - nibble * 4U;
      if (digest[word * 8U + nibble] != kHexDigits[(state[word] >> shift) & 0xfU]) {
        return false;
      }
    }
  }
  return true;
}


bool fail(std::string* error_text, const std::string& message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

bool is_lowercase_sha256(std::string_view digest) {
  if (digest.size() != 64) return false;
  for (const char byte : digest) {
    if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) return false;
  }
  return true;
}

}  // namespace

bool validate_kernel_descriptors(const std::vector<KernelDescriptor>& descriptors,
                                 std::string* error_text) {
  if (descriptors.empty()) return fail(error_text, "at least one kernel descriptor is required");
  for (std::size_t index = 0; index < descriptors.size(); ++index) {
    const KernelDescriptor& descriptor = descriptors[index];
    if (descriptor.name.empty()) return fail(error_text, "kernel name is required");
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (descriptors[prior].name == descriptor.name) {
        return fail(error_text, "kernel names must be unique: " + descriptor.name);
      }
    }
    if (!is_lowercase_sha256(descriptor.sha256)) {
      return fail(error_text, "kernel digest must be a 64-character lowercase SHA-256");
    }
    if (descriptor.code.empty()) return fail(error_text, "kernel code must be nonempty");
    if (!sha256_matches(descriptor.code, descriptor.sha256)) {
      return fail(error_text, "kernel digest does not match kernel code");
    }
    if (descriptor.rsrc1 == 0 || descriptor.rsrc2 == 0 || descriptor.rsrc3 == 0) {
      return fail(error_text, "kernel resource registers must be nonzero");
    }
    if (descriptor.workgroup_x == 0 || descriptor.workgroup_y == 0 ||
        descriptor.workgroup_z == 0 || descriptor.global_x == 0 || descriptor.global_y == 0 ||
        descriptor.global_z == 0) {
      return fail(error_text, "kernel workgroup and global dimensions must be nonzero");
    }
    if (descriptor.kernarg_bytes == 0) {
      return fail(error_text, "kernel kernarg bytes must be nonzero");
    }
  }
  return true;
}

const KernelDescriptor* find_kernel(std::string_view name) {
  for (const KernelDescriptor& descriptor : kCatalog) {
    if (descriptor.name == name) return &descriptor;
  }
  return nullptr;
}

}  // namespace native_r9700

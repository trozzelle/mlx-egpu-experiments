#include "prefill_npz.h"

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace native_r9700 {
namespace {

constexpr uint32_t kNumLayers = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kCacheCapacityTokens = 128;
constexpr uint32_t kHeadDim = 64;
constexpr char kProducerKind[] = "r9700_native";

bool fail(std::string* error_text, const std::string& text) {
  if (error_text != nullptr) *error_text = text;
  return false;
}

void append_u16_le(std::string* out, uint16_t value) {
  out->push_back(static_cast<char>(value & 0xFFu));
  out->push_back(static_cast<char>(value >> 8));
}

void append_u32_le(std::string* out, uint32_t value) {
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<char>((value >> shift) & 0xFFu));
  }
}

uint32_t crc32_ieee(const uint8_t* data, size_t size) {
  static uint32_t table[256];
  static bool initialized = false;
  if (!initialized) {
    for (uint32_t index = 0; index < 256; ++index) {
      uint32_t value = index;
      for (uint32_t bit = 0; bit < 8; ++bit) {
        value = (value & 1u) ? (0xEDB88320u ^ (value >> 1)) : (value >> 1);
      }
      table[index] = value;
    }
    initialized = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t index = 0; index < size; ++index) {
    crc = table[(crc ^ data[index]) & 0xFFu] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// NumPy v1.0 header: magic, version, uint16 header length, then a Python
// dict literal padded with spaces so the total preamble plus header is a
// multiple of 64 bytes and terminated by a newline.
std::string npy_wrap(const std::string& descr, const std::string& shape,
                     const std::vector<uint8_t>& data) {
  const std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                           shape + ", }";
  const size_t preamble = 10;  // 6 magic + 2 version + 2 header length
  const size_t padding = (64 - ((preamble + dict.size() + 1) % 64)) % 64;
  const std::string header = dict + std::string(padding, ' ') + '\n';
  std::string out;
  out.reserve(preamble + header.size() + data.size());
  out += "\x93NUMPY";
  out.push_back(static_cast<char>(1));
  out.push_back(static_cast<char>(0));
  append_u16_le(&out, static_cast<uint16_t>(header.size()));
  out += header;
  out.append(reinterpret_cast<const char*>(data.data()), data.size());
  return out;
}

std::vector<uint8_t> scalar_i64_bytes(int64_t value) {
  std::vector<uint8_t> data(8);
  for (uint32_t shift = 0; shift < 64; shift += 8) {
    data[shift / 8] = static_cast<uint8_t>((static_cast<uint64_t>(value) >> shift) & 0xFFu);
  }
  return data;
}

bool decode_utf8(const std::string& text, std::vector<uint32_t>* codepoints) {
  codepoints->clear();
  for (size_t index = 0; index < text.size();) {
    const uint8_t lead = static_cast<uint8_t>(text[index]);
    uint32_t codepoint = 0;
    size_t length = 0;
    if (lead < 0x80u) {
      codepoint = lead;
      length = 1;
    } else if ((lead & 0xE0u) == 0xC0u) {
      codepoint = lead & 0x1Fu;
      length = 2;
    } else if ((lead & 0xF0u) == 0xE0u) {
      codepoint = lead & 0x0Fu;
      length = 3;
    } else if ((lead & 0xF8u) == 0xF0u) {
      codepoint = lead & 0x07u;
      length = 4;
    } else {
      return false;
    }
    if (index + length > text.size()) return false;
    for (size_t offset = 1; offset < length; ++offset) {
      const uint8_t next = static_cast<uint8_t>(text[index + offset]);
      if ((next & 0xC0u) != 0x80u) return false;
      codepoint = (codepoint << 6) | (next & 0x3Fu);
    }
    codepoints->push_back(codepoint);
    index += length;
  }
  return true;
}

bool npy_unicode_scalar(const std::string& value, std::string* entry, std::string* error_text) {
  std::vector<uint32_t> codepoints;
  if (!decode_utf8(value, &codepoints) || codepoints.empty()) {
    return fail(error_text, "NPZ text scalar must be non-empty valid UTF-8");
  }
  std::vector<uint8_t> data(codepoints.size() * 4);
  for (size_t index = 0; index < codepoints.size(); ++index) {
    for (uint32_t shift = 0; shift < 32; shift += 8) {
      data[index * 4 + shift / 8] = static_cast<uint8_t>((codepoints[index] >> shift) & 0xFFu);
    }
  }
  *entry = npy_wrap("<U" + std::to_string(codepoints.size()), "()", data);
  return true;
}

// Slices the accepted (1, 8, n_prefix, 64) fp16 prefix out of one raw
// head-major [kv_head][capacity][head_dim] cache buffer. Pure byte
// permutation: out[h][t][d] = raw[(h * capacity + t) * 64 + d].
std::vector<uint8_t> prefix_slice_fp16(const std::vector<uint8_t>& raw, uint32_t n_prefix,
                                       uint32_t capacity) {
  std::vector<uint8_t> out(static_cast<uint64_t>(kKvHeads) * n_prefix * kHeadDim * 2);
  uint8_t* dst = out.data();
  for (uint32_t head = 0; head < kKvHeads; ++head) {
    const uint8_t* src = raw.data() + static_cast<uint64_t>(head) * capacity * kHeadDim * 2;
    const size_t run = static_cast<size_t>(n_prefix) * kHeadDim * 2;
    std::copy(src, src + run, dst);
    dst += run;
  }
  return out;
}

struct ZipEntry {
  std::string name;
  std::string bytes;
};

bool write_stored_zip(const std::vector<ZipEntry>& entries, const std::string& out_path,
                      std::string* error_text) {
  const std::string temp_path = [&]() {
    const size_t slash = out_path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : out_path.substr(0, slash);
    const std::string name =
        slash == std::string::npos ? out_path : out_path.substr(slash + 1);
    return dir + "/." + name + ".tmp." + std::to_string(static_cast<long long>(getpid())) +
           ".npz";
  }();


  std::string locals;
  std::string central;
  uint64_t offset = 0;
  for (const ZipEntry& entry : entries) {
    const uint32_t crc = crc32_ieee(reinterpret_cast<const uint8_t*>(entry.bytes.data()),
                                    entry.bytes.size());
    const uint32_t size = static_cast<uint32_t>(entry.bytes.size());
    const uint32_t name_size = static_cast<uint32_t>(entry.name.size());
    append_u32_le(&locals, 0x04034b50u);
    append_u16_le(&locals, 20);      // version needed
    append_u16_le(&locals, 0);       // flags
    append_u16_le(&locals, 0);       // stored
    append_u16_le(&locals, 0);       // mod time: fixed 1980-01-01 00:00
    append_u16_le(&locals, 0x0021u); // mod date: 1980-01-01
    append_u32_le(&locals, crc);
    append_u32_le(&locals, size);
    append_u32_le(&locals, size);
    append_u16_le(&locals, static_cast<uint16_t>(name_size));
    append_u16_le(&locals, 0);  // extra length
    locals += entry.name;
    locals += entry.bytes;

    append_u32_le(&central, 0x02014b50u);
    append_u16_le(&central, 20);  // version made by
    append_u16_le(&central, 20);  // version needed
    append_u16_le(&central, 0);
    append_u16_le(&central, 0);
    append_u16_le(&central, 0);
    append_u16_le(&central, 0x0021u);
    append_u32_le(&central, crc);
    append_u32_le(&central, size);
    append_u32_le(&central, size);
    append_u16_le(&central, static_cast<uint16_t>(name_size));
    append_u16_le(&central, 0);  // extra
    append_u16_le(&central, 0);  // comment
    append_u16_le(&central, 0);  // disk number
    append_u16_le(&central, 0);  // internal attrs
    append_u32_le(&central, 0);  // external attrs
    append_u32_le(&central, static_cast<uint32_t>(offset));
    central += entry.name;
    offset += 30 + name_size + size;
  }

  std::string final_bytes = std::move(locals);
  const uint64_t central_offset = final_bytes.size();
  final_bytes += central;
  append_u32_le(&final_bytes, 0x06054b50u);
  append_u16_le(&final_bytes, 0);
  append_u16_le(&final_bytes, 0);
  append_u16_le(&final_bytes, static_cast<uint16_t>(entries.size()));
  append_u16_le(&final_bytes, static_cast<uint16_t>(entries.size()));
  append_u32_le(&final_bytes, static_cast<uint32_t>(central.size()));
  append_u32_le(&final_bytes, static_cast<uint32_t>(central_offset));
  append_u16_le(&final_bytes, 0);

  {
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) return fail(error_text, "failed to open NPZ temp path: " + temp_path);
    out.write(final_bytes.data(), static_cast<std::streamsize>(final_bytes.size()));
    out.flush();
    if (!out) {
      out.close();
      std::remove(temp_path.c_str());
      return fail(error_text, "failed to write NPZ temp path: " + temp_path);
    }
  }
  if (std::rename(temp_path.c_str(), out_path.c_str()) != 0) {
    std::remove(temp_path.c_str());
    return fail(error_text, "failed to rename NPZ temp path over: " + out_path);
  }
  return true;
}

}  // namespace

bool validate_native_prefill_kv_finite(
    const NativePrefillNpzPayload& payload, std::string* error_text) {
  if (payload.n_prefix == 0 ||
      payload.cache_capacity_tokens != kCacheCapacityTokens ||
      payload.n_prefix > payload.cache_capacity_tokens) {
    return fail(error_text,
                "NPZ payload requires cache capacity 128 and a positive live prefix");
  }
  if (payload.kv_readback_bytes.size() != 2U * kNumLayers) {
    return fail(error_text, "NPZ payload requires exactly 32 K/V readback buffers");
  }
  const uint64_t expected_cache_bytes =
      static_cast<uint64_t>(payload.cache_capacity_tokens) * kKvHeads *
      kHeadDim * sizeof(uint16_t);
  for (std::size_t buffer_index = 0;
       buffer_index < payload.kv_readback_bytes.size(); ++buffer_index) {
    const std::vector<uint8_t>& buffer =
        payload.kv_readback_bytes[buffer_index];
    if (buffer.size() != expected_cache_bytes) {
      return fail(error_text,
                  "NPZ payload K/V readback buffer byte count mismatch");
    }
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      const uint64_t head_base =
          static_cast<uint64_t>(head) * payload.cache_capacity_tokens *
          kHeadDim * sizeof(uint16_t);
      const uint64_t live_value_count =
          static_cast<uint64_t>(payload.n_prefix) * kHeadDim;
      for (uint64_t value_index = 0; value_index < live_value_count;
           ++value_index) {
        const uint64_t byte_offset =
            head_base + value_index * sizeof(uint16_t);
        const uint16_t bits =
            static_cast<uint16_t>(buffer[byte_offset]) |
            static_cast<uint16_t>(
                static_cast<uint16_t>(buffer[byte_offset + 1U]) << 8U);
        if ((bits & 0x7C00U) == 0x7C00U) {
          return fail(error_text,
                      "NPZ payload live K/V prefix contains non-finite fp16");
        }
      }
    }
  }
  if (error_text != nullptr) error_text->clear();
  return true;
}

bool write_native_prefill_npz(const NativePrefillNpzPayload& payload,
                              const std::string& out_path, std::string* error_text) {
  if (payload.model.empty()) return fail(error_text, "NPZ payload model must be non-empty");
  if (payload.n_prefix == 0) return fail(error_text, "NPZ payload n_prefix must be positive");
  if (payload.cache_capacity_tokens == 0 ||
      payload.n_prefix > payload.cache_capacity_tokens) {
    return fail(error_text, "NPZ payload n_prefix must not exceed cache capacity");
  }
  if (payload.kv_readback_bytes.size() != 2 * kNumLayers) {
    return fail(error_text, "NPZ payload requires exactly 32 K/V readback buffers");
  }
  const uint64_t expected_cache_bytes =
      static_cast<uint64_t>(payload.cache_capacity_tokens) * kKvHeads * kHeadDim * 2;
  for (const std::vector<uint8_t>& buffer : payload.kv_readback_bytes) {
    if (buffer.size() != expected_cache_bytes) {
      return fail(error_text, "NPZ payload K/V readback buffer byte count mismatch");
    }
  }
  if (!validate_native_prefill_kv_finite(payload, error_text)) return false;
  if (out_path.empty()) return fail(error_text, "NPZ output path must be non-empty");

  std::vector<ZipEntry> entries;
  entries.reserve(4 + 2 * kNumLayers);
  std::string entry;
  if (!npy_unicode_scalar(payload.model, &entry, error_text)) return false;
  entries.push_back({"model.npy", std::move(entry)});
  entries.push_back({"n_prefix.npy",
                     npy_wrap("<i8", "()", scalar_i64_bytes(payload.n_prefix))});
  entries.push_back({"num_layers.npy",
                     npy_wrap("<i8", "()", scalar_i64_bytes(kNumLayers))});
  if (!npy_unicode_scalar(kProducerKind, &entry, error_text)) return false;
  entries.push_back({"producer_kind.npy", std::move(entry)});

  const std::string shape = "(1, " + std::to_string(kKvHeads) + ", " +
                            std::to_string(payload.n_prefix) + ", " +
                            std::to_string(kHeadDim) + ")";
  for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
    for (uint32_t kv = 0; kv < 2; ++kv) {
      const std::vector<uint8_t>& raw = payload.kv_readback_bytes[2 * layer + kv];
      std::vector<uint8_t> slice =
          prefix_slice_fp16(raw, payload.n_prefix, payload.cache_capacity_tokens);
      entries.push_back({"layer" + std::to_string(layer) + (kv == 0 ? "_K.npy" : "_V.npy"),
                         npy_wrap("<f2", shape, slice)});
    }
  }
  return write_stored_zip(entries, out_path, error_text);
}

}  // namespace native_r9700

#include "amdev_session.h"
#include "hsa_code_image_asset.h"
#include "runtime.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr char kKernelName[] = "wmma_lane_map_gfx1201";
constexpr char kManifestName[] = "wmma_lane_map_gfx1201.json";
constexpr char kImageName[] = "wmma_lane_map_gfx1201.image";
constexpr char kSourcePath[] = "native_r9700/kernels/wmma_lane_map_gfx1201.cpp";
constexpr char kInstruction[] = "v_wmma_f32_16x16x16_f16";
constexpr char kNumericalPolicy[] = "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1";
constexpr char kPackDomain[] = "r9700-wmma-lane-map-diagnostic-pack-v1";
constexpr std::size_t kMatrixSide = 16U;
constexpr std::size_t kMatrixElements = kMatrixSide * kMatrixSide;
constexpr std::size_t kWaveSize = 32U;
constexpr std::size_t kRawWordsPerLane = 16U;
constexpr std::size_t kReadbackByteCount = 2048U;
constexpr std::size_t kObservationCaseCount = 3U;
constexpr std::size_t kBufferCount = 4U;
constexpr std::size_t kMaximumImageBytes = 4U * 1024U * 1024U;
constexpr std::size_t kMaximumManifestBytes = 64U * 1024U;
constexpr std::array<const char*, kRawWordsPerLane> kRawWordNames = {
    "A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3",
    "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
};
constexpr std::array<const char*, kObservationCaseCount> kCaseNames = {
    "a_map", "b_map", "d_map",
};

struct RunnerArguments {
  std::filesystem::path asset_root;
  std::filesystem::path log;
};

struct LaneMapAsset {
  native_r9700::HsaCodeImageAsset image;
  std::filesystem::path manifest_path;
  std::string manifest_sha256;
  std::string source_path;
  std::string source_sha256;
  std::string image_path;
  std::string image_sha256;
  std::string pack_sha256;
};

using RawWords = std::array<std::array<std::uint32_t, kRawWordsPerLane>, kWaveSize>;

struct CaseObservation {
  RawWords raw_words{};
};

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

void print_usage(const char* argv0) {
  const char* program = argv0 == nullptr ? "wmma_lane_map_gfx1201" : argv0;
  std::printf("usage: %s --asset-root <asset-root> --log <path>\n", program);
  std::printf("       %s --help\n", program);
  std::printf("Runs the request-bound gfx1201 WMMA lane-map proof.\n");
  std::printf("  --asset-root <asset-root>  frozen WMMA HSA asset directory\n");
  std::printf("  --log <path>               observed JSON proof log\n");
  std::printf("  --help                     show this message\n");
}

bool parse_arguments(int argc, char** argv, RunnerArguments* arguments) {
  if (arguments == nullptr) return false;
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    print_usage(argv[0]);
    return false;
  }
  bool saw_asset_root = false;
  bool saw_log = false;
  for (int index = 1; index < argc;) {
    const char* option = argv[index];
    if (std::strcmp(option, "--asset-root") == 0 && !saw_asset_root && index + 1 < argc) {
      saw_asset_root = true;
      arguments->asset_root = argv[index + 1];
      index += 2;
    } else if (std::strcmp(option, "--log") == 0 && !saw_log && index + 1 < argc) {
      saw_log = true;
      arguments->log = argv[index + 1];
      index += 2;
    } else {
      return false;
    }
  }
  return saw_asset_root && saw_log && !arguments->asset_root.empty() && !arguments->log.empty();
}

bool read_regular_file(const std::filesystem::path& path, std::size_t maximum_bytes,
                       std::vector<std::uint8_t>* bytes, std::string* error) {
  if (bytes == nullptr) return fail(error, "file output is required");
  bytes->clear();
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return fail(error, "cannot open " + path.string() + ": " + std::strerror(errno));
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uintmax_t>(status.st_size) > maximum_bytes) {
    ::close(fd);
    return fail(error, "file is not a bounded regular file: " + path.string());
  }
  bytes->assign(static_cast<std::size_t>(status.st_size), 0U);
  std::size_t offset = 0U;
  while (offset < bytes->size()) {
    const ssize_t count = ::read(fd, bytes->data() + offset, bytes->size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      ::close(fd);
      return fail(error, "cannot read complete file: " + path.string());
    }
  }
  if (::close(fd) != 0) return fail(error, "cannot close file: " + path.string());
  return true;
}

void skip_json_space(const std::string& text, std::size_t* position) {
  while (*position < text.size()) {
    const char character = text[*position];
    if (character != ' ' && character != '\t' && character != '\r' && character != '\n') break;
    ++*position;
  }
}

bool parse_json_string_at(const std::string& text, std::size_t* position,
                          std::string* value) {
  if (position == nullptr || value == nullptr) return false;
  skip_json_space(text, position);
  if (*position >= text.size() || text[*position] != '"') return false;
  ++*position;
  std::string parsed;
  while (*position < text.size()) {
    const char character = text[*position];
    if (character == '"') {
      ++*position;
      *value = std::move(parsed);
      return true;
    }
    if (character == '\\' || static_cast<unsigned char>(character) < 0x20U) return false;
    parsed.push_back(character);
    ++*position;
  }
  return false;
}

bool locate_json_value(const std::string& text, const char* key, std::size_t* position) {
  if (position == nullptr) return false;
  const std::string marker = std::string("\"") + key + "\"";
  const std::size_t key_position = text.find(marker);
  if (key_position == std::string::npos) return false;
  std::size_t cursor = key_position + marker.size();
  skip_json_space(text, &cursor);
  if (cursor >= text.size() || text[cursor] != ':') return false;
  ++cursor;
  skip_json_space(text, &cursor);
  *position = cursor;
  return true;
}

bool json_string_field(const std::string& text, const char* key, std::string* value) {
  std::size_t position = 0U;
  return locate_json_value(text, key, &position) && parse_json_string_at(text, &position, value);
}

bool json_uint_field(const std::string& text, const char* key, std::uint64_t* value) {
  if (value == nullptr) return false;
  std::size_t position = 0U;
  if (!locate_json_value(text, key, &position) || position >= text.size()) return false;
  if (text[position] == '0') {
    ++position;
    if (position < text.size() && text[position] >= '0' && text[position] <= '9') return false;
    *value = 0U;
    return true;
  }
  if (text[position] < '1' || text[position] > '9') return false;
  std::uint64_t parsed = 0U;
  while (position < text.size() && text[position] >= '0' && text[position] <= '9') {
    const std::uint64_t digit = static_cast<std::uint64_t>(text[position] - '0');
    if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) return false;
    parsed = parsed * 10U + digit;
    ++position;
  }
  *value = parsed;
  return true;
}

bool json_bool_field(const std::string& text, const char* key, bool expected) {
  std::size_t position = 0U;
  if (!locate_json_value(text, key, &position)) return false;
  const char* literal = expected ? "true" : "false";
  const std::size_t length = std::strlen(literal);
  return text.compare(position, length, literal) == 0;
}

bool json_array_strings_field(const std::string& text, const char* key,
                              const std::array<const char*, kRawWordsPerLane>& expected,
                              std::string* error) {
  std::size_t position = 0U;
  if (!locate_json_value(text, key, &position)) return fail(error, "manifest is missing " + std::string(key));
  if (position >= text.size() || text[position] != '[') return fail(error, "manifest " + std::string(key) + " is not an array");
  ++position;
  for (const char* expected_value : expected) {
    std::string actual;
    if (!parse_json_string_at(text, &position, &actual) || actual != expected_value) {
      return fail(error, "manifest " + std::string(key) + " has an unexpected value");
    }
    skip_json_space(text, &position);
    if (expected_value != expected.back()) {
      if (position >= text.size() || text[position] != ',') {
        return fail(error, "manifest " + std::string(key) + " is incomplete");
      }
      ++position;
    }
  }
  skip_json_space(text, &position);
  if (position >= text.size() || text[position] != ']') {
    return fail(error, "manifest " + std::string(key) + " has extra values");
  }
  return true;
}

bool json_case_array_field(const std::string& text, const char* key, std::string* error) {
  std::size_t position = 0U;
  if (!locate_json_value(text, key, &position)) return fail(error, "manifest is missing " + std::string(key));
  if (position >= text.size() || text[position] != '[') return fail(error, "manifest cases is not an array");
  ++position;
  for (std::size_t index = 0U; index < kCaseNames.size(); ++index) {
    std::string actual;
    if (!parse_json_string_at(text, &position, &actual) || actual != kCaseNames[index]) {
      return fail(error, "manifest observation cases are not exactly a_map, b_map, and d_map");
    }
    skip_json_space(text, &position);
    if (index + 1U < kCaseNames.size()) {
      if (position >= text.size() || text[position] != ',') {
        return fail(error, "manifest observation cases are incomplete");
      }
      ++position;
    }
  }
  skip_json_space(text, &position);
  if (position >= text.size() || text[position] != ']') {
    return fail(error, "manifest observation cases have extra values");
  }
  return true;
}

bool json_contains_string_value(const std::string& text, const char* key, const char* expected) {
  const std::string marker = std::string("\"") + key + "\"";
  std::size_t search_position = 0U;
  while (true) {
    const std::size_t key_position = text.find(marker, search_position);
    if (key_position == std::string::npos) return false;
    std::size_t position = key_position + marker.size();
    skip_json_space(text, &position);
    if (position < text.size() && text[position] == ':') {
      ++position;
      std::string actual;
      if (parse_json_string_at(text, &position, &actual) && actual == expected) return true;
    }
    search_position = key_position + marker.size();
  }
}

bool require_manifest_uint(const std::string& text, const char* key, std::uint64_t expected,
                           std::string* error) {
  std::uint64_t actual = 0U;
  if (!json_uint_field(text, key, &actual) || actual != expected) {
    return fail(error, "manifest field " + std::string(key) + " is not the frozen value");
  }
  return true;
}

bool require_manifest_string(const std::string& text, const char* key, const char* expected,
                             std::string* error) {
  std::string actual;
  if (!json_string_field(text, key, &actual) || actual != expected) {
    return fail(error, "manifest field " + std::string(key) + " is not the frozen value");
  }
  return true;
}

std::string json_quote(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (unsigned char character : value) {
    switch (character) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (character < 0x20U) {
          static constexpr char kHex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(kHex[(character >> 4U) & 0x0fU]);
          escaped.push_back(kHex[character & 0x0fU]);
        } else {
          escaped.push_back(static_cast<char>(character));
        }
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string diagnostic_pack_preimage(const LaneMapAsset& asset) {
  std::string preimage =
      "{\"domain\":\"" + std::string(kPackDomain) + "\",\"pack\":{";
  preimage += "\"abi\":{\"bytes\":32,\"fields\":[";
  preimage += "{\"name\":\"a\",\"offset\":0,\"type\":\"uint64\"},";
  preimage += "{\"name\":\"b\",\"offset\":8,\"type\":\"uint64\"},";
  preimage += "{\"name\":\"c\",\"offset\":16,\"type\":\"uint64\"},";
  preimage += "{\"name\":\"observations\",\"offset\":24,\"type\":\"uint64\"}],";
  preimage += "\"name\":\"wmma-lane-map-gfx1201-v1\"},";
  preimage += "\"geometry\":{\"global\":[32,1,1],\"observation_cases\":[\"a_map\",\"b_map\",\"d_map\"],";
  preimage += "\"raw_words_per_lane\":16,\"readback_bytes\":2048,\"wave_size\":32,\"workgroup\":[32,1,1]},";
  preimage += "\"image_path\":" + json_quote(asset.image_path) + ",";
  preimage += "\"image_sha256\":" + json_quote(asset.image_sha256) + ",";
  preimage += "\"instruction\":\"" + std::string(kInstruction) + "\",";
  preimage += "\"manifest_path\":" + json_quote(asset.manifest_path.generic_string()) + ",";
  preimage += "\"manifest_sha256\":" + json_quote(asset.manifest_sha256) + ",";
  preimage += "\"numerical_policy\":\"" + std::string(kNumericalPolicy) + "\",";
  preimage += "\"raw_word_order\":[";
  for (std::size_t index = 0U; index < kRawWordNames.size(); ++index) {
    if (index != 0U) preimage.push_back(',');
    preimage += json_quote(kRawWordNames[index]);
  }
  preimage += "],\"schema_version\":1,";
  preimage += "\"source_path\":" + json_quote(asset.source_path) + ",";
  preimage += "\"source_sha256\":" + json_quote(asset.source_sha256) + ",";
  preimage += "\"target\":\"gfx1201\"}}";
  return preimage;
}

bool load_lane_map_asset(const std::filesystem::path& root, LaneMapAsset* asset,
                         std::string* error) {
  if (asset == nullptr) return fail(error, "lane-map asset output is required");
  std::error_code filesystem_error;
  const std::filesystem::file_status root_status = std::filesystem::symlink_status(root, filesystem_error);
  if (filesystem_error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    return fail(error, "lane-map asset root must be a non-symlink directory");
  }

  const std::filesystem::path manifest_path = root / kManifestName;
  std::vector<std::uint8_t> manifest_bytes;
  if (!read_regular_file(manifest_path, kMaximumManifestBytes, &manifest_bytes, error) ||
      manifest_bytes.empty()) {
    return fail(error, "lane-map manifest is missing or empty");
  }
  const std::string manifest_text(reinterpret_cast<const char*>(manifest_bytes.data()), manifest_bytes.size());
  if (!json_contains_string_value(manifest_text, "name", kKernelName)) {
    return fail(error, "lane-map manifest name is not the reviewed kernel");
  }
  if (!require_manifest_string(manifest_text, "target", "gfx1201", error) ||
      !require_manifest_string(manifest_text, "instruction", kInstruction, error) ||
      !require_manifest_string(manifest_text, "image_path", kImageName, error) ||
      !require_manifest_string(manifest_text, "source_path", kSourcePath, error) ||
      !require_manifest_string(manifest_text, "numerical_policy", kNumericalPolicy, error) ||
      !json_bool_field(manifest_text, "diagnostic_only", true) ||
      !json_bool_field(manifest_text, "model_selectable", false)) {
    return fail(error, "lane-map manifest identity or diagnostic admission is invalid");
  }
  if (!require_manifest_uint(manifest_text, "schema_version", 1U, error) ||
      !require_manifest_uint(manifest_text, "descriptor_offset", 1600U, error) ||
      !require_manifest_uint(manifest_text, "entry_offset", 5888U, error) ||
      !require_manifest_uint(manifest_text, "descriptor_rsrc1", 3222208515U, error) ||
      !require_manifest_uint(manifest_text, "descriptor_rsrc2", 132U, error) ||
      !require_manifest_uint(manifest_text, "descriptor_rsrc3", 112U, error) ||
      !require_manifest_uint(manifest_text, "rsrc1", 3222208515U, error) ||
      !require_manifest_uint(manifest_text, "rsrc2", 132U, error) ||
      !require_manifest_uint(manifest_text, "rsrc3", 112U, error) ||
      !require_manifest_uint(manifest_text, "kernel_code_properties", 1032U, error) ||
      !require_manifest_uint(manifest_text, "image_size", 15473U, error) ||
      !require_manifest_uint(manifest_text, "kernarg_bytes", 32U, error) ||
      !require_manifest_uint(manifest_text, "kernarg_alignment", 8U, error) ||
      !require_manifest_uint(manifest_text, "kernarg_preload_bytes", 0U, error) ||
      !require_manifest_uint(manifest_text, "tail_padding_bytes", 0U, error) ||
      !require_manifest_uint(manifest_text, "group_segment_bytes", 0U, error) ||
      !require_manifest_uint(manifest_text, "private_segment_bytes", 0U, error) ||
      !require_manifest_uint(manifest_text, "wave_size", 32U, error) ||
      !require_manifest_uint(manifest_text, "workgroup_x", 32U, error) ||
      !require_manifest_uint(manifest_text, "workgroup_y", 1U, error) ||
      !require_manifest_uint(manifest_text, "workgroup_z", 1U, error) ||
      !require_manifest_uint(manifest_text, "global_x", 32U, error) ||
      !require_manifest_uint(manifest_text, "global_y", 1U, error) ||
      !require_manifest_uint(manifest_text, "global_z", 1U, error) ||
      !require_manifest_uint(manifest_text, "readback_bytes", kReadbackByteCount, error) ||
      !require_manifest_uint(manifest_text, "raw_words_per_lane", kRawWordsPerLane, error)) {
    return false;
  }
  if (!json_case_array_field(manifest_text, "observation_cases", error)) return false;
  if (!json_array_strings_field(manifest_text, "raw_word_order", kRawWordNames, error)) return false;

  std::string image_sha256;
  std::string source_sha256;
  if (!json_string_field(manifest_text, "image_sha256", &image_sha256) ||
      !json_string_field(manifest_text, "source_sha256", &source_sha256)) {
    return fail(error, "lane-map manifest is missing source or image digest");
  }
  const std::filesystem::path image_path = root / kImageName;
  std::vector<std::uint8_t> image_bytes;
  if (!read_regular_file(image_path, kMaximumImageBytes, &image_bytes, error) || image_bytes.size() != 15473U) {
    return fail(error, "lane-map image is missing or has an invalid size");
  }
  if (native_r9700::sha256_hex(image_bytes) != image_sha256) {
    return fail(error, "lane-map image digest does not match its manifest");
  }
  const std::filesystem::path source_path = std::filesystem::path(kSourcePath);
  std::vector<std::uint8_t> source_bytes;
  if (!read_regular_file(source_path, kMaximumImageBytes, &source_bytes, error) ||
      native_r9700::sha256_hex(source_bytes) != source_sha256) {
    return fail(error, "lane-map source digest does not match its manifest");
  }
  if (!native_r9700::image_is_wave32(image_bytes, 1600U)) {
    return fail(error, "lane-map image descriptor does not enable wave32");
  }

  LaneMapAsset candidate;
  candidate.manifest_path = manifest_path;
  candidate.manifest_sha256 = native_r9700::sha256_hex(manifest_bytes);
  candidate.source_path = kSourcePath;
  candidate.source_sha256 = source_sha256;
  candidate.image_path = kImageName;
  candidate.image_sha256 = image_sha256;
  candidate.image.image = std::move(image_bytes);
  candidate.image.image_sha256 = image_sha256;
  candidate.image.descriptor_offset = 1600U;
  candidate.image.entry_offset = 5888U;
  candidate.image.rsrc1 = 3222208515U;
  candidate.image.rsrc2 = 132U;
  candidate.image.rsrc3 = 112U;
  candidate.image.wave32 = true;
  candidate.image.schema =
      R"({"name":"wmma-lane-map-gfx1201-v1","bytes":32,"fields":[{"name":"a","offset":0,"type":"uint64"},{"name":"b","offset":8,"type":"uint64"},{"name":"c","offset":16,"type":"uint64"},{"name":"observations","offset":24,"type":"uint64"}]})";
  candidate.image.source_path = candidate.source_path;
  candidate.image.source_sha256 = candidate.source_sha256;
  const std::string pack_preimage = diagnostic_pack_preimage(candidate);
  candidate.pack_sha256 = native_r9700::sha256_hex(
      std::vector<std::uint8_t>(pack_preimage.begin(), pack_preimage.end()));
  *asset = std::move(candidate);
  return true;
}

std::uint16_t fp16_bits_from_integer(std::uint32_t value) {
  if (value == 0U) return 0U;
  unsigned int highest_bit = 0U;
  while ((1U << (highest_bit + 1U)) <= value) ++highest_bit;
  const int exponent = static_cast<int>(highest_bit) - 8;
  const std::uint16_t exponent_bits = static_cast<std::uint16_t>((exponent + 15) << 10U);
  const std::uint32_t remainder = value - (1U << highest_bit);
  const std::uint16_t fraction = static_cast<std::uint16_t>(remainder << (10U - highest_bit));
  return static_cast<std::uint16_t>(exponent_bits | fraction);
}

void fill_fp16_tags(std::array<std::uint16_t, kMatrixElements>* tags) {
  for (std::size_t row = 0U; row < kMatrixSide; ++row) {
    for (std::size_t column = 0U; column < kMatrixSide; ++column) {
      const std::uint32_t element = static_cast<std::uint32_t>(row * 16U + column + 1U);
      (*tags)[row * kMatrixSide + column] = fp16_bits_from_integer(element);
    }
  }
}

void fill_fp32_tags(std::array<float, kMatrixElements>* tags) {
  for (std::size_t row = 0U; row < kMatrixSide; ++row) {
    for (std::size_t column = 0U; column < kMatrixSide; ++column) {
      const std::uint32_t element = static_cast<std::uint32_t>(row * 16U + column + 1U);
      (*tags)[row * kMatrixSide + column] = static_cast<float>(element);
    }
  }
}

std::vector<std::uint8_t> fp16_bytes(const std::array<std::uint16_t, kMatrixElements>& values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(std::uint16_t));
  for (std::uint16_t value : values) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
  }
  return bytes;
}

std::vector<std::uint8_t> fp32_bytes(const std::array<float, kMatrixElements>& values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(float));
  for (float value : values) {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "float must be IEEE-754 binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    bytes.push_back(static_cast<std::uint8_t>(bits & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>((bits >> 16U) & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>(bits >> 24U));
  }
  return bytes;
}

std::vector<std::uint8_t> zero_bytes(std::size_t count) {
  return std::vector<std::uint8_t>(count, 0U);
}

native_r9700::ResidentHsaDispatch make_dispatch(
    const native_r9700::HsaCodeImageAsset* image,
    const std::vector<std::uint8_t>& a_bytes,
    const std::vector<std::uint8_t>& b_bytes,
    const std::vector<std::uint8_t>& c_bytes) {
  native_r9700::ResidentHsaDispatch request;
  request.hsa_image = image;
  request.buffers.reserve(kBufferCount);
  native_r9700::ResidentHsaBuffer a;
  a.name = "a";
  a.upload_bytes = a_bytes;
  a.allocation_byte_count = a_bytes.size();
  request.buffers.push_back(std::move(a));
  native_r9700::ResidentHsaBuffer b;
  b.name = "b";
  b.upload_bytes = b_bytes;
  b.allocation_byte_count = b_bytes.size();
  request.buffers.push_back(std::move(b));
  native_r9700::ResidentHsaBuffer c;
  c.name = "c";
  c.upload_bytes = c_bytes;
  c.allocation_byte_count = c_bytes.size();
  request.buffers.push_back(std::move(c));
  native_r9700::ResidentHsaBuffer observations;
  observations.name = "observations";
  observations.upload_bytes = zero_bytes(kReadbackByteCount);
  observations.allocation_byte_count = kReadbackByteCount;
  observations.readback_byte_count = kReadbackByteCount;
  request.buffers.push_back(std::move(observations));
  request.kernargs.assign(32U, 0U);
  request.kernarg_bindings = {
      {0U, 0U}, {1U, 8U}, {2U, 16U}, {3U, 24U},
  };
  request.workgroup_x = 32U;
  request.workgroup_y = 1U;
  request.workgroup_z = 1U;
  request.global_x = 32U;
  request.global_y = 1U;
  request.global_z = 1U;
  return request;
}

bool decode_observation(const native_r9700::ResidentHsaDispatchResult& result,
                        CaseObservation* observation, std::string* error) {
  if (observation == nullptr) return fail(error, "observation output is required");
  if (result.readback_bytes.size() != kBufferCount ||
      result.readback_bytes[0].size() != 0U || result.readback_bytes[1].size() != 0U ||
      result.readback_bytes[2].size() != 0U ||
      result.readback_bytes[3].size() != kReadbackByteCount) {
    return fail(error, "lane-map readback did not return exactly 2048 observation bytes");
  }
  const std::vector<std::uint8_t>& bytes = result.readback_bytes[3];
  for (std::size_t lane = 0U; lane < kWaveSize; ++lane) {
    for (std::size_t word = 0U; word < kRawWordsPerLane; ++word) {
      const std::size_t offset = (lane * kRawWordsPerLane + word) * sizeof(std::uint32_t);
      (*observation).raw_words[lane][word] =
          static_cast<std::uint32_t>(bytes[offset]) |
          (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
          (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
          (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
    }
  }
  return true;
}

bool dispatch_case(native_r9700::AMDevSession* session, const LaneMapAsset& asset,
                   const std::vector<std::uint8_t>& a_bytes,
                   const std::vector<std::uint8_t>& b_bytes,
                   const std::vector<std::uint8_t>& c_bytes,
                   CaseObservation* observation, std::string* error) {
  if (session == nullptr) return fail(error, "AMDev session is required");
  const native_r9700::ResidentHsaDispatch request =
      make_dispatch(&asset.image, a_bytes, b_bytes, c_bytes);
  native_r9700::ResidentHsaDispatchResult result;
  std::string detail;
  if (!session->dispatch_resident_hsa(request, &result, &detail)) {
    return fail(error, "lane-map dispatch failed: " + detail);
  }
  if (result.hardware_identity.find(native_r9700::kRuntimeSubstrate) == std::string::npos ||
      result.hardware_identity.find("pci_id=1002:7551") == std::string::npos ||
      result.hardware_identity.find("arch=gfx1201") == std::string::npos) {
    return fail(error, "lane-map dispatch did not report the admitted hardware identity");
  }
  return decode_observation(result, observation, error);
}

void append_raw_words_json(const RawWords& raw_words, std::string* output) {
  output->push_back('[');
  for (std::size_t lane = 0U; lane < kWaveSize; ++lane) {
    if (lane != 0U) output->push_back(',');
    output->push_back('[');
    for (std::size_t word = 0U; word < kRawWordsPerLane; ++word) {
      if (word != 0U) output->push_back(',');
      output->append(std::to_string(raw_words[lane][word]));
    }
    output->push_back(']');
  }
  output->push_back(']');
}

std::string observed_json(const LaneMapAsset& asset, const std::array<CaseObservation, kObservationCaseCount>& observations) {
  const std::string request_id = "f2-wmma-lane-map-gfx1201-" + asset.image_sha256.substr(0U, 16U);
  std::string output = "{\"schema_version\":1,\"request_id\":" + json_quote(request_id);
  output += ",\"runtime_substrate\":\"" + std::string(native_r9700::kRuntimeSubstrate) + "\"";
  output += ",\"pci_id\":\"1002:7551\",\"arch\":\"gfx1201\",\"wave_size\":32";
  output += ",\"instruction\":\"" + std::string(kInstruction) + "\"";
  output += ",\"source_path\":" + json_quote(asset.source_path);
  output += ",\"source_sha256\":" + json_quote(asset.source_sha256);
  output += ",\"image_path\":" + json_quote(asset.image_path);
  output += ",\"image_sha256\":" + json_quote(asset.image_sha256);
  output += ",\"manifest_path\":" + json_quote(asset.manifest_path.generic_string());
  output += ",\"manifest_sha256\":" + json_quote(asset.manifest_sha256);
  output += ",\"pack_sha256\":" + json_quote(asset.pack_sha256);
  output += ",\"readback_byte_count\":2048,\"raw_words_per_lane\":16";
  output += ",\"cases\":{";
  for (std::size_t index = 0U; index < kCaseNames.size(); ++index) {
    if (index != 0U) output.push_back(',');
    output += json_quote(kCaseNames[index]);
    output += ":{\"raw_words\":";
    append_raw_words_json(observations[index].raw_words, &output);
    output.push_back('}');
  }
  output += "},\"observation_cases\":[\"a_map\",\"b_map\",\"d_map\"]}\n";
  return output;
}

bool write_observed_log(const std::filesystem::path& path, const std::string& contents,
                        std::string* error) {
  if (path.empty()) return fail(error, "lane-map log path is required");
  std::error_code filesystem_error;
  const std::filesystem::file_status existing = std::filesystem::symlink_status(path, filesystem_error);
  if (!filesystem_error && std::filesystem::is_symlink(existing)) {
    return fail(error, "lane-map log path must not be a symlink");
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) return fail(error, "cannot create lane-map log directory: " + filesystem_error.message());
  }
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::filesystem::remove(temporary, filesystem_error);
  filesystem_error.clear();
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) return fail(error, "cannot open temporary lane-map log");
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.flush();
    if (!output) {
      output.close();
      std::filesystem::remove(temporary, filesystem_error);
      return fail(error, "cannot write temporary lane-map log");
    }
  }
  std::filesystem::rename(temporary, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(temporary, filesystem_error);
    return fail(error, "cannot publish lane-map log: " + filesystem_error.message());
  }
  return true;
}

int run_lane_map(const RunnerArguments& arguments, std::string* error) {
  LaneMapAsset asset;
  if (!load_lane_map_asset(arguments.asset_root, &asset, error)) return 1;

  std::array<std::uint16_t, kMatrixElements> fp16_tags{};
  std::array<float, kMatrixElements> fp32_tags{};
  fill_fp16_tags(&fp16_tags);
  fill_fp32_tags(&fp32_tags);
  const std::vector<std::uint8_t> tagged_fp16 = fp16_bytes(fp16_tags);
  const std::vector<std::uint8_t> tagged_fp32 = fp32_bytes(fp32_tags);
  const std::vector<std::uint8_t> zero_fp16 = zero_bytes(tagged_fp16.size());
  const std::vector<std::uint8_t> zero_fp32 = zero_bytes(tagged_fp32.size());

  native_r9700::AMDevSession session;
  std::array<CaseObservation, kObservationCaseCount> observations{};
  if (!dispatch_case(&session, asset, tagged_fp16, zero_fp16, zero_fp32,
                     &observations[0], error) ||
      !dispatch_case(&session, asset, zero_fp16, tagged_fp16, zero_fp32,
                     &observations[1], error) ||
      !dispatch_case(&session, asset, zero_fp16, zero_fp16, tagged_fp32,
                     &observations[2], error)) {
    return 1;
  }
  if (!write_observed_log(arguments.log, observed_json(asset, observations), error)) return 1;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  RunnerArguments arguments;
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    print_usage(argv[0]);
    return 0;
  }
  if (!parse_arguments(argc, argv, &arguments)) {
    print_usage(argv[0]);
    return 2;
  }
  std::string error;
  const int status = run_lane_map(arguments, &error);
  if (status != 0) {
    std::fprintf(stderr, "wmma lane-map runner failed: %s\n", error.c_str());
  }
  return status;
}

#include "model_weight_binder.h"
#include "llama_stage_layout.h"


#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace native_r9700 {
namespace {

bool fail(std::string* error_text, const std::string& message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

constexpr uint64_t kMaxSafetensorsMetadataBytes = 16ULL * 1024 * 1024;

bool checked_multiply(uint64_t left, uint64_t right, uint64_t* product) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) return false;
  *product = left * right;
  return true;
}

class JsonReader {
 public:
  explicit JsonReader(const std::string& text) : text_(text) {}

  bool parse_string(std::string* value, std::string* error) {
    skip_whitespace();
    if (!consume_raw('"')) return set_error(error, "expected JSON string");
    value->clear();
    while (position_ < text_.size()) {
      const char character = text_[position_++];
      if (character == '"') return true;
      if (static_cast<unsigned char>(character) < 0x20) {
        return set_error(error, "control character in JSON string");
      }
      if (character != '\\') {
        value->push_back(character);
        continue;
      }
      if (position_ == text_.size()) return set_error(error, "unterminated JSON escape");
      switch (text_[position_++]) {
        case '"': value->push_back('"'); break;
        case '\\': value->push_back('\\'); break;
        case '/': value->push_back('/'); break;
        case 'b': value->push_back('\b'); break;
        case 'f': value->push_back('\f'); break;
        case 'n': value->push_back('\n'); break;
        case 'r': value->push_back('\r'); break;
        case 't': value->push_back('\t'); break;
        case 'u':
          if (!parse_unicode_escape(value, error)) return false;
          break;
        default: return set_error(error, "invalid JSON escape");
      }
    }
    return set_error(error, "unterminated JSON string");
  }

  bool parse_uint64(uint64_t* value, std::string* error) {
    skip_whitespace();
    if (position_ == text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_]))) {
      return set_error(error, "expected unsigned JSON integer");
    }
    uint64_t result = 0;
    do {
      const uint64_t digit = static_cast<uint64_t>(text_[position_] - '0');
      if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
        return set_error(error, "JSON integer overflows uint64");
      }
      result = result * 10 + digit;
      ++position_;
    } while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])));
    if (position_ < text_.size() && (text_[position_] == '.' || text_[position_] == 'e' || text_[position_] == 'E')) {
      return set_error(error, "expected integer JSON number");
    }
    *value = result;
    return true;
  }

  bool parse_uint64_array(std::vector<uint64_t>* values, std::string* error) {
    if (!expect('[', error)) return false;
    values->clear();
    if (consume_if(']')) return true;
    while (true) {
      uint64_t value = 0;
      if (!parse_uint64(&value, error)) return false;
      values->push_back(value);
      if (consume_if(']')) return true;
      if (!expect(',', error)) return false;
    }
  }

  bool skip_value(std::string* error) {
    skip_whitespace();
    if (position_ == text_.size()) return set_error(error, "missing JSON value");
    const char character = text_[position_];
    if (character == '"') {
      std::string ignored;
      return parse_string(&ignored, error);
    }
    if (character == '{') return skip_object(error);
    if (character == '[') return skip_array(error);
    if (std::isdigit(static_cast<unsigned char>(character)) || character == '-') return skip_number(error);
    if (skip_literal("true") || skip_literal("false") || skip_literal("null")) return true;
    return set_error(error, "invalid JSON value");
  }

  bool expect(char expected, std::string* error) {
    if (!consume_if(expected)) {
      std::ostringstream stream;
      stream << "expected '" << expected << "'";
      return set_error(error, stream.str());
    }
    return true;
  }

  bool consume_if(char expected) {
    skip_whitespace();
    return consume_raw(expected);
  }

  bool finish(std::string* error) {
    skip_whitespace();
    return position_ == text_.size() || set_error(error, "trailing JSON data");
  }

  bool parse_unicode_escape(std::string* value, std::string* error) {
    uint32_t codepoint = 0;
    if (!read_hex_quad(&codepoint, error)) return false;
    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
      if (position_ + 2 > text_.size() || text_[position_] != '\\' || text_[position_ + 1] != 'u') {
        return set_error(error, "unpaired JSON high surrogate");
      }
      position_ += 2;
      uint32_t low_surrogate = 0;
      if (!read_hex_quad(&low_surrogate, error) || low_surrogate < 0xDC00 || low_surrogate > 0xDFFF) {
        return set_error(error, "unpaired JSON high surrogate");
      }
      codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low_surrogate - 0xDC00);
    } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
      return set_error(error, "unpaired JSON low surrogate");
    }
    if (codepoint <= 0x7F) {
      value->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
      value->push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
      value->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
      value->push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
      value->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
      value->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
      value->push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
      value->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
      value->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
      value->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return true;
  }

  bool read_hex_quad(uint32_t* value, std::string* error) {
    if (position_ + 4 > text_.size()) return set_error(error, "short JSON unicode escape");
    uint32_t result = 0;
    for (size_t index = 0; index < 4; ++index) {
      const char character = text_[position_++];
      uint32_t nibble = 0;
      if (character >= '0' && character <= '9') {
        nibble = static_cast<uint32_t>(character - '0');
      } else if (character >= 'a' && character <= 'f') {
        nibble = static_cast<uint32_t>(character - 'a' + 10);
      } else if (character >= 'A' && character <= 'F') {
        nibble = static_cast<uint32_t>(character - 'A' + 10);
      } else {
        return set_error(error, "invalid JSON unicode escape");
      }
      result = (result << 4) | nibble;
    }
    *value = result;
    return true;
  }

 private:
  bool consume_raw(char expected) {
    if (position_ < text_.size() && text_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void skip_whitespace() {
    while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
  }

  bool skip_object(std::string* error) {
    if (!expect('{', error)) return false;
    if (consume_if('}')) return true;
    while (true) {
      std::string key;
      if (!parse_string(&key, error) || !expect(':', error) || !skip_value(error)) return false;
      if (consume_if('}')) return true;
      if (!expect(',', error)) return false;
    }
  }

  bool skip_array(std::string* error) {
    if (!expect('[', error)) return false;
    if (consume_if(']')) return true;
    while (true) {
      if (!skip_value(error)) return false;
      if (consume_if(']')) return true;
      if (!expect(',', error)) return false;
    }
  }

  bool skip_number(std::string* error) {
    skip_whitespace();
    if (consume_raw('-') && position_ == text_.size()) return set_error(error, "invalid JSON number");
    if (position_ == text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_]))) {
      return set_error(error, "invalid JSON number");
    }
    while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      if (position_ == text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_]))) {
        return set_error(error, "invalid JSON number");
      }
      while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
    }
    if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
      if (position_ == text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_]))) {
        return set_error(error, "invalid JSON number");
      }
      while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
    }
    return true;
  }

  bool skip_literal(const char* literal) {
    const size_t length = std::char_traits<char>::length(literal);
    if (text_.compare(position_, length, literal) != 0) return false;
    position_ += length;
    return true;
  }

  static bool set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  }

  const std::string& text_;
  size_t position_ = 0;
};

struct ParsedTensorRecord {
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
};

bool parse_tensor_record(JsonReader* reader, ParsedTensorRecord* record, std::string* error) {
  if (!reader->expect('{', error)) return false;
  bool have_dtype = false;
  bool have_shape = false;
  bool have_offsets = false;
  if (reader->consume_if('}')) return fail(error, "tensor metadata object is empty");
  while (true) {
    std::string key;
    if (!reader->parse_string(&key, error) || !reader->expect(':', error)) return false;
    if (key == "dtype") {
      if (have_dtype || !reader->parse_string(&record->dtype, error)) return false;
      have_dtype = true;
    } else if (key == "shape") {
      if (have_shape || !reader->parse_uint64_array(&record->shape, error)) return false;
      have_shape = true;
    } else if (key == "data_offsets") {
      std::vector<uint64_t> offsets;
      if (have_offsets || !reader->parse_uint64_array(&offsets, error)) return false;
      if (offsets.size() != 2) return fail(error, "data_offsets must contain exactly two integers");
      record->begin = offsets[0];
      record->end = offsets[1];
      have_offsets = true;
    } else if (!reader->skip_value(error)) {
      return false;
    }
    if (reader->consume_if('}')) break;
    if (!reader->expect(',', error)) return false;
  }
  if (!have_dtype || !have_shape || !have_offsets) {
    return fail(error, "tensor metadata requires dtype, shape, and data_offsets");
  }
  return true;
}

bool parse_safetensors_header(const std::string& text,
                              std::unordered_map<std::string, ParsedTensorRecord>* records,
                              std::string* error) {
  JsonReader reader(text);
  if (!reader.expect('{', error)) return false;
  if (reader.consume_if('}')) return fail(error, "safetensors header has no tensor entries");
  while (true) {
    std::string name;
    if (!reader.parse_string(&name, error) || !reader.expect(':', error)) return false;
    if (name == "__metadata__") {
      if (!reader.skip_value(error)) return false;
    } else {
      ParsedTensorRecord record;
      if (!parse_tensor_record(&reader, &record, error)) return false;
      if (!records->emplace(name, std::move(record)).second) {
        return fail(error, "duplicate tensor entry " + name);
      }
    }
    if (reader.consume_if('}')) break;
    if (!reader.expect(',', error)) return false;
  }
  return reader.finish(error);
}

bool parse_weight_map(JsonReader* reader,
                      std::unordered_map<std::string, std::string>* weight_map,
                      std::string* error) {
  if (!reader->expect('{', error)) return false;
  if (reader->consume_if('}')) return true;
  while (true) {
    std::string tensor_name;
    std::string shard_name;
    if (!reader->parse_string(&tensor_name, error) || !reader->expect(':', error) ||
        !reader->parse_string(&shard_name, error)) {
      return false;
    }
    if (!weight_map->emplace(std::move(tensor_name), std::move(shard_name)).second) {
      return fail(error, "duplicate weight_map tensor name");
    }
    if (reader->consume_if('}')) return true;
    if (!reader->expect(',', error)) return false;
  }
}

bool parse_safetensors_index(const std::string& text,
                             std::unordered_map<std::string, std::string>* weight_map,
                             std::string* error) {
  JsonReader reader(text);
  if (!reader.expect('{', error)) return false;
  bool have_weight_map = false;
  if (reader.consume_if('}')) return fail(error, "safetensors index has no weight_map");
  while (true) {
    std::string key;
    if (!reader.parse_string(&key, error) || !reader.expect(':', error)) return false;
    if (key == "weight_map") {
      if (have_weight_map || !parse_weight_map(&reader, weight_map, error)) return false;
      have_weight_map = true;
    } else if (!reader.skip_value(error)) {
      return false;
    }
    if (reader.consume_if('}')) break;
    if (!reader.expect(',', error)) return false;
  }
  if (!have_weight_map || weight_map->empty()) return fail(error, "safetensors index has an empty weight_map");
  return reader.finish(error);
}

bool read_text_file(const std::filesystem::path& path, std::string* text, std::string* error_text) {
  std::error_code filesystem_error;
  const uintmax_t file_size = std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error || file_size == 0 || file_size > kMaxSafetensorsMetadataBytes) {
    return fail(error_text, "invalid safetensors metadata file " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) return fail(error_text, "cannot open " + path.string());
  text->resize(static_cast<size_t>(file_size));
  input.read(&(*text)[0], static_cast<std::streamsize>(text->size()));
  if (input.gcount() != static_cast<std::streamsize>(text->size())) {
    return fail(error_text, "cannot read " + path.string());
  }
  return true;
}

bool is_beneath_directory(const std::filesystem::path& path,
                          const std::filesystem::path& directory) {
  auto path_component = path.begin();
  for (const auto& directory_component : directory) {
    if (path_component == path.end() || *path_component != directory_component) return false;
    ++path_component;
  }
  return path_component != path.end();
}

std::string format_shape(const std::vector<uint64_t>& shape) {
  std::ostringstream stream;
  stream << '[';
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) stream << ',';
    stream << shape[index];
  }
  stream << ']';
  return stream.str();
}

bool validate_geometry(const LlamaModelGeometry& geometry, uint64_t* kv_hidden,
                       std::string* error_text) {
  if (geometry.vocab_size == 0 || geometry.hidden_size == 0 || geometry.intermediate_size == 0 ||
      geometry.n_kv_heads == 0 || geometry.head_dim == 0) {
    return fail(error_text, "invalid Llama geometry: vocab_size, hidden_size, intermediate_size, n_kv_heads, and head_dim must be nonzero");
  }
  if (!checked_multiply(geometry.n_kv_heads, geometry.head_dim, kv_hidden)) {
    return fail(error_text, "invalid Llama geometry: n_kv_heads * head_dim overflows uint64");
  }
  return true;
}

}  // namespace

bool ModelWeightBinder::open(const std::string& model_dir, std::string* error_text) {
  model_dir_.clear();
  single_shard_.clear();
  indexed_shards_.clear();
  shard_payload_offsets_.clear();
  shard_headers_.clear();

  std::error_code filesystem_error;
  model_dir_ = std::filesystem::path(model_dir);
  if (model_dir.empty() || !std::filesystem::is_directory(model_dir_, filesystem_error) || filesystem_error) {
    return fail(error_text, "model directory not found: " + model_dir);
  }
  model_dir_ = std::filesystem::canonical(model_dir_, filesystem_error);
  if (filesystem_error) return fail(error_text, "cannot resolve model directory: " + model_dir);

  const std::filesystem::path index_path = model_dir_ / "model.safetensors.index.json";
  if (std::filesystem::is_regular_file(index_path, filesystem_error) && !filesystem_error) {
    std::string index_text;
    if (!read_text_file(index_path, &index_text, error_text)) return false;
    std::unordered_map<std::string, std::string> weight_map;
    std::string parse_error;
    if (!parse_safetensors_index(index_text, &weight_map, &parse_error)) {
      return fail(error_text, "malformed safetensors index " + index_path.string() + ": " + parse_error);
    }
    for (const auto& entry : weight_map) {
      const std::filesystem::path shard_name(entry.second);
      if (shard_name.empty() || shard_name.is_absolute() || shard_name.has_parent_path() ||
          shard_name.extension() != ".safetensors") {
        return fail(error_text, "unsafe safetensors shard path for tensor " + entry.first + ": " + entry.second);
      }
      const std::filesystem::path shard_path =
          std::filesystem::canonical(model_dir_ / shard_name, filesystem_error);
      if (filesystem_error || !is_beneath_directory(shard_path, model_dir_)) {
        return fail(error_text, "unsafe safetensors shard path for tensor " + entry.first + ": " + entry.second);
      }
      indexed_shards_.emplace(entry.first, shard_path);
    }
    return true;
  }
  if (filesystem_error) return fail(error_text, "cannot inspect safetensors index: " + index_path.string());

  const std::filesystem::path unresolved_shard_path = model_dir_ / "model.safetensors";
  const std::filesystem::path shard_path =
      std::filesystem::canonical(unresolved_shard_path, filesystem_error);
  if (filesystem_error) {
    return fail(error_text, "no model.safetensors or model.safetensors.index.json found in " + model_dir);
  }
  if (!is_beneath_directory(shard_path, model_dir_)) {
    return fail(error_text, "unsafe safetensors shard path: " + unresolved_shard_path.string());
  }
  if (!std::filesystem::is_regular_file(shard_path, filesystem_error) || filesystem_error) {
    return fail(error_text, "no model.safetensors or model.safetensors.index.json found in " + model_dir);
  }
  single_shard_ = shard_path;
  return true;
}

bool ModelWeightBinder::load_shard_header(const std::filesystem::path& shard_path,
                                          std::string* error_text) {
  if (shard_headers_.find(shard_path) != shard_headers_.end()) return true;

  std::error_code filesystem_error;
  const uintmax_t file_size = std::filesystem::file_size(shard_path, filesystem_error);
  if (filesystem_error || file_size < 8) {
    return fail(error_text, "invalid safetensors file " + shard_path.string() + ": missing 8-byte header length");
  }

  std::ifstream input(shard_path, std::ios::binary);
  if (!input) return fail(error_text, "cannot open safetensors shard " + shard_path.string());
  uint8_t length_bytes[8] = {};
  input.read(reinterpret_cast<char*>(length_bytes), sizeof(length_bytes));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(length_bytes))) {
    return fail(error_text, "invalid safetensors file " + shard_path.string() + ": cannot read header length");
  }
  uint64_t header_length = 0;
  for (uint32_t index = 0; index < 8; ++index) {
    header_length |= static_cast<uint64_t>(length_bytes[index]) << (8 * index);
  }
  if (header_length == 0 || header_length > file_size - 8 ||
      header_length > kMaxSafetensorsMetadataBytes ||
      header_length > std::numeric_limits<size_t>::max() ||
      header_length > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return fail(error_text, "invalid safetensors header length in " + shard_path.string());
  }

  std::string header(static_cast<size_t>(header_length), '\0');
  input.read(&header[0], static_cast<std::streamsize>(header.size()));
  if (input.gcount() != static_cast<std::streamsize>(header.size())) {
    return fail(error_text, "invalid safetensors file " + shard_path.string() + ": truncated header");
  }
  std::unordered_map<std::string, ParsedTensorRecord> parsed_records;
  std::string parse_error;
  if (!parse_safetensors_header(header, &parsed_records, &parse_error)) {
    return fail(error_text, "malformed safetensors header " + shard_path.string() + ": " + parse_error);
  }

  std::unordered_map<std::string, TensorRecord> records;
  records.reserve(parsed_records.size());
  for (auto& entry : parsed_records) {
    TensorRecord record;
    record.dtype = std::move(entry.second.dtype);
    record.shape = std::move(entry.second.shape);
    record.begin = entry.second.begin;
    record.end = entry.second.end;
    records.emplace(std::move(entry.first), std::move(record));
  }
  shard_payload_offsets_.emplace(shard_path, header_length + 8);
  shard_headers_.emplace(shard_path, std::move(records));
  return true;
}

bool ModelWeightBinder::bind_tensor(const char* name,
                                    const std::vector<uint64_t>& expected_shape,
                                    Fp16WeightSpan* span,
                                    std::string* error_text) {
  if (span == nullptr) return fail(error_text, "output byte span is required for tensor " + std::string(name));
  const auto indexed = indexed_shards_.find(name);
  const std::filesystem::path shard_path = indexed_shards_.empty() ? single_shard_ :
                                            (indexed == indexed_shards_.end() ? std::filesystem::path{} : indexed->second);
  if (shard_path.empty()) return fail(error_text, "required Llama tensor missing: " + std::string(name));
  if (!load_shard_header(shard_path, error_text)) return false;

  const auto& records = shard_headers_.at(shard_path);
  const auto record = records.find(name);
  if (record == records.end()) {
    return fail(error_text, "required Llama tensor missing from shard " + shard_path.string() + ": " + name);
  }
  if (record->second.dtype != "F16") {
    return fail(error_text, "unsupported dtype " + record->second.dtype + " for tensor " + name + "; expected F16");
  }
  if (record->second.shape != expected_shape) {
    return fail(error_text, "unexpected shape for tensor " + std::string(name) +
                                "; expected " + format_shape(expected_shape) +
                                ", got " + format_shape(record->second.shape));
  }
  if (record->second.end < record->second.begin) {
    return fail(error_text, "invalid data_offsets for tensor " + std::string(name));
  }

  uint64_t element_count = 1;
  for (uint64_t dimension : record->second.shape) {
    if (dimension == 0 || !checked_multiply(element_count, dimension, &element_count)) {
      return fail(error_text, "invalid fp16 shape for tensor " + std::string(name));
    }
  }
  uint64_t byte_length = 0;
  if (!checked_multiply(element_count, 2, &byte_length) || record->second.end - record->second.begin != byte_length) {
    return fail(error_text, "fp16 byte span does not match shape for tensor " + std::string(name));
  }

  std::error_code filesystem_error;
  const uintmax_t file_size = std::filesystem::file_size(shard_path, filesystem_error);
  const uint64_t payload_offset = shard_payload_offsets_.at(shard_path);
  if (filesystem_error || payload_offset > file_size ||
      record->second.begin > std::numeric_limits<uint64_t>::max() - payload_offset) {
    return fail(error_text, "invalid safetensors offsets for tensor " + std::string(name));
  }
  const uint64_t data_offset = payload_offset + record->second.begin;
  if (record->second.end > file_size - payload_offset) {
    return fail(error_text, "data_offsets exceed payload for tensor " + std::string(name));
  }

  span->name = name;
  span->shard_path = shard_path;
  span->payload_offset = payload_offset;
  span->data_offset = data_offset;
  span->byte_length = byte_length;
  span->shape = record->second.shape;
  return true;
}

bool ModelWeightBinder::bind_llama_layer0(const LlamaModelGeometry& geometry,
                                          LlamaLayer0WeightSpans* weights,
                                          std::string* error_text) {
  if (weights == nullptr) return fail(error_text, "layer-0 weight spans are required");
  if (model_dir_.empty()) return fail(error_text, "model weights must be opened before binding Llama layer 0");

  uint64_t kv_hidden = 0;
  if (!validate_geometry(geometry, &kv_hidden, error_text)) return false;
  const std::vector<uint64_t> hidden = {geometry.hidden_size};
  const std::vector<uint64_t> embedding = {geometry.vocab_size, geometry.hidden_size};
  const std::vector<uint64_t> square = {geometry.hidden_size, geometry.hidden_size};
  const std::vector<uint64_t> kv_projection = {kv_hidden, geometry.hidden_size};
  const std::vector<uint64_t> up_projection = {geometry.intermediate_size, geometry.hidden_size};
  const std::vector<uint64_t> down_projection = {geometry.hidden_size, geometry.intermediate_size};
  LlamaLayer0WeightSpans bound_weights;

  if (!bind_tensor("model.embed_tokens.weight", embedding, &bound_weights.embed_tokens, error_text) ||
      !bind_tensor("model.layers.0.input_layernorm.weight", hidden, &bound_weights.input_layernorm, error_text) ||
      !bind_tensor("model.layers.0.post_attention_layernorm.weight", hidden,
                   &bound_weights.post_attention_layernorm, error_text) ||
      !bind_tensor("model.layers.0.self_attn.q_proj.weight", square, &bound_weights.q_proj, error_text) ||
      !bind_tensor("model.layers.0.self_attn.k_proj.weight", kv_projection, &bound_weights.k_proj, error_text) ||
      !bind_tensor("model.layers.0.self_attn.v_proj.weight", kv_projection, &bound_weights.v_proj, error_text) ||
      !bind_tensor("model.layers.0.self_attn.o_proj.weight", square, &bound_weights.o_proj, error_text) ||
      !bind_tensor("model.layers.0.mlp.gate_proj.weight", up_projection, &bound_weights.gate_proj, error_text) ||
      !bind_tensor("model.layers.0.mlp.up_proj.weight", up_projection, &bound_weights.up_proj, error_text) ||
      !bind_tensor("model.layers.0.mlp.down_proj.weight", down_projection, &bound_weights.down_proj, error_text)) {
    return false;
  }

  const Fp16WeightSpan* const spans[] = {
      &bound_weights.embed_tokens,
      &bound_weights.input_layernorm,
      &bound_weights.post_attention_layernorm,
      &bound_weights.q_proj,
      &bound_weights.k_proj,
      &bound_weights.v_proj,
      &bound_weights.o_proj,
      &bound_weights.gate_proj,
      &bound_weights.up_proj,
      &bound_weights.down_proj,
  };
  constexpr size_t span_count = sizeof(spans) / sizeof(spans[0]);
  for (size_t left = 0; left < span_count; ++left) {
    const Fp16WeightSpan& left_span = *spans[left];
    if (left_span.byte_length > std::numeric_limits<uint64_t>::max() - left_span.data_offset) {
      return fail(error_text, "invalid fp16 byte span for tensor " + left_span.name);
    }
    const uint64_t left_end = left_span.data_offset + left_span.byte_length;
    for (size_t right = left + 1; right < span_count; ++right) {
      const Fp16WeightSpan& right_span = *spans[right];
      if (left_span.shard_path != right_span.shard_path) continue;
      if (right_span.byte_length > std::numeric_limits<uint64_t>::max() - right_span.data_offset) {
        return fail(error_text, "invalid fp16 byte span for tensor " + right_span.name);
      }
      const uint64_t right_end = right_span.data_offset + right_span.byte_length;
      if (left_span.data_offset < right_end && right_span.data_offset < left_end) {
        return fail(error_text, "overlapping required Llama tensor spans in shard " +
                                    left_span.shard_path.string() + ": " + left_span.name +
                                    " and " + right_span.name);
      }
    }
  }

  *weights = std::move(bound_weights);
  return true;
}

bool ModelWeightBinder::bind_llama_stage_layer(
    const LlamaModelGeometry& geometry,
    uint32_t layer_index,
    LlamaLayerWeightSpans* weights,
    std::string* error_text) {
  if (weights == nullptr) {
    return fail(error_text, "Llama stage layer weight spans are required");
  }
  if (geometry.hidden_size != 2048 || geometry.intermediate_size != 8192 ||
      geometry.n_kv_heads != 8 || geometry.head_dim != 64) {
    return fail(error_text,
                "Llama stage geometry must be hidden_size=2048, "
                "intermediate_size=8192, n_kv_heads=8, and head_dim=64");
  }
  if (layer_index >= kLlamaStageLayerCount) {
    return fail(error_text, "Llama stage layer index is outside the 16-layer model");
  }
  if (model_dir_.empty()) {
    return fail(error_text,
                "model weights must be opened before binding a Llama stage layer");
  }

  const uint64_t kv_hidden = geometry.n_kv_heads * geometry.head_dim;
  const std::vector<uint64_t> hidden = {geometry.hidden_size};
  const std::vector<uint64_t> square = {geometry.hidden_size, geometry.hidden_size};
  const std::vector<uint64_t> kv_projection = {kv_hidden, geometry.hidden_size};
  const std::vector<uint64_t> up_projection = {
      geometry.intermediate_size, geometry.hidden_size};
  const std::vector<uint64_t> down_projection = {
      geometry.hidden_size, geometry.intermediate_size};
  const std::string prefix = "model.layers." + std::to_string(layer_index) + ".";
  LlamaLayerWeightSpans bound_weights;
  bound_weights.layer_index = layer_index;

  if (!bind_tensor((prefix + "input_layernorm.weight").c_str(), hidden,
                   &bound_weights.input_layernorm, error_text) ||
      !bind_tensor((prefix + "post_attention_layernorm.weight").c_str(), hidden,
                   &bound_weights.post_attention_layernorm, error_text) ||
      !bind_tensor((prefix + "self_attn.q_proj.weight").c_str(), square,
                   &bound_weights.q_proj, error_text) ||
      !bind_tensor((prefix + "self_attn.k_proj.weight").c_str(), kv_projection,
                   &bound_weights.k_proj, error_text) ||
      !bind_tensor((prefix + "self_attn.v_proj.weight").c_str(), kv_projection,
                   &bound_weights.v_proj, error_text) ||
      !bind_tensor((prefix + "self_attn.o_proj.weight").c_str(), square,
                   &bound_weights.o_proj, error_text) ||
      !bind_tensor((prefix + "mlp.gate_proj.weight").c_str(), up_projection,
                   &bound_weights.gate_proj, error_text) ||
      !bind_tensor((prefix + "mlp.up_proj.weight").c_str(), up_projection,
                   &bound_weights.up_proj, error_text) ||
      !bind_tensor((prefix + "mlp.down_proj.weight").c_str(), down_projection,
                   &bound_weights.down_proj, error_text)) {
    return false;
  }

  const Fp16WeightSpan* const spans[] = {
      &bound_weights.input_layernorm,
      &bound_weights.post_attention_layernorm,
      &bound_weights.q_proj,
      &bound_weights.k_proj,
      &bound_weights.v_proj,
      &bound_weights.o_proj,
      &bound_weights.gate_proj,
      &bound_weights.up_proj,
      &bound_weights.down_proj,
  };
  constexpr size_t span_count = sizeof(spans) / sizeof(spans[0]);
  for (size_t left = 0; left < span_count; ++left) {
    const Fp16WeightSpan& left_span = *spans[left];
    if (left_span.byte_length >
        std::numeric_limits<uint64_t>::max() - left_span.data_offset) {
      return fail(error_text, "invalid fp16 byte span for tensor " + left_span.name);
    }
    const uint64_t left_end = left_span.data_offset + left_span.byte_length;
    for (size_t right = left + 1; right < span_count; ++right) {
      const Fp16WeightSpan& right_span = *spans[right];
      if (left_span.shard_path != right_span.shard_path) continue;
      if (right_span.byte_length >
          std::numeric_limits<uint64_t>::max() - right_span.data_offset) {
        return fail(error_text,
                    "invalid fp16 byte span for tensor " + right_span.name);
      }
      const uint64_t right_end = right_span.data_offset + right_span.byte_length;
      if (left_span.data_offset < right_end && right_span.data_offset < left_end) {
        return fail(error_text, "overlapping required Llama tensor spans in shard " +
                                    left_span.shard_path.string() + ": " +
                                    left_span.name + " and " + right_span.name);
      }
    }
  }

  *weights = std::move(bound_weights);
  return true;
}

}  // namespace native_r9700

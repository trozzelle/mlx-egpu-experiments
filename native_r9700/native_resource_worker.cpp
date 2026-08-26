#include "native_resource_worker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>

namespace native_r9700 {
namespace {

constexpr std::size_t kMaxErrorStringBytes = 16U * 1024U;

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  std::string string;
  std::string number;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;
};

bool valid_utf8(std::string_view text) {
  for (std::size_t index = 0; index < text.size();) {
    const unsigned char first = static_cast<unsigned char>(text[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    std::size_t width = 0;
    uint32_t codepoint = 0;
    uint32_t minimum = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
      width = 2;
      codepoint = first & 0x1fU;
      minimum = 0x80U;
    } else if (first >= 0xe0U && first <= 0xefU) {
      width = 3;
      codepoint = first & 0x0fU;
      minimum = 0x800U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      width = 4;
      codepoint = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (index + width > text.size()) return false;
    for (std::size_t offset = 1; offset < width; ++offset) {
      const unsigned char continuation =
          static_cast<unsigned char>(text[index + offset]);
      if ((continuation & 0xc0U) != 0x80U) return false;
      codepoint = (codepoint << 6U) | (continuation & 0x3fU);
    }
    if (codepoint < minimum || codepoint > 0x10ffffU ||
        (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
      return false;
    }
    index += width;
  }
  return true;
}

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  bool parse(JsonValue* value) {
    if (value == nullptr || !valid_utf8(input_)) return false;
    skip_whitespace();
    if (!parse_value(value, 0U)) return false;
    skip_whitespace();
    return position_ == input_.size();
  }

 private:
  static bool is_whitespace(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
  }

  static bool is_digit(char value) { return value >= '0' && value <= '9'; }

  void skip_whitespace() {
    while (position_ < input_.size() && is_whitespace(input_[position_])) ++position_;
  }

  bool parse_value(JsonValue* value, std::size_t depth) {
    if (depth > 64U || value == nullptr || position_ >= input_.size()) return false;
    switch (input_[position_]) {
      case 'n':
        if (!consume("null")) return false;
        value->type = JsonValue::Type::Null;
        return true;
      case 't':
        if (!consume("true")) return false;
        value->type = JsonValue::Type::Bool;
        value->boolean = true;
        return true;
      case 'f':
        if (!consume("false")) return false;
        value->type = JsonValue::Type::Bool;
        value->boolean = false;
        return true;
      case '"':
        value->type = JsonValue::Type::String;
        return parse_string(&value->string);
      case '[':
        value->type = JsonValue::Type::Array;
        return parse_array(value, depth + 1U);
      case '{':
        value->type = JsonValue::Type::Object;
        return parse_object(value, depth + 1U);
      default:
        if (input_[position_] == '-' || is_digit(input_[position_])) {
          value->type = JsonValue::Type::Number;
          return parse_number(&value->number);
        }
        return false;
    }
  }

  bool consume(std::string_view expected) {
    if (input_.substr(position_, expected.size()) != expected) return false;
    position_ += expected.size();
    return true;
  }

  bool parse_hex_quad(uint32_t* value) {
    if (value == nullptr || position_ + 4U > input_.size()) return false;
    uint32_t result = 0;
    for (std::size_t index = 0; index < 4U; ++index) {
      const char digit = input_[position_ + index];
      uint32_t nibble = 0;
      if (digit >= '0' && digit <= '9') {
        nibble = static_cast<uint32_t>(digit - '0');
      } else if (digit >= 'a' && digit <= 'f') {
        nibble = static_cast<uint32_t>(digit - 'a') + 10U;
      } else if (digit >= 'A' && digit <= 'F') {
        nibble = static_cast<uint32_t>(digit - 'A') + 10U;
      } else {
        return false;
      }
      result = (result << 4U) | nibble;
    }
    position_ += 4U;
    *value = result;
    return true;
  }

  static void append_codepoint(std::string* output, uint32_t codepoint) {
    if (output == nullptr) return;
    if (codepoint <= 0x7fU) {
      output->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
      output->push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
      output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
      output->push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
      output->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
      output->push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
      output->push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
      output->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
  }

  bool parse_string(std::string* value) {
    if (value == nullptr || position_ >= input_.size() || input_[position_] != '"') {
      return false;
    }
    ++position_;
    value->clear();
    while (position_ < input_.size()) {
      const unsigned char character = static_cast<unsigned char>(input_[position_++]);
      if (character == '"') return true;
      if (character < 0x20U) return false;
      if (character != '\\') {
        value->push_back(static_cast<char>(character));
        continue;
      }
      if (position_ >= input_.size()) return false;
      const char escape = input_[position_++];
      switch (escape) {
        case '"': value->push_back('"'); break;
        case '\\': value->push_back('\\'); break;
        case '/': value->push_back('/'); break;
        case 'b': value->push_back('\b'); break;
        case 'f': value->push_back('\f'); break;
        case 'n': value->push_back('\n'); break;
        case 'r': value->push_back('\r'); break;
        case 't': value->push_back('\t'); break;
        case 'u': {
          uint32_t high = 0;
          if (!parse_hex_quad(&high)) return false;
          if (high >= 0xd800U && high <= 0xdbffU) {
            if (position_ + 6U > input_.size() || input_[position_] != '\\' ||
                input_[position_ + 1U] != 'u') {
              return false;
            }
            position_ += 2U;
            uint32_t low = 0;
            if (!parse_hex_quad(&low) || low < 0xdc00U || low > 0xdfffU) return false;
            append_codepoint(value, 0x10000U + ((high - 0xd800U) << 10U) +
                                      (low - 0xdc00U));
          } else if (high >= 0xdc00U && high <= 0xdfffU) {
            return false;
          } else {
            append_codepoint(value, high);
          }
          break;
        }
        default: return false;
      }
    }
    return false;
  }

  bool parse_number(std::string* value) {
    if (value == nullptr) return false;
    const std::size_t begin = position_;
    if (input_[position_] == '-') {
      ++position_;
      if (position_ == input_.size()) return false;
    }
    if (position_ < input_.size() && input_[position_] == '0') {
      ++position_;
      if (position_ < input_.size() && is_digit(input_[position_])) return false;
    } else {
      if (position_ == input_.size() || input_[position_] < '1' || input_[position_] > '9') {
        return false;
      }
      while (position_ < input_.size() && is_digit(input_[position_])) ++position_;
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fraction_begin = position_;
      while (position_ < input_.size() && is_digit(input_[position_])) ++position_;
      if (fraction_begin == position_) return false;
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent_begin = position_;
      while (position_ < input_.size() && is_digit(input_[position_])) ++position_;
      if (exponent_begin == position_) return false;
    }
    *value = std::string(input_.substr(begin, position_ - begin));
    return true;
  }

  bool parse_array(JsonValue* value, std::size_t depth) {
    ++position_;  // '['
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == ']') {
      ++position_;
      return true;
    }
    while (position_ < input_.size()) {
      JsonValue element;
      if (!parse_value(&element, depth)) return false;
      value->array.push_back(std::move(element));
      skip_whitespace();
      if (position_ >= input_.size()) return false;
      if (input_[position_] == ']') {
        ++position_;
        return true;
      }
      if (input_[position_] != ',') return false;
      ++position_;
      skip_whitespace();
    }
    return false;
  }

  bool parse_object(JsonValue* value, std::size_t depth) {
    ++position_;  // '{'
    skip_whitespace();
    if (position_ < input_.size() && input_[position_] == '}') {
      ++position_;
      return true;
    }
    while (position_ < input_.size()) {
      if (input_[position_] != '"') return false;
      std::string key;
      if (!parse_string(&key)) return false;
      for (const auto& member : value->object) {
        if (member.first == key) return false;  // duplicate-key rejection
      }
      skip_whitespace();
      if (position_ >= input_.size() || input_[position_] != ':') return false;
      ++position_;
      skip_whitespace();
      JsonValue member_value;
      if (!parse_value(&member_value, depth)) return false;
      value->object.emplace_back(std::move(key), std::move(member_value));
      skip_whitespace();
      if (position_ >= input_.size()) return false;
      if (input_[position_] == '}') {
        ++position_;
        return true;
      }
      if (input_[position_] != ',') return false;
      ++position_;
      skip_whitespace();
    }
    return false;
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

const JsonValue* member(const JsonValue& object, std::string_view name) {
  if (object.type != JsonValue::Type::Object) return nullptr;
  for (const auto& entry : object.object) {
    if (entry.first == name) return &entry.second;
  }
  return nullptr;
}

bool exact_members(const JsonValue& object,
                  std::initializer_list<std::string_view> names) {
  if (object.type != JsonValue::Type::Object || object.object.size() != names.size()) {
    return false;
  }
  for (const std::string_view name : names) {
    if (member(object, name) == nullptr) return false;
  }
  return true;
}

bool is_string(const JsonValue* value) {
  return value != nullptr && value->type == JsonValue::Type::String;
}

bool is_number(const JsonValue* value) {
  return value != nullptr && value->type == JsonValue::Type::Number;
}


bool valid_sha256(std::string_view value) {
  if (value.size() != 71U || value.substr(0U, 7U) != "sha256:") return false;
  for (std::size_t index = 7U; index < value.size(); ++index) {
    const char digit = value[index];
    if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f'))) return false;
  }
  return true;
}

bool parse_u64(const JsonValue* value, uint64_t* result) {
  if (!is_number(value) || result == nullptr || value->number.empty() ||
      value->number.front() == '-' || value->number.front() == '+' ||
      value->number.find_first_of(".eE") != std::string::npos ||
      (value->number.size() > 1U && value->number.front() == '0')) {
    return false;
  }
  uint64_t parsed = 0;
  for (const char digit : value->number) {
    if (digit < '0' || digit > '9') return false;
    const uint64_t next_digit = static_cast<uint64_t>(digit - '0');
    if (parsed > (std::numeric_limits<uint64_t>::max() - next_digit) / 10U) return false;
    parsed = parsed * 10U + next_digit;
  }
  *result = parsed;
  return true;
}

bool parse_u32(const JsonValue* value, uint32_t* result) {
  uint64_t parsed = 0;
  if (!parse_u64(value, &parsed) || parsed > std::numeric_limits<uint32_t>::max() ||
      result == nullptr) {
    return false;
  }
  *result = static_cast<uint32_t>(parsed);
  return true;
}

bool finite_number(const JsonValue* value) {
  if (!is_number(value)) return false;
  char* end = nullptr;
  const double parsed = std::strtod(value->number.c_str(), &end);
  return end != value->number.c_str() && *end == '\0' && std::isfinite(parsed);
}

bool valid_request_id(std::string_view value) {
  if (value.empty() || value.size() > 128U || value == "." || value == "..") return false;
  for (const unsigned char character : value) {
    if (character > 0x7fU ||
        !((character >= 'A' && character <= 'Z') ||
          (character >= 'a' && character <= 'z') ||
          (character >= '0' && character <= '9') || character == '.' ||
          character == '_' || character == '-')) {
      return false;
    }
  }
  return true;
}

bool valid_path_string(std::string_view value) {
  return !value.empty() && value.find('\0') == std::string_view::npos && valid_utf8(value);
}

std::string jcs_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (const unsigned char character : value) {
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
          std::ostringstream code;
          code << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<unsigned int>(character);
          escaped += code.str();
        } else {
          escaped.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string jcs_number(std::string_view value) {
  // ResourceSpec numbers that cross the worker boundary are unsigned integer
  // limits.  This path is retained for model-fingerprint identity values and
  // follows the finite JSON number grammar without accepting NaN/Infinity.
  const std::string number(value);
  char* end = nullptr;
  const double parsed = std::strtod(number.c_str(), &end);
  if (end == nullptr || *end != '\0' || !std::isfinite(parsed)) return "null";
  std::ostringstream rendered;
  rendered << std::setprecision(17) << parsed;
  std::string output = rendered.str();
  if (output.find('e') != std::string::npos || output.find('E') != std::string::npos) {
    for (char& character : output) {
      if (character == 'E') character = 'e';
    }
    const std::size_t exponent = output.find('e');
    if (exponent != std::string::npos && exponent + 1U < output.size() &&
        output[exponent + 1U] == '+') {
      output.erase(exponent + 1U, 1U);
    }
  }
  if (output == "-0" || output == "-0.0") output = "0";
  return output;
}

std::string jcs_serialize(const JsonValue& value) {
  switch (value.type) {
    case JsonValue::Type::Null: return "null";
    case JsonValue::Type::Bool: return value.boolean ? "true" : "false";
    case JsonValue::Type::Number: return jcs_number(value.number);
    case JsonValue::Type::String: return jcs_escape(value.string);
    case JsonValue::Type::Array: {
      std::string output = "[";
      for (std::size_t index = 0; index < value.array.size(); ++index) {
        if (index != 0U) output.push_back(',');
        output += jcs_serialize(value.array[index]);
      }
      output.push_back(']');
      return output;
    }
    case JsonValue::Type::Object: {
      std::vector<std::pair<std::string, std::string>> members;
      members.reserve(value.object.size());
      for (const auto& entry : value.object) {
        members.emplace_back(entry.first, jcs_serialize(entry.second));
      }
      std::sort(members.begin(), members.end(),
                [](const auto& left, const auto& right) { return left.first < right.first; });
      std::string output = "{";
      for (std::size_t index = 0; index < members.size(); ++index) {
        if (index != 0U) output.push_back(',');
        output += jcs_escape(members[index].first);
        output.push_back(':');
        output += members[index].second;
      }
      output.push_back('}');
      return output;
    }
  }
  return "null";
}

bool validate_model_fingerprint(const JsonValue* value, const std::string& model_digest) {
  if (value == nullptr || value->type != JsonValue::Type::Object ||
      !exact_members(*value, {"model_digest", "format", "quantization", "model_family",
                              "model_type", "architectures", "geometry", "rms_norm_eps",
                              "rope_theta", "rope_scaling"})) {
    return false;
  }
  const JsonValue* fingerprint_digest = member(*value, "model_digest");
  const JsonValue* format = member(*value, "format");
  const JsonValue* quantization = member(*value, "quantization");
  const JsonValue* model_family = member(*value, "model_family");
  const JsonValue* model_type = member(*value, "model_type");
  const JsonValue* architectures = member(*value, "architectures");
  const JsonValue* geometry = member(*value, "geometry");
  const JsonValue* rms_norm_eps = member(*value, "rms_norm_eps");
  const JsonValue* rope_theta = member(*value, "rope_theta");
  const JsonValue* rope_scaling = member(*value, "rope_scaling");
  if (!is_string(fingerprint_digest) || fingerprint_digest->string != model_digest ||
      !is_string(format) || !is_string(quantization) || !is_string(model_family) ||
      !is_string(model_type) || format->string != "safetensors" ||
      quantization->string != "fp16" || model_family->string != "llama" ||
      model_type->string != "llama" || architectures->type != JsonValue::Type::Array ||
      architectures->array.size() != 1U || !is_string(&architectures->array[0]) ||
      architectures->array[0].string != "LlamaForCausalLM" ||
      geometry->type != JsonValue::Type::Object ||
      !exact_members(*geometry, {"num_layers", "num_heads", "n_kv_heads", "head_dim",
                                 "hidden_size", "intermediate_size", "vocab_size",
                                 "max_position_embeddings"}) ||
      !finite_number(rms_norm_eps) || !finite_number(rope_theta) ||
      rope_scaling->type != JsonValue::Type::Object ||
      !exact_members(*rope_scaling, {"rope_type", "factor", "high_freq_factor",
                                     "low_freq_factor", "original_max_position_embeddings"})) {
    return false;
  }
  for (const auto& entry : geometry->object) {
    uint64_t ignored = 0;
    if (!parse_u64(&entry.second, &ignored)) return false;
  }
  const JsonValue* rope_type = member(*rope_scaling, "rope_type");
  if (!is_string(rope_type) || rope_type->string != "llama3") return false;
  if (!finite_number(member(*rope_scaling, "factor")) ||
      !finite_number(member(*rope_scaling, "high_freq_factor")) ||
      !finite_number(member(*rope_scaling, "low_freq_factor")) ||
      !finite_number(member(*rope_scaling, "original_max_position_embeddings"))) {
    return false;
  }
  return true;
}

bool parse_resource_spec(const JsonValue* value, NativeResourceSpec* spec) {
  if (value == nullptr || spec == nullptr || value->type != JsonValue::Type::Object ||
      !exact_members(*value, {"model_uri", "model_digest", "model_fingerprint",
                              "cache_capacity", "kernel_pack", "resource_budget"})) {
    return false;
  }
  const JsonValue* model_uri = member(*value, "model_uri");
  const JsonValue* model_digest = member(*value, "model_digest");
  const JsonValue* fingerprint = member(*value, "model_fingerprint");
  const JsonValue* cache_capacity = member(*value, "cache_capacity");
  const JsonValue* kernel_pack = member(*value, "kernel_pack");
  const JsonValue* resource_budget = member(*value, "resource_budget");
  if (!is_string(model_uri) || !valid_path_string(model_uri->string) ||
      !is_string(model_digest) || !valid_sha256(model_digest->string) ||
      !validate_model_fingerprint(fingerprint, model_digest->string) ||
      cache_capacity->type != JsonValue::Type::Object ||
      !exact_members(*cache_capacity, {"batch", "prefix_positions"}) ||
      kernel_pack->type != JsonValue::Type::Object ||
      !exact_members(*kernel_pack, {"name", "version", "digests"}) ||
      resource_budget->type != JsonValue::Type::Object ||
      !exact_members(*resource_budget, {"resident_bytes_max", "scratch_bytes_max",
                                        "total_bytes_max"})) {
    return false;
  }
  uint64_t batch = 0;
  uint64_t prefix_positions = 0;
  if (!parse_u64(member(*cache_capacity, "batch"), &batch) || batch != 1U ||
      !parse_u64(member(*cache_capacity, "prefix_positions"), &prefix_positions) ||
      prefix_positions != 128U) {
    return false;
  }
  const JsonValue* pack_name = member(*kernel_pack, "name");
  const JsonValue* pack_version = member(*kernel_pack, "version");
  const JsonValue* pack_digests = member(*kernel_pack, "digests");
  if (!is_string(pack_name) || !valid_path_string(pack_name->string) ||
      !is_string(pack_version) || !valid_path_string(pack_version->string) ||
      pack_digests->type != JsonValue::Type::Array || pack_digests->array.empty()) {
    return false;
  }
  std::vector<std::string> digests;
  digests.reserve(pack_digests->array.size());
  for (const JsonValue& digest : pack_digests->array) {
    if (!is_string(&digest) || !valid_sha256(digest.string)) return false;
    digests.push_back(digest.string);
  }
  uint64_t resident = 0;
  uint64_t scratch = 0;
  uint64_t total = 0;
  if (!parse_u64(member(*resource_budget, "resident_bytes_max"), &resident) ||
      !parse_u64(member(*resource_budget, "scratch_bytes_max"), &scratch) ||
      !parse_u64(member(*resource_budget, "total_bytes_max"), &total) ||
      resident > total || scratch > total - resident) {
    return false;
  }
  spec->model_uri = model_uri->string;
  spec->model_digest = model_digest->string;
  spec->model_fingerprint = jcs_serialize(*fingerprint);
  spec->cache_capacity = {batch, prefix_positions};
  spec->kernel_pack = {pack_name->string, pack_version->string, std::move(digests)};
  spec->resource_budget = {resident, scratch, total};
  return true;
}

bool parse_generation_body(const JsonValue& body, uint64_t* generation) {
  return exact_members(body, {"resource_generation"}) &&
         parse_u64(member(body, "resource_generation"), generation);
}

bool parse_prefill_body(const JsonValue& body, NativeResourcePrefillRequest* request) {
  if (request == nullptr || !exact_members(body, {"resource_generation", "request_id", "token_ids",
                                                  "prefill_npz_path", "hardware_log_path"})) {
    return false;
  }
  uint64_t generation = 0;
  const JsonValue* request_id = member(body, "request_id");
  const JsonValue* token_ids = member(body, "token_ids");
  const JsonValue* prefill_path = member(body, "prefill_npz_path");
  const JsonValue* log_path = member(body, "hardware_log_path");
  if (!parse_u64(member(body, "resource_generation"), &generation) || !is_string(request_id) ||
      !valid_request_id(request_id->string) || token_ids->type != JsonValue::Type::Array ||
      token_ids->array.size() > 128U || !is_string(prefill_path) ||
      !valid_path_string(prefill_path->string) || !is_string(log_path) ||
      !valid_path_string(log_path->string)) {
    return false;
  }
  std::vector<uint32_t> parsed_tokens;
  parsed_tokens.reserve(token_ids->array.size());
  for (const JsonValue& token : token_ids->array) {
    uint32_t parsed = 0;
    if (!parse_u32(&token, &parsed)) return false;
    parsed_tokens.push_back(parsed);
  }
  request->resource_generation = generation;
  request->request_id = request_id->string;
  request->token_ids = std::move(parsed_tokens);
  request->prefill_npz_path = prefill_path->string;
  request->hardware_log_path = log_path->string;
  return true;
}

std::string bounded(std::string value, std::string_view fallback) {
  if (value.empty() || value.size() > kMaxErrorStringBytes || !valid_utf8(value)) {
    return std::string(fallback);
  }
  return value;
}

NativeResourceError normalized_error(const NativeResourceError& error,
                                     std::string_view fallback_stage) {
  NativeResourceError normalized = error;
  if (normalized.domain != "invalid_request" &&
      normalized.domain != "resource_exhaustion" &&
      normalized.domain != "device_lost_or_faulted" &&
      normalized.domain != "executable_rejection") {
    normalized.domain = "device_lost_or_faulted";
  }
  normalized.message = bounded(std::move(normalized.message), "native resource operation failed");
  normalized.failure_stage = bounded(std::move(normalized.failure_stage), fallback_stage);
  return normalized;
}

std::string error_json(const NativeResourceError& error) {
  const NativeResourceError normalized = normalized_error(error, "operation");
  return std::string("{\"domain\":") + jcs_escape(normalized.domain) +
         ",\"message\":" + jcs_escape(normalized.message) +
         ",\"failure_stage\":" + jcs_escape(normalized.failure_stage) + "}";
}

std::string response_json(const char* request_id, const char* operation, std::string_view status,
                          std::string_view result, const NativeResourceError* error) {
  std::string response = "{\"protocol_version\":" +
                         jcs_escape(kNativeResourceProtocolVersion) + ",\"request_id\":";
  response += request_id == nullptr ? "null" : jcs_escape(request_id);
  response += ",\"operation\":";
  response += operation == nullptr ? "null" : jcs_escape(operation);
  response += ",\"status\":" + jcs_escape(status) + ",\"result\":" + std::string(result);
  response += ",\"error\":";
  response += error == nullptr ? "null" : error_json(*error);
  response.push_back('}');
  return response;
}

bool write_response(std::ostream& output, const char* request_id, const char* operation,
                    std::string_view status, std::string_view result,
                    const NativeResourceError* error) {
  const std::string response = response_json(request_id, operation, status, result, error);
  if (response.size() + 1U > kNativeResourceMaxFrameBytes) return false;
  output << response << '\n';
  output.flush();
  return static_cast<bool>(output);
}

bool write_raw_error(std::ostream& output, std::string_view failure_stage) {
  NativeResourceError error;
  error.domain = "invalid_request";
  error.message = "raw frame rejected before decode";
  error.failure_stage = std::string(failure_stage);
  return write_response(output, nullptr, nullptr, "error", "{}", &error);
}

bool operation_known(std::string_view operation) {
  return operation == "Prepare" || operation == "Commit" || operation == "Rollback" ||
         operation == "Release" || operation == "Prefill" || operation == "Health" ||
         operation == "Shutdown";
}

std::string prepare_result_json(const NativePrepareResult& result) {
  std::string output =
      "{\"resource_generation\":" + std::to_string(result.resource_generation) +
      ",\"state\":" + jcs_escape(result.state) +
      ",\"producer_fingerprint\":" + jcs_escape(result.producer_fingerprint);
  // The concrete runner always publishes its executable identity.  Keeping
  // the field conditional preserves the abstract backend seam for
  // hardware-free fakes that intentionally have no child binary.
  if (!result.runner_binary_sha256.empty()) {
    output += ",\"runner_binary_sha256\":" + jcs_escape(result.runner_binary_sha256);
  }
  output.push_back('}');
  return output;
}

std::string commit_result_json(const NativeCommitResult& result) {
  return std::string("{\"resource_generation\":") + std::to_string(result.resource_generation) +
         ",\"state\":" + jcs_escape(result.state) +
         ",\"producer_fingerprint\":" + jcs_escape(result.producer_fingerprint) + "}";
}

std::string cleanup_result_json(const NativeCleanupResult& result) {
  return std::string("{\"resource_generation\":") + std::to_string(result.resource_generation) +
         ",\"state\":" + jcs_escape(result.state) +
         ",\"already_released\":" + (result.already_released ? "true" : "false") + "}";
}

std::string prefill_result_json(const NativeResourcePrefillResult& result) {
  return std::string("{\"resource_generation\":") + std::to_string(result.resource_generation) +
         ",\"producer_fingerprint\":" + jcs_escape(result.producer_fingerprint) +
         ",\"native_prefill_acceptance\":" + jcs_escape(result.native_prefill_acceptance) +
         ",\"native_prefill_full_layer_loop_status\":" +
         jcs_escape(result.native_prefill_full_layer_loop_status) +
         ",\"runtime_substrate\":" + jcs_escape(result.runtime_substrate) +
         ",\"hardware_log_path\":" + jcs_escape(result.hardware_log_path) +
         ",\"compute_completion_policy\":" + jcs_escape(result.compute_completion_policy) +
         ",\"compute_barrier_policy\":" + jcs_escape(result.compute_barrier_policy) +
         ",\"prefill_npz_path\":" + jcs_escape(result.prefill_npz_path) +
         ",\"kernel_count\":" + std::to_string(result.kernel_count) +
         ",\"transfer_bytes\":" + std::to_string(result.transfer_bytes) +
         ",\"block_tokens\":" + std::to_string(result.block_tokens) +
         ",\"block_count\":" + std::to_string(result.block_count) +
         ",\"failure_stage\":" + jcs_escape(result.failure_stage) +
         ",\"exit_status\":" + std::to_string(result.exit_status) +
         ",\"failure_text\":" + jcs_escape(result.failure_text) + "}";
}

std::string health_result_json(const NativeHealthResult& result, std::string_view state,
                               bool has_generation, uint64_t generation,
                               std::string_view fingerprint,
                               const NativeResourceError* error_summary) {
  std::string output = "{\"child_state\":" + jcs_escape(result.child_state) +
                       ",\"resource_generation\":";
  output += has_generation ? std::to_string(generation) : "null";
  output += ",\"resource_state\":" + jcs_escape(state) + ",\"producer_fingerprint\":";
  output += fingerprint.empty() ? "null" : jcs_escape(fingerprint);
  output += ",\"error_summary\":";
  output += error_summary == nullptr ? "null" : error_json(*error_summary);
  output.push_back('}');
  return output;
}

struct WorkerState {
  enum class ResourceState { None, Prepared, ResidentReady, ReleaseFailed, Shutdown };

  ResourceState resource_state = ResourceState::None;
  bool has_generation = false;
  uint64_t generation = 0;
  std::string fingerprint;
  bool has_released_generation = false;
  uint64_t released_generation = 0;
  std::string released_operation;
  NativeResourceError cleanup_error;
};

const char* state_name(WorkerState::ResourceState state) {
  switch (state) {
    case WorkerState::ResourceState::None: return "none";
    case WorkerState::ResourceState::Prepared: return "prepared";
    case WorkerState::ResourceState::ResidentReady: return "resident-ready";
    case WorkerState::ResourceState::ReleaseFailed: return "release-failed";
    case WorkerState::ResourceState::Shutdown: return "shutdown";
  }
  return "none";
}

NativeResourceError validation_error(std::string_view message, std::string_view stage) {
  NativeResourceError error;
  error.domain = "invalid_request";
  error.message = std::string(message);
  error.failure_stage = std::string(stage);
  return error;
}

bool write_blocked(std::ostream& output, const char* request_id, const char* operation,
                   std::string_view message = "operation is not allowed in current resource state") {
  const NativeResourceError error = validation_error(message, "resource_state");
  return write_response(output, request_id, operation, "blocked", "{}", &error);
}

bool write_backend_error(std::ostream& output, const char* request_id, const char* operation,
                         const NativeResourceError& error) {
  return write_response(output, request_id, operation, "error", "{}", &error);
}

bool read_frame(std::istream& input, std::string* frame, bool* oversized) {
  if (frame == nullptr || oversized == nullptr) return false;
  frame->clear();
  *oversized = false;
  bool saw_input = false;
  char character = 0;
  while (input.get(character)) {
    saw_input = true;
    frame->push_back(character);
    if (frame->size() > kNativeResourceMaxFrameBytes) {
      *oversized = true;
      if (character != '\n') {
        while (input.get(character) && character != '\n') {
        }
      }
      return true;
    }
    // JSONL framing is physical: malformed syntax on one line must never
    // consume the following request, even when braces remain unbalanced.
    if (character == '\n') return true;
  }
  return saw_input;
}

// SHA-256 JCS canonical preimage hashing is kept private to this worker so the producer identity is computed
// from the deterministic JCS preimage, never from model/request/path data.
std::string sha256_hex(std::string_view bytes) {
  static constexpr std::array<uint32_t, 64> kRoundConstants = {
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
  std::array<uint32_t, 8> hash = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                  0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::vector<unsigned char> padded(bytes.begin(), bytes.end());
  const uint64_t bit_length = static_cast<uint64_t>(padded.size()) * 8U;
  padded.push_back(0x80U);
  while ((padded.size() % 64U) != 56U) padded.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8) {
    padded.push_back(static_cast<unsigned char>((bit_length >> shift) & 0xffU));
  }
  const auto rotate_right = [](uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32U - amount));
  };
  for (std::size_t offset = 0; offset < padded.size(); offset += 64U) {
    std::array<uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      words[index] = (static_cast<uint32_t>(padded[offset + index * 4U]) << 24U) |
                     (static_cast<uint32_t>(padded[offset + index * 4U + 1U]) << 16U) |
                     (static_cast<uint32_t>(padded[offset + index * 4U + 2U]) << 8U) |
                     static_cast<uint32_t>(padded[offset + index * 4U + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                          rotate_right(words[index - 15U], 18U) ^ (words[index - 15U] >> 3U);
      const uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                          rotate_right(words[index - 2U], 19U) ^ (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    uint32_t a = hash[0];
    uint32_t b = hash[1];
    uint32_t c = hash[2];
    uint32_t d = hash[3];
    uint32_t e = hash[4];
    uint32_t f = hash[5];
    uint32_t g = hash[6];
    uint32_t h = hash[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const uint32_t choice = (e & f) ^ ((~e) & g);
      const uint32_t temporary1 = h + s1 + choice + kRoundConstants[index] + words[index];
      const uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }
  std::ostringstream digest;
  digest << std::hex << std::setfill('0');
  for (const uint32_t word : hash) digest << std::setw(8) << word;
  return digest.str();
}

}  // namespace

std::string compute_native_producer_fingerprint(const NativeProducerIdentity& identity) {
  // Exact JCS object: "domain", "protocol_version", "runner_binary_sha256",
  // "ordered_kernel_pack_sha256", "target", "runtime_substrate",
  // "completion_policy", "barrier_policy", and "device_identity" with
  // "vendor_id"/"device_id".  JCS UTF-16 key ordering is ASCII ordering here.
  std::string preimage = "{\"barrier_policy\":" + jcs_escape(identity.barrier_policy) +
                         ",\"completion_policy\":" + jcs_escape(identity.completion_policy) +
                         ",\"device_identity\":{\"device_id\":" +
                         jcs_escape(identity.device_id) + ",\"vendor_id\":" +
                         jcs_escape(identity.vendor_id) + "},\"domain\":\"r9700-producer-fingerprint-v1\",\"ordered_kernel_pack_sha256\":[";
  for (std::size_t index = 0; index < identity.ordered_kernel_pack_sha256.size(); ++index) {
    if (index != 0U) preimage.push_back(',');
    preimage += jcs_escape(identity.ordered_kernel_pack_sha256[index]);
  }
  preimage += "],\"protocol_version\":\"r9700_native_resource_v1\",\"runtime_substrate\":";
  preimage += jcs_escape(identity.runtime_substrate);
  preimage += ",\"runner_binary_sha256\":" + jcs_escape(identity.runner_binary_sha256) +
              ",\"target\":" + jcs_escape(identity.target) + "}";
  return "sha256:" + sha256_hex(preimage);
}
int run_native_resource_worker(std::istream& input, std::ostream& output,
                               NativeResourceBackend& backend) {
  WorkerState state;
  std::string raw_frame;
  while (true) {
    raw_frame.clear();
    bool oversized = false;
    if (!read_frame(input, &raw_frame, &oversized)) break;
    if (oversized) {
      if (!write_raw_error(output, "frame_size")) return 1;
      continue;
    }
    if (!raw_frame.empty() && raw_frame.back() == '\n') raw_frame.pop_back();

    JsonValue root;
    JsonParser parser(raw_frame);
    if (!parser.parse(&root) || root.type != JsonValue::Type::Object) {
      if (!write_raw_error(output, "frame_decode")) return 1;
      continue;
    }
    const JsonValue* protocol_version = member(root, "protocol_version");
    const JsonValue* request_id = member(root, "request_id");
    const JsonValue* operation = member(root, "operation");
    const JsonValue* body = member(root, "body");
    const bool request_id_is_valid = is_string(request_id) && valid_request_id(request_id->string);
    const bool operation_is_valid = is_string(operation) && operation_known(operation->string);
    const char* response_request_id = request_id_is_valid ? request_id->string.c_str() : nullptr;
    const char* response_operation = operation_is_valid ? operation->string.c_str() : nullptr;
    NativeResourceError validation;
    if (!exact_members(root, {"protocol_version", "request_id", "operation", "body"}) ||
        !is_string(protocol_version) || protocol_version->string != kNativeResourceProtocolVersion ||
        !request_id_is_valid || !operation_is_valid || body == nullptr ||
        body->type != JsonValue::Type::Object) {
      validation = validation_error("request envelope rejected", "envelope_validation");
      if (!is_string(protocol_version) ||
          (is_string(protocol_version) &&
           protocol_version->string != kNativeResourceProtocolVersion)) {
        validation.failure_stage = "protocol_version";
      } else if (!operation_is_valid) {
        validation.failure_stage = "operation_validation";
      } else if (!request_id_is_valid) {
        validation.failure_stage = "request_id_validation";
      }
      if (!write_response(output, response_request_id, response_operation, "blocked", "{}",
                          &validation)) {
        return 1;
      }
      continue;
    }

    const std::string& request_id_string = request_id->string;
    const std::string& operation_string = operation->string;
    const char* request_id_cstr = request_id_string.c_str();
    const char* operation_cstr = operation_string.c_str();
    if (state.resource_state == WorkerState::ResourceState::Shutdown) {
      if (!write_blocked(output, request_id_cstr, operation_cstr, "worker is shut down")) return 1;
      continue;
    }
    const bool cleanup_retry =
        (operation_string == "Rollback" || operation_string == "Release") &&
        state.resource_state == WorkerState::ResourceState::ReleaseFailed;
    if (state.resource_state == WorkerState::ResourceState::ReleaseFailed &&
        operation_string != "Health" && !cleanup_retry) {
      if (!write_blocked(output, request_id_cstr, operation_cstr)) return 1;
    } else if (operation_string == "Prepare") {
      if (!exact_members(*body, {"resource_spec"}) ||
          state.resource_state != WorkerState::ResourceState::None) {
        validation = validation_error("Prepare is not valid in the current resource state",
                                      state.resource_state == WorkerState::ResourceState::None
                                          ? "operation_validation"
                                          : "resource_state");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativeResourceSpec spec;
      if (!parse_resource_spec(member(*body, "resource_spec"), &spec)) {
        validation = validation_error("resource spec rejected", "operation_validation");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativePrepareResult result;
      NativeResourceError error;
      if (!backend.prepare(spec, &result, &error)) {
        state = WorkerState{};  // Prepare owns and cleans every partial allocation.
        if (!write_backend_error(output, request_id_cstr, operation_cstr,
                                 normalized_error(error, "prepare"))) {
          return 1;
        }
        continue;
      }
      state.resource_state = WorkerState::ResourceState::Prepared;
      state.has_generation = true;
      state.generation = result.resource_generation;
      state.fingerprint = result.producer_fingerprint;
      state.has_released_generation = false;
      if (!write_response(output, request_id_cstr, operation_cstr, "pass",
                          prepare_result_json(result), nullptr)) {
        return 1;
      }
    } else if (operation_string == "Commit") {
      uint64_t generation = 0;
      if (!parse_generation_body(*body, &generation) ||
          state.resource_state != WorkerState::ResourceState::Prepared ||
          !state.has_generation || generation != state.generation) {
        validation = validation_error("Commit generation or state rejected", "operation_validation");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativeCommitResult result;
      NativeResourceError error;
      if (!backend.commit(generation, &result, &error)) {
        state = WorkerState{};  // Commit consumes and self-cleans its prepared value.
        if (!write_backend_error(output, request_id_cstr, operation_cstr,
                                 normalized_error(error, "commit"))) {
          return 1;
        }
        continue;
      }
      state.resource_state = WorkerState::ResourceState::ResidentReady;
      state.generation = result.resource_generation;
      state.fingerprint = result.producer_fingerprint;
      if (!write_response(output, request_id_cstr, operation_cstr, "pass",
                          commit_result_json(result), nullptr)) {
        return 1;
      }
    } else if (operation_string == "Rollback" || operation_string == "Release") {
      uint64_t generation = 0;
      if (!parse_generation_body(*body, &generation)) {
        validation = validation_error("cleanup body rejected", "operation_validation");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      const bool first_cleanup =
          operation_string == "Rollback"
              ? (state.resource_state == WorkerState::ResourceState::Prepared &&
                 state.has_generation && generation == state.generation)
              : (state.resource_state == WorkerState::ResourceState::ResidentReady &&
                 state.has_generation && generation == state.generation);
      const bool repeated_cleanup =
          state.resource_state == WorkerState::ResourceState::None &&
          state.has_released_generation && state.released_generation == generation &&
          state.released_operation == operation_string;
      const bool release_failed_retry =
          state.resource_state == WorkerState::ResourceState::ReleaseFailed &&
          cleanup_retry && generation == state.generation &&
          operation_string == state.released_operation;
      if (!first_cleanup && !repeated_cleanup && !release_failed_retry) {
        validation = validation_error("cleanup generation or state rejected", "resource_state");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativeCleanupResult result;
      NativeResourceError error;
      const bool success = operation_string == "Rollback"
                               ? backend.rollback(generation, &result, &error)
                               : backend.release(generation, &result, &error);
      if (!success) {
        state.resource_state = WorkerState::ResourceState::ReleaseFailed;
        state.has_generation = true;
        state.generation = generation;
        state.released_operation = operation_string;
        state.cleanup_error = normalized_error(error, "cleanup");
        if (!write_backend_error(output, request_id_cstr, operation_cstr, state.cleanup_error)) {
          return 1;
        }
        continue;
      }
      if (release_failed_retry) result.already_released = false;
      state.resource_state = WorkerState::ResourceState::None;
      state.has_generation = false;
      state.fingerprint.clear();
      state.has_released_generation = true;
      state.released_generation = generation;
      state.released_operation = operation_string;
      state.cleanup_error = {};
      if (!write_response(output, request_id_cstr, operation_cstr, "pass",
                          cleanup_result_json(result), nullptr)) {
        return 1;
      }
    } else if (operation_string == "Prefill") {
      NativeResourcePrefillRequest request;
      if (!parse_prefill_body(*body, &request) ||
          state.resource_state != WorkerState::ResourceState::ResidentReady ||
          !state.has_generation || request.resource_generation != state.generation) {
        validation = validation_error("Prefill generation, request, or state rejected",
                                      "operation_validation");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativeResourcePrefillResult result;
      NativeResourceError error;
      if (!backend.prefill(request, &result, &error)) {
        if (!write_backend_error(output, request_id_cstr, operation_cstr,
                                 normalized_error(error, "prefill"))) {
          return 1;
        }
        continue;
      }
      if (!write_response(output, request_id_cstr, operation_cstr, "pass",
                          prefill_result_json(result), nullptr)) {
        return 1;
      }
    } else if (operation_string == "Health") {
      if (!exact_members(*body, {})) {
        validation = validation_error("Health body rejected", "operation_validation");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativeHealthResult result;
      NativeResourceError error;
      if (!backend.health(&result, &error)) {
        if (!write_backend_error(output, request_id_cstr, operation_cstr,
                                 normalized_error(error, "health"))) {
          return 1;
        }
        continue;
      }
      const bool has_generation = state.resource_state != WorkerState::ResourceState::None &&
                                  state.resource_state != WorkerState::ResourceState::Shutdown;
      const uint64_t generation = has_generation ? state.generation : result.resource_generation;
      const std::string_view resource_state = state.resource_state == WorkerState::ResourceState::None
                                                   ? "none"
                                                   : state_name(state.resource_state);
      const std::string_view fingerprint =
          has_generation ? state.fingerprint : std::string_view{};
      const NativeResourceError* summary =
          state.resource_state == WorkerState::ResourceState::ReleaseFailed
              ? &state.cleanup_error
              : nullptr;
      if (!write_response(output, request_id_cstr, operation_cstr, "pass",
                          health_result_json(result, resource_state, has_generation, generation,
                                             fingerprint, summary),
                          nullptr)) {
        return 1;
      }
    } else if (operation_string == "Shutdown") {
      if (!exact_members(*body, {}) || state.resource_state != WorkerState::ResourceState::None) {
        validation = validation_error("Shutdown requires completed native cleanup", "resource_state");
        if (!write_response(output, request_id_cstr, operation_cstr, "blocked", "{}", &validation)) {
          return 1;
        }
        continue;
      }
      NativeShutdownResult result;
      NativeResourceError error;
      if (!backend.shutdown(&result, &error)) {
        if (!write_backend_error(output, request_id_cstr, operation_cstr,
                                 normalized_error(error, "shutdown"))) {
          return 1;
        }
        continue;
      }
      state.resource_state = WorkerState::ResourceState::Shutdown;
      if (!write_response(output, request_id_cstr, operation_cstr, "pass",
                          std::string("{\"state\":") + jcs_escape(result.state) + "}",
                          nullptr)) {
        return 1;
      }
      break;
    }
  }
  return 0;
}

}  // namespace native_r9700

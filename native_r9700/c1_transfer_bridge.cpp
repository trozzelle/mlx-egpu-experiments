// native_r9700/c1_transfer_bridge.cpp — CLI adapter for the C1 streaming transfer proof.
//
// AMDevSession is the sole owner of the source-grounded C0 TinyGPU lifecycle,
// VM setup, and SDMA transfer mechanics. This file intentionally only maps the
// stable command-line contract onto that session.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>

#include "amdev_session.h"

namespace {

bool parse_u64_arg(const char* text, uint64_t* out) {
  if (text == nullptr || text[0] == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') return false;
  if (value > std::numeric_limits<uint64_t>::max()) return false;
  *out = static_cast<uint64_t>(value);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  native_r9700::AMDevSession session;
  if (argc == 3 && std::strcmp(argv[1], "--byte-count") == 0) {
    uint64_t byte_count = 0;
    if (!parse_u64_arg(argv[2], &byte_count) || byte_count == 0) {
      std::fprintf(stderr, "error: --byte-count expects N > 0\n");
      return 2;
    }
    return session.streaming_transfer_proof(byte_count);
  }
  if (argc == 4 && std::strcmp(argv[1], "--roundtrip-file") == 0) {
    std::string error_text;
    const int result = session.transfer_round_trip_file(argv[2], argv[3], &error_text);
    if (result != 0 && !error_text.empty()) std::fprintf(stderr, "error: %s\n", error_text.c_str());
    return result;
  }
  std::fprintf(stderr, "usage: %s --byte-count N | --roundtrip-file IN OUT\n", argv[0]);
  return 2;
}

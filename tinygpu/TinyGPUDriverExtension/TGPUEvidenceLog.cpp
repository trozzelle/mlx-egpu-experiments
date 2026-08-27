#include "TGPUEvidenceLog.h"

#include <cstdio>
#include <filesystem>
#include <system_error>

namespace {

bool WriteFormatted(FILE* stream, const char* format, unsigned long long value) {
  return stream != nullptr && std::fprintf(stream, format, value) >= 0;
}

bool WriteFailureText(FILE* stream,
                      const uint8_t* bytes,
                      std::size_t byte_count) {
  if (!stream || !bytes || std::fputs("failure_text=", stream) == EOF) {
    return false;
  }
  for (std::size_t index = 0; index < byte_count; ++index) {
    const unsigned char byte = bytes[index];
    if (byte == 0) break;
    const int output = (byte < 0x20U || byte == 0x7fU) ? '?' : byte;
    if (std::fputc(output, stream) == EOF) return false;
  }
  return std::fputc('\n', stream) != EOF;
}

}  // namespace

bool TGPUEvidenceLog::Write(const char* path,
                            const TGPUEvidenceRecord& record) {
  if (!path || path[0] == '\0') return false;

  try {
    const std::filesystem::path target(path);
    const std::filesystem::path parent = target.parent_path();
    if (!parent.empty()) {
      std::error_code error;
      (void)std::filesystem::create_directories(parent, error);
      if (error || !std::filesystem::is_directory(parent, error) || error) {
        return false;
      }
    }

    FILE* stream = std::fopen(path, "wb");
    if (!stream) return false;

    bool ok = WriteFormatted(stream, "abi_major=%llu\n",
                             static_cast<unsigned long long>(record.abi_major));
    ok = ok && WriteFormatted(
                  stream, "abi_minor=%llu\n",
                  static_cast<unsigned long long>(record.abi_minor));
    ok = ok && WriteFormatted(
                  stream, "selector=%llu\n",
                  static_cast<unsigned long long>(record.selector));
    ok = ok && WriteFormatted(
                  stream, "status=%llu\n",
                  static_cast<unsigned long long>(record.status));
    ok = ok && WriteFormatted(
                  stream, "failure_stage=%llu\n",
                  static_cast<unsigned long long>(record.failure_stage));
    ok = ok && WriteFormatted(
                  stream, "device_epoch=%llu\n",
                  static_cast<unsigned long long>(record.device_epoch));
    ok = ok && WriteFormatted(
                  stream, "exit_status=%llu\n",
                  static_cast<unsigned long long>(record.exit_status));
    ok = ok && WriteFailureText(
                  stream, record.failure_text,
                  sizeof(record.failure_text) - 1U);
    if (ok) ok = std::fflush(stream) == 0 && std::ferror(stream) == 0;
    const int close_result = std::fclose(stream);
    return ok && close_result == 0;
  } catch (...) {
    return false;
  }
}

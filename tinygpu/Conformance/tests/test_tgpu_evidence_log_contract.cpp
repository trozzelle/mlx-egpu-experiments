// RED contract for the TinyGPU conformance evidence artifact.
//
// This is a host-only contract for the evidence boundary.  It uses a real
// temporary filesystem but no DriverKit, PCI device, provider, or generic
// logging framework.
#include "TGPUEvidenceLog.h"
#include "TGPUABI.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace {

constexpr uint32_t kAbiMajor = 1;
constexpr uint32_t kAbiMinor = 0;
constexpr uint32_t kHealthSelector = 0x0d;
constexpr uint32_t kDeviceLostStatus = 14;
constexpr uint32_t kFirmwareFailureStage = 3;
constexpr uint64_t kDeviceEpoch = 7;
constexpr uint32_t kFailedExitStatus = 1;
constexpr size_t kMaxFailureTextBytes = 192;

bool expect(bool condition, const char* message) {
  if (condition) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

TGPUEvidenceRecord BaseRecord() {
  TGPUEvidenceRecord record{};
  record.abi_major = kAbiMajor;
  record.abi_minor = kAbiMinor;
  record.selector = kHealthSelector;
  record.status = kDeviceLostStatus;
  record.failure_stage = kFirmwareFailureStage;
  record.device_epoch = kDeviceEpoch;
  record.exit_status = kFailedExitStatus;
  return record;
}

TGPUHealthFaultQueryResponse HealthResponse(const std::string& text) {
  TGPUHealthFaultQueryResponse response{};
  response.text_length = static_cast<uint32_t>(
      std::min(text.size(), sizeof(response.failure_text)));
  std::memcpy(response.failure_text, text.data(), response.text_length);
  return response;
}

void SetFailureTextFromHealth(TGPUEvidenceRecord* record,
                              const TGPUHealthFaultQueryResponse& health) {
  std::memset(record->failure_text, 0, sizeof(record->failure_text));
  const size_t bytes = std::min<size_t>(
      health.text_length,
      std::min(sizeof(record->failure_text) - 1,
               sizeof(health.failure_text)));
  std::memcpy(record->failure_text, health.failure_text, bytes);
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("tgpu-evidence-contract-" + std::to_string(::getpid()));
  std::error_code error;
  std::filesystem::remove_all(root, error);
  if (!expect(std::filesystem::create_directory(root, error) && !error,
              "temporary evidence root can be created")) {
    return 1;
  }

  // Parent directories are intentionally absent.  Write must create the
  // complete nested path or report failure; printing a diagnostic to stderr is
  // not durable evidence.
  const std::filesystem::path nested_log =
      root / "not" / "yet" / "existing" / "health.log";
  TGPUHealthFaultQueryResponse health =
      HealthResponse("cold_stage=PspSosTmr");
  TGPUEvidenceRecord record = BaseRecord();
  SetFailureTextFromHealth(&record, health);
  if (!expect(TGPUEvidenceLog::Write(nested_log.string().c_str(), record),
              "evidence write creates nested parent directories") ||
      !expect(std::filesystem::is_regular_file(nested_log),
              "nested evidence path is a regular file")) {
    return 1;
  }

  const std::string expected =
      "abi_major=1\n"
      "abi_minor=0\n"
      "selector=13\n"
      "status=14\n"
      "failure_stage=3\n"
      "device_epoch=7\n"
      "exit_status=1\n"
      "failure_text=cold_stage=PspSosTmr\n";
  if (!expect(ReadFile(nested_log) == expected,
              "evidence contains exactly the bounded required fields")) {
    return 1;
  }

  // Newlines and other control bytes in health-derived text must not become
  // extra records.  The contract uses '?' as the deterministic replacement.
  const std::filesystem::path sanitized_log = root / "sanitized.log";
  record = BaseRecord();
  std::string injected = "cold_stage=PspSosTmr\nstatus=0";
  injected.push_back('\x1b');
  health = HealthResponse(injected);
  SetFailureTextFromHealth(&record, health);
  if (!expect(TGPUEvidenceLog::Write(sanitized_log.string().c_str(), record),
              "sanitized evidence write succeeds") ||
      !expect(ReadFile(sanitized_log) ==
                  "abi_major=1\n"
                  "abi_minor=0\n"
                  "selector=13\n"
                  "status=14\n"
                  "failure_stage=3\n"
                  "device_epoch=7\n"
                  "exit_status=1\n"
                  "failure_text=cold_stage=PspSosTmr?status=0?\n",
              "failure text is one line with control injection removed")) {
    return 1;
  }

  // The frozen health text field is 192 bytes including its terminator.  Even
  // a response reporting all 192 bytes must emit no more than 191 text bytes.
  const std::filesystem::path bounded_log = root / "bounded.log";
  record = BaseRecord();
  health = TGPUHealthFaultQueryResponse{};
  health.text_length = sizeof(health.failure_text);
  std::memset(health.failure_text, 'X', sizeof(health.failure_text));
  SetFailureTextFromHealth(&record, health);
  if (!expect(TGPUEvidenceLog::Write(bounded_log.string().c_str(), record),
              "bounded evidence write succeeds")) {
    return 1;
  }
  const std::string bounded_text(kMaxFailureTextBytes - 1, 'X');
  const std::string expected_bounded =
      std::string("abi_major=1\n") +
      "abi_minor=0\n"
      "selector=13\n"
      "status=14\n"
      "failure_stage=3\n"
      "device_epoch=7\n"
      "exit_status=1\n"
      "failure_text=" +
      bounded_text + "\n";
  if (!expect(ReadFile(bounded_log) == expected_bounded,
              "failure text is bounded to the frozen field length")) {
    return 1;
  }

  // A regular file used as a parent is a deterministic unwritable target.  A
  // failed open/write/close must be reported to the caller.
  const std::filesystem::path not_a_directory = root / "not-a-directory";
  {
    std::ofstream marker(not_a_directory);
    marker << "marker";
  }
  const std::filesystem::path unwritable_log =
      not_a_directory / "health.log";
  record = BaseRecord();
  SetFailureTextFromHealth(&record, health);
  if (!expect(!TGPUEvidenceLog::Write(unwritable_log.string().c_str(), record),
              "evidence write reports an unwritable path")) {
    return 1;
  }

  std::filesystem::remove_all(root, error);
  return 0;
}

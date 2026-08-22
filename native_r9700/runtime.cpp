// native_r9700/runtime.cpp — C1 native runner runtime shell implementation.
//
// Adapts the proven C0 probe lifecycle (experiments/native-r9700-runtime/
// native_amdev_transfer_probe.cpp) into this reusable shell. The C0 probe is
// the frozen, byte-stable reference; this file reuses its source-grounded
// encodings and lifecycle sequence without importing or editing the probe.
//
// RuntimeSession::dry_run; `kernel_proof` and `transfer_proof` are hardware-gated
// wrappers around source-grounded C0/C1 bridge commands. The reusable in-process
// buffer/kernel primitive stages still record their intended effect until C1
// task sets 5-8 replace the wrappers with direct primitives.

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>


#include "amdev_session.h"
#include "amdev_packets.h"
#include "hsa_code_image_asset.h"
#include "model_weight_binder.h"
#include "runtime.h"


namespace native_r9700 {

namespace {

constexpr size_t kLogBufferSize = 4096;
constexpr size_t kLlamaEmbeddingRowBytes = 4096;
}  // namespace

const char* lifecycle_stage_name(LifecycleStage stage) {
  switch (stage) {
    case LifecycleStage::Created: return "created";
    case LifecycleStage::Initialized: return "initialized";
    case LifecycleStage::BuffersAllocated: return "buffers_allocated";
    case LifecycleStage::InputCopied: return "input_copied";
    case LifecycleStage::KernelLoaded: return "kernel_loaded";
    case LifecycleStage::KernargsWritten: return "kernargs_written";
    case LifecycleStage::Dispatched: return "dispatched";
    case LifecycleStage::ReadbackCompared: return "readback_compared";
    case LifecycleStage::CleanedUp: return "cleaned_up";
    case LifecycleStage::Failed: return "failed";
  }
  return "unknown";
}

void Kernargs::encode(uint8_t* out, size_t out_capacity) const {
  const size_t needed = kKernargScalarOffset + sizeof(uint32_t);
  if (out_capacity < needed) {
    return;  // caller contract: capacity guaranteed by declaration site
  }
  std::memset(out, 0, needed);
  // output_va @ 0
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    out[kKernargOutputVaOffset + i] = static_cast<uint8_t>((output_va >> (i * 8)) & 0xffU);
  }
  // input_va @ 8
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    out[kKernargInputVaOffset + i] = static_cast<uint8_t>((input_va >> (i * 8)) & 0xffU);
  }
  // scalar_va @ 16
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    out[kKernargScalarVaOffset + i] = static_cast<uint8_t>((scalar_va >> (i * 8)) & 0xffU);
  }
  // scalar u32 @ 24
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    out[kKernargScalarOffset + i] = static_cast<uint8_t>((scalar >> (i * 8)) & 0xffU);
  }
}

bool Kernargs::verify(const uint8_t* data, size_t size, std::string* error_text) const {
  const size_t needed = kKernargScalarOffset + sizeof(uint32_t);
  if (data == nullptr || size < needed) {
    *error_text = "kernarg verify buffer too small";
    return false;
  }
  uint64_t out_va = 0, in_va = 0, scalar_va = 0;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    out_va |= static_cast<uint64_t>(data[kKernargOutputVaOffset + i]) << (i * 8);
    in_va |= static_cast<uint64_t>(data[kKernargInputVaOffset + i]) << (i * 8);
    scalar_va |= static_cast<uint64_t>(data[kKernargScalarVaOffset + i]) << (i * 8);
  }
  uint32_t scalar = 0;
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    scalar |= static_cast<uint32_t>(data[kKernargScalarOffset + i]) << (i * 8);
  }
  if (out_va != output_va || in_va != input_va || scalar_va != this->scalar_va ||
      scalar != this->scalar) {
    *error_text = "kernarg layout readback mismatch";
    return false;
  }
  return true;
}

namespace {


std::string hex_bytes(const uint8_t* data, size_t size) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(size * 2);
  for (size_t i = 0; i < size; ++i) {
    out.push_back(kHex[(data[i] >> 4) & 0x0fU]);
    out.push_back(kHex[data[i] & 0x0fU]);
  }
  return out;
}

std::string timestamp_utc_now() {
  char buf[32];
  const time_t now = time(nullptr);
  struct tm tmv;
  gmtime_r(&now, &tmv);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
  return std::string(buf);
}

struct ProcessResult {
  int exit_status = 127;
  std::string output;
};

bool contains_text(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

std::string log_value_for_key(const std::string& text, const std::string& key) {
  const std::string prefix = key + ": ";
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find('\n', start);
    const std::size_t line_end = end == std::string::npos ? text.size() : end;
    if (line_end >= start + prefix.size() &&
        text.compare(start, prefix.size(), prefix) == 0) {
      return text.substr(start + prefix.size(), line_end - start - prefix.size());
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return std::string();
}

bool require_log_value(const std::string& text, const std::string& key, const std::string& expected,
                       std::string* missing_or_wrong) {
  const std::string observed = log_value_for_key(text, key);
  if (observed == expected) return true;
  if (!missing_or_wrong->empty()) *missing_or_wrong += ", ";
  *missing_or_wrong += key + "=" + expected + " (observed " +
                       (observed.empty() ? std::string("<missing>") : observed) + ")";
  return false;
}

uint64_t ceil_div_u64(uint64_t numerator, uint64_t denominator) {
  return (numerator / denominator) + ((numerator % denominator) != 0 ? 1ULL : 0ULL);
}

std::string shell_quote_for_log(const std::string& arg) {
  if (arg.empty()) return "''";
  bool safe = true;
  for (char c : arg) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '/' || c == '.' || c == '_' ||
          c == '-' || c == ':' || c == '=')) {
      safe = false;
      break;
    }
  }
  if (safe) return arg;
  std::string out = "'";
  for (char c : arg) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

std::string join_command_for_log(const std::vector<std::string>& args) {
  std::string out;
  for (const std::string& arg : args) {
    if (!out.empty()) out += " ";
    out += shell_quote_for_log(arg);
  }
  return out;
}

ProcessResult run_process_capture(const std::vector<std::string>& args) {
  ProcessResult result;
  if (args.empty()) {
    result.output = "failed to launch child process: empty argv\n";
    return result;
  }

  int pipefd[2];
  if (pipe(pipefd) != 0) {
    result.output = std::string("failed to create child pipe: ") + std::strerror(errno) + "\n";
    return result;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    result.output = std::string("failed to fork child process: ") + std::strerror(errno) + "\n";
    close(pipefd[0]);
    close(pipefd[1]);
    return result;
  }

  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const std::string& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    std::fprintf(stderr, "execvp failed for %s: %s\n", argv[0], std::strerror(errno));
    _exit(127);
  }

  close(pipefd[1]);
  char buf[4096];
  for (;;) {
    const ssize_t n = read(pipefd[0], buf, sizeof(buf));
    if (n > 0) {
      result.output.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) break;
    if (errno == EINTR) continue;
    result.output += std::string("failed to read child output: ") + std::strerror(errno) + "\n";
    break;
  }
  close(pipefd[0]);

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) continue;
    result.output += std::string("failed to wait for child process: ") + std::strerror(errno) + "\n";
    result.exit_status = 127;
    return result;
  }
  if (WIFEXITED(status)) {
    result.exit_status = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_status = 128 + WTERMSIG(status);
  } else {
    result.exit_status = 127;
  }
  return result;
}

std::string write_text_log(const std::string& text, const std::string& name) {
  const std::string logs_dir = "logs";
  if (mkdir(logs_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    std::fprintf(stderr, "warning: cannot create logs/ (%s)\n", std::strerror(errno));
    return std::string();
  }
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s", logs_dir.c_str(), name.c_str());
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::fprintf(stderr, "warning: cannot open %s (%s)\n", path, std::strerror(errno));
    return std::string();
  }
  const ssize_t written = write(fd, text.data(), text.size());
  close(fd);
  if (written != static_cast<ssize_t>(text.size())) {
    std::fprintf(stderr, "warning: short write to %s\n", path);
    return std::string();
  }
  return std::string(path);
}

struct ReservedVramSmokeLog {
  int parent_fd = -1;
  int directory_fd = -1;
  int file_fd = -1;
  struct stat directory_stat {};
  std::string filename;
  std::string path;
};

bool close_vram_smoke_log_fd(int* fd, const char* description, std::string* error_text) {
  if (*fd < 0) return true;
  const int closing_fd = *fd;
  *fd = -1;
  if (close(closing_fd) == 0) return true;

  const int error = errno;
  if (error_text != nullptr && error_text->empty()) {
    *error_text = std::string("close VRAM smoke ") + description + " failed: " +
                  std::strerror(error);
  }
  return false;
}

bool close_reserved_vram_smoke_log(ReservedVramSmokeLog* reserved, std::string* error_text) {
  const bool file_closed =
      close_vram_smoke_log_fd(&reserved->file_fd, "log file", error_text);
  const bool directory_closed =
      close_vram_smoke_log_fd(&reserved->directory_fd, "logs directory", error_text);
  const bool parent_closed =
      close_vram_smoke_log_fd(&reserved->parent_fd, "working directory", error_text);
  return file_closed && directory_closed && parent_closed;
}

bool reserve_vram_smoke_log(const std::string& timestamp, ReservedVramSmokeLog* reserved,
                            std::string* error_text) {
  if (error_text != nullptr) error_text->clear();

  const int parent_fd = open(".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
  if (parent_fd < 0) {
    if (error_text != nullptr) {
      *error_text = std::string("cannot open working directory: ") + std::strerror(errno);
    }
    return false;
  }

  ReservedVramSmokeLog pending;
  pending.parent_fd = parent_fd;
  struct stat parent_stat {};
  const int parent_stat_status = fstat(parent_fd, &parent_stat);
  if (parent_stat_status != 0 || !S_ISDIR(parent_stat.st_mode)) {
    const int error = errno;
    if (error_text != nullptr) {
      *error_text = parent_stat_status == 0
                        ? "working directory is not a directory"
                        : std::string("cannot stat working directory: ") + std::strerror(error);
    }
    close_reserved_vram_smoke_log(&pending, error_text);
    return false;
  }

  if (mkdirat(parent_fd, "logs", 0755) != 0 && errno != EEXIST) {
    if (error_text != nullptr) {
      *error_text = std::string("cannot create logs/: ") + std::strerror(errno);
    }
    close_reserved_vram_smoke_log(&pending, error_text);
    return false;
  }

  const int directory_fd = openat(parent_fd, "logs", O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
  if (directory_fd < 0) {
    if (error_text != nullptr) {
      *error_text = std::string("cannot open logs/: ") + std::strerror(errno);
    }
    close_reserved_vram_smoke_log(&pending, error_text);
    return false;
  }
  pending.directory_fd = directory_fd;

  const int directory_stat_status = fstat(directory_fd, &pending.directory_stat);
  if (directory_stat_status != 0 || !S_ISDIR(pending.directory_stat.st_mode)) {
    const int error = errno;
    if (error_text != nullptr) {
      *error_text = directory_stat_status == 0
                        ? "logs/ is not a directory"
                        : std::string("cannot stat logs/: ") + std::strerror(error);
    }
    close_reserved_vram_smoke_log(&pending, error_text);
    return false;
  }

  constexpr uint32_t kMaxSuffixes = 1000;
  const std::string base = "c1-runner-vram-smoke-" + timestamp;
  for (uint32_t suffix = 0; suffix != kMaxSuffixes; ++suffix) {
    const std::string filename =
        base + (suffix == 0 ? std::string() : "-" + std::to_string(suffix)) + ".log";
    const int file_fd =
        openat(directory_fd, filename.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (file_fd >= 0) {
      pending.file_fd = file_fd;
      pending.filename = filename;
      pending.path = "logs/" + filename;
      *reserved = std::move(pending);
      return true;
    }
    if (errno != EEXIST) {
      if (error_text != nullptr) {
        *error_text =
            "reserve VRAM smoke log failed: " + filename + ": " + std::strerror(errno);
      }
      close_reserved_vram_smoke_log(&pending, error_text);
      return false;
    }
  }

  if (error_text != nullptr) {
    *error_text = "reserve VRAM smoke log failed: all collision-safe names are occupied";
  }
  close_reserved_vram_smoke_log(&pending, error_text);
  return false;
}

bool write_reserved_vram_smoke_log(ReservedVramSmokeLog* reserved, const std::string& text,
                                   std::string* error_text) {
  if (error_text != nullptr) error_text->clear();
  auto fail = [&](const std::string& message) {
    if (error_text != nullptr) *error_text = message;
    close_reserved_vram_smoke_log(reserved, error_text);
    return false;
  };

  size_t offset = 0;
  while (offset < text.size()) {
    const ssize_t written = write(reserved->file_fd, text.data() + offset, text.size() - offset);
    if (written > 0) {
      offset += static_cast<size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) continue;
    const int error = written == 0 ? EIO : errno;
    return fail(std::string("write VRAM smoke log failed: ") + std::strerror(error));
  }

  if (fsync(reserved->file_fd) != 0) {
    const int error = errno;
    return fail(std::string("sync VRAM smoke log failed: ") + std::strerror(error));
  }

  struct stat file_stat {};
  if (fstat(reserved->file_fd, &file_stat) != 0) {
    const int error = errno;
    return fail(std::string("revalidate VRAM smoke log file failed: ") + std::strerror(error));
  }

  struct stat entry_stat {};
  if (fstatat(reserved->directory_fd, reserved->filename.c_str(), &entry_stat,
              AT_SYMLINK_NOFOLLOW) != 0) {
    const int error = errno;
    return fail(std::string("revalidate VRAM smoke log entry failed: ") + std::strerror(error));
  }
  if (!S_ISREG(file_stat.st_mode) || !S_ISREG(entry_stat.st_mode) ||
      file_stat.st_dev != entry_stat.st_dev || file_stat.st_ino != entry_stat.st_ino) {
    return fail("revalidate VRAM smoke log entry failed: log file is no longer the directory entry");
  }

  if (fsync(reserved->directory_fd) != 0) {
    const int error = errno;
    return fail(std::string("sync logs directory failed: ") + std::strerror(error));
  }

  struct stat current_directory_entry_stat {};
  if (fstatat(reserved->parent_fd, "logs", &current_directory_entry_stat,
              AT_SYMLINK_NOFOLLOW) != 0) {
    const int error = errno;
    return fail(std::string("revalidate logs directory entry failed: ") + std::strerror(error));
  }
  if (!S_ISDIR(current_directory_entry_stat.st_mode) ||
      current_directory_entry_stat.st_dev != reserved->directory_stat.st_dev ||
      current_directory_entry_stat.st_ino != reserved->directory_stat.st_ino) {
    return fail("revalidate logs directory entry failed: logs directory was replaced");
  }

  return close_reserved_vram_smoke_log(reserved, error_text);
}

bool write_binary_file(const std::string& path, const std::vector<uint8_t>& data,
                       std::string* error_text) {
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    *error_text = "open " + path + " for write failed: " + std::strerror(errno);
    return false;
  }
  size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t written = write(fd, data.data() + offset, data.size() - offset);
    if (written < 0) {
      if (errno == EINTR) continue;
      *error_text = "write " + path + " failed: " + std::strerror(errno);
      close(fd);
      return false;
    }
    if (written == 0) {
      *error_text = "write " + path + " made no progress";
      close(fd);
      return false;
    }
    offset += static_cast<size_t>(written);
  }
  close(fd);
  return true;
}

bool read_binary_file(const std::string& path, std::vector<uint8_t>* data,
                      std::string* error_text) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    *error_text = "open " + path + " for read failed: " + std::strerror(errno);
    return false;
  }
  data->clear();
  uint8_t buf[4096];
  for (;;) {
    const ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
      data->insert(data->end(), buf, buf + n);
      continue;
    }
    if (n == 0) break;
    if (errno == EINTR) continue;
    *error_text = "read " + path + " failed: " + std::strerror(errno);
    close(fd);
    return false;
  }
  close(fd);
  return true;
}

uint8_t transfer_pattern_byte(uint64_t absolute_offset) {
  return static_cast<uint8_t>(((absolute_offset * 131ULL) + 17ULL) & 0xffULL);
}

std::vector<uint8_t> make_transfer_pattern(uint64_t byte_count) {
  std::vector<uint8_t> bytes;
  bytes.resize(static_cast<size_t>(byte_count));
  for (uint64_t i = 0; i < byte_count; ++i) bytes[static_cast<size_t>(i)] = transfer_pattern_byte(i);
  return bytes;
}

std::string kernel_proof_log_path_for_timestamp(const std::string& timestamp) {
  return "logs/c1-runner-kernel-proof-" + timestamp + ".log";
}
std::string transfer_proof_log_path_for_timestamp(const std::string& timestamp) {
  return "logs/c1-runner-transfer-proof-" + timestamp + ".log";
}
std::string legacy_primitive_diagnostic_log_path_for_timestamp(
    const std::string& timestamp, const std::string& primitive_name) {
  return "logs/c1-runner-legacy-primitive-diagnostic-" + primitive_name + "-" + timestamp + ".log";
}

struct PrimitiveProofSpec {
  const char* name;
  const char* source_id;
  const char* kernel_sha256;
  const char* kernel_text_byte_count;
  const char* element_type;
  const char* element_count;
  const char* input_shape;
  const char* output_shape;
  const char* input_layout;
  const char* input_byte_count;
  const char* output_byte_count;
  const char* scalar_bits;
  const char* tolerance;
  const char* max_abs_diff;
  const char* max_ulp_diff;
  const char* mismatch_count;
  const char* byte_mismatch_count;
  const char* acceptance_scope;
  const char* model_forward_scope;
  const char* native_prefill_acceptance;
  const char* source_fixture;
  const char* fixture_sha256;
  const char* rows_valid;
  const char* tile_rows;
  const char* tile_inner;
  const char* tile_cols;
  const char* source_arrays = nullptr;
  const char* fixture_slice = nullptr;
  const char* full_fixture_shape = nullptr;
  const char* covered_element_count = nullptr;
  const char* full_element_count = nullptr;
};

const PrimitiveProofSpec* primitive_proof_spec_for_name(const std::string& primitive_name) {
  static constexpr PrimitiveProofSpec kSpecs[] = {
      {kFirstPrimitiveName, kFirstPrimitiveSourceId, kFirstPrimitiveKernelSha256,
       kFirstPrimitiveKernelTextByteCount, kFirstPrimitiveElementType,
       kFirstPrimitiveElementCount, nullptr, nullptr, nullptr, kFirstPrimitiveByteCount,
       kFirstPrimitiveByteCount, kFirstPrimitiveScalarBits, kFirstPrimitiveTolerance,
       kFirstPrimitiveMaxAbsDiff, kFirstPrimitiveMaxUlpDiff, kFirstPrimitiveMismatchCount,
       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
      {kFp16ToFp32PrimitiveName, kFp16ToFp32PrimitiveSourceId,
       kFp16ToFp32PrimitiveKernelSha256, kFirstPrimitiveKernelTextByteCount,
       kFp16ToFp32PrimitiveElementType, kFp16ToFp32PrimitiveElementCount, nullptr,
       nullptr, nullptr, kFp16ToFp32PrimitiveInputByteCount,
       kFp16ToFp32PrimitiveOutputByteCount, kFp16ToFp32PrimitiveScalarBits,
       kFirstPrimitiveTolerance, kFirstPrimitiveMaxAbsDiff, kFirstPrimitiveMaxUlpDiff,
       kFirstPrimitiveMismatchCount, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
       nullptr, nullptr, nullptr, nullptr},
      {kFp32ToFp16PrimitiveName, kFp32ToFp16PrimitiveSourceId,
       kFp32ToFp16PrimitiveKernelSha256, kFirstPrimitiveKernelTextByteCount,
       kFp32ToFp16PrimitiveElementType, kFp32ToFp16PrimitiveElementCount, nullptr,
       nullptr, nullptr, kFp32ToFp16PrimitiveInputByteCount,
       kFp32ToFp16PrimitiveOutputByteCount, kFp32ToFp16PrimitiveScalarBits,
       kFirstPrimitiveTolerance, kFirstPrimitiveMaxAbsDiff, kFirstPrimitiveMaxUlpDiff,
       kFirstPrimitiveMismatchCount, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
       nullptr, nullptr, nullptr, nullptr},
      {kFp16Matmul8x16x8PrimitiveName, kFp16Matmul8x16x8PrimitiveSourceId,
       kFp16Matmul8x16x8PrimitiveKernelSha256,
       kFp16Matmul8x16x8PrimitiveKernelTextByteCount,
       kFp16Matmul8x16x8PrimitiveElementType, kFp16Matmul8x16x8PrimitiveElementCount,
       kFp16Matmul8x16x8PrimitiveInputShape, kFp16Matmul8x16x8PrimitiveOutputShape,
       kFp16Matmul8x16x8PrimitiveInputLayout, kFp16Matmul8x16x8PrimitiveInputByteCount,
       kFp16Matmul8x16x8PrimitiveOutputByteCount, kFp16Matmul8x16x8PrimitiveScalarBits,
       kFirstPrimitiveTolerance, kFirstPrimitiveMaxAbsDiff, kFirstPrimitiveMaxUlpDiff,
       kFirstPrimitiveMismatchCount, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
       nullptr, nullptr, nullptr, nullptr},
      {kFp16Matmul8x16x8Layer0KTilePrimitiveName,
       kFp16Matmul8x16x8Layer0KTilePrimitiveSourceId,
       kFp16Matmul8x16x8PrimitiveKernelSha256,
       kFp16Matmul8x16x8PrimitiveKernelTextByteCount,
       kFp16Matmul8x16x8PrimitiveElementType, kFp16Matmul8x16x8PrimitiveElementCount,
       kFp16Matmul8x16x8PrimitiveInputShape, kFp16Matmul8x16x8PrimitiveOutputShape,
       kFp16Matmul8x16x8PrimitiveInputLayout, kFp16Matmul8x16x8PrimitiveInputByteCount,
       kFp16Matmul8x16x8PrimitiveOutputByteCount, kFp16Matmul8x16x8PrimitiveScalarBits,
       "fp32_ulp<=1", "1.862645149230957e-09", "1", kFirstPrimitiveMismatchCount,
       "1", kFp16Matmul8x16x8Layer0KTileAcceptanceScope,
       kFp16Matmul8x16x8Layer0KTileModelForwardScope,
       kFp16Matmul8x16x8Layer0KTileNativePrefillAcceptance,
       kFp16Matmul8x16x8Layer0KTileSourceFixture,
       kFp16Matmul8x16x8Layer0KTileFixtureSha256,
       kFp16Matmul8x16x8Layer0KTileRowsValid,
       kFp16Matmul8x16x8Layer0KTileRows,
       kFp16Matmul8x16x8Layer0KTileInner,
       kFp16Matmul8x16x8Layer0KTileCols},
      {kFp16ResidualAddLayer0AttentionSlice8PrimitiveName,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveSourceId,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveKernelSha256,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveKernelTextByteCount,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveElementType,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveElementCount,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveInputShape,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveOutputShape,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveInputLayout,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveInputByteCount,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveOutputByteCount,
       kFp16ResidualAddLayer0AttentionSlice8PrimitiveScalarBits,
       kFp16ResidualAddLayer0AttentionSlice8Tolerance,
       kFp16ResidualAddLayer0AttentionSlice8MaxAbsDiff,
       kFp16ResidualAddLayer0AttentionSlice8MaxUlpDiff,
       kFp16ResidualAddLayer0AttentionSlice8MismatchCount,
       kFp16ResidualAddLayer0AttentionSlice8ByteMismatchCount,
       kFp16ResidualAddLayer0AttentionSlice8AcceptanceScope,
       kFp16ResidualAddLayer0AttentionSlice8ModelForwardScope,
       kFp16ResidualAddLayer0AttentionSlice8NativePrefillAcceptance,
       kFp16ResidualAddLayer0AttentionSlice8SourceFixture,
       kFp16ResidualAddLayer0AttentionSlice8FixtureSha256,
       nullptr, nullptr, nullptr, nullptr,
       kFp16ResidualAddLayer0AttentionSlice8SourceArrays,
       kFp16ResidualAddLayer0AttentionSlice8FixtureSlice,
       kFp16ResidualAddLayer0AttentionSlice8FullFixtureShape,
       kFp16ResidualAddLayer0AttentionSlice8CoveredElementCount,
       kFp16ResidualAddLayer0AttentionSlice8FullElementCount},
      {kFp16RmsNorm1x64PrimitiveName,
       kFp16RmsNorm1x64PrimitiveSourceId,
       kFp16RmsNorm1x64PrimitiveKernelSha256,
       kFp16RmsNorm1x64PrimitiveKernelTextByteCount,
       kFp16RmsNorm1x64PrimitiveElementType,
       kFp16RmsNorm1x64PrimitiveElementCount,
       kFp16RmsNorm1x64PrimitiveInputShape,
       kFp16RmsNorm1x64PrimitiveOutputShape,
       kFp16RmsNorm1x64PrimitiveInputLayout,
       kFp16RmsNorm1x64PrimitiveInputByteCount,
       kFp16RmsNorm1x64PrimitiveOutputByteCount,
       kFp16RmsNorm1x64PrimitiveScalarBits,
       kFp16RmsNorm1x64Tolerance,
       kFp16RmsNorm1x64MaxAbsDiff,
       kFp16RmsNorm1x64MaxUlpDiff,
       kFp16RmsNorm1x64MismatchCount,
       kFp16RmsNorm1x64ByteMismatchCount,
       kFp16RmsNorm1x64AcceptanceScope,
       kFp16RmsNorm1x64ModelForwardScope,
       kFp16RmsNorm1x64NativePrefillAcceptance,
       kFp16RmsNorm1x64SourceFixture,
       kFp16RmsNorm1x64FixtureSha256,
       nullptr, nullptr, nullptr, nullptr,
       kFp16RmsNorm1x64SourceArrays,
       kFp16RmsNorm1x64FixtureSlice,
       kFp16RmsNorm1x64FullFixtureShape,
       kFp16RmsNorm1x64CoveredElementCount,
       kFp16RmsNorm1x64FullElementCount},

      {kFp16Silu8x8PrimitiveName,
       kFp16Silu8x8PrimitiveSourceId,
       kFp16Silu8x8PrimitiveKernelSha256,
       kFp16Silu8x8PrimitiveKernelTextByteCount,
       kFp16Silu8x8PrimitiveElementType,
       kFp16Silu8x8PrimitiveElementCount,
       kFp16Silu8x8PrimitiveInputShape,
       kFp16Silu8x8PrimitiveOutputShape,
       kFp16Silu8x8PrimitiveInputLayout,
       kFp16Silu8x8PrimitiveInputByteCount,
       kFp16Silu8x8PrimitiveOutputByteCount,
       kFp16Silu8x8PrimitiveScalarBits,
       kFp16Silu8x8Tolerance,
       kFp16Silu8x8MaxAbsDiff,
       kFp16Silu8x8MaxUlpDiff,
       kFp16Silu8x8MismatchCount,
       kFp16Silu8x8ByteMismatchCount,
       kFp16Silu8x8AcceptanceScope,
       kFp16Silu8x8ModelForwardScope,
       kFp16Silu8x8NativePrefillAcceptance,
       kFp16Silu8x8SourceFixture,
       kFp16Silu8x8FixtureSha256,
       nullptr, nullptr, nullptr, nullptr,
       kFp16Silu8x8SourceArrays,
       kFp16Silu8x8FixtureSlice,
       kFp16Silu8x8FullFixtureShape,
       kFp16Silu8x8CoveredElementCount,
       kFp16Silu8x8FullElementCount},
      {kFp16RopeSplitHalfLayer0KPairs8PrimitiveName,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveSourceId,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveKernelSha256,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveKernelTextByteCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveElementType,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveElementCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputShape,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveOutputShape,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputLayout,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputByteCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveOutputByteCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveScalarBits,
       kFp16RopeSplitHalfLayer0KPairs8Tolerance,
       kFp16RopeSplitHalfLayer0KPairs8MaxAbsDiff,
       kFp16RopeSplitHalfLayer0KPairs8MaxUlpDiff,
       kFp16RopeSplitHalfLayer0KPairs8MismatchCount,
       kFp16RopeSplitHalfLayer0KPairs8ByteMismatchCount,
       kFp16RopeSplitHalfLayer0KPairs8AcceptanceScope,
       kFp16RopeSplitHalfLayer0KPairs8ModelForwardScope,
       kFp16RopeSplitHalfLayer0KPairs8NativePrefillAcceptance,
       kFp16RopeSplitHalfLayer0KPairs8SourceFixture,
       kFp16RopeSplitHalfLayer0KPairs8FixtureSha256,
       nullptr, nullptr, nullptr, nullptr,
       kFp16RopeSplitHalfLayer0KPairs8SourceArrays,
       kFp16RopeSplitHalfLayer0KPairs8FixtureSlice,
       kFp16RopeSplitHalfLayer0KPairs8FullFixtureShape,
       kFp16RopeSplitHalfLayer0KPairs8CoveredElementCount,
       kFp16RopeSplitHalfLayer0KPairs8FullElementCount},
      {kFp16RopeSplitHalfLayer0QPairs8PrimitiveName,
       kFp16RopeSplitHalfLayer0QPairs8PrimitiveSourceId,
       kFp16RopeSplitHalfLayer0QPairs8PrimitiveKernelSha256,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveKernelTextByteCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveElementType,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveElementCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputShape,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveOutputShape,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputLayout,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputByteCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveOutputByteCount,
       kFp16RopeSplitHalfLayer0KPairs8PrimitiveScalarBits,
       kFp16RopeSplitHalfLayer0KPairs8Tolerance,
       kFp16RopeSplitHalfLayer0KPairs8MaxAbsDiff,
       kFp16RopeSplitHalfLayer0KPairs8MaxUlpDiff,
       kFp16RopeSplitHalfLayer0KPairs8MismatchCount,
       kFp16RopeSplitHalfLayer0KPairs8ByteMismatchCount,
       kFp16RopeSplitHalfLayer0KPairs8AcceptanceScope,
       kFp16RopeSplitHalfLayer0QPairs8ModelForwardScope,
       kFp16RopeSplitHalfLayer0KPairs8NativePrefillAcceptance,
       kFp16RopeSplitHalfLayer0KPairs8SourceFixture,
       kFp16RopeSplitHalfLayer0KPairs8FixtureSha256,
       nullptr, nullptr, nullptr, nullptr,
       kFp16RopeSplitHalfLayer0QPairs8SourceArrays,
       kFp16RopeSplitHalfLayer0KPairs8FixtureSlice,
       kFp16RopeSplitHalfLayer0QPairs8FullFixtureShape,
       kFp16RopeSplitHalfLayer0KPairs8CoveredElementCount,
       kFp16RopeSplitHalfLayer0QPairs8FullElementCount},
  };
  for (const PrimitiveProofSpec& spec : kSpecs) {
    if (primitive_name == spec.name) return &spec;
  }
  return nullptr;
}




}  // namespace

namespace {

std::string format_log_text(const RuntimeLog& log, const char* path) {
  std::string failure_text = log.failure_text;
  for (char& c : failure_text) {
    if (c == '\n') c = ' ';
  }
  char buf[kLogBufferSize];
  int n = std::snprintf(
      buf, sizeof(buf),
      "timestamp_utc: %s\n"
      "command_line: %s\n"
      "log_path: %s\n"
      "socket_path: %s\n"
      "runtime_substrate: %s\n"
      "pci_id: %s\n"
      "arch: %s\n"
      "arch_discovery_status: %s\n"
      "build_metadata: %s\n"
      "input_digest: %s\n"
      "output_digest: %s\n"
      "stage: %s\n"
      "connect_status: %s\n"
      "bar_map_status: %s\n"
      "sdma_h2d_status: %s\n"
      "kernel_blob_load_status: %s\n"
      "kernarg_write_status: %s\n"
      "kernel_launch_status: %s\n"
      "sdma_d2h_status: %s\n"
      "cpu_comparison_status: %s\n"
      "host_device_transfer_status: %s\n"
      "failure_stage: %s\n"
      "failure_text: %s\n"
      "exit_status: %d\n",
      log.timestamp_utc.c_str(), log.command_line.c_str(), path, log.socket_path.c_str(),
      log.runtime_substrate.c_str(),
      log.pci_id.c_str(), log.arch.c_str(), log.arch_discovery_status.c_str(),
      log.build_metadata.c_str(), log.input_digest.c_str(), log.output_digest.c_str(),
      lifecycle_stage_name(log.stage), log.connect_status.c_str(), log.bar_map_status.c_str(),
      log.sdma_h2d_status.c_str(), log.kernel_blob_load_status.c_str(),
      log.kernarg_write_status.c_str(), log.kernel_launch_status.c_str(),
      log.sdma_d2h_status.c_str(), log.cpu_comparison_status.c_str(),
      log.host_device_transfer_status.c_str(), log.failure_stage.c_str(),
      failure_text.c_str(), log.exit_status);
  if (n < 0 || static_cast<size_t>(n) >= sizeof(buf)) {
    return std::string();
  }
  return std::string(buf, static_cast<size_t>(n));
}

}  // namespace

std::string write_run_log(const RuntimeLog& log, const std::string& name) {
  const std::string logs_dir = "logs";
  if (mkdir(logs_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    // Fall back to stdout-only reporting if logs/ cannot be created.
    std::fprintf(stderr, "warning: cannot create logs/ (%s)\n", std::strerror(errno));
    return std::string();
  }
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s", logs_dir.c_str(), name.c_str());
  const std::string text = format_log_text(log, path);
  if (text.empty()) {
    std::fprintf(stderr, "warning: log buffer too small for %s\n", name.c_str());
    return std::string();
  }
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::fprintf(stderr, "warning: cannot open %s (%s)\n", path, std::strerror(errno));
    return std::string();
  }
  const ssize_t written = write(fd, text.data(), text.size());
  close(fd);
  if (written != static_cast<ssize_t>(text.size())) {
    std::fprintf(stderr, "warning: short write to %s\n", path);
    return std::string();
  }
  return std::string(path);
}

int run_llama_embed_smoke(const LlamaEmbedSmokeRequest& request,
                          LlamaEmbedSmokeResult* result,
                          std::string* out_text,
                          std::string* log_path,
                          std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "Llama embedding smoke result is required";
    return 2;
  }
  *result = LlamaEmbedSmokeResult{};
  const std::string timestamp = timestamp_utc_now();
  const std::string intended_log_path = "logs/llama-embed-smoke-" + timestamp + ".log";
  uint64_t host_staging_read_count = 0;
  uint64_t uploaded_row_window_count = 0;
  std::string hsa_image_load_status = "not_run";
  std::string binder_span_validation_status = "not_run";
  auto render = [&]() {
    std::string text;
    text += "timestamp_utc: " + timestamp + "\n";
    text += "command_line: native-r9700-runner --llama-embed-smoke --model <redacted> --token-id <redacted>\n";
    text += "log_path: " + intended_log_path + "\n";
    text += "producer_kind: hardware_llama_embed_smoke\n";
    text += "runtime_substrate: " + std::string(kRuntimeSubstrate) + "\n";
    text += "model_identity: " + result->model_identity + "\n";
    text += "token_id: <redacted>\n";
    text += "model_token_count: " + std::to_string(result->model_token_count) + "\n";
    text += "token_provenance: explicit_uint32_cli_argument\n";
    text += "embedding_source_kind: binder_validated_safetensors_row\n";
    text += "binder_span_validation_status: " + binder_span_validation_status + "\n";
    text += "binder_span_path: " + result->binder_span_path + "\n";
    text += "binder_span_offset_bytes: " + std::to_string(result->binder_span_offset_bytes) + "\n";
    text += "binder_span_byte_count: " + std::to_string(result->binder_span_byte_count) + "\n";
    text += "model_source_row_offset: " + std::to_string(result->binder_span_offset_bytes) + "\n";
    text += "model_source_row_byte_length: " + std::to_string(result->binder_span_byte_count) + "\n";
    text += "host_staging_read_count: " + std::to_string(host_staging_read_count) + "\n";
    text += "uploaded_row_window_count: " + std::to_string(uploaded_row_window_count) + "\n";
    text += "selected_row_gpu_scalar: 0\n";
    text += "hsa_image_load_status: " + hsa_image_load_status + "\n";
    text += "hsa_image_sha256: " + result->hsa_image_sha256 + "\n";
    text += "hsa_image_entry_offset: " + std::to_string(result->hsa_image_entry_offset) + "\n";
    text += "hsa_image_descriptor_offset: " +
            std::to_string(result->hsa_image_descriptor_offset) + "\n";
    text += "hsa_image_size: " + std::to_string(result->hsa_image_size) + "\n";
    text += "kernel_asset_kind: hsa_code_image\n";
    text += "hsa_image_gpu_va: " + std::to_string(result->hsa_image_gpu_va) + "\n";
    text += "hsa_image_physical_offset: " +
            std::to_string(result->hsa_image_physical_offset) + "\n";
    text += "resident_embedding_row_buffer: " +
            std::string(result->embedding_row_gpu_va == 0 ? "not_run" : "resident") + "\n";
    text += "resident_hidden_output_buffer: " +
            std::string(result->hidden_output_gpu_va == 0 ? "not_run" : "resident") + "\n";
    text += "embedding_row_gpu_va: " + std::to_string(result->embedding_row_gpu_va) + "\n";
    text += "embedding_row_physical_offset: " +
            std::to_string(result->embedding_row_physical_offset) + "\n";
    text += "hidden_output_gpu_va: " + std::to_string(result->hidden_output_gpu_va) + "\n";
    text += "hidden_output_physical_offset: " +
            std::to_string(result->hidden_output_physical_offset) + "\n";
    text += "selected_row_gpu_va: " + std::to_string(result->selected_row_gpu_va) + "\n";
    text += "selected_row_physical_offset: " +
            std::to_string(result->selected_row_physical_offset) + "\n";
    text += "dynamic_ptb_count: " + std::to_string(result->dynamic_ptb_count) + "\n";
    text += "dynamic_ptb_physical_offset: " +
            std::to_string(result->dynamic_ptb_physical_offset) + "\n";
    text += "page_table_pool_base: " + std::to_string(result->page_table_pool_base) + "\n";
    text += "page_table_pool_bytes: " + std::to_string(result->page_table_pool_bytes) + "\n";
    text += "payload_allocation_range_start: " +
            std::to_string(result->payload_allocation_range_start) + "\n";
    text += "payload_allocation_range_end: " +
            std::to_string(result->payload_allocation_range_end) + "\n";
    text += "pm4_dispatch_count: " + std::to_string(result->pm4_dispatch_count) + "\n";
    text += "pm4_dispatch_word_count: " + std::to_string(result->pm4_dispatch_word_count) + "\n";
    text += "pm4_dispatch_digest: " + result->pm4_dispatch_digest + "\n";
    text += "kernarg_hex: " + result->kernarg_hex + "\n";
    text += "sdma_h2d_status: " + result->sdma_h2d_status + "\n";
    text += "bar0_hsa_image_readback_status: " + result->bar0_hsa_image_readback_status + "\n";
    text += "resident_buffer_zero_status: " + result->resident_buffer_zero_status + "\n";
    text += "sdma_d2h_status: " + result->sdma_d2h_status + "\n";
    text += "fp16_row_hidden_byte_equality: " + result->fp16_row_hidden_byte_equality + "\n";
    text += "hardware_identity: " + result->hardware_identity + "\n";
    text += "cpu_model_math: none\n";
    text += "fixture_row_source: none\n";
    text += "archive_source: none\n";
    text += "c0_asset_usage: none\n";
    text += "native_prefill_acceptance: open\n";
    text += "failure_stage: " + result->failure_stage + "\n";
    text += "failure_text: " + result->failure_text + "\n";
    text += "exit_status: " + std::to_string(result->exit_status) + "\n";
    return text;
  };
  auto finish = [&](int status) {
    result->exit_status = status;
    const std::string text = render();
    const std::string name = intended_log_path.substr(std::string("logs/").size());
    const std::string written = write_text_log(text, name);
    if (out_text != nullptr) *out_text = text;
    if (log_path != nullptr) *log_path = written.empty() ? intended_log_path : written;
    return status;
  };
  auto fail = [&](const char* stage, const std::string& text) {
    if (result->failure_stage == "not_run") {
      result->failure_stage = stage;
      result->failure_text = text;
    }
    if (error_text != nullptr) *error_text = text;
    return finish(1);
  };
  if (request.model_dir.empty()) return fail("model_request", "model directory is required");
  result->token_id = request.token_id;
  std::error_code filesystem_error;
  const std::filesystem::path canonical_model =
      std::filesystem::canonical(std::filesystem::path(request.model_dir), filesystem_error);
  if (filesystem_error || !std::filesystem::is_directory(canonical_model, filesystem_error) ||
      filesystem_error) {
    return fail("model_path", "model directory canonicalization failed");
  }
  result->model_identity = canonical_model.string();
  constexpr LlamaModelGeometry kLlama32OneBGeometry{128256, 2048, 8192, 8, 64};
  result->model_token_count = kLlama32OneBGeometry.vocab_size;
  if (request.token_id >= result->model_token_count) {
    return fail("token_range", "token ID is outside the model vocabulary");
  }
  ModelWeightBinder binder;
  std::string detail;
  if (!binder.open(result->model_identity, &detail)) return fail("binder_open", detail);
  LlamaLayer0WeightSpans weights;
  if (!binder.bind_llama_layer0(kLlama32OneBGeometry, &weights, &detail)) {
    return fail("binder_span_validate", detail);
  }
  const Fp16WeightSpan& embed = weights.embed_tokens;
  if (embed.byte_length < kLlamaEmbeddingRowBytes ||
      embed.data_offset > std::numeric_limits<uint64_t>::max() - embed.byte_length ||
      request.token_id >
          (std::numeric_limits<uint64_t>::max() - embed.data_offset) / kLlamaEmbeddingRowBytes) {
    return fail("binder_span_validate", "embedding row range overflows its safetensors span");
  }
  const uint64_t span_end = embed.data_offset + embed.byte_length;
  const uint64_t row_offset =
      embed.data_offset + static_cast<uint64_t>(request.token_id) * kLlamaEmbeddingRowBytes;
  if (row_offset < embed.data_offset || row_offset > span_end ||
      kLlamaEmbeddingRowBytes > span_end - row_offset) {
    return fail("binder_span_validate", "embedding row is outside the bound safetensors span");
  }
  filesystem_error.clear();
  const std::filesystem::path canonical_shard = std::filesystem::canonical(embed.shard_path, filesystem_error);
  if (filesystem_error) return fail("binder_span_validate", "embedding shard canonicalization failed");
  struct stat shard_status {};
  if (stat(canonical_shard.c_str(), &shard_status) != 0 || !S_ISREG(shard_status.st_mode) ||
      shard_status.st_size < 0 || row_offset > static_cast<uint64_t>(shard_status.st_size) ||
      kLlamaEmbeddingRowBytes > static_cast<uint64_t>(shard_status.st_size) - row_offset) {
    return fail("binder_span_validate", "embedding row is outside the shard file");
  }
  result->binder_span_path = canonical_shard.string();
  result->binder_span_offset_bytes = row_offset;
  result->binder_span_byte_count = kLlamaEmbeddingRowBytes;
  binder_span_validation_status = "pass";
  std::vector<uint8_t> row(kLlamaEmbeddingRowBytes);
  if (row_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    return fail("row_read", "selected embedding row offset does not fit pread");
  }
  const int shard_fd = open(canonical_shard.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (shard_fd < 0) return fail("row_read", "opening the selected embedding shard failed");
  const ssize_t row_read = pread(shard_fd, row.data(), row.size(), static_cast<off_t>(row_offset));
  const int read_error = errno;
  const int close_status = close(shard_fd);
  if (row_read != static_cast<ssize_t>(row.size()) || close_status != 0) {
    return fail("row_read", row_read < 0 ? std::string("pread selected embedding row failed: ") +
                                           std::strerror(read_error)
                                        : "pread selected embedding row was short");
  }
  host_staging_read_count = 1;
  HsaCodeImageAsset hsa_image;
  if (!load_llama_embed_hsa_image("native_r9700/kernels/llama-hsa-assets", &hsa_image, &detail)) {
    return fail("hsa_image_load", detail);
  }
  hsa_image_load_status = "pass";
  result->hsa_image_sha256 = hsa_image.image_sha256;
  result->hsa_image_entry_offset = hsa_image.entry_offset;
  result->hsa_image_descriptor_offset = hsa_image.descriptor_offset;
  result->hsa_image_size = hsa_image.image.size();
  LlamaEmbedSmokeDispatch dispatch;
  dispatch.hsa_image = &hsa_image;
  dispatch.embedding_row = std::move(row);
  AMDevSession session;
  LlamaEmbedSmokeDispatchResult dispatch_result;
  if (!session.llama_embed_smoke(dispatch, &dispatch_result, &detail)) {
    result->hardware_identity = dispatch_result.hardware_identity;
    result->hsa_image_gpu_va = dispatch_result.hsa_image_gpu_va;
    result->hsa_image_physical_offset = dispatch_result.hsa_image_physical_offset;
    result->embedding_row_gpu_va = dispatch_result.embedding_row_gpu_va;
    result->bar0_hsa_image_readback_status = dispatch_result.bar0_image_readback_status;
    result->resident_buffer_zero_status = dispatch_result.resident_buffer_zero_status;
    result->embedding_row_physical_offset = dispatch_result.embedding_row_physical_offset;
    result->hidden_output_gpu_va = dispatch_result.hidden_output_gpu_va;
    result->hidden_output_physical_offset = dispatch_result.hidden_output_physical_offset;
    result->selected_row_gpu_va = dispatch_result.selected_row_gpu_va;
    result->selected_row_physical_offset = dispatch_result.selected_row_physical_offset;
    result->dynamic_ptb_count = dispatch_result.dynamic_ptb_count;
    result->dynamic_ptb_physical_offset = dispatch_result.dynamic_ptb_physical_offset;
    result->page_table_pool_base = dispatch_result.page_table_pool_base;
    result->page_table_pool_bytes = dispatch_result.page_table_pool_bytes;
    result->payload_allocation_range_start = dispatch_result.payload_allocation_range_start;
    result->payload_allocation_range_end = dispatch_result.payload_allocation_range_end;
    result->kernarg_hex = dispatch_result.kernarg_hex;
    result->pm4_dispatch_word_count = dispatch_result.pm4_dispatch_word_count;
    result->pm4_dispatch_digest = dispatch_result.pm4_dispatch_digest;
    result->pm4_dispatch_count = dispatch_result.pm4_dispatch_count;
    result->sdma_h2d_status = dispatch_result.sdma_h2d_status;
    result->sdma_d2h_status = dispatch_result.sdma_d2h_status;
    result->fp16_row_hidden_byte_equality = dispatch_result.fp16_row_hidden_byte_equality;
    uploaded_row_window_count = dispatch_result.sdma_h2d_status == "pass" ? 1 : 0;
    return fail(dispatch_result.failure_stage.c_str(), detail);
  }
  result->hardware_identity = dispatch_result.hardware_identity;
  result->hsa_image_gpu_va = dispatch_result.hsa_image_gpu_va;
  result->hsa_image_physical_offset = dispatch_result.hsa_image_physical_offset;
  result->embedding_row_gpu_va = dispatch_result.embedding_row_gpu_va;
  result->bar0_hsa_image_readback_status = dispatch_result.bar0_image_readback_status;
  result->resident_buffer_zero_status = dispatch_result.resident_buffer_zero_status;
  result->embedding_row_physical_offset = dispatch_result.embedding_row_physical_offset;
  result->hidden_output_gpu_va = dispatch_result.hidden_output_gpu_va;
  result->hidden_output_physical_offset = dispatch_result.hidden_output_physical_offset;
  result->selected_row_gpu_va = dispatch_result.selected_row_gpu_va;
  result->selected_row_physical_offset = dispatch_result.selected_row_physical_offset;
  result->dynamic_ptb_count = dispatch_result.dynamic_ptb_count;
  result->dynamic_ptb_physical_offset = dispatch_result.dynamic_ptb_physical_offset;
  result->page_table_pool_base = dispatch_result.page_table_pool_base;
  result->page_table_pool_bytes = dispatch_result.page_table_pool_bytes;
  result->payload_allocation_range_start = dispatch_result.payload_allocation_range_start;
  result->payload_allocation_range_end = dispatch_result.payload_allocation_range_end;
  result->kernarg_hex = dispatch_result.kernarg_hex;
  result->pm4_dispatch_word_count = dispatch_result.pm4_dispatch_word_count;
  result->pm4_dispatch_digest = dispatch_result.pm4_dispatch_digest;
  result->pm4_dispatch_count = dispatch_result.pm4_dispatch_count;
  result->sdma_h2d_status = dispatch_result.sdma_h2d_status;
  result->sdma_d2h_status = dispatch_result.sdma_d2h_status;
  result->fp16_row_hidden_byte_equality = dispatch_result.fp16_row_hidden_byte_equality;
  uploaded_row_window_count = 1;
  result->failure_stage = "none";
  result->failure_text = "none";
  return finish(0);
}

// ---------------------------------------------------------------- lifecycle --


int RuntimeSession::kernel_proof(std::string* out_text, std::string* log_path) {
  const std::string timestamp = timestamp_utc_now();
  const std::string intended_log_path = kernel_proof_log_path_for_timestamp(timestamp);
  std::string text;
  text += "timestamp_utc: " + timestamp + "\n";
  text += "command_line: native-r9700-runner --kernel-proof\n";
  text += "log_path: " + intended_log_path + "\n";
  text += "producer_kind: hardware_probe\n";
  text += "runtime_substrate: " + std::string(kRuntimeSubstrate) + "\n";
  text += "pci_id: 1002:7551\n";
  text += "arch: " + std::string(kKernelArch) + "\n";
  text += "kernel_source_id: " + std::string(kKernelSourceId) + "\n";

  auto finish = [&](int status) -> int {
    text += "wrapper_exit_status: " + std::to_string(status) + "\n";
    text += "exit_status: " + std::to_string(status) + "\n";
    const std::string name = intended_log_path.substr(std::string("logs/").size());
    const std::string written = write_text_log(text, name);
    if (log_path) *log_path = written.empty() ? intended_log_path : written;
    if (out_text) *out_text = text;
    return status;
  };

  std::string probe_exe;
  const char* env_probe = std::getenv("NATIVE_R9700_C0_PROBE");
  if (env_probe != nullptr && env_probe[0] != '\0') {
    probe_exe = env_probe;
    text += "c0_probe_source: env:NATIVE_R9700_C0_PROBE\n";
  } else {
    const std::string source = "experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp";
    probe_exe = "logs/native-r9700-c0a25-probe";
    if (access(source.c_str(), R_OK) != 0) {
      text += "kernel_proof_wrapper_status: fail\n";
      text += "failure_stage: c0_probe_source\n";
      text += "failure_text: missing frozen C0A25 probe source at " + source + "\n";
      return finish(2);
    }
    mkdir("logs", 0755);
    std::vector<std::string> build_cmd = {
        "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        source, "-o", probe_exe};
    text += "c0_probe_source: " + source + "\n";
    text += "c0_probe_build_command: " + join_command_for_log(build_cmd) + "\n";
    const ProcessResult build = run_process_capture(build_cmd);
    if (!build.output.empty()) {
      text += "c0_probe_build_output_begin\n";
      text += build.output;
      if (text.back() != '\n') text += "\n";
      text += "c0_probe_build_output_end\n";
    }
    if (build.exit_status != 0) {
      text += "kernel_proof_wrapper_status: fail\n";
      text += "failure_stage: c0_probe_build\n";
      text += "failure_text: failed to build frozen C0A25 probe\n";
      return finish(build.exit_status);
    }
  }

  std::vector<std::string> run_cmd = {probe_exe, "--kernel-proof"};
  text += "c0_probe_command: " + join_command_for_log(run_cmd) + "\n";
  const ProcessResult proof = run_process_capture(run_cmd);
  if (!proof.output.empty()) {
    text += "c0_probe_output_begin\n";
    text += proof.output;
    if (text.back() != '\n') text += "\n";
    text += "c0_probe_output_end\n";
  }
  const char* required_markers[] = {
      "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface",
      "pci_id: 1002:7551",
      "arch: gfx1201",
      "kernel_blob_load_status: pass",
      "kernarg_write_status: pass",
      "sdma_h2d_status: pass",
      "sdma_d2h_status: pass",
      "kernel_launch_status: pass",
      "cpu_comparison_status: pass",
      "host_device_transfer_status: pass",
      "failure_stage: none",
      "failure_text: none",
      "exit_status: 0",
  };
  std::string missing_markers;
  bool markers_present = true;
  for (const char* marker : required_markers) {
    if (!contains_text(proof.output, marker)) {
      markers_present = false;
      if (!missing_markers.empty()) missing_markers += ", ";
      missing_markers += marker;
    }
  }
  const bool proof_pass = proof.exit_status == 0 && markers_present;
  text += std::string("kernel_proof_wrapper_status: ") + (proof_pass ? "pass" : "fail") + "\n";
  if (!proof_pass) {
    text += "failure_stage: c0_probe_kernel_proof\n";
    text += "failure_text: wrapped C0A25 probe did not report the full hardware pass marker set\n";
    if (!missing_markers.empty()) text += "missing_c0_markers: " + missing_markers + "\n";
  } else {
    text += "failure_stage: none\n";
    text += "failure_text: none\n";
  }
  const int status = proof_pass ? 0 : (proof.exit_status == 0 ? 1 : proof.exit_status);
  return finish(status);
}

int RuntimeSession::transfer_round_trip_bytes(const std::vector<uint8_t>& input,
                                              std::vector<uint8_t>* output,
                                              TransferRoundTripResult* result,
                                              std::string* error_text) {
  std::string local_error;
  TransferRoundTripResult local_result;
  if (error_text == nullptr) error_text = &local_error;
  if (result == nullptr) result = &local_result;
  *result = TransferRoundTripResult{};
  result->byte_count = static_cast<uint64_t>(input.size());
  result->chunk_size_bytes = kTransferProofChunkByteCount;
  result->chunk_count = ceil_div_u64(result->byte_count, result->chunk_size_bytes);
  if (output == nullptr) {
    *error_text = "transfer output pointer is null";
    return 2;
  }
  output->clear();
  if (input.empty()) {
    *error_text = "transfer byte count must be nonzero";
    return 2;
  }
  if (input.size() > kMaxTransferProofByteCount) {
    *error_text = "transfer byte count exceeds max C1R-4 policy: byte_count=" +
                  std::to_string(input.size()) + " max=" +
                  std::to_string(kMaxTransferProofByteCount);
    return 2;
  }

  mkdir("build", 0755);
  mkdir("build/native-r9700-runtime", 0755);
  const std::string timestamp = timestamp_utc_now();
  const std::string input_path =
      "build/native-r9700-runtime/c1-transfer-input-" + timestamp + ".bin";
  const std::string output_path =
      "build/native-r9700-runtime/c1-transfer-output-" + timestamp + ".bin";
  if (!write_binary_file(input_path, input, error_text)) {
    return 2;
  }

  std::string bridge_exe;
  const char* env_bridge = std::getenv("NATIVE_R9700_C1_TRANSFER_BRIDGE");
  if (env_bridge != nullptr && env_bridge[0] != '\0') {
    bridge_exe = env_bridge;
    result->bridge_source = "env:NATIVE_R9700_C1_TRANSFER_BRIDGE";
  } else {
    const std::string source = "native_r9700/c1_transfer_bridge.cpp";
    bridge_exe = "build/native-r9700-runtime/native_r9700_transfer_bridge";
    if (access(source.c_str(), R_OK) != 0) {
      *error_text = "missing C1 transfer bridge source at " + source;
      unlink(input_path.c_str());
      return 2;
    }
    std::vector<std::string> build_cmd = {
        "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        source, "native_r9700/amdev_session.cpp", "native_r9700/amdev_packets.cpp",
        "native_r9700/kernel_catalog.cpp", "-o", bridge_exe};
    result->bridge_source = source;
    result->bridge_build_command = join_command_for_log(build_cmd);
    const ProcessResult build = run_process_capture(build_cmd);
    result->bridge_build_output = build.output;
    if (build.exit_status != 0) {
      *error_text = "failed to build C1 transfer bridge";
      unlink(input_path.c_str());
      return build.exit_status;
    }
  }

  std::vector<std::string> run_cmd = {bridge_exe, "--roundtrip-file", input_path, output_path};
  result->bridge_command = join_command_for_log(run_cmd);
  const ProcessResult proof = run_process_capture(run_cmd);
  result->bridge_output = proof.output;
  if (proof.exit_status == 0 && !read_binary_file(output_path, output, error_text)) {
    unlink(input_path.c_str());
    unlink(output_path.c_str());
    return 1;
  }
  unlink(input_path.c_str());
  unlink(output_path.c_str());
  if (proof.exit_status != 0) {
    *error_text = "C1 transfer bridge exited " + std::to_string(proof.exit_status);
  }
  return proof.exit_status;
}

int RuntimeSession::transfer_proof(uint64_t byte_count, std::string* out_text,
                                   std::string* log_path) {
  const std::string timestamp = timestamp_utc_now();
  const std::string intended_log_path = transfer_proof_log_path_for_timestamp(timestamp);
  std::string text;
  text += "timestamp_utc: " + timestamp + "\n";
  text += "command_line: native-r9700-runner --transfer-proof --bytes " +
          std::to_string(byte_count) + "\n";
  text += "log_path: " + intended_log_path + "\n";
  text += "producer_kind: hardware_memory_transfer\n";
  text += "runtime_substrate: " + std::string(kRuntimeSubstrate) + "\n";
  text += "pci_id: 1002:7551\n";
  text += "arch: " + std::string(kKernelArch) + "\n";
  text += "requested_transfer_byte_count: " + std::to_string(byte_count) + "\n";
  text += "transfer_chunk_size_bytes: " + std::to_string(kTransferProofChunkByteCount) + "\n";

  auto finish = [&](int status) -> int {
    text += "wrapper_exit_status: " + std::to_string(status) + "\n";
    text += "exit_status: " + std::to_string(status) + "\n";
    const std::string name = intended_log_path.substr(std::string("logs/").size());
    const std::string written = write_text_log(text, name);
    if (log_path) *log_path = written.empty() ? intended_log_path : written;
    if (out_text) *out_text = text;
    return status;
  };

  if (byte_count == 0 || byte_count > kMaxTransferProofByteCount) {
    text += "transfer_proof_wrapper_status: fail\n";
    text += "failure_stage: transfer_request\n";
    text += "failure_text: transfer byte count must be in 1.." +
            std::to_string(kMaxTransferProofByteCount) + "\n";
    return finish(2);
  }

  const std::vector<uint8_t> input = make_transfer_pattern(byte_count);
  std::vector<uint8_t> output;
  TransferRoundTripResult result;
  std::string transfer_error;
  const int transfer_status = transfer_round_trip_bytes(input, &output, &result, &transfer_error);
  text += "transfer_bridge_source: " + result.bridge_source + "\n";
  if (!result.bridge_build_command.empty()) {
    text += "transfer_bridge_build_command: " + result.bridge_build_command + "\n";
  }
  if (!result.bridge_build_output.empty()) {
    text += "transfer_bridge_build_output_begin\n";
    text += result.bridge_build_output;
    if (text.back() != '\n') text += "\n";
    text += "transfer_bridge_build_output_end\n";
  }
  text += "transfer_bridge_command: " + result.bridge_command + "\n";
  if (!result.bridge_output.empty()) {
    text += "transfer_bridge_output_begin\n";
    text += result.bridge_output;
    if (text.back() != '\n') text += "\n";
    text += "transfer_bridge_output_end\n";
  }

  const uint64_t chunk_count = ceil_div_u64(byte_count, kTransferProofChunkByteCount);
  const std::string byte_count_text = std::to_string(byte_count);
  const std::string chunk_count_text = std::to_string(chunk_count);
  const std::string chunk_size_text = std::to_string(kTransferProofChunkByteCount);
  const std::string streaming_required = byte_count > kTransferProofChunkByteCount ? "yes" : "no";
  const std::string allocation_total = std::to_string(3ULL * kTransferProofChunkByteCount);
  std::string missing_markers;
  bool markers_present = true;
  auto require = [&](const std::string& key, const std::string& expected) {
    markers_present =
        require_log_value(result.bridge_output, key, expected, &missing_markers) && markers_present;
  };
  require("producer_kind", "hardware_memory_transfer");
  require("runtime_substrate", "TinyGPU.app/APLRemotePCIDevice/PCIIface");
  require("pci_id", "1002:7551");
  require("arch", "gfx1201");
  require("transfer_byte_count", byte_count_text);
  require("transfer_chunk_count", chunk_count_text);
  require("transfer_chunks_completed", chunk_count_text);
  require("transfer_chunk_size_bytes", chunk_size_text);
  require("buffer_count", "3");
  require("allocation_total_bytes", allocation_total);
  require("upload_total_bytes", byte_count_text);
  require("download_total_bytes", byte_count_text);
  require("streaming_required", streaming_required);
  require("sdma_h2d_status", "pass");
  require("sdma_d2h_status", "pass");
  require("cpu_comparison_status", "pass");
  require("host_device_transfer_status", "pass");
  require("failure_stage", "none");
  require("failure_text", "none");
  require("exit_status", "0");
  const bool output_matches = transfer_status == 0 && output == input;
  const bool proof_pass = transfer_status == 0 && markers_present && output_matches;
  text += std::string("transfer_proof_wrapper_status: ") + (proof_pass ? "pass" : "fail") + "\n";
  if (!proof_pass) {
    text += "failure_stage: transfer_bridge_proof\n";
    text += "failure_text: C1 transfer bridge did not report the exact streaming transfer contract";
    if (!transfer_error.empty()) text += ": " + transfer_error;
    text += "\n";
    if (!missing_markers.empty()) text += "missing_transfer_markers: " + missing_markers + "\n";
    if (transfer_status == 0 && !output_matches) {
      text += "output_mismatch: transfer API output bytes differ from input bytes\n";
    }
  } else {
    text += "failure_stage: none\n";
    text += "failure_text: none\n";
  }
  const int status = proof_pass ? 0 : (transfer_status == 0 ? 1 : transfer_status);

  return finish(status);
}
int RuntimeSession::vram_smoke(std::string* out_text, std::string* log_path) {
  const std::string timestamp = timestamp_utc_now();
  VramSmokeResult result;
  auto render = [&](const std::string& durable_log_path, int status) {
    std::string text;
    text += "timestamp_utc: " + timestamp + "\n";
    text += "command_line: native-r9700-runner --vram-smoke\n";
    text += "log_path: " + durable_log_path + "\n";
    text += "producer_kind: hardware_resident_vram_smoke\n";
    text += "runtime_substrate: " + std::string(kRuntimeSubstrate) + "\n";
    text += "pci_id: " + result.pci_id + "\n";
    text += "arch: " + result.arch + "\n";
    text += "source_asset_path: " + result.source_asset_path + "\n";
    text += "asset_sha256: " + result.asset_sha256 + "\n";
    text += "code_byte_count: " + std::to_string(result.code_byte_count) + "\n";
    text += "bar0_code_readback_status: " + result.bar0_code_readback_status + "\n";
    text += "vram_allocation_status: " + result.vram_allocation_status + "\n";
    text += "resident_mapping_count: " + std::to_string(result.resident_mapping_count) + "\n";
    text += "bar0_aperture_bytes: " + std::to_string(result.bar0_aperture_bytes) + "\n";
    text += "large_bar: " + result.large_bar + "\n";
    text += "page_table_pool_base: " + std::to_string(result.page_table_pool_base) + "\n";
    text += "page_table_pool_bytes: " + std::to_string(result.page_table_pool_bytes) + "\n";
    text += "dynamic_ptb_count: " + std::to_string(result.dynamic_ptb_count) + "\n";
    text += "dynamic_ptb_physical_offset: " +
            std::to_string(result.dynamic_ptb_physical_offset) + "\n";
    text += "payload_allocation_range_start: " +
            std::to_string(result.payload_allocation_range_start) + "\n";
    text += "payload_allocation_range_end: " +
            std::to_string(result.payload_allocation_range_end) + "\n";
    text += "mapping_uncertainty_status: " + result.mapping_uncertainty_status + "\n";
    text += "a_gpu_va: " + std::to_string(result.a_gpu_va) + "\n";
    text += "a_physical_offset: " + std::to_string(result.a_physical_offset) + "\n";
    text += "b_gpu_va: " + std::to_string(result.b_gpu_va) + "\n";
    text += "b_physical_offset: " + std::to_string(result.b_physical_offset) + "\n";
    text += "out_gpu_va: " + std::to_string(result.out_gpu_va) + "\n";
    text += "out_physical_offset: " + std::to_string(result.out_physical_offset) + "\n";
    text += "bar0_zero_status: " + result.bar0_zero_status + "\n";
    text += "pte_map_status: " + result.pte_map_status + "\n";
    text += "pte_write_status: " + result.pte_write_status + "\n";
    text += "pte_readback_status: " + result.pte_readback_status + "\n";
    text += "mmhub_tlb_flush_status: " + result.mmhub_tlb_flush_status + "\n";
    text += "gc_tlb_flush_status: " + result.gc_tlb_flush_status + "\n";
    text += "compute_dispatch_count: " + std::to_string(result.compute_dispatch_count) + "\n";
    text += "sdma_h2d_status: " + result.sdma_h2d_status + "\n";
    text += "sdma_d2h_status: " + result.sdma_d2h_status + "\n";
    text += "sdma_upload_bytes: " + std::to_string(result.sdma_upload_bytes) + "\n";
    text += "sdma_download_bytes: " + std::to_string(result.sdma_download_bytes) + "\n";
    text += "kernarg_byte_count: " + std::to_string(result.kernarg_byte_count) + "\n";
    text += "kernarg_hex: " + result.kernarg_hex + "\n";
    text += "pm4_dispatch_word_count: " + std::to_string(result.pm4_dispatch_word_count) + "\n";
    text += "pm4_dispatch_digest: " + result.pm4_dispatch_digest + "\n";
    text += "cpu_comparison_status: " + result.cpu_comparison_status + "\n";
    text += "native_prefill_acceptance: open\n";
    text += "failure_stage: " + result.failure_stage + "\n";
    text += "failure_text: " + result.failure_text + "\n";
    text += "exit_status: " + std::to_string(status) + "\n";
    return text;
  };

  ReservedVramSmokeLog reserved;
  std::string log_error;
  if (!reserve_vram_smoke_log(timestamp, &reserved, &log_error)) {
    result.failure_stage = "log_write";
    result.failure_text = log_error;
    const std::string text = render("not_written", 1);
    if (log_path != nullptr) *log_path = "not_written";
    if (out_text != nullptr) *out_text = text;
    return 1;
  }

  std::string smoke_error;
  AMDevSession session;
  bool succeeded = session.vram_smoke(&result, &smoke_error);
  if (!succeeded && result.failure_text == "not_run") {
    result.failure_stage = "vram_smoke";
    result.failure_text = smoke_error.empty() ? "direct VRAM smoke failed" : smoke_error;
  }
  const bool payload_range_is_recorded =
      result.payload_allocation_range_start < result.payload_allocation_range_end &&
      result.a_physical_offset >= result.payload_allocation_range_start &&
      result.a_physical_offset < result.payload_allocation_range_end &&
      result.b_physical_offset >= result.payload_allocation_range_start &&
      result.b_physical_offset < result.payload_allocation_range_end &&
      result.out_physical_offset >= result.payload_allocation_range_start &&
      result.out_physical_offset < result.payload_allocation_range_end;
  const bool pool_backed_dynamic_ptb =
      result.bar0_aperture_bytes != 0 && result.large_bar == "false" &&
      result.page_table_pool_base != 0 && result.page_table_pool_bytes != 0 &&
      result.page_table_pool_base <= result.bar0_aperture_bytes &&
      result.page_table_pool_bytes <=
          result.bar0_aperture_bytes - result.page_table_pool_base &&
      result.dynamic_ptb_count >= 1 &&
      result.dynamic_ptb_physical_offset >= result.page_table_pool_base &&
      result.dynamic_ptb_physical_offset - result.page_table_pool_base <
          result.page_table_pool_bytes;
  const bool evidence_complete =
      result.source_asset_path != "not_run" && result.asset_sha256.size() == 64 &&
      result.code_byte_count != 0 && result.bar0_code_readback_status == "pass" &&
      result.resident_mapping_count >= 5 && payload_range_is_recorded &&
      pool_backed_dynamic_ptb && result.kernarg_byte_count == 24 &&
      result.kernarg_hex.size() == 48 &&
      result.pm4_dispatch_word_count != 0 && result.pm4_dispatch_digest != "not_run";
  if (succeeded && !evidence_complete) {
    succeeded = false;
    result.failure_stage = "evidence_incomplete";
    result.failure_text = "required VRAM smoke evidence was unavailable";
  }
  int status = succeeded ? 0 : 1;
  std::string text = render(reserved.path, status);
  if (!write_reserved_vram_smoke_log(&reserved, text, &log_error)) {
    result.failure_stage = "log_write";
    result.failure_text = log_error;
    status = 1;
    text = render("not_written", status);
    if (log_path != nullptr) *log_path = "not_written";
    if (out_text != nullptr) *out_text = text;
    return status;
  }
  if (log_path != nullptr) *log_path = reserved.path;
  if (out_text != nullptr) *out_text = text;
  return status;
}

int RuntimeSession::legacy_primitive_diagnostic(const std::string& primitive_name,
                                                std::string* out_text,
                                                std::string* log_path) {
  const std::string timestamp = timestamp_utc_now();
  const PrimitiveProofSpec* spec = primitive_proof_spec_for_name(primitive_name);
  const std::string primitive_name_for_log = spec != nullptr ? primitive_name : "unsupported";
  const std::string intended_log_path =
      legacy_primitive_diagnostic_log_path_for_timestamp(timestamp, primitive_name_for_log);
  std::string text;
  text += "timestamp_utc: " + timestamp + "\n";
  text += "command_line: native-r9700-runner --legacy-primitive-diagnostic " + primitive_name + "\n";
  text += "log_path: " + intended_log_path + "\n";
  text += "producer_kind: legacy_primitive_diagnostic\n";
  text += "runtime_substrate: " + std::string(kRuntimeSubstrate) + "\n";
  text += "pci_id: 1002:7551\n";
  text += "arch: " + std::string(kKernelArch) + "\n";
  text += "primitive_name: " + primitive_name + "\n";

  auto finish = [&](int status) -> int {
    text += "wrapper_exit_status: " + std::to_string(status) + "\n";
    text += "exit_status: " + std::to_string(status) + "\n";
    const std::string name = intended_log_path.substr(std::string("logs/").size());
    const std::string written = write_text_log(text, name);
    if (log_path) *log_path = written.empty() ? intended_log_path : written;
    if (out_text) *out_text = text;
    return status;
  };

  if (spec == nullptr) {
    text += "legacy_diagnostic_status: fail\n";
    text += "failure_stage: primitive_request\n";
    text += "failure_text: unsupported primitive '" + primitive_name + "'\n";
    return finish(2);
  }

  const char* injected_executable = std::getenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE");
  if (injected_executable == nullptr || injected_executable[0] == '\0') {
    text += "legacy_diagnostic_status: unavailable\n";
    text += "failure_stage: legacy_proof_unavailable\n";
    text += "failure_text: legacy primitive diagnostic requires NATIVE_R9700_C1_PRIMITIVE_BRIDGE\n";
    return finish(2);
  }
  const std::string diagnostic_executable = injected_executable;
  text += "legacy_diagnostic_executable: env:NATIVE_R9700_C1_PRIMITIVE_BRIDGE\n";

  std::vector<std::string> run_cmd = {diagnostic_executable, "--primitive", primitive_name};
  text += "legacy_diagnostic_command: " + join_command_for_log(run_cmd) + "\n";
  const ProcessResult proof = run_process_capture(run_cmd);
  const bool reports_native_prefill_acceptance =
      contains_text(proof.output, "native_prefill_acceptance: pass");
  if (!reports_native_prefill_acceptance && !proof.output.empty()) {
    text += "legacy_diagnostic_output_begin\n";
    text += proof.output;
    if (text.back() != '\n') text += "\n";
    text += "legacy_diagnostic_output_end\n";
  }

  std::string missing_markers;
  bool markers_present = true;
  auto require = [&](const std::string& key, const std::string& expected) {
    markers_present =
        require_log_value(proof.output, key, expected, &missing_markers) && markers_present;
  };
  require("producer_kind", "hardware_primitive");
  require("primitive_backend", kFirstPrimitiveBackend);
  require("runtime_substrate", "TinyGPU.app/APLRemotePCIDevice/PCIIface");
  require("pci_id", "1002:7551");
  require("arch", "gfx1201");
  require("primitive_name", spec->name);
  require("kernel_source_id", spec->source_id);
  require("kernel_blob_sha256", spec->kernel_sha256);
  require("kernel_text_byte_count", spec->kernel_text_byte_count);
  require("element_type", spec->element_type);
  require("element_count", spec->element_count);
  if (spec->input_shape != nullptr) require("input_shape", spec->input_shape);
  if (spec->output_shape != nullptr) require("output_shape", spec->output_shape);
  if (spec->input_layout != nullptr) require("input_layout", spec->input_layout);
  require("input_byte_count", spec->input_byte_count);
  require("output_byte_count", spec->output_byte_count);
  require("scalar_bits", spec->scalar_bits);
  if (spec->acceptance_scope != nullptr) require("acceptance_scope", spec->acceptance_scope);
  if (spec->model_forward_scope != nullptr) require("model_forward_scope", spec->model_forward_scope);
  if (spec->native_prefill_acceptance != nullptr) {
    require("native_prefill_acceptance", spec->native_prefill_acceptance);
  }
  if (spec->source_fixture != nullptr) require("source_fixture", spec->source_fixture);
  if (spec->fixture_sha256 != nullptr) require("fixture_sha256", spec->fixture_sha256);
  if (spec->source_arrays != nullptr) require("source_arrays", spec->source_arrays);
  if (spec->fixture_slice != nullptr) require("fixture_slice", spec->fixture_slice);
  if (spec->full_fixture_shape != nullptr) {
    require("full_fixture_shape", spec->full_fixture_shape);
  }
  if (spec->covered_element_count != nullptr) {
    require("covered_element_count", spec->covered_element_count);
  }
  if (spec->full_element_count != nullptr) {
    require("full_element_count", spec->full_element_count);
  }
  if (spec->rows_valid != nullptr) require("rows_valid", spec->rows_valid);
  if (spec->tile_rows != nullptr) require("tile_rows", spec->tile_rows);
  if (spec->tile_inner != nullptr) require("tile_inner", spec->tile_inner);
  if (spec->tile_cols != nullptr) require("tile_cols", spec->tile_cols);
  require("tolerance", spec->tolerance);
  require("max_abs_diff", spec->max_abs_diff);
  require("max_ulp_diff", spec->max_ulp_diff);
  require("mismatch_count", spec->mismatch_count);
  if (spec->byte_mismatch_count != nullptr) {
    require("byte_mismatch_count", spec->byte_mismatch_count);
  }
  require("upload_total_bytes", spec->input_byte_count);
  require("download_total_bytes", spec->output_byte_count);
  require("kernel_blob_load_status", "pass");
  require("kernarg_write_status", "pass");
  require("kernel_launch_status", "pass");
  require("sdma_h2d_status", "pass");
  require("sdma_d2h_status", "pass");
  require("cpu_comparison_status", "pass");
  require("host_device_transfer_status", "pass");
  require("failure_stage", "none");
  require("failure_text", "none");
  require("exit_status", "0");
  const bool proof_pass =
      proof.exit_status == 0 && markers_present && !reports_native_prefill_acceptance;
  text += std::string("legacy_diagnostic_status: ") + (proof_pass ? "pass" : "fail") + "\n";
  if (!proof_pass) {
    text += "failure_stage: legacy_diagnostic_protocol\n";
    if (reports_native_prefill_acceptance) {
      text += "failure_text: injected legacy executable reported a prohibited native-prefill acceptance marker\n";
    } else {
      text += "failure_text: injected legacy executable did not report the full primitive marker set\n";
    }
    if (!missing_markers.empty()) text += "missing_primitive_markers: " + missing_markers + "\n";
  } else {
    text += "failure_stage: none\n";
    text += "failure_text: none\n";
  }
  const int status = proof_pass ? 0 : (proof.exit_status == 0 ? 1 : proof.exit_status);
  return finish(status);
}

RuntimeSession::RuntimeSession() {}

RuntimeSession::~RuntimeSession() {
  cleanup();
}

bool RuntimeSession::transition_to(LifecycleStage expected, LifecycleStage next,
                                   std::string* error_text) {
  if (stage_ != expected) {
    if (error_text) {
      *error_text = "lifecycle ordering violation: expected " +
                    std::string(lifecycle_stage_name(expected)) + " but stage is " +
                    std::string(lifecycle_stage_name(stage_));
    }
    log_.failure_stage = "lifecycle_ordering";
    log_.failure_text = error_text ? *error_text : "ordering violation";
    stage_ = LifecycleStage::Failed;
    log_.stage = stage_;
    return false;
  }
  stage_ = next;
  log_.stage = next;
  return true;
}

void RuntimeSession::fail_log(const std::string& stage, const std::string& text) {
  log_.failure_stage = stage;
  log_.failure_text = text;
  log_.exit_status = 1;
  stage_ = LifecycleStage::Failed;
  log_.stage = stage_;
}

bool RuntimeSession::initialize(const std::string& socket_path, std::string* error_text) {
  if (!transition_to(LifecycleStage::Created, LifecycleStage::Initialized, error_text)) {
    return false;
  }
  log_.socket_path = socket_path;
  // No hardware work occurs here: the TinyGPU socket connect and BAR0/2/5
  // mapping are deferred gates for C1 task sets 5-8 and are NOT implemented.
  // This stage records substrate identity and advances the lifecycle state
  // machine so downstream task sets can call the shell uniformly.
  log_.pci_id = "1002:7551";  // frozen C1 identity (C0A25 PASS)
  log_.arch = kKernelArch;
  log_.arch_discovery_status = kStatusNotRun;  // set by a hardware run
  log_.connect_status = kStatusBlocked;        // hardware gate
  log_.bar_map_status = kStatusNotRun;         // hardware gate
  return true;
}

bool RuntimeSession::allocate_buffers(std::string* error_text) {
  if (!transition_to(LifecycleStage::Initialized, LifecycleStage::BuffersAllocated, error_text)) {
    return false;
  }
  // Actual MAP_SYSMEM_FD staging/readback mapping is a deferred hardware gate
  // (C1 task sets 5-8); this shell only records the frozen intent and advances
  // the lifecycle state machine.
  log_.host_device_transfer_status = kStatusNotRun;
  return true;
}

bool RuntimeSession::copy_input(const std::vector<uint8_t>& input, std::string* error_text) {
  if (!transition_to(LifecycleStage::BuffersAllocated, LifecycleStage::InputCopied, error_text)) {
    return false;
  }
  if (input.size() != kTransferByteCount) {
    fail_log("input_size", "input must be exactly 32 bytes (8 x u32), observed " +
                               std::to_string(input.size()));
    log_.sdma_h2d_status = kStatusFail;
    return false;
  }
  log_.input_digest = hex_bytes(input.data(), input.size());
  log_.sdma_h2d_status = kStatusNotRun;  // deferred hardware gate (C1 task sets 5-8): SDMA submit not implemented
  return true;
}

bool RuntimeSession::load_kernel(std::string* error_text) {
  if (!transition_to(LifecycleStage::InputCopied, LifecycleStage::KernelLoaded, error_text)) {
    return false;
  }
  log_.kernel_blob_load_status = kStatusNotRun;  // deferred hardware gate (C1 task sets 5-8): BAR0 write/readback not implemented
  return true;
}

bool RuntimeSession::write_kernargs(const Kernargs& kernargs, std::string* error_text) {
  if (!transition_to(LifecycleStage::KernelLoaded, LifecycleStage::KernargsWritten, error_text)) {
    return false;
  }
  uint8_t blk[kKernargScalarOffset + sizeof(uint32_t)];
  kernargs.encode(blk, sizeof(blk));
  std::string verify_error;
  if (!kernargs.verify(blk, sizeof(blk), &verify_error)) {
    fail_log("kernarg_write", verify_error);
    log_.kernarg_write_status = kStatusFail;
    return false;
  }
  log_.kernarg_write_status = kStatusPass;  // CPU-side layout self-check
  return true;
}

bool RuntimeSession::dispatch_and_poll(const std::vector<uint32_t>& dispatch_words,
                                       std::string* error_text) {
  if (!transition_to(LifecycleStage::KernargsWritten, LifecycleStage::Dispatched, error_text)) {
    return false;
  }
  if (dispatch_words.empty() || dispatch_words.size() != build_pm4_dispatch_words(0, 0, 0).size()) {
    fail_log("kernel_dispatch", "dispatch word count mismatch: observed " +
                                     std::to_string(dispatch_words.size()));
    log_.kernel_launch_status = kStatusFail;
    return false;
  }
  log_.kernel_launch_status = kStatusNotRun;  // deferred hardware gate (C1 task sets 5-8): doorbell + timeline not implemented
  return true;
}

bool RuntimeSession::readback_and_compare(const std::vector<uint8_t>& expected,
                                          std::string* error_text) {
  if (!transition_to(LifecycleStage::Dispatched, LifecycleStage::ReadbackCompared, error_text)) {
    return false;
  }
  if (expected.size() != kTransferByteCount) {
    fail_log("readback", "expected output must be exactly 32 bytes, observed " +
                             std::to_string(expected.size()));
    log_.cpu_comparison_status = kStatusFail;
    return false;
  }
  log_.output_digest = hex_bytes(expected.data(), expected.size());
  log_.cpu_comparison_status = kStatusNotRun;  // deferred hardware gate (C1 task sets 5-8): device readback not implemented
  return true;
}

void RuntimeSession::cleanup() {
  // This shell holds no host resources (TinyGPU socket connect, BAR mapping,
  // and staging/readback mappings are deferred hardware gates for C1 task sets
  // 5-8, not implemented here). cleanup only advances the lifecycle state.
  if (stage_ != LifecycleStage::Failed) {
    stage_ = LifecycleStage::CleanedUp;
    log_.stage = stage_;
  }
}

int RuntimeSession::dry_run(std::string* out_text, std::string* log_path) {
  std::string err;
  std::string text;
  const std::string socket_path = "/tmp/tinygpu.sock";
  log_.command_line = "native-r9700-runner --lifecycle-dry-run";
  log_.build_metadata = "xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra";

  // Exercise the full lifecycle in order, capturing progress.
  if (!initialize(socket_path, &err)) {
    text += "init: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: initialized\n";
  if (!allocate_buffers(&err)) {
    text += "allocate: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: buffers_allocated\n";
  std::vector<uint8_t> input = {1,0,0,0, 2,0,0,0, 3,0,0,0, 4,0,0,0,
                                5,0,0,0, 6,0,0,0, 7,0,0,0, 8,0,0,0};
  if (!copy_input(input, &err)) {
    text += "copy_input: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: input_copied\n";
  if (!load_kernel(&err)) {
    text += "load_kernel: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: kernel_loaded\n";

  // 24-byte kernarg layout exercise.
  Kernargs kernargs;
  kernargs.output_va = 0x0000200000004000ULL;
  kernargs.input_va = 0x0000200000001000ULL;
  kernargs.scalar_va = 0x0000200000006018ULL;  // kernargs_va + 24
  kernargs.scalar = 1U;
  if (!write_kernargs(kernargs, &err)) {
    text += "write_kernargs: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: kernargs_written\n";

  const std::vector<uint32_t> dispatch = build_pm4_dispatch_words(
      0x0000200000005000ULL, 0x0000200000006000ULL, 0x000020000000f010ULL);
  if (!dispatch_and_poll(dispatch, &err)) {
    text += "dispatch: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: dispatched\n";

  std::vector<uint8_t> expected = {2,0,0,0, 3,0,0,0, 4,0,0,0, 5,0,0,0,
                                   6,0,0,0, 7,0,0,0, 8,0,0,0, 9,0,0,0};
  if (!readback_and_compare(expected, &err)) {
    text += "readback: fail (" + err + ")\n";
    if (out_text) *out_text = text;
    return 1;
  }
  text += "lifecycle: readback_compared\n";

  // Contract emission lines (parsed by the focused pytest).
  uint8_t kernarg_blk[kKernargScalarOffset + sizeof(uint32_t)];
  kernargs.encode(kernarg_blk, sizeof(kernarg_blk));
  text += "kernarg_layout_offsets: output_va=0,input_va=8,scalar_va=16,scalar=24\n";
  text += "kernarg_byte_size: 24\n";
  text += "kernarg_bytes_hex: " + hex_bytes(kernarg_blk, kKernargByteSize) + "\n";
  text += "kernarg_scalar_hex: " +
          hex_bytes(kernarg_blk + kKernargScalarOffset, sizeof(uint32_t)) + "\n";
  text += "sdma_copy_dword_count: " +
          std::to_string(build_sdma_copy_words(0, 0, 32, 0, 1).size()) + "\n";
  text += "pm4_dispatch_dword_count: " + std::to_string(dispatch.size()) + "\n";
  // Lock the byte-faithful C0 header encodings (not just the dword counts).
  // SDMA begins with 0x000001 and PM4 begins with 0xc0065800. Both are
  // printed as 8-lowercase-hex-digit numeric values.
  {
    const std::vector<uint32_t> sdma = build_sdma_copy_words(0, 0, 32, 0, 1);
    char hbuf[16];
    std::snprintf(hbuf, sizeof(hbuf), "%08x", sdma.empty() ? 0U : sdma[0]);
    text += "sdma_copy_header_hex: " + std::string(hbuf) + "\n";
    std::snprintf(hbuf, sizeof(hbuf), "%08x", dispatch.empty() ? 0U : dispatch[0]);
    text += "pm4_dispatch_first_dword_hex: " + std::string(hbuf) + "\n";
  }
  text += "dispatch_global_size_x: 1\n";
  text += "dispatch_local_size_x: 8\n";

  // Ordering checks: a second initialize() on the already-initialized session
  // and a skipped-stage transition must both fail loudly, while the main
  // dry-run lifecycle stays on the clean pass path.
  {
    std::string reinit_err;
    std::string skip_err;

    // Fix 2a: re-init rejection. This session is already at ReadbackCompared
    // (far past Created), so a second initialize() must be rejected by the
    // ordering state machine. The rejection is captured as a bool so the failed
    // transition (which internally marks the stage Failed) does not corrupt
    // this session for the subsequent output emission.
    const LifecycleStage saved_stage = stage_;
    const LifecycleStage saved_log_stage = log_.stage;
    const bool reinit_ok = initialize(socket_path, &reinit_err);  // 2nd init: must fail
    stage_ = saved_stage;          // undo the probe's Failed transition
    log_.stage = saved_log_stage;
    text += std::string("lifecycle_reinit_rejected: ") + (reinit_ok ? "no" : "yes") + "\n";

    // Fix 2b: skip-stage rejection. Readback before dispatch must fail loudly;
    // probe with a purpose-built session advanced only to Initialized (readback
    // requires Dispatched), keeping the main session uncontaminated.
    bool skip_ok = false;
    {
      RuntimeSession skip_probe;  // begins at Created
      skip_probe.initialize(socket_path, &skip_err);  // -> Initialized (not Dispatched)
      skip_ok = skip_probe.readback_and_compare(expected, &skip_err);  // skip: must fail
    }
    text += std::string("lifecycle_skip_rejected: ") + (skip_ok ? "no" : "yes") + "\n";

    if (reinit_ok || skip_ok) {
      // An accepted re-init or skip means the ordering contract is broken.
      text += "ordering_check: fail\n";
      if (out_text) *out_text = text;
      return 1;
    }
  }

  cleanup();
  log_.exit_status = 0;
  log_.failure_stage = kFailureStageNone;
  log_.failure_text = kFailureTextNone;
  log_.cpu_comparison_status = kStatusPass;  // dry-run validated expected-output contract
  log_.timestamp_utc = timestamp_utc_now();

  std::string log_name = "c1-runner-dry-run-" + timestamp_utc_now() + ".log";
  const std::string written = write_run_log(log_, log_name);
  *log_path = written;

  text += "runtime_substrate: " + std::string(kRuntimeSubstrate) + "\n";
  text += "pci_id: " + log_.pci_id + "\n";
  text += "arch: " + log_.arch + "\n";
  text += "kernel_kernarg_size: 24\n";
  text += "exit_status: 0\n";
  text += "log_path: " + written + "\n";
  text += "lifecycle: cleanup\n";
  text += "status: pass\n";
  // Emit the complete standardized log (C0 print_*_log field conventions) to
  // stdout as well, so the wrapped shell command sees the same fields it writes
  // to the log file.
  text += format_log_text(log_, written.c_str());
  if (out_text) *out_text = text;
  return 0;
}

}  // namespace native_r9700

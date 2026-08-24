#include "hardware_lock.h"
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

namespace native_r9700 {
struct HardwareLock::Impl { int fd = -1; };
HardwareLock::HardwareLock() : impl_(std::make_unique<Impl>()) {}
HardwareLock::~HardwareLock() { release(); }

bool HardwareLock::acquire(std::string* error_text) {
  namespace fs = std::filesystem;
  fs::path dir = fs::path(getenv("HOME")) / ".cache/native-r9700";
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path lock_path = dir / "hardware.lock";
  impl_->fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
  if (impl_->fd < 0) { if (error_text) *error_text = "open hardware.lock failed"; return false; }
  if (::flock(impl_->fd, LOCK_EX) != 0) { if (error_text) *error_text = "flock hardware.lock failed"; return false; }
  return true;
}
void HardwareLock::release() {
  if (impl_->fd >= 0) { ::flock(impl_->fd, LOCK_UN); ::close(impl_->fd); impl_->fd = -1; }
}

bool hardware_lock_health_check(const std::string& socket_path, std::string* error_text) {
  if (socket_path.empty()) {
    if (error_text != nullptr) *error_text = "TinyGPU socket path is empty";
    return false;
  }
  struct stat socket_stat;
  if (::stat(socket_path.c_str(), &socket_stat) != 0) {
    if (error_text != nullptr) *error_text = "TinyGPU socket does not exist: " + socket_path;
    return false;
  }
  if (!S_ISSOCK(socket_stat.st_mode)) {
    if (error_text != nullptr) *error_text = "TinyGPU socket path is not a socket: " + socket_path;
    return false;
  }
  const std::string command = "pgrep -f \"TinyGPU server " + socket_path + "\"";
  std::string output;
  FILE* pipe = ::popen(command.c_str(), "r");
  if (pipe == nullptr) {
    if (error_text != nullptr) *error_text = "pgrep could not start for hardware lock health check";
    return false;
  }
  char buffer[256];
  while (::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const int pclose_status = ::pclose(pipe);
  if (pclose_status == -1) {
    if (error_text != nullptr) *error_text = "pgrep failed for hardware lock health check";
    return false;
  }
  int owner_count = 0;
  for (char c : output) {
    if (c == '\n') ++owner_count;
  }
  if (owner_count != 1) {
    if (error_text != nullptr) {
      *error_text = "expected exactly one TinyGPU server owner, observed " +
                    std::to_string(owner_count);
    }
    return false;
  }
  return true;
}
}  // namespace native_r9700

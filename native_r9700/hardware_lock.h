#pragma once

#include <memory>
#include <string>

namespace native_r9700 {

// Exclusive cross-process ownership of the shared R9700/TinyGPU hardware.
// acquire() blocks on an flock(LOCK_EX) of ~/.cache/native-r9700/hardware.lock;
// ownership is held RAII-scoped and released on destruction or release().
class HardwareLock {
 public:
  HardwareLock();
  ~HardwareLock();

  HardwareLock(const HardwareLock&) = delete;
  HardwareLock& operator=(const HardwareLock&) = delete;
  HardwareLock(HardwareLock&&) = delete;
  HardwareLock& operator=(HardwareLock&&) = delete;

  // Blocking acquisition. On failure, fills `error_text` (when non-null) and
  // returns false. Once true is returned the lock is held until release().
  bool acquire(std::string* error_text);
  void release();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Asserts that exactly one TinyGPU server process owns `socket_path`: the path
// is nonempty, names an existing socket, and `pgrep -f "TinyGPU server <path>"`
// reports exactly one match. Returns false with an error naming the observed
// count (or the failed assertion) otherwise.
bool hardware_lock_health_check(const std::string& socket_path, std::string* error_text);

}  // namespace native_r9700

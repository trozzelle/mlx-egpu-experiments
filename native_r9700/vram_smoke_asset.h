#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>
#include <unistd.h>

#include "kernel_catalog.h"

namespace native_r9700 {

// A preflighted, canonical source asset and descriptor. The source path is
// retained so a hardware result can identify exactly which reviewed bytes it
// consumed.
struct VramSmokeAsset {
  KernelDescriptor descriptor;
  std::string source_asset_path;
};

// Resolves and validates the sole reviewed resident-VRAM smoke asset before
// the caller opens a TinyGPU connection or maps any device resource.
inline bool preflight_vram_smoke_add_asset(VramSmokeAsset* asset, std::string* error_text) {
  if (asset == nullptr) {
    if (error_text != nullptr) *error_text = "VRAM smoke asset output is required";
    return false;
  }

  constexpr char kAssetRoot[] = "native_r9700/kernels/vram-smoke-assets";
  constexpr char kCodeName[] = "vram_smoke_add_gfx1201.code";
  const std::string requested_path = std::string(kAssetRoot) + "/" + kCodeName;
  struct stat requested_status {};
  if (lstat(requested_path.c_str(), &requested_status) != 0) {
    if (error_text != nullptr) {
      *error_text = "inspect reviewed VRAM smoke asset failed: " + requested_path + ": " +
                    std::strerror(errno);
    }
    return false;
  }
  if (S_ISLNK(requested_status.st_mode) || !S_ISREG(requested_status.st_mode)) {
    if (error_text != nullptr) {
      *error_text = "reviewed VRAM smoke asset must be a regular non-symlink file: " +
                    requested_path;
    }
    return false;
  }

  char canonical_root[PATH_MAX];
  if (realpath(kAssetRoot, canonical_root) == nullptr) {
    if (error_text != nullptr) {
      *error_text = std::string("canonicalize VRAM smoke asset root failed: ") + kAssetRoot +
                    ": " + std::strerror(errno);
    }
    return false;
  }
  char canonical_asset[PATH_MAX];
  if (realpath(requested_path.c_str(), canonical_asset) == nullptr) {
    if (error_text != nullptr) {
      *error_text = "canonicalize reviewed VRAM smoke asset failed: " + requested_path + ": " +
                    std::strerror(errno);
    }
    return false;
  }
  const std::string root(canonical_root);
  const std::string source_path(canonical_asset);
  const std::string root_prefix = root + "/";
  if (source_path.compare(0, root_prefix.size(), root_prefix) != 0) {
    if (error_text != nullptr) {
      *error_text = "reviewed VRAM smoke asset escapes fixed asset root: " + source_path;
    }
    return false;
  }

  const int fd = open(source_path.c_str(), O_RDONLY | O_NOFOLLOW);
  if (fd < 0) {
    if (error_text != nullptr) {
      *error_text = "open reviewed VRAM smoke asset failed: " + source_path + ": " +
                    std::strerror(errno);
    }
    return false;
  }
  struct stat opened_status {};
  if (fstat(fd, &opened_status) != 0 || !S_ISREG(opened_status.st_mode) ||
      opened_status.st_size < 0) {
    if (error_text != nullptr) {
      *error_text = "opened VRAM smoke asset is not a regular file: " + source_path;
    }
    close(fd);
    return false;
  }
  std::vector<uint8_t> code(static_cast<size_t>(opened_status.st_size));
  size_t offset = 0;
  while (offset < code.size()) {
    const ssize_t read_count = read(fd, code.data() + offset, code.size() - offset);
    if (read_count > 0) {
      offset += static_cast<size_t>(read_count);
      continue;
    }
    if (read_count < 0 && errno == EINTR) continue;
    if (error_text != nullptr) {
      *error_text = "read reviewed VRAM smoke asset failed: " + source_path + ": " +
                    (read_count == 0 ? "unexpected EOF" : std::strerror(errno));
    }
    close(fd);
    return false;
  }
  if (close(fd) != 0) {
    if (error_text != nullptr) {
      *error_text = "close reviewed VRAM smoke asset failed: " + source_path + ": " +
                    std::strerror(errno);
    }
    return false;
  }

  KernelDescriptor descriptor{
      "vram_smoke_add",
      "7a3f2a21c612c2a6b32401b6129a59a8e08b497e5b17afad1133cac780a7601d",
      std::move(code),
      3758882816U,
      132U,
      16U,
      64U,
      1U,
      1U,
      64U,
      1U,
      1U,
      24U,
  };
  if (!validate_kernel_descriptors(std::vector<KernelDescriptor>{descriptor}, error_text)) {
    return false;
  }
  asset->descriptor = std::move(descriptor);
  asset->source_asset_path = source_path;
  return true;
}

}  // namespace native_r9700

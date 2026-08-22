"""RED behavioral contracts for compact native kernel descriptors."""

from pathlib import Path
import subprocess


KERNEL_CATALOG_HEADER = Path("native_r9700/kernel_catalog.h")
KERNEL_CATALOG_SOURCE = Path("native_r9700/kernel_catalog.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_catalog_probe(tmp_path: Path) -> Path:
    """Compile a probe against the public descriptor-validation contract."""
    assert KERNEL_CATALOG_HEADER.is_file(), "kernel catalog header is missing"
    assert KERNEL_CATALOG_SOURCE.is_file(), "kernel catalog source is missing"

    probe_source = tmp_path / "kernel_catalog_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kernel_catalog.h"

namespace {

native_r9700::KernelDescriptor descriptor(std::string name, std::string sha256,
                                          uint32_t workgroup_x = 1,
                                          uint32_t workgroup_y = 1,
                                          uint32_t workgroup_z = 1) {
  native_r9700::KernelDescriptor value;
  value.name = std::move(name);
  value.sha256 = std::move(sha256);
  value.code = {0, 0, 0, 0};
  value.rsrc1 = 1;
  value.rsrc2 = 1;
  value.rsrc3 = 1;
  value.workgroup_x = workgroup_x;
  value.workgroup_y = workgroup_y;
  value.workgroup_z = workgroup_z;
  value.global_x = 1;
  value.global_y = 1;
  value.global_z = 1;
  value.kernarg_bytes = 64;
  return value;
}

bool rejects(const std::vector<native_r9700::KernelDescriptor>& descriptors) {
  std::string error_text;
  return !native_r9700::validate_kernel_descriptors(descriptors, &error_text) &&
         !error_text.empty();
}

}  // namespace

int main() {
  constexpr const char* kValidSha256 =
      "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119";

  if (!native_r9700::validate_kernel_descriptors(
          {descriptor("valid", kValidSha256)}, nullptr)) {
    return 1;
  }
  if (!rejects({descriptor("duplicate", kValidSha256),
                descriptor("duplicate", kValidSha256)})) {
    return 2;
  }
  if (!rejects({descriptor("zero-x", kValidSha256, 0, 1, 1)})) return 3;
  if (!rejects({descriptor("zero-y", kValidSha256, 1, 0, 1)})) return 4;
  if (!rejects({descriptor("zero-z", kValidSha256, 1, 1, 0)})) return 5;
  if (!rejects({descriptor("short-digest", "abc")})) return 6;
  if (!rejects({descriptor(
          "upper-case-digest",
          "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF")})) {
    return 7;
  }
  if (native_r9700::find_kernel("not-a-catalog-kernel") != nullptr) return 8;
  return 0;
}
'''.lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "kernel_catalog_probe"
    completed = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(KERNEL_CATALOG_SOURCE),
            str(probe_source),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return exe


def test_kernel_descriptor_validation_rejects_malformed_catalog_entries(tmp_path: Path) -> None:
    """Duplicate names, zero launch dimensions, and noncanonical digests fail loudly."""
    completed = subprocess.run(
        [str(compile_catalog_probe(tmp_path))], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unknown_kernel_lookup_returns_null(tmp_path: Path) -> None:
    """Catalog lookup must not manufacture a descriptor for an unknown name."""
    completed = subprocess.run(
        [str(compile_catalog_probe(tmp_path))], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

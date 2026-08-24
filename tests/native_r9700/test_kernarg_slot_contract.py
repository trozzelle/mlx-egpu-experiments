"""No-hardware contract for the in-page compute kernarg slots (Task 2.1).

Asserts, without any TinyGPU connection:
  (a) every stage kernarg schema is <= 256 bytes and 8-byte aligned,
  (b) ten 256-byte slots fit the existing 4 KiB kernarg page, and
  (c) the slot binder writes only its own slot, zero-pads to 256 bytes,
      and returns slot_va == kKernargsVa + slot * 256.
"""

import json
import glob
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_INCLUDE_DIR = REPO_ROOT / "native_r9700"

CLOSURE_SOURCES = (
    REPO_ROOT / "native_r9700/kernel_catalog.cpp",
    REPO_ROOT / "native_r9700/amdev_packets.cpp",
    REPO_ROOT / "native_r9700/hardware_lock.cpp",
    REPO_ROOT / "native_r9700/vram_layout.cpp",
    REPO_ROOT / "native_r9700/vram_allocator.cpp",
    REPO_ROOT / "native_r9700/dynamic_page_table.cpp",
    REPO_ROOT / "native_r9700/resident_memory.cpp",
    REPO_ROOT / "native_r9700/vram_smoke_asset.cpp",
)

PROBE_SOURCE = r'''
#include "amdev_session.cpp"
#include <cstdio>
#include <cstring>
int main() {
  if (am_compute::kKernargSlotCount * am_compute::kKernargSlotByteCount > 4096) {
    std::printf("FAIL slots exceed page\n"); return 1;
  }
  SysmemMapping mapping;
  mapping.data = new uint8_t[am_compute::kComputeControlByteCount];
  mapping.size = am_compute::kComputeControlByteCount;
  std::memset(mapping.data, 0xA5, mapping.size);
  native_r9700::ResidentKernelDispatch request;
  request.kernargs.resize(32);
  for (size_t i = 0; i < 32; ++i) request.kernargs[i] = (uint8_t)(i + 1);
  std::string error;
  uint64_t slot3_va = 0;
  if (!native_r9700::bind_resident_kernel_kernargs_slot(request, &mapping, 3U,
                                                       &slot3_va, &error)) {
    std::printf("FAIL bind slot3: %s\n", error.c_str()); return 1;
  }
  if (slot3_va != am_compute::kKernargsVa + 3U * am_compute::kKernargSlotByteCount) {
    std::printf("FAIL slot3 va\n"); return 1;
  }
  const uint8_t* base = static_cast<const uint8_t*>(mapping.data) +
                        am_compute::kComputeControlKernargsCpuOffset;
  if (std::memcmp(base + 3U * 256U, request.kernargs.data(), 32) != 0) {
    std::printf("FAIL slot3 bytes\n"); return 1;
  }
  for (size_t i = 32; i < 256; ++i) {
    if (base[3U * 256U + i] != 0) { std::printf("FAIL slot3 zero pad\n"); return 1; }
  }
  // Neighbouring slots (2 and 4) keep the 0xA5 sentinel — only slot 3 was written.
  for (size_t i = 0; i < 256; ++i) {
    if (base[2U * 256U + i] != 0xA5 || base[4U * 256U + i] != 0xA5) {
      std::printf("FAIL neighbour slot clobbered\n"); return 1;
    }
  }
  uint64_t slot9_va = 0;
  if (!native_r9700::bind_resident_kernel_kernargs_slot(request, &mapping, 9U,
                                                       &slot9_va, &error)) {
    std::printf("FAIL bind slot9: %s\n", error.c_str()); return 1;
  }
  uint64_t slot10_va = 0;
  if (native_r9700::bind_resident_kernel_kernargs_slot(request, &mapping, 10U,
                                                      &slot10_va, &error)) {
    std::printf("FAIL slot10 accepted\n"); return 1;
  }
  std::printf("status: pass\n");
  return 0;
}
'''


def compile_probe(tmp_path: Path) -> Path:
    assert (REPO_ROOT / "native_r9700/amdev_session.cpp").exists()
    probe_source = tmp_path / "kernarg_slot_probe.cpp"
    probe_source.write_text(PROBE_SOURCE.lstrip(), encoding="utf-8")
    exe = tmp_path / "kernarg_slot_probe"
    subprocess.run(
        [
            "xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2",
            "-Wall", "-Wextra",
            str(probe_source),
            *map(str, CLOSURE_SOURCES),
            "-I", str(NATIVE_INCLUDE_DIR),
            "-o", str(exe),
        ],
        check=True, capture_output=True, text=True,
    )
    return exe


def test_every_stage_kernarg_schema_fits_a_256_byte_slot():
    found = 0
    for path in glob.glob("native_r9700/kernels/*-hsa-assets/*.json"):
        data = json.loads(Path(path).read_text())
        if "kernarg_schema" not in data:
            continue
        found += 1
        nbytes = data["kernarg_schema"]["bytes"]
        assert nbytes <= 256, f"{path}: kernarg bytes {nbytes} > 256"
        assert nbytes % 8 == 0, f"{path}: kernarg bytes {nbytes} not 8-byte aligned"
    assert found >= 6, "expected at least six stage kernarg schemas"


def test_slot_layout_and_binding(tmp_path: Path) -> None:
    exe = compile_probe(tmp_path)
    completed = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "status: pass" in completed.stdout

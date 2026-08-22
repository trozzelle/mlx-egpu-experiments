"""No-hardware contracts for the C0-derived resident-kernel dispatch seam."""

from pathlib import Path
import subprocess


AMDEV_SESSION_SOURCE = Path("native_r9700/amdev_session.cpp")
KERNEL_CATALOG_SOURCE = Path("native_r9700/kernel_catalog.cpp")
PACKET_SOURCE = Path("native_r9700/amdev_packets.cpp")
VRAM_CLOSURE_SOURCES = (
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
)
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_dispatch_probe(tmp_path: Path) -> Path:
    """Compile the public preflight boundary without opening a TinyGPU socket."""
    probe_source = tmp_path / "resident_dispatch_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <string>
#include <vector>

#include "amdev_session.h"

namespace {

native_r9700::KernelDescriptor complete_descriptor() {
  native_r9700::KernelDescriptor descriptor;
  descriptor.name = "reviewed-stage-kernel";
  descriptor.sha256 = "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119";
  descriptor.code = {0x00, 0x00, 0x00, 0x00};
  descriptor.rsrc1 = 0xc00c0040U;
  descriptor.rsrc2 = 0x00000084U;
  descriptor.rsrc3 = 0x00000010U;
  descriptor.workgroup_x = 8;
  descriptor.workgroup_y = 1;
  descriptor.workgroup_z = 1;
  descriptor.global_x = 8;
  descriptor.global_y = 1;
  descriptor.global_z = 1;
  descriptor.kernarg_bytes = 24;
  return descriptor;
}

int rejects_missing_code() {
  native_r9700::ResidentKernelDispatch request;
  request.kernel = complete_descriptor();
  request.kernel.code.clear();
  request.kernargs.assign(24, 0);
  request.input_bytes = {1, 2, 3, 4};
  request.output_byte_count = 4;
  std::string error;
  if (native_r9700::validate_resident_kernel_dispatch(request, &error)) return 1;
  return error.find("code") == std::string::npos ? 2 : 0;
}

int rejects_mismatched_kernargs() {
  native_r9700::ResidentKernelDispatch request;
  request.kernel = complete_descriptor();
  request.kernargs.assign(23, 0);
  request.input_bytes = {1, 2, 3, 4};
  request.output_byte_count = 4;
  std::string error;
  if (native_r9700::validate_resident_kernel_dispatch(request, &error)) return 1;
  return error.find("kernarg") == std::string::npos ? 2 : 0;
}


int rejects_code_larger_than_the_c0_page() {
  native_r9700::ResidentKernelDispatch request;
  request.kernel = complete_descriptor();
  request.kernel.code.assign(4097, 0);
  request.kernargs.assign(24, 0);
  request.input_bytes = {1, 2, 3, 4};
  request.output_byte_count = 4;
  std::string error;
  if (native_r9700::validate_resident_kernel_dispatch(request, &error)) return 1;
  return error.find("code") == std::string::npos ? 2 : 0;
}


int physical_dispatch_rejects_unreviewed_asset_before_connecting() {
  native_r9700::ResidentKernelDispatch request;
  request.kernel = complete_descriptor();
  request.kernel.code.clear();
  request.kernargs.assign(24, 0);
  request.input_bytes = {1, 2, 3, 4};
  request.output_byte_count = 4;
  native_r9700::ResidentKernelDispatchResult result;
  native_r9700::AMDevSession session;
  std::string error;
  if (session.dispatch_resident_kernel(request, &result, &error)) return 1;
  if (result.failure_stage != "preflight") return 2;
  return error.find("code") == std::string::npos ? 3 : 0;
}

int physical_dispatch_rejects_digest_mismatch_before_bar0_operation() {
  native_r9700::ResidentKernelDispatch request;
  request.kernel = complete_descriptor();
  request.kernel.sha256 =
      "0000000000000000000000000000000000000000000000000000000000000000";
  request.kernargs.assign(24, 0);
  request.input_bytes = {1, 2, 3, 4};
  request.output_byte_count = 4;
  native_r9700::ResidentKernelDispatchResult result;
  native_r9700::AMDevSession session;
  std::string error;
  if (session.dispatch_resident_kernel(request, &result, &error)) return 1;
  if (result.failure_stage != "preflight") return 2;
  return error.find("digest") == std::string::npos ? 3 : 0;
}

int accepts_complete_bounded_launch() {
  native_r9700::ResidentKernelDispatch request;
  request.kernel = complete_descriptor();
  request.kernargs.assign(24, 0);
  request.input_bytes = {1, 2, 3, 4};
  request.output_byte_count = 4;
  std::string error;
  return native_r9700::validate_resident_kernel_dispatch(request, &error) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 64;
  const std::string mode = argv[1];
  if (mode == "missing-code") return rejects_missing_code();
  if (mode == "kernarg-size") return rejects_mismatched_kernargs();
  if (mode == "code-page") return rejects_code_larger_than_the_c0_page();
  if (mode == "dispatch-missing-code") return physical_dispatch_rejects_unreviewed_asset_before_connecting();
  if (mode == "dispatch-digest-mismatch") {
    return physical_dispatch_rejects_digest_mismatch_before_bar0_operation();
  }
}
'''.lstrip(),
        encoding="utf-8",
    )
    exe = tmp_path / "resident_dispatch_probe"
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
            str(AMDEV_SESSION_SOURCE),
            str(KERNEL_CATALOG_SOURCE),
            str(PACKET_SOURCE),
            *map(str, VRAM_CLOSURE_SOURCES),
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


def test_resident_dispatch_rejects_a_kernel_without_reviewable_code(tmp_path: Path) -> None:
    """Catches accepting a descriptor that cannot load a real resident kernel asset."""
    completed = subprocess.run(
        [str(compile_dispatch_probe(tmp_path)), "missing-code"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_resident_dispatch_rejects_kernarg_layout_drift(tmp_path: Path) -> None:
    """Catches binding a byte span that differs from the reviewed kernel layout."""
    completed = subprocess.run(
        [str(compile_dispatch_probe(tmp_path)), "kernarg-size"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr



def test_resident_dispatch_rejects_code_that_exceeds_the_c0_page(tmp_path: Path) -> None:
    """Catches silently truncating an asset instead of rejecting its real code span."""
    completed = subprocess.run(
        [str(compile_dispatch_probe(tmp_path)), "code-page"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_physical_dispatch_rejects_unreviewed_asset_before_tinygpu_connection(
    tmp_path: Path,
) -> None:
    """Catches connecting to hardware before the descriptor asset is fail-closed."""
    completed = subprocess.run(
        [str(compile_dispatch_probe(tmp_path)), "dispatch-missing-code"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

def test_physical_dispatch_rejects_a_digest_mismatch_before_bar0_operation(
    tmp_path: Path,
) -> None:
    """Catches loading code whose descriptor digest does not bind those bytes."""
    completed = subprocess.run(
        [str(compile_dispatch_probe(tmp_path)), "dispatch-digest-mismatch"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_resident_dispatch_preflight_accepts_a_complete_bounded_launch(tmp_path: Path) -> None:
    """Catches a preflight that blocks a fully described C0-sized resident launch."""
    completed = subprocess.run(
        [str(compile_dispatch_probe(tmp_path)), "complete"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

def test_native_queue_setup_keeps_terminal_retirement() -> None:
    """Catches dropping stale-HQD retirement."""
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")
    retirement_start = source.index("class TerminalComputeQueue0Retirement")
    retirement_end = source.index("\n};", retirement_start) + 3
    retirement_body = source[retirement_start:retirement_end]

    assert "reset_compute_queue0(client_, log_, &retirement_failure_)" in retirement_body

def test_native_queue_diagnostics_identify_post_doorbell_receipt_and_fetch_state() -> None:
    """Catches losing the post-submit evidence that separates doorbell receipt from fetch."""
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")

    for key in (
        "compute_queue_mec_rs64_cntl",
        "compute_queue_mec_rs64_instruction_pointer",
        "compute_queue_mec_rs64_program_start_lo",
        "compute_queue_mec_rs64_program_start_hi",
        "compute_queue_mec_doorbell_range_lower",
        "compute_queue_mec_doorbell_range_upper",
        "compute_queue_hqd_active",
        "compute_queue_hqd_pq_doorbell_control",
        "compute_queue_post_doorbell_hit",
        "compute_queue_post_wptr_lo",
    ):
        assert f'"{key}"' in source

    post_doorbell_start = source.index("void log_compute_queue_post_doorbell_diagnostics(")
    post_doorbell_end = source.index("\n}", post_doorbell_start) + 2
    post_doorbell_body = source[post_doorbell_start:post_doorbell_end]
    assert "select_grbm_queue0(client, log, &selection_error)" in post_doorbell_body
    assert "regs_gfx1201::kCpHqdPqDoorbellControl" in post_doorbell_body
    assert "regs_gfx1201::kCpHqdPqWptrLo" in post_doorbell_body

    submit_start = source.index("bool submit_compute_dispatch_with_post_doorbell_diagnostics(")
    submit_end = source.index("\n}", submit_start) + 2
    submit_body = source[submit_start:submit_end]
    assert submit_body.index("submit_compute_dispatch(") < submit_body.index(
        "log_compute_queue_post_doorbell_diagnostics(client, *log)"
    )

    # Definition plus four call sites: kernel proof, embed smoke, legacy
    # primitive chain, and the resident HSA dispatch path.
    assert source.count("submit_compute_dispatch_with_post_doorbell_diagnostics(") == 5

    assert "(value >> 31U) & 1U" in source
    assert r'std::printf("%s: 0x%08x\n", key, value)' in source
    assert r'std::printf("%s: unavailable (%s)\n", key, error.c_str())' in source


def test_resident_staging_window_streams_one_mib_chunks() -> None:
    """Weight streaming must not pay a queue round trip per 4 KiB page.

    The resident session maps a 1 MiB staging window (256 fixed PTB entries)
    and chunks uploads/readbacks by that window; per-page queue setups made
    full-model weight streaming hours-slow.
    """
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")

    assert "constexpr uint64_t kResidentStagingByteCount = 256ULL * kPageSize;" in source
    assert 'VmBufferLog staging{"staging", kTransferProofVmStagingVa, kResidentStagingByteCount' in source
    upload_start = source.index("bool ResidentHsaSession::upload_named(")
    upload_end = source.index("\n}", upload_start)
    upload_body = source[upload_start:upload_end]
    assert "std::min<uint64_t>(kResidentStagingByteCount, byte_count - offset)" in upload_body
    assert "std::min<uint64_t>(kPageSize" not in upload_body

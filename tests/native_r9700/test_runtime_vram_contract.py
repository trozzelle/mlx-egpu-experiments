"""No-hardware public contract for the future resident-VRAM smoke command.

This test deliberately compiles the runner's resident-VRAM dependency closure and
invokes only ``--help``.  It never selects a mode that can open TinyGPU.
"""

from collections.abc import Mapping
from pathlib import Path
import re
import subprocess

from native_r9700.llama_stage_oracle import STAGE_SPECS


AMDEV_SESSION_SOURCE = Path("native_r9700/amdev_session.cpp")

RUNNER_SOURCES = (
    Path("native_r9700/amdev_packets.cpp"),
    Path("native_r9700/runtime_contract.cpp"),
    Path("native_r9700/prefill_npz.cpp"),
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/hsa_code_image_asset.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
    Path("native_r9700/kernel_assets.cpp"),
    Path("native_r9700/device_memory.cpp"),
    Path("native_r9700/llama_stage_layout.cpp"),
    Path("native_r9700/llama_layer_executor.cpp"),
    Path("native_r9700/model_weight_binder.cpp"),
    Path("native_r9700/amdev_session.cpp"),
    Path("native_r9700/kernel_catalog.cpp"),
    Path("native_r9700/runtime.cpp"),
    Path("native_r9700/runner.cpp"),
)

# This is a minimum schema: the standard runtime log may retain additional
# diagnostic fields, but a successful hardware smoke must provide every field
# below with these exact success conditions.
VRAM_SMOKE_RESULT_FIELDS = frozenset(
    {
        "command_line",
        "producer_kind",
        "runtime_substrate",
        "pci_id",
        "arch",
        "vram_allocation_status",
        "resident_mapping_count",
        "bar0_zero_status",
        "pte_write_status",
        "pte_readback_status",
        "mmhub_tlb_flush_status",
        "gc_tlb_flush_status",
        "compute_dispatch_count",
        "sdma_h2d_status",
        "sdma_d2h_status",
        "sdma_upload_bytes",
        "sdma_download_bytes",
        "cpu_comparison_status",
        "failure_stage",
        "failure_text",
        "exit_status",
    }
)


def compile_runner(tmp_path: Path) -> Path:
    """Compile the runner plus every resident-VRAM production dependency."""
    assert all(source.exists() for source in RUNNER_SOURCES), (
        "native_r9700 resident-VRAM runner sources missing"
    )
    executable = tmp_path / "native_r9700_runner"
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
            *map(str, RUNNER_SOURCES),
            "-I",
            "native_r9700",
            "-o",
            str(executable),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def assert_hardware_vram_smoke_success(result: Mapping[str, str]) -> None:
    """Validate the hardware-only result that a future smoke invocation emits."""
    missing_fields = VRAM_SMOKE_RESULT_FIELDS.difference(result)
    assert not missing_fields, f"missing vram-smoke result fields: {sorted(missing_fields)}"
    assert "--vram-smoke" in result["command_line"]
    assert result["producer_kind"] == "hardware_resident_vram_smoke"
    assert result["runtime_substrate"] == "TinyGPU.app/APLRemotePCIDevice/PCIIface"
    assert result["pci_id"] == "1002:7551"
    assert result["arch"] == "gfx1201"
    for field in (
        "vram_allocation_status",
        "bar0_zero_status",
        "pte_write_status",
        "pte_readback_status",
        "mmhub_tlb_flush_status",
        "gc_tlb_flush_status",
        "sdma_h2d_status",
        "sdma_d2h_status",
        "cpu_comparison_status",
    ):
        assert result[field] == "pass", f"{field}: {result[field]!r}"
    assert int(result["resident_mapping_count"]) >= 3
    assert int(result["compute_dispatch_count"]) == 1
    assert int(result["sdma_upload_bytes"]) > 0
    assert int(result["sdma_download_bytes"]) > 0
    assert result["failure_stage"] == "none"
    assert result["failure_text"] == "none"
    assert int(result["exit_status"]) == 0


def test_help_lists_vram_smoke_without_opening_tinygpu(tmp_path: Path) -> None:
    """Catches a runner that omits the resident-VRAM smoke command entirely."""
    executable = compile_runner(tmp_path)

    completed = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "--vram-smoke" in completed.stdout

def test_fixed_vm_gc_flush_uses_req_ack_without_semaphore() -> None:
    """Catches a GC flush that reintroduces unsupported ENG17 semaphore I/O."""
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")
    flush_start = source.index("bool flush_gc_tlb_vmid0_native(")
    flush_end = source.index("\n}\n", flush_start) + 2
    flush_body = source[flush_start:flush_end]
    setup_start = source.index("bool setup_fixed_vm_mapping(", flush_end)
    setup_end = source.index("\n}\n", setup_start) + 2
    setup_body = source[setup_start:setup_end]

    assert "flush_hdp(client, *log, error_text)" in flush_body
    assert "regs_gfx1201::kGcInvalidateEng17Req" in flush_body
    assert "encode_invalidate_req_vmid0()" in flush_body
    assert "regs_gfx1201::kGcInvalidateEng17Ack" in flush_body
    assert "regs_gfx1201::kGcInvalidateEng17Sem" not in flush_body
    assert flush_body.index("kGcInvalidateEng17Req") < flush_body.index("kGcInvalidateEng17Ack")
    assert "flush_gc_tlb_vmid0_native(client, log, &error)" in setup_body

def test_vram_smoke_readback_mismatch_reports_observed_value() -> None:
    """A resident-compute mismatch must identify the first divergent word."""
    source = AMDEV_SESSION_SOURCE.read_text(encoding="utf-8")

    assert '"vector-add mismatch at element " + std::to_string(index)' in source
    assert '" expected=" + std::to_string(expected_value)' in source
    assert '" observed=" + std::to_string(observed_value)' in source



def test_llama_stage_trace_help_and_invalid_arguments_never_open_tinygpu(tmp_path: Path) -> None:
    """Trace CLI parsing must complete before any route can prepare a device session."""
    executable = compile_runner(tmp_path)

    help_result = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    invalid_result = subprocess.run(
        [
            str(executable),
            "--llama-stage-trace",
            "--model",
            "missing",
            "--token-id",
            "1",
            "--layer",
            "1",
            "--position",
            "0",
            "--stage",
            "hidden",
            "--trace-dir",
            str(tmp_path / "trace"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "--llama-stage-trace" in help_result.stdout
    assert invalid_result.returncode == 2
    assert "only layer 0 and position 0" in invalid_result.stderr
    assert not (tmp_path / "trace").exists()
    unknown_stage_result = subprocess.run(
        [
            str(executable),
            "--llama-stage-trace",
            "--model",
            "missing",
            "--token-id",
            "1",
            "--layer",
            "0",
            "--position",
            "0",
            "--stage",
            "not-a-boundary",
            "--trace-dir",
            str(tmp_path / "unknown-stage-trace"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert unknown_stage_result.returncode == 1
    assert "unknown layer-0 trace stage" in unknown_stage_result.stdout
    assert not (tmp_path / "unknown-stage-trace").exists()


def test_llama_stage_trace_contract_is_bounded_and_non_accepting() -> None:
    """The public trace schema stays bounded and separate from prefill output."""
    source = Path("native_r9700/runtime_contract.cpp").read_text(encoding="utf-8")
    header = Path("native_r9700/runtime.h").read_text(encoding="utf-8")
    result_start = header.index("struct LlamaStageTraceResult")
    result_declaration = header[result_start : header.index("\n};", result_start)]

    for field in (
        "token_index",
        "layer_index",
        "stage",
        "buffer",
        "shape_json",
        "dtype",
        "byte_count",
        "sha256",
        "finite_count",
        "raw_path",
        "kernarg_hex",
        "hsa_image_sha256",
        "gpu_va",
        "scalars_json",
    ):
        assert re.search(rf"\b{field}\b", result_declaration)
    trace_body = source[source.index("int run_llama_stage_trace(") :]
    assert "kLlamaStageTraceStages" in source
    assert "write_native_prefill_npz" not in trace_body
    assert "NativePrefill" not in header[header.index("struct LlamaStageTraceRequest"):]


def test_llama_stage_trace_native_table_matches_canonical_oracle_schema() -> None:
    """The two producers must agree before any numerical comparator is allowed."""
    source = Path("native_r9700/runtime_contract.cpp").read_text(encoding="utf-8")
    table_start = source.index("constexpr std::array<LlamaStageTraceSpec")
    table_end = source.index("find_trace_stage", table_start)
    rows = re.findall(
        r'\{"([^"]+)",\s*"([^"]+)",\s*"(\[[^"]+\])",\s*"([^"]+)",\s*(\d+)'
        r"(?:,\s*[^}]*)?\}",
        source[table_start:table_end],
    )
    assert len(rows) == len(STAGE_SPECS)
    native = {
        stage: (buffer, tuple(map(int, shape[1:-1].split(","))), dtype, int(byte_count))
        for stage, buffer, shape, dtype, byte_count in rows
    }
    assert native == {
        stage: (spec.buffer, spec.shape, spec.dtype, spec.byte_count)
        for stage, spec in STAGE_SPECS.items()
    }


def test_llama_stage_trace_scalar_schema_contains_only_dispatched_fields() -> None:
    """Scalar labels describe only values materialized in native kernargs."""
    source = Path("native_r9700/runtime_contract.cpp").read_text(encoding="utf-8")

    assert '"epsilon"' in source
    assert '"sequence_length"' in source
    assert '"position"' in source
    assert '"cache_capacity_tokens"' in source
    assert "output_columns" not in source
    assert "head_count" not in source

def test_llama_trace_publication_failure_seam_and_scalar_values(tmp_path: Path) -> None:
    """Publication faults leave no visible pair, while scalar JSON reads kernarg bytes."""
    harness = tmp_path / "trace_publication_harness.cpp"
    harness.write_text(
        r'''
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "native_r9700/runtime_contract.cpp"

namespace {
struct FailurePlan {
  std::string failure;
  bool cleanup_failure = false;
  bool cleanup_parent_sync_failure = false;
  bool cleanup_completed = false;
  bool failed = false;
  std::vector<std::string> calls;
};

bool plan_matches(FailurePlan* plan, const std::string& operation) {
  plan->calls.push_back(operation);
  if (!plan->failed && plan->failure == operation) {
    plan->failed = true;
    return true;
  }
  return false;
}

bool write_file(void* context, const std::filesystem::path& path, const char* data, size_t size,
                std::string* error_text) {
  auto* plan = static_cast<FailurePlan*>(context);
  const std::string operation = path.extension() == ".bin" ? "write_raw" : "write_json";
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(data, static_cast<std::streamsize>(size));
  output.close();
  if (!output || plan_matches(plan, operation)) {
    if (error_text != nullptr) *error_text = operation;
    return false;
  }
  return true;
}

bool sync_path(void* context, const std::filesystem::path& path, bool directory,
               std::string* error_text) {
  auto* plan = static_cast<FailurePlan*>(context);
  const std::string operation = directory
      ? (path.filename().string().find(".staging") != std::string::npos
             ? "sync_staging"
             : "sync_parent")
      : (path.extension() == ".bin" ? "sync_raw" : "sync_json");
  if (operation == "sync_parent" && plan->cleanup_parent_sync_failure &&
      plan->cleanup_completed) {
    plan->calls.push_back("sync_parent_after_cleanup");
    if (error_text != nullptr) *error_text = "sync_parent_after_cleanup";
    return false;
  }
  if (plan_matches(plan, operation)) {
    if (error_text != nullptr) *error_text = operation;
    return false;
  }
  return true;
}

bool rename_path(void* context, const std::filesystem::path& from, const std::filesystem::path& to,
                 std::string* error_text) {
  auto* plan = static_cast<FailurePlan*>(context);
  if (plan_matches(plan, "rename")) {
    if (error_text != nullptr) *error_text = "rename";
    return false;
  }
  std::error_code error;
  std::filesystem::rename(from, to, error);
  if (error) {
    if (error_text != nullptr) *error_text = error.message();
    return false;
  }
  return true;
}

bool remove_tree(void* context, const std::filesystem::path& path, std::string* error_text) {
  auto* plan = static_cast<FailurePlan*>(context);
  plan->calls.push_back("cleanup");
  if (plan->cleanup_failure) {
    if (error_text != nullptr) *error_text = "cleanup";
    return false;
  }
  std::error_code error;
  std::filesystem::remove_all(path, error);
  if (error) {
    if (error_text != nullptr) *error_text = error.message();
    return false;
  }
  plan->cleanup_completed = true;
  return true;
}

bool run_case(const std::filesystem::path& root, const std::string& failure) {
  const std::filesystem::path staging = root / ".trace.staging";
  const std::filesystem::path artifact = root / "trace";
  std::error_code error;
  std::filesystem::create_directory(root, error);
  if (error) {
    std::cerr << "trace publication setup root=" << root << " error=" << error.message() << "\n";
    return false;
  }
  std::filesystem::create_directory(staging, error);
  if (error) {
    std::cerr << "trace publication setup staging=" << staging << " error=" << error.message()
              << "\n";
    return false;
  }

  const bool cleanup_parent_sync_failure = failure == "cleanup_parent_sync";
  FailurePlan plan{failure == "cleanup" || cleanup_parent_sync_failure ? "sync_raw" : failure,
                   failure == "cleanup", cleanup_parent_sync_failure};
  native_r9700::TracePublicationOps ops{
      &plan, write_file, sync_path, rename_path, remove_tree};
  std::string detail;
  const bool published = native_r9700::publish_trace_artifact(
      staging, artifact, root, "trace.bin", "trace.json", "raw", "json", ops, &detail);
  const auto check = [&](bool passed, const std::vector<std::string>& expected) {
    if (passed && plan.calls == expected) return true;
    std::cerr << "trace publication case=" << (failure.empty() ? "success" : failure)
              << " published=" << published << " detail=" << detail << " calls=";
    for (const std::string& call : plan.calls) std::cerr << call << ",";
    std::cerr << " expected=";
    for (const std::string& call : expected) std::cerr << call << ",";
    std::cerr << " staging=" << std::filesystem::exists(staging)
              << " artifact=" << std::filesystem::exists(artifact) << "\n";
    return false;
  };
  const std::vector<std::string> success = {
      "write_raw", "sync_raw", "write_json", "sync_json", "sync_staging", "rename",
      "sync_parent"};
  if (failure.empty()) {
    return check(published && std::filesystem::exists(artifact / "trace.bin") &&
                     std::filesystem::exists(artifact / "trace.json") &&
                     !std::filesystem::exists(staging),
                 success);
  }
  if (failure == "cleanup") {
    return check(!published && detail == "sync_raw; cleanup failed: cleanup" &&
                     std::filesystem::exists(staging) && !std::filesystem::exists(artifact),
                 {"write_raw", "sync_raw", "cleanup"});
  }
  if (failure == "cleanup_parent_sync") {
    return check(!published && plan.cleanup_completed &&
                     detail == "sync_raw; cleanup failed: sync_parent_after_cleanup" &&
                     !std::filesystem::exists(staging) && !std::filesystem::exists(artifact),
                 {"write_raw", "sync_raw", "cleanup", "sync_parent_after_cleanup"});
  }
  std::vector<std::string> expected;
  for (const std::string& operation : success) {
    expected.push_back(operation);
    if (operation == failure) break;
  }
  expected.push_back("cleanup");
  expected.push_back("sync_parent");
  return check(!published && detail == failure && !std::filesystem::exists(staging) &&
                   !std::filesystem::exists(artifact),
               expected);
}

void put_u32(std::vector<uint8_t>* bytes, size_t offset, uint32_t value) {
  (*bytes)[offset] = static_cast<uint8_t>(value);
  (*bytes)[offset + 1] = static_cast<uint8_t>(value >> 8U);
  (*bytes)[offset + 2] = static_cast<uint8_t>(value >> 16U);
  (*bytes)[offset + 3] = static_cast<uint8_t>(value >> 24U);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  size_t case_index = 0;
  for (const char* failure : {"", "write_raw", "sync_raw", "write_json", "sync_json",
                              "sync_staging", "rename", "sync_parent"}) {
    if (!run_case(root / (std::string("case-") + (failure[0] == '\0' ? "success" : failure)),
                  failure)) {
      std::cerr << "trace publication fault index=" << case_index << "\n";
      return 3;
    }
    ++case_index;
  }
  if (!run_case(root / "case-cleanup", "cleanup")) return 4;
  if (!run_case(root / "case-cleanup-parent-sync", "cleanup_parent_sync")) return 5;

  std::vector<uint8_t> kernargs(48, 0);
  uint32_t epsilon_bits = 0;
  const float epsilon = 0.001F;
  std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
  put_u32(&kernargs, 24, epsilon_bits);
  std::string scalars;
  std::string detail;
  const auto scalars_match = [&kernargs, &scalars, &detail](int stage_index,
                                                              const char* expected) {
    return native_r9700::trace_scalars_json(stage_index, kernargs, &scalars, &detail) &&
           scalars == expected;
  };
  if (!scalars_match(0, "{\"epsilon\":0.001000}")) return 6;

  put_u32(&kernargs, 24, 101);
  if (!scalars_match(1, "{\"sequence_length\":101}") ||
      !scalars_match(2, "{\"sequence_length\":101}")) {
    return 7;
  }

  put_u32(&kernargs, 32, 7);
  put_u32(&kernargs, 36, 6);
  put_u32(&kernargs, 40, 128);
  if (!scalars_match(3, "{\"sequence_length\":7,\"position\":6,\"cache_capacity_tokens\":128}") ||
      !scalars_match(4, "{\"sequence_length\":7,\"position\":6,\"cache_capacity_tokens\":128}")) {
    return 8;
  }

  put_u32(&kernargs, 16, 11);
  put_u32(&kernargs, 20, 12);
  put_u32(&kernargs, 24, 13);
  if (!scalars_match(5, "{\"sequence_length\":11,\"position\":12,\"cache_capacity_tokens\":13}")) {
    return 9;
  }

  put_u32(&kernargs, 24, 21);
  put_u32(&kernargs, 28, 22);
  put_u32(&kernargs, 32, 23);
  if (!scalars_match(6, "{\"sequence_length\":21,\"position\":22,\"cache_capacity_tokens\":23}")) {
    return 10;
  }

  put_u32(&kernargs, 32, 31);
  if (!scalars_match(7, "{\"sequence_length\":31}")) return 11;
  return 0;
}
''',
        encoding="utf-8",
    )
    executable = tmp_path / "trace_publication_harness"
    compile_result = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            str(harness),
            *(
                str(source)
                for source in RUNNER_SOURCES
                if source.name not in {"runtime_contract.cpp", "runner.cpp"}
            ),
            "-I",
            ".",
            "-I",
            "native_r9700",
            "-o",
            str(executable),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    completed = subprocess.run(
        [str(executable), str(tmp_path / "publication-cases")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

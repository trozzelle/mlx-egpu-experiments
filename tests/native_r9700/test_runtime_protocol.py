"""Injected-executable protocol contracts for the native runner.

Fake bridge executables validate wrapper marker and command handling without hardware.
"""

from pathlib import Path
import re
import subprocess
import pytest


VRAM_CLOSURE_SOURCES = (
    Path("native_r9700/vram_layout.cpp"),
    Path("native_r9700/vram_allocator.cpp"),
    Path("native_r9700/dynamic_page_table.cpp"),
    Path("native_r9700/resident_memory.cpp"),
    Path("native_r9700/vram_smoke_asset.cpp"),
)

RUNNER_SOURCES = [
    Path("native_r9700/amdev_packets.cpp"),
    Path("native_r9700/runtime_contract.cpp"),
    Path("native_r9700/prefill_npz.cpp"),
    *VRAM_CLOSURE_SOURCES,
    Path("native_r9700/hsa_code_image_asset.cpp"),
    Path("native_r9700/model_weight_binder.cpp"),
    Path("native_r9700/amdev_session.cpp"),
    Path("native_r9700/kernel_catalog.cpp"),
    Path("native_r9700/device_memory.cpp"),
    Path("native_r9700/hardware_lock.cpp"),
    Path("native_r9700/llama_stage_layout.cpp"),
    Path("native_r9700/llama_layer_executor.cpp"),
    Path("native_r9700/kernel_assets.cpp"),
    Path("native_r9700/runtime.cpp"),
    Path("native_r9700/runner.cpp"),
]


C1R4_LAYER_SLICE_BYTES = "20480"  # prompt-0 prefix activation: 5 * 2048 * fp16.


def compile_runner(tmp_path):
    assert all(s.exists() for s in RUNNER_SOURCES), (
        "native_r9700 runner sources missing"
    )
    exe = tmp_path / "native_r9700_runner"
    subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
        ]
        + [str(s) for s in RUNNER_SOURCES]
        + ["-I", "native_r9700", "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe

def default_transfer_bridge_sources() -> list[Path]:
    runtime_source = Path("native_r9700/runtime.cpp")
    runtime_text = runtime_source.read_text(encoding="utf-8")
    bridge_build = re.search(
        r'const std::string source = "(?P<source>[^"]*c1_transfer_bridge\.cpp)";'
        r".*?std::vector<std::string> build_cmd = \{(?P<entries>.*?)\};",
        runtime_text,
        flags=re.DOTALL,
    )
    assert bridge_build, "default transfer bridge build command is missing"
    sources = [Path(bridge_build.group("source"))] + [
        Path(path)
        for path in re.findall(
            r'"(native_r9700/[^"]+\.cpp)"', bridge_build.group("entries")
        )
    ]
    return sources + [source for source in VRAM_CLOSURE_SOURCES if source not in sources]


def test_default_transfer_bridge_build_links_all_amdev_modules(tmp_path: Path) -> None:
    """The runtime fallback must compile the complete transfer bridge without hardware."""
    sources = default_transfer_bridge_sources()
    required_sources = {
        Path("native_r9700/c1_transfer_bridge.cpp"),
        Path("native_r9700/amdev_session.cpp"),
        Path("native_r9700/amdev_packets.cpp"),
        Path("native_r9700/kernel_catalog.cpp"),
        *VRAM_CLOSURE_SOURCES,
    }
    assert required_sources <= set(sources)

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
            *map(str, sources),
            "-o",
            str(tmp_path / "c1_transfer_bridge"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def compile_runtime_api_probe(tmp_path, source_text):
    source = tmp_path / "runtime_api_probe.cpp"
    source.write_text(source_text, encoding="utf-8")
    exe = tmp_path / "runtime_api_probe"
    subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "native_r9700/amdev_packets.cpp",
            *map(str, VRAM_CLOSURE_SOURCES),
            "native_r9700/hsa_code_image_asset.cpp",
            "native_r9700/model_weight_binder.cpp",
            "native_r9700/amdev_session.cpp",
            "native_r9700/kernel_catalog.cpp",
            "native_r9700/runtime.cpp",
            str(source),
            "-I",
            "native_r9700",
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return exe


def fake_transfer_bridge_script(exact_markers=True, copy_output=True):
    chunk_expr = "$(( (bytes + 4095) / 4096 ))"
    completed_expr = "$chunks" if exact_markers else "$(( chunks + 45 ))"
    copy_line = 'cp "$2" "$3"\n' if copy_output else ': > "$3"\n'
    return (
        "#!/bin/sh\n"
        "if [ \"$1\" != \"--roundtrip-file\" ]; then echo bad-arg; exit 7; fi\n"
        "bytes=$(wc -c < \"$2\" | tr -d ' ')\n"
        f"chunks={chunk_expr}\n"
        "if [ \"$bytes\" -gt 4096 ]; then streaming=yes; else streaming=no; fi\n"
        f"{copy_line}"
        "cat <<EOF\n"
        "producer_kind: hardware_memory_transfer\n"
        "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface\n"
        "pci_id: 1002:7551\n"
        "arch: gfx1201\n"
        "transfer_byte_count: $bytes\n"
        "transfer_chunk_count: $chunks\n"
        f"transfer_chunks_completed: {completed_expr}\n"
        "transfer_chunk_size_bytes: 4096\n"
        "buffer_count: 3\n"
        "allocation_total_bytes: 12288\n"
        "upload_total_bytes: $bytes\n"
        "download_total_bytes: $bytes\n"
        "streaming_required: $streaming\n"
        "sdma_h2d_status: pass\n"
        "sdma_d2h_status: pass\n"
        "cpu_comparison_status: pass\n"
        "host_device_transfer_status: pass\n"
        "failure_stage: none\n"
        "failure_text: none\n"
        "exit_status: 0\n"
        "EOF\n"
    )


def fake_legacy_primitive_diagnostic_script(
    *, native_prefill_acceptance: str | None = None
) -> str:
    markers = """producer_kind: hardware_primitive
primitive_backend: hardware
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201
primitive_name: fp32_add_scalar
kernel_source_id: c1r5-fp32-add-scalar-v1
kernel_blob_sha256: 697ba0c938e34d6f8db6498a803fb1d82181b111b28fe8c60acaac6a8d6011fd
kernel_text_byte_count: 64
element_type: fp32
element_count: 8
input_byte_count: 32
output_byte_count: 32
scalar_bits: 0x3f800000
tolerance: exact_bytes
max_abs_diff: 0
max_ulp_diff: 0
mismatch_count: 0
upload_total_bytes: 32
download_total_bytes: 32
kernel_blob_load_status: pass
kernarg_write_status: pass
kernel_launch_status: pass
sdma_h2d_status: pass
sdma_d2h_status: pass
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
failure_text: none
exit_status: 0
"""
    if native_prefill_acceptance is not None:
        markers += f"native_prefill_acceptance: {native_prefill_acceptance}\n"
    return "#!/bin/sh\ncat <<'EOF'\n" + markers + "EOF\n"


def test_kernel_proof_wraps_supplied_c0_probe_and_logs_hardware_identity(tmp_path, monkeypatch):
    """No GPU is required: this proves the runner wrapper/identity/log contract."""

    exe = compile_runner(tmp_path)
    fake_probe = tmp_path / "fake-c0-probe"
    fake_probe.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface\n"
        "socket_path: /tmp/tinygpu.sock\n"
        "pci_id: 1002:7551\n"
        "arch: gfx1201\n"
        "kernel_blob_load_status: pass\n"
        "kernarg_write_status: pass\n"
        "sdma_h2d_status: pass\n"
        "sdma_d2h_status: pass\n"
        "kernel_launch_status: pass\n"
        "cpu_comparison_status: pass\n"
        "host_device_transfer_status: pass\n"
        "failure_stage: none\n"
        "failure_text: none\n"
        "exit_status: 0\n"
        "EOF\n",
        encoding="utf-8",
    )
    fake_probe.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C0_PROBE", str(fake_probe))

    completed = subprocess.run(
        [str(exe), "--kernel-proof"], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "gated" not in completed.stdout
    assert "gated" not in completed.stderr
    assert "producer_kind: hardware_probe" in completed.stdout
    assert "kernel_launch_status: pass" in completed.stdout
    assert "cpu_comparison_status: pass" in completed.stdout
    assert "host_device_transfer_status: pass" in completed.stdout
    log_lines = [
        line.removeprefix("log_path: ")
        for line in completed.stdout.splitlines()
        if line.startswith("log_path: ")
    ]
    assert log_lines, completed.stdout
    log_path = Path(log_lines[-1])
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "producer_kind: hardware_probe" in log_text
    assert "kernel_launch_status: pass" in log_text


def test_kernel_proof_rejects_missing_c0_hardware_pass_marker(tmp_path, monkeypatch):
    """The C1R wrapper must not pass when a C0A25 marker is omitted."""
    exe = compile_runner(tmp_path)
    fake_probe = tmp_path / "fake-c0-probe"
    fake_probe.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface\n"
        "pci_id: 1002:7551\n"
        "arch: gfx1201\n"
        "kernel_blob_load_status: pass\n"
        "kernarg_write_status: pass\n"
        "sdma_h2d_status: pass\n"
        "kernel_launch_status: pass\n"
        "cpu_comparison_status: pass\n"
        "host_device_transfer_status: pass\n"
        "failure_stage: none\n"
        "failure_text: none\n"
        "exit_status: 0\n"
        "EOF\n",
        encoding="utf-8",
    )
    fake_probe.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C0_PROBE", str(fake_probe))

    completed = subprocess.run(
        [str(exe), "--kernel-proof"], capture_output=True, text=True, check=False
    )

    assert completed.returncode != 0
    assert "kernel_proof_wrapper_status: fail" in completed.stdout
    assert "wrapped C0A25 probe did not report the full hardware pass marker set" in completed.stdout


def test_transfer_proof_wraps_supplied_bridge_and_logs_streaming_transfer(tmp_path, monkeypatch):
    """C1R-4 transfer proof is hardware-gated but wrapper-validatable with a fake bridge."""
    exe = compile_runner(tmp_path)
    fake_bridge = tmp_path / "fake-c1-transfer-bridge"
    fake_bridge.write_text(fake_transfer_bridge_script(), encoding="utf-8")
    fake_bridge.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C1_TRANSFER_BRIDGE", str(fake_bridge))

    completed = subprocess.run(
        [str(exe), "--transfer-proof", "--bytes", C1R4_LAYER_SLICE_BYTES],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "producer_kind: hardware_memory_transfer" in completed.stdout
    assert f"transfer_byte_count: {C1R4_LAYER_SLICE_BYTES}" in completed.stdout
    assert "transfer_proof_wrapper_status: pass" in completed.stdout
    assert "wrapper_exit_status: 0" in completed.stdout
    log_lines = [
        line.removeprefix("log_path: ")
        for line in completed.stdout.splitlines()
        if line.startswith("log_path: ")
    ]
    assert log_lines, completed.stdout
    log_text = Path(log_lines[-1]).read_text(encoding="utf-8")
    assert "transfer_chunk_count: 5" in log_text
    assert "host_device_transfer_status: pass" in log_text


def test_transfer_round_trip_bytes_returns_caller_owned_output(tmp_path, monkeypatch):
    fake_bridge = tmp_path / "fake-c1-transfer-bridge"
    fake_bridge.write_text(fake_transfer_bridge_script(), encoding="utf-8")
    fake_bridge.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C1_TRANSFER_BRIDGE", str(fake_bridge))
    exe = compile_runtime_api_probe(
        tmp_path,
        r'''
#include "runtime.h"

#include <cstdio>
#include <string>
#include <vector>

int main() {
  native_r9700::RuntimeSession session;
  const std::vector<uint8_t> input = {0x00, 0x01, 0xfe, 0xff, 0x10, 0x20, 0x30, 0x40};
  std::vector<uint8_t> output;
  native_r9700::TransferRoundTripResult result;
  std::string error;
  const int status = session.transfer_round_trip_bytes(input, &output, &result, &error);
  if (status != 0 || output != input || result.byte_count != input.size() ||
      result.chunk_count != 1 || result.bridge_command.find("--roundtrip-file") == std::string::npos) {
    std::printf("status=%d error=%s output_size=%zu byte_count=%llu chunk_count=%llu command=%s\n",
                status, error.c_str(), output.size(),
                static_cast<unsigned long long>(result.byte_count),
                static_cast<unsigned long long>(result.chunk_count),
                result.bridge_command.c_str());
    return 1;
  }
  return 0;
}
''',
    )

    completed = subprocess.run([str(exe)], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_transfer_proof_rejects_missing_transfer_marker(tmp_path, monkeypatch):
    exe = compile_runner(tmp_path)
    fake_bridge = tmp_path / "fake-c1-transfer-bridge"
    fake_bridge.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"--roundtrip-file\" ]; then echo bad-arg; exit 7; fi\n"
        "cp \"$2\" \"$3\"\n"
        "cat <<'EOF'\n"
        "producer_kind: hardware_memory_transfer\n"
        "runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface\n"
        "pci_id: 1002:7551\n"
        "arch: gfx1201\n"
        "transfer_byte_count: 20480\n"
        "upload_total_bytes: 20480\n"
        "download_total_bytes: 20480\n"
        "sdma_h2d_status: pass\n"
        "cpu_comparison_status: pass\n"
        "host_device_transfer_status: pass\n"
        "failure_stage: none\n"
        "failure_text: none\n"
        "exit_status: 0\n"
        "EOF\n",
        encoding="utf-8",
    )
    fake_bridge.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C1_TRANSFER_BRIDGE", str(fake_bridge))

    completed = subprocess.run(
        [str(exe), "--transfer-proof", "--bytes", C1R4_LAYER_SLICE_BYTES],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "transfer_proof_wrapper_status: fail" in completed.stdout
    assert "missing_transfer_markers:" in completed.stdout




def test_transfer_proof_rejects_inexact_transfer_marker_value(tmp_path, monkeypatch):
    exe = compile_runner(tmp_path)
    fake_bridge = tmp_path / "fake-c1-transfer-bridge"
    fake_bridge.write_text(fake_transfer_bridge_script(exact_markers=False), encoding="utf-8")
    fake_bridge.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C1_TRANSFER_BRIDGE", str(fake_bridge))

    completed = subprocess.run(
        [str(exe), "--transfer-proof", "--bytes", C1R4_LAYER_SLICE_BYTES],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "transfer_proof_wrapper_status: fail" in completed.stdout
    assert "transfer_chunks_completed=5 (observed 50)" in completed.stdout



def test_legacy_primitive_diagnostic_accepts_explicitly_injected_protocol(tmp_path, monkeypatch):
    exe = compile_runner(tmp_path)
    diagnostic = tmp_path / "fake-legacy-primitive-diagnostic"
    diagnostic.write_text(
        fake_legacy_primitive_diagnostic_script(),
        encoding="utf-8",
    )
    diagnostic.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", str(diagnostic))

    completed = subprocess.run(
        [str(exe), "--legacy-primitive-diagnostic", "fp32_add_scalar"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "producer_kind: legacy_primitive_diagnostic" in completed.stdout
    assert "legacy_diagnostic_status: pass" in completed.stdout
    assert "failure_stage: none" in completed.stdout


def test_legacy_primitive_diagnostic_reports_legacy_proof_unavailable_without_bridge(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [str(exe), "--legacy-primitive-diagnostic", "fp32_add_scalar"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: legacy_proof_unavailable" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout


def test_legacy_primitive_diagnostic_rejects_injected_native_prefill_acceptance(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    diagnostic = tmp_path / "fake-legacy-primitive-diagnostic"
    diagnostic.write_text(
        fake_legacy_primitive_diagnostic_script(native_prefill_acceptance="pass"),
        encoding="utf-8",
    )
    diagnostic.chmod(0o755)
    monkeypatch.setenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", str(diagnostic))

    completed = subprocess.run(
        [str(exe), "--legacy-primitive-diagnostic", "fp32_add_scalar"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "legacy_diagnostic_status: fail" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout


@pytest.mark.parametrize(
    "model_dir,token_ids_json",
    [
        ("missing", "[1]"),
        ("synthetic-model", "[]"),
        ("synthetic-model", "[-1]"),
        ("synthetic-model", "[1.5]"),
        ("synthetic-model", '["1"]'),
        ("synthetic-model", "[1,]"),
    ],
    ids=(
        "missing_model",
        "empty_tokens",
        "negative_token",
        "fractional_token",
        "string_token",
        "malformed_json",
    ),
)
def test_native_prefill_proof_rejects_invalid_request_before_hardware_or_npz(
    tmp_path, model_dir, token_ids_json
):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            model_dir,
            "--token-ids-json",
            token_ids_json,
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "producer_kind: r9700_native" in completed.stdout
    assert "native_prefill_acceptance: open" in completed.stdout
    assert "failure_stage: native_prefill_request" in completed.stdout
    assert not out_path.exists()



def test_native_prefill_proof_reports_layer_kernel_sequence_blocker(tmp_path, monkeypatch):
    """The native prefill seam fails closed at model weight binding, not retired diagnostics."""
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


    assert completed.returncode != 0
    assert "native_prefill_full_layer_loop_status: blocked" in completed.stdout
    assert "failure_stage: layer_weight_table" in completed.stdout
    assert "model directory not found: synthetic-model" in completed.stdout
    assert "legacy_proof_unavailable" not in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert not out_path.exists()

def test_native_prefill_proof_redacts_token_ids_from_stdout_and_hardware_log(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "token_ids_json: <redacted>" in completed.stdout
    assert "[1,2,3]" not in completed.stdout
    assert log_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "token_ids_json: <redacted>" in log_text
    assert "[1,2,3]" not in log_text
    assert "gpu_stage_profile_sample_count: 0" in completed.stdout
    assert "block_tokens: 4" in completed.stdout
    assert "block_count: 0" in completed.stdout
    assert "block_tokens: 4" in log_text
    assert "block_count: 0" in log_text


def test_native_prefill_proof_rejects_equal_output_and_log_paths(tmp_path, monkeypatch):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_conflict" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert not out_path.exists()


def test_native_prefill_proof_rejects_lexically_distinct_output_and_log_path_aliases(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = f"{tmp_path}/./native-prefill.npz"
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            log_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_conflict" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert not out_path.exists()

def test_native_prefill_proof_rejects_parent_symlink_dotdot_output_log_alias(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    output_parent = tmp_path / "physical-output-parent"
    symlink_target = output_parent / "symlink-target"
    symlink_target.mkdir(parents=True)
    symlink_parent = tmp_path / "symlink-parent"
    try:
        symlink_parent.symlink_to(symlink_target, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")
    out_path = output_parent / "native-prefill.npz"
    log_path = symlink_parent / ".." / "native-prefill.npz"
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_conflict" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert not out_path.exists()

def test_native_prefill_proof_rejects_relative_output_and_absolute_log_aliases(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    runner_cwd = tmp_path / "runner-cwd"
    runner_cwd.mkdir()
    target_path = runner_cwd / "native-prefill.npz"
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            "native-prefill.npz",
            "--log",
            str(target_path),
        ],
        cwd=runner_cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_conflict" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert not target_path.exists()


def test_native_prefill_proof_reports_output_cleanup_failure(tmp_path, monkeypatch):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    out_path.mkdir()
    sentinel_path = out_path / "keep"
    sentinel_path.write_text("keep", encoding="utf-8")
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_cleanup" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert out_path.is_dir()
    assert sentinel_path.read_text(encoding="utf-8") == "keep"

def test_native_prefill_proof_rejects_empty_directory_output_target_before_cleanup(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    out_path.mkdir()
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_cleanup" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert out_path.is_dir()


def test_native_prefill_proof_rejects_log_symlink_to_absent_output_target(
    tmp_path, monkeypatch
):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"
    try:
        log_path.symlink_to(out_path)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")
    monkeypatch.delenv("NATIVE_R9700_C1_PRIMITIVE_BRIDGE", raising=False)

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "failure_stage: output_path_conflict" in completed.stdout
    assert "native_prefill_acceptance: pass" not in completed.stdout
    assert not out_path.exists()


@pytest.mark.parametrize(
    "final_argument,expected_failure_stage",
    [
        ("--gpu-stage-profile", "layer_weight_table"),
        ("--gpu-stage-profile=1", "native_prefill_request"),
        ("--unknown", "native_prefill_request"),
    ],
)
def test_native_prefill_gpu_stage_profile_is_a_strict_optional_final_flag(
    tmp_path, final_argument, expected_failure_stage
):
    exe = compile_runner(tmp_path)
    out_path = tmp_path / "native-prefill.npz"
    log_path = tmp_path / "native-prefill.log"

    completed = subprocess.run(
        [
            str(exe),
            "--native-prefill-proof",
            "--model",
            "synthetic-model",
            "--token-ids-json",
            "[1,2,3]",
            "--out",
            str(out_path),
            "--log",
            str(log_path),
            final_argument,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert f"failure_stage: {expected_failure_stage}" in completed.stdout
    assert "token_ids_json: <redacted>" in completed.stdout
    assert "[1,2,3]" not in completed.stdout
    assert "gpu_stage_profile_sample_count: 0" in completed.stdout

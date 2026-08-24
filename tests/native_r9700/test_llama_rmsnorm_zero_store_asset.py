"""Contracts for the trace-only RMSNorm zero-store HSA diagnostic asset."""

import hashlib
import json
from pathlib import Path


SOURCE = Path("native_r9700/kernels/llama_rmsnorm_zero_store_f16.cpp")
ASSET_ROOT = Path("native_r9700/kernels/llama-rmsnorm-zero-store-hsa-assets")
MANIFEST_PATH = ASSET_ROOT / "llama_rmsnorm_zero_store_f16.json"
IMAGE_PATH = ASSET_ROOT / "llama_rmsnorm_zero_store_f16.image"
EPSILON_SOURCE = Path("native_r9700/kernels/llama_rmsnorm_epsilon_arithmetic_f16.cpp")
EPSILON_ASSET_ROOT = Path("native_r9700/kernels/llama-rmsnorm-epsilon-arithmetic-hsa-assets")
EPSILON_MANIFEST_PATH = EPSILON_ASSET_ROOT / "llama_rmsnorm_epsilon_arithmetic_f16.json"
EPSILON_IMAGE_PATH = EPSILON_ASSET_ROOT / "llama_rmsnorm_epsilon_arithmetic_f16.image"

GENERATOR = Path("experiments/native-r9700-runtime/generate_hsa_code_image.py")
EXECUTOR = Path("native_r9700/llama_layer_executor.cpp")
TRACE_RUNTIME = Path("native_r9700/runtime_contract.cpp")
RUNTIME_HEADER = Path("native_r9700/runtime.h")
RUNNER = Path("native_r9700/runner.cpp")
KERNEL_ASSETS = Path("native_r9700/kernel_assets.cpp")

KERNARG_SCHEMA = {
    "name": "llama-rmsnorm-f16-v1",
    "bytes": 32,
    "fields": [
        {"name": "hidden_input", "offset": 0, "type": "uint64"},
        {"name": "scale", "offset": 8, "type": "uint64"},
        {"name": "hidden_output", "offset": 16, "type": "uint64"},
        {"name": "epsilon", "offset": 24, "type": "float32"},
    ],
}


def test_rmsnorm_zero_store_source_and_manifest_are_digest_bound_gfx1201_asset() -> None:
    """The diagnostic is a real 32-byte-ABI gfx1201 HSA asset, not a host shortcut."""
    source = SOURCE.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    image = IMAGE_PATH.read_bytes()

    assert 'extern "C" __attribute__((global)) void llama_rmsnorm_zero_store_f16(' in source
    assert "const unsigned short* hidden_input" in source
    assert "const unsigned short* scale" in source
    assert "unsigned short* hidden_output" in source
    assert "float epsilon" in source
    assert "__builtin_amdgcn_workgroup_id_x" in source
    assert "__builtin_amdgcn_workitem_id_x" in source
    assert "row * 2048ULL" in source
    assert "column < 2048U" in source
    assert "hidden_output[row_offset + column] = 0U;" in source

    assert manifest["name"] == "llama_rmsnorm_zero_store_f16"
    assert manifest["target"] == "gfx1201"
    assert manifest["kernarg_schema"] == KERNARG_SCHEMA
    assert manifest["source_path"] == SOURCE.as_posix()
    assert manifest["image_path"] == IMAGE_PATH.name
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert manifest["image_size"] == len(image)
    assert manifest["entry_offset"] % 256 == 0
    assert manifest["elf_admission"]["symbol_target_count"] == 1


def test_rmsnorm_epsilon_arithmetic_source_and_manifest_are_digest_bound_gfx1201_asset() -> None:
    """The arithmetic probe retains RMSNorm's ABI and derives its output from epsilon."""
    source = EPSILON_SOURCE.read_text(encoding="utf-8")
    manifest = json.loads(EPSILON_MANIFEST_PATH.read_text(encoding="utf-8"))
    image = EPSILON_IMAGE_PATH.read_bytes()

    assert 'extern "C" __attribute__((global)) void llama_rmsnorm_epsilon_arithmetic_f16(' in source
    assert "const unsigned short* hidden_input" in source
    assert "const unsigned short* scale" in source
    assert "unsigned short* hidden_output" in source
    assert "float epsilon" in source
    assert "zero_input_mean_square + epsilon" in source
    assert "1.0f / __builtin_sqrtf" in source
    assert "(_Float16)inverse_rms" in source
    assert "row * 2048ULL" in source
    assert "column < 2048U" in source

    assert manifest["name"] == "llama_rmsnorm_epsilon_arithmetic_f16"
    assert manifest["target"] == "gfx1201"
    assert manifest["kernarg_schema"] == KERNARG_SCHEMA
    assert manifest["source_path"] == EPSILON_SOURCE.as_posix()
    assert manifest["image_path"] == EPSILON_IMAGE_PATH.name
    assert manifest["source_sha256"] == hashlib.sha256(EPSILON_SOURCE.read_bytes()).hexdigest()
    assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert manifest["image_size"] == len(image)
    assert manifest["entry_offset"] % 256 == 0
    assert manifest["elf_admission"]["symbol_target_count"] == 1


def test_rmsnorm_zero_store_is_trace_only_and_keeps_prefill_rmsnorm_asset() -> None:
    """The diagnostic substitutes only the bounded normalized trace dispatch."""
    generator = GENERATOR.read_text(encoding="utf-8")
    executor = EXECUTOR.read_text(encoding="utf-8")
    runtime = TRACE_RUNTIME.read_text(encoding="utf-8")
    header = RUNTIME_HEADER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    kernel_assets = KERNEL_ASSETS.read_text(encoding="utf-8")

    assert "llama_rmsnorm_epsilon_arithmetic_f16" in generator
    assert "llama_rmsnorm_epsilon_arithmetic_f16" in executor
    assert '"llama_rmsnorm_epsilon_arithmetic_f16"' in kernel_assets
    assert "e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037" in kernel_assets
    assert "llama_rmsnorm_zero_store_f16" in generator
    assert "llama_rmsnorm_zero_store_f16" in executor
    assert '"llama_rmsnorm_f16", "native_r9700/kernels/llama-rmsnorm-hsa-assets"' in executor
    assert '"llama_rmsnorm_zero_store_f16"' in kernel_assets
    assert "8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47" in kernel_assets
    assert "request.rmsnorm_zero_store" in runtime
    assert "request.rmsnorm_output_sentinel" in runtime
    assert "request.rmsnorm_epsilon_arithmetic" in runtime
    assert "rmsnorm_expected_output" in runtime
    assert "f16_0x5cf1_316.25" in runtime
    assert "trace_expected_output" in runtime
    assert 'request.stage != "normalized"' in runtime
    assert "rmsnorm_kernel" in runtime
    assert "bool rmsnorm_zero_store = false;" in header
    assert "--rmsnorm-zero-store" in runner
    assert "bool rmsnorm_epsilon_arithmetic = false;" in header
    assert "--rmsnorm-epsilon-arithmetic" in runner
    trace_builder = executor[executor.index("bool build_llama_layer0_stage_trace_dispatch"):]
    persistent_builder = executor[executor.index("bool build_llama_persistent_dispatch"):]
    assert "stage.workgroup_x = 64;" in executor
    assert "append_stage(0, {{0, 0}, {1, 8}, {11, 16}}, {{24, 0x3727c5acU}}, 1);" in executor
    assert "kLlamaRmsNormZeroStoreTraceAssetConfig" in trace_builder
    assert "kLlamaRmsNormEpsilonArithmeticTraceAssetConfig" in trace_builder
    assert "kLlamaRmsNormEpsilonArithmeticTraceAssetConfig" not in persistent_builder
    assert "kLlamaRmsNormZeroStoreTraceAssetConfig" not in persistent_builder

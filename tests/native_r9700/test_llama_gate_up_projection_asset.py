"""No-hardware contracts for the cooperative Llama gate/up projection asset."""

import hashlib
import json
from pathlib import Path
import struct


SOURCE = Path("native_r9700/kernels/llama_gate_up_projection_f16.cpp")
ASSET_ROOT = Path("native_r9700/kernels/llama-gate-up-projection-hsa-assets")
IMAGE = ASSET_ROOT / "llama_gate_up_projection_f16.image"
MANIFEST = ASSET_ROOT / "llama_gate_up_projection_f16.json"
KERNEL_ASSETS_SOURCE = Path("native_r9700/kernel_assets.cpp")
EXPECTED_SCHEMA = {
    "name": "llama-gate-up-projection-f16-v1",
    "bytes": 56,
    "fields": [
        {"name": "post_attention_hidden", "offset": 0, "type": "uint64"},
        {"name": "post_attention_layernorm_weight", "offset": 8, "type": "uint64"},
        {"name": "gate_projection_weight", "offset": 16, "type": "uint64"},
        {"name": "up_projection_weight", "offset": 24, "type": "uint64"},
        {"name": "gate_output", "offset": 32, "type": "uint64"},
        {"name": "up_output", "offset": 40, "type": "uint64"},
        {"name": "sequence_length", "offset": 48, "type": "uint32"},
    ],
}


def test_gate_up_source_uses_one_cooperative_shared_rmsnorm_tile() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    compact = " ".join(source.split())
    assert "__attribute__((shared)) unsigned short normalized_tile[2048]" in source
    assert "__attribute__((shared)) float shared_inverse_rms" in source
    assert source.count("__builtin_amdgcn_s_barrier()") == 2
    assert source.count('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup")') == 2
    assert source.count('__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup")') == 2
    assert "if (lane == 0U)" in source
    assert "normalized_tile[column] = raw_bits" in source
    assert "shared_inverse_rms = 1.0f / __builtin_sqrtf" in source
    assert "column = lane; column < kHiddenSize; column += kLanesPerWorkgroup" in compact
    assert "post_attention_layernorm_weight[column]" in source
    assert "normalized_tile[column] = __builtin_bit_cast" in compact
    assert "_Float16, normalized_tile[column]" in compact
    assert source.count("post_attention_hidden[") == 1


def test_gate_up_asset_is_digest_bound_and_uses_exact_compiler_descriptor() -> None:
    image = IMAGE.read_bytes()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    descriptor_offset = manifest["descriptor_offset"]

    assert manifest["name"] == "llama_gate_up_projection_f16"
    assert manifest["target"] == "gfx1201"
    assert manifest["kernarg_schema"] == EXPECTED_SCHEMA
    assert manifest["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert manifest["image_size"] == len(image)
    assert manifest["elf_admission"]["relocation_count"] == 0
    assert struct.unpack_from("<IIQ", image, descriptor_offset) == (4100, 0, 56)
    assert struct.unpack_from("<H", image, descriptor_offset + 56)[0] == 0x408
    assert struct.unpack_from("<H", image, descriptor_offset + 58)[0] == 0
    assert manifest["group_segment_bytes"] == 4100
    assert manifest["private_segment_bytes"] == 0
    assert manifest["kernarg_bytes"] == 56
    assert manifest["kernel_code_properties"] == 0x408
    assert manifest["kernarg_preload_bytes"] == 0
    assert manifest["entry_offset"] == descriptor_offset + struct.unpack_from(
        "<q", image, descriptor_offset + 16
    )[0]
    raw_resources = {
        "descriptor_rsrc3": struct.unpack_from("<I", image, descriptor_offset + 44)[0],
        "descriptor_rsrc1": struct.unpack_from("<I", image, descriptor_offset + 48)[0],
        "descriptor_rsrc2": struct.unpack_from("<I", image, descriptor_offset + 52)[0],
    }
    for name, value in raw_resources.items():
        assert manifest[name] == value
        assert value > 0
    assert manifest["rsrc1"] == raw_resources["descriptor_rsrc1"]
    assert manifest["rsrc3"] == raw_resources["descriptor_rsrc3"]
    lds_size = ((manifest["group_segment_bytes"] + 511) // 512) & 0x1FF
    assert lds_size == 9
    assert manifest["rsrc2"] == raw_resources["descriptor_rsrc2"] | (lds_size << 15)
    assert manifest["rsrc2"] == 0x00048084


def test_kernel_asset_manifest_uses_generated_gate_up_digest_and_lds_bytes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = KERNEL_ASSETS_SOURCE.read_text(encoding="utf-8")
    gate_up = source.index('{"llama_gate_up_projection_f16"')
    entry_end = source.index("llama-gate-up-projection-f16-v1", gate_up)
    entry = source[gate_up:entry_end]

    assert entry.count(manifest["image_sha256"]) == 2
    assert '"gfx1201", 0, 0, 4100, "source_amdgpu_metadata"' in entry
    assert f"{{}}, {manifest['rsrc1']}U, {manifest['rsrc2']}U, {manifest['rsrc3']}U" in entry
    assert "64U, 1U, 1U, 64U, 1U, 1U, 56U" in entry

"""No-hardware contracts for reviewed native Llama K/V HSA assets."""

import hashlib
import json
from pathlib import Path


ASSETS = (
    (
        Path("native_r9700/kernels/llama-k-projection-hsa-assets"),
        "llama_k_projection_f16",
        "llama-k-projection-f16-v1",
        "native_r9700/kernels/llama_k_projection_f16.cpp",
    ),
    (
        Path("native_r9700/kernels/llama-v-projection-hsa-assets"),
        "llama_v_projection_f16",
        "llama-v-projection-f16-v1",
        "native_r9700/kernels/llama_v_projection_f16.cpp",
    ),
)


def test_kv_projection_assets_bind_each_device_source_to_its_exact_abi() -> None:
    for root, kernel_name, schema_name, source_path in ASSETS:
        manifest_path = root / f"{kernel_name}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_path = root / manifest["image_path"]
        image = image_path.read_bytes()
        assert manifest["name"] == kernel_name
        assert manifest["target"] == "gfx1201"
        assert manifest["kernarg_schema"]["name"] == schema_name
        assert manifest["kernarg_schema"]["bytes"] == 32
        assert manifest["source_path"] == source_path
        assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
        assert manifest["image_size"] == len(image)
        assert manifest["rsrc1"] > 0 and manifest["rsrc2"] > 0 and manifest["rsrc3"] > 0


def test_kv_projection_assets_keep_separate_fresh_k_and_fresh_v_sources() -> None:
    k_manifest = json.loads(
        (ASSETS[0][0] / "llama_k_projection_f16.json").read_text(encoding="utf-8")
    )
    v_manifest = json.loads(
        (ASSETS[1][0] / "llama_v_projection_f16.json").read_text(encoding="utf-8")
    )
    assert k_manifest["source_path"] != v_manifest["source_path"]
    assert k_manifest["image_sha256"] != v_manifest["image_sha256"]

"""No-hardware contracts for reviewed text-only Qwen native HSA assets."""

import hashlib
import json
from pathlib import Path


ASSETS = (
    (
        Path("native_r9700/kernels/qwen-affine4-hsa-assets"),
        "qwen_affine4_linear",
        "qwen-affine4-linear-v1",
        "native_r9700/kernels/qwen_affine4_linear.cpp",
    ),
    (
        Path("native_r9700/kernels/qwen-deltanet-hsa-assets"),
        "qwen_deltanet_state",
        "qwen-deltanet-state-v1",
        "native_r9700/kernels/qwen_deltanet_state.cpp",
    ),
    (
        Path("native_r9700/kernels/qwen-full-attention-hsa-assets"),
        "qwen_full_attention",
        "qwen-full-attention-v1",
        "native_r9700/kernels/qwen_full_attention.cpp",
    ),
)


def test_qwen_text_stage_hsa_assets_are_manifest_bound() -> None:
    for root, kernel_name, schema_name, source_path in ASSETS:
        manifest = json.loads((root / f"{kernel_name}.json").read_text(encoding="utf-8"))
        image = (root / manifest["image_path"]).read_bytes()
        assert manifest["name"] == kernel_name
        assert manifest["target"] == "gfx1201"
        assert manifest["kernarg_schema"]["name"] == schema_name
        assert manifest["source_path"] == source_path
        assert manifest["image_sha256"] == hashlib.sha256(image).hexdigest()
        assert manifest["image_size"] == len(image)
        assert manifest["rsrc1"] > 0 and manifest["rsrc2"] > 0 and manifest["rsrc3"] > 0


def test_qwen_affine_asset_carries_every_raw_window_capacity() -> None:
    manifest = json.loads((ASSETS[0][0] / "qwen_affine4_linear.json").read_text(encoding="utf-8"))
    assert [field["name"] for field in manifest["kernarg_schema"]["fields"]] == [
        "input", "packed_weight", "scales", "biases", "output", "input_features",
        "output_features", "input_capacity_elements", "packed_weight_capacity_bytes",
        "affine_group_capacity", "output_capacity_elements",
    ]


def test_qwen_stage_sources_exclude_llama_multimodal_and_host_paths() -> None:
    forbidden = ("llama", "vision", "image", "video", "fixture", "archive", "cpu")
    for _, _, _, source_path in ASSETS:
        source = Path(source_path).read_text(encoding="utf-8").lower()
        assert not any(marker in source for marker in forbidden)

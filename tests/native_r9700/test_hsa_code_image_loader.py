"""No-hardware contract for loading the generated Llama HSA code image."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


HSA_CODE_IMAGE_HEADER = Path("native_r9700/hsa_code_image_asset.h")
HSA_CODE_IMAGE_SOURCE = Path("native_r9700/hsa_code_image_asset.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")
ASSET_DIRECTORY = Path("native_r9700/kernels/llama-hsa-assets")
IMAGE_NAME = "llama_embed_row_f16.image"
MANIFEST_NAME = "llama_embed_row_f16.json"
IMAGE_SHA256 = "389d8726a5a3e0d827f05680fb73b0d08e2dade34d8ae2ef79d5010c0bfdb53e"
SOURCE_PATH = "native_r9700/kernels/llama_embed_row_f16.cpp"
SOURCE_SHA256 = "a4c6be25193895d54549530beb9c3224addda22562999c8bc949a6d87153043f"
CANONICAL_SCHEMA = (
    '{"name":"llama-embed-row-f16-v1","bytes":24,"fields":['
    '{"name":"embedding_rows","offset":0,"type":"uint64"},'
    '{"name":"hidden_output","offset":8,"type":"uint64"},'
    '{"name":"selected_row","offset":16,"type":"uint64"}]}'
)
GENERATOR = Path("experiments/native-r9700-runtime/generate_hsa_code_image.py")
FRESH_HIP_SOURCE = Path("native_r9700/kernels/llama_embed_row_f16.cpp")
WORKSPACE_TINYGRAD_ROOT = Path(__file__).resolve().parents[5] / "tinygrad"


def _copy_asset_directory(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(ASSET_DIRECTORY, destination)
    return destination


def _load_manifest(asset_dir: Path) -> dict[str, object]:
    return json.loads((asset_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(asset_dir: Path, manifest: dict[str, object]) -> None:
    (asset_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )


def _compile_loader_probe(tmp_path: Path) -> Path:
    """Compile the asset boundary only; the probe opens neither driver nor device."""
    assert HSA_CODE_IMAGE_HEADER.is_file(), "HSA code-image loader header is missing"
    assert HSA_CODE_IMAGE_SOURCE.is_file(), "HSA code-image loader source is missing"
    probe_source = tmp_path / "hsa_code_image_loader_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "hsa_code_image_asset.h"

namespace {

constexpr const char* kImageSha256 = "389d8726a5a3e0d827f05680fb73b0d08e2dade34d8ae2ef79d5010c0bfdb53e";
constexpr const char* kSourcePath = "native_r9700/kernels/llama_embed_row_f16.cpp";
constexpr const char* kSourceSha256 = "a4c6be25193895d54549530beb9c3224addda22562999c8bc949a6d87153043f";
constexpr const char* kSchema = R"({"name":"llama-embed-row-f16-v1","bytes":24,"fields":[{"name":"embedding_rows","offset":0,"type":"uint64"},{"name":"hidden_output","offset":8,"type":"uint64"},{"name":"selected_row","offset":16,"type":"uint64"}]})";

native_r9700::HsaCodeImageAsset sentinel_asset() {
  native_r9700::HsaCodeImageAsset asset;
  asset.image = {0xde, 0xad, 0xbe, 0xef};
  asset.image_sha256 = "sentinel-image-sha";
  asset.descriptor_offset = 101;
  asset.entry_offset = 202;
  asset.rsrc1 = 303;
  asset.rsrc2 = 404;
  asset.rsrc3 = 505;
  asset.schema = "sentinel-schema";
  asset.source_path = "sentinel-source";
  asset.source_sha256 = "sentinel-source-sha";
  return asset;
}

bool is_sentinel(const native_r9700::HsaCodeImageAsset& asset) {
  return asset.image == std::vector<std::uint8_t>({0xde, 0xad, 0xbe, 0xef}) &&
         asset.image_sha256 == "sentinel-image-sha" &&
         asset.descriptor_offset == 101 && asset.entry_offset == 202 &&
         asset.rsrc1 == 303 && asset.rsrc2 == 404 && asset.rsrc3 == 505 &&
         asset.schema == "sentinel-schema" && asset.source_path == "sentinel-source" &&
         asset.source_sha256 == "sentinel-source-sha";
}

int valid(const std::filesystem::path& asset_dir) {
  native_r9700::HsaCodeImageAsset asset;
  std::string error;
  if (!native_r9700::load_llama_embed_hsa_image(asset_dir, &asset, &error)) return 1;
  if (!error.empty()) return 2;
  if (asset.image.size() != 14833 || asset.image_sha256 != kImageSha256) return 3;
  if (asset.descriptor_offset != 1536 || asset.entry_offset != 5888) return 4;
  if (asset.rsrc1 != 3222208512U || asset.rsrc2 != 132U || asset.rsrc3 != 32U) return 5;
  if (asset.schema != kSchema) return 6;
  if (asset.source_path != kSourcePath || asset.source_sha256 != kSourceSha256) return 7;
  return 0;
}

int rejected_preserves_output(const std::filesystem::path& asset_dir) {
  native_r9700::HsaCodeImageAsset asset = sentinel_asset();
  std::string error;
  if (native_r9700::load_llama_embed_hsa_image(asset_dir, &asset, &error)) return 1;
  if (!is_sentinel(asset)) return 2;
  return error.empty() ? 3 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 64;
  const std::string mode = argv[1];
  const std::filesystem::path asset_dir(argv[2]);
  if (mode == "valid") return valid(asset_dir);
  if (mode == "reject") return rejected_preserves_output(asset_dir);
  return 65;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "hsa_code_image_loader_probe"
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
            str(HSA_CODE_IMAGE_SOURCE),
            str(probe_source),
            "-I",
            str(NATIVE_INCLUDE_DIR),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return executable


def _run_probe(executable: Path, mode: str, asset_dir: Path) -> None:
    completed = subprocess.run(
        [str(executable), mode, str(asset_dir)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_loader_accepts_fresh_regenerated_default_zero_lds_manifest(
    tmp_path: Path,
) -> None:
    """Default-zero generation remains exactly compatible with the strict embed loader."""
    if not WORKSPACE_TINYGRAD_ROOT.is_dir():
        pytest.skip("optional capability: no workspace Tinygrad checkout")
    output_dir = tmp_path / "fresh-default-zero-lds"
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--source",
            str(FRESH_HIP_SOURCE),
            "--target",
            "gfx1201",
            "--tinygrad-root",
            str(WORKSPACE_TINYGRAD_ROOT),
            "--schema",
            CANONICAL_SCHEMA,
            "--out-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = _load_manifest(output_dir)
    assert not {
        "descriptor_rsrc1",
        "descriptor_rsrc2",
        "descriptor_rsrc3",
        "group_segment_bytes",
        "private_segment_bytes",
        "kernarg_bytes",
        "kernel_code_properties",
        "kernarg_preload_bytes",
    } & manifest.keys()
    _run_probe(_compile_loader_probe(tmp_path), "valid", output_dir)


def test_loader_admits_only_the_generated_manifest_bound_hsa_image(tmp_path: Path) -> None:
    """The loader returns one attested image and fails closed before device work."""
    assert ASSET_DIRECTORY.is_dir(), "generated Llama HSA asset directory is missing"
    assert not ASSET_DIRECTORY.is_symlink(), "generated Llama HSA asset directory must be real"
    assert (ASSET_DIRECTORY / IMAGE_NAME).is_file()
    assert (ASSET_DIRECTORY / MANIFEST_NAME).is_file()
    assert hashlib.sha256((ASSET_DIRECTORY / IMAGE_NAME).read_bytes()).hexdigest() == IMAGE_SHA256
    manifest = _load_manifest(ASSET_DIRECTORY)
    assert manifest["source_path"] == SOURCE_PATH
    assert manifest["source_sha256"] == SOURCE_SHA256
    assert manifest["kernarg_schema"] == json.loads(CANONICAL_SCHEMA)

    executable = _compile_loader_probe(tmp_path)
    _run_probe(executable, "valid", ASSET_DIRECTORY)

    modified_image = _copy_asset_directory(tmp_path, "modified-image")
    image_path = modified_image / IMAGE_NAME
    image = bytearray(image_path.read_bytes())
    image[-1] ^= 1
    image_path.write_bytes(image)
    _run_probe(executable, "reject", modified_image)

    malformed_manifest = _copy_asset_directory(tmp_path, "malformed-manifest")
    (malformed_manifest / MANIFEST_NAME).write_text("{", encoding="utf-8")
    _run_probe(executable, "reject", malformed_manifest)

    escaped_image = _copy_asset_directory(tmp_path, "escaped-image")
    escaped_path = escaped_image / IMAGE_NAME
    escaped_path.unlink()
    escaped_path.symlink_to((ASSET_DIRECTORY / IMAGE_NAME).resolve())
    _run_probe(executable, "reject", escaped_image)
    escaped_manifest = _copy_asset_directory(tmp_path, "escaped-manifest")
    escaped_path = escaped_manifest / MANIFEST_NAME
    escaped_path.unlink()
    escaped_path.symlink_to((ASSET_DIRECTORY / MANIFEST_NAME).resolve())
    _run_probe(executable, "reject", escaped_manifest)


    zero_entry = _copy_asset_directory(tmp_path, "zero-entry")
    manifest = _load_manifest(zero_entry)
    manifest["entry_offset"] = 0
    _write_manifest(zero_entry, manifest)
    _run_probe(executable, "reject", zero_entry)

    invalid_entry = _copy_asset_directory(tmp_path, "invalid-entry")
    manifest = _load_manifest(invalid_entry)
    manifest["entry_offset"] = 5889
    _write_manifest(invalid_entry, manifest)
    _run_probe(executable, "reject", invalid_entry)
    invalid_descriptor = _copy_asset_directory(tmp_path, "invalid-descriptor")
    manifest = _load_manifest(invalid_descriptor)
    manifest["descriptor_offset"] = 1537
    _write_manifest(invalid_descriptor, manifest)
    _run_probe(executable, "reject", invalid_descriptor)


    wrong_schema = _copy_asset_directory(tmp_path, "wrong-schema")
    manifest = _load_manifest(wrong_schema)
    schema = manifest["kernarg_schema"]
    assert isinstance(schema, dict)
    schema["bytes"] = 32
    _write_manifest(wrong_schema, manifest)
    _run_probe(executable, "reject", wrong_schema)

    raw_fallback = _copy_asset_directory(tmp_path, "raw-fallback")
    fallback_path = raw_fallback / "llama_embed_row_f16.code"
    fallback_path.write_bytes((raw_fallback / IMAGE_NAME).read_bytes())
    manifest = _load_manifest(raw_fallback)
    manifest["image_path"] = fallback_path.name
    manifest["image_sha256"] = hashlib.sha256(fallback_path.read_bytes()).hexdigest()
    _write_manifest(raw_fallback, manifest)
    _run_probe(executable, "reject", raw_fallback)

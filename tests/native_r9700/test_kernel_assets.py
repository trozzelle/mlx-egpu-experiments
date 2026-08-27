"""RED contracts for verified, file-backed Llama kernel assets."""

from pathlib import Path
import subprocess


KERNEL_CATALOG_SOURCE = Path("native_r9700/kernel_catalog.cpp")
KERNEL_ASSETS_HEADER = Path("native_r9700/kernel_assets.h")
KERNEL_ASSETS_SOURCE = Path("native_r9700/kernel_assets.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")


def compile_kernel_assets_probe(tmp_path: Path) -> Path:
    """Compile the public asset-loader contract without a GPU or driver."""
    assert KERNEL_CATALOG_SOURCE.is_file(), "kernel catalog source is missing"
    assert KERNEL_ASSETS_HEADER.is_file(), "kernel assets header is missing"
    assert KERNEL_ASSETS_SOURCE.is_file(), "kernel assets source is missing"

    probe_source = tmp_path / "kernel_assets_probe.cpp"
    probe_source.write_text(
        r'''
#include <CommonCrypto/CommonDigest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/stat.h>
#include <vector>

#include "kernel_assets.h"

namespace {

constexpr const char* kCodeDigest =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
constexpr const char* kSchema = "task9-kernarg-v1";
constexpr std::size_t kMaxCodeBytes = 4 * 1024 * 1024;

native_r9700::KernelDescriptor descriptor(std::string digest = kCodeDigest) {
  native_r9700::KernelDescriptor value;
  value.name = "future-llama-kernel";
  value.sha256 = std::move(digest);
  value.rsrc1 = 1;
  value.rsrc2 = 1;
  value.rsrc3 = 1;
  value.workgroup_x = 1;
  value.workgroup_y = 1;
  value.workgroup_z = 1;
  value.global_x = 1;
  value.global_y = 1;
  value.global_z = 1;
  value.kernarg_bytes = 64;
  return value;
}

native_r9700::LlamaKernelAsset asset(std::string code_path = "kernel.code") {
  native_r9700::LlamaKernelAsset value;
  value.descriptor = descriptor();
  value.location.code_path = std::move(code_path);
  value.location.sha256 = kCodeDigest;
  value.location.target = "gfx1201";
  value.location.sgpr_count = 1;
  value.location.vgpr_count = 1;
  value.location.lds_bytes = 0;
  value.location.resource_metadata_provenance = "source_amdgpu_metadata";
  value.kernarg_schema = kSchema;
  return value;
}

bool same_descriptor(const native_r9700::KernelDescriptor& left,
                     const native_r9700::KernelDescriptor& right) {
  return left.name == right.name && left.sha256 == right.sha256 &&
         left.code == right.code && left.rsrc1 == right.rsrc1 &&
         left.rsrc2 == right.rsrc2 && left.rsrc3 == right.rsrc3 &&
         left.workgroup_x == right.workgroup_x &&
         left.workgroup_y == right.workgroup_y &&
         left.workgroup_z == right.workgroup_z &&
         left.global_x == right.global_x && left.global_y == right.global_y &&
         left.global_z == right.global_z && left.kernarg_bytes == right.kernarg_bytes;
}

bool rejects_without_output_mutation(const native_r9700::LlamaKernelAsset& value,
                                     const std::filesystem::path& asset_root,
                                     std::string_view schema) {
  native_r9700::KernelDescriptor output = descriptor("sentinel-digest");
  const native_r9700::KernelDescriptor original = output;
  std::string error_text;
  return !native_r9700::load_verified_kernel_code(value, asset_root, schema,
                                                   &output, &error_text) &&
         !error_text.empty() && same_descriptor(output, original);
}

bool write_code(const std::filesystem::path& path) {
  std::ofstream stream(path, std::ios::binary);
  stream.write("abc", 3);
  return stream.good();
}

bool write_sparse_code(const std::filesystem::path& path) {
  std::ofstream stream(path, std::ios::binary);
  stream.write("abc", 3);
  stream.seekp(kMaxCodeBytes);
  stream.put('\0');
  return stream.good();
}

std::string sha256(const std::vector<uint8_t>& code) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH] = {};
  CC_SHA256(code.data(), static_cast<CC_LONG>(code.size()), digest);
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string text;
  text.reserve(CC_SHA256_DIGEST_LENGTH * 2);
  for (unsigned char byte : digest) {
    text.push_back(kHexDigits[byte >> 4]);
    text.push_back(kHexDigits[byte & 0x0f]);
  }
  return text;
}

}  // namespace

int main() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "native-r9700-kernel-assets-probe";
  std::error_code error;
  std::filesystem::remove_all(root, error);
  if (error || !std::filesystem::create_directories(root, error) || error) return 1;
  if (!write_code(root / "kernel.code")) return 2;
  if (!std::filesystem::create_directories(root / "assets", error) || error ||
      !write_code(root / "assets" / "kernel.code")) {
    return 23;
  }

  const native_r9700::LlamaKernelAsset valid = asset();
  if (!valid.descriptor.code.empty()) return 3;
  native_r9700::KernelDescriptor loaded;
  std::string error_text;
  if (!native_r9700::load_verified_kernel_code(valid, root, kSchema, &loaded, &error_text)) {
    return 4;
  }
  if (loaded.code != std::vector<uint8_t>({'a', 'b', 'c'}) ||
      !native_r9700::validate_kernel_descriptors({loaded}, &error_text)) {
    return 5;
  }

  const std::filesystem::path fifo_path = root / "fifo.code";
  if (::mkfifo(fifo_path.c_str(), 0600) != 0) return 27;
  if (!rejects_without_output_mutation(asset("fifo.code"), root, kSchema)) return 28;

  if (!rejects_without_output_mutation(asset("assets/kernel.code"), root, kSchema)) return 24;

  const std::filesystem::path oversized_path = root / "oversized.code";
  if (!write_sparse_code(oversized_path)) return 25;
  native_r9700::LlamaKernelAsset oversized = asset("oversized.code");
  std::vector<uint8_t> oversized_code(kMaxCodeBytes + 1);
  oversized_code[0] = 'a';
  oversized_code[1] = 'b';
  oversized_code[2] = 'c';
  oversized.location.sha256 = sha256(oversized_code);
  oversized.descriptor.sha256 = oversized.location.sha256;
  if (!rejects_without_output_mutation(oversized, root, kSchema)) return 26;

  constexpr const char* kLlamaNames[] = {
      "llama_k_projection_f16",
      "llama_v_projection_f16",
      "llama_rmsnorm_f16",
      "llama_rmsnorm_zero_store_f16",
      "llama_rmsnorm_epsilon_arithmetic_f16",
      "llama_rope_kv_f16",
      "llama_causal_attention_score_f16",
      "llama_causal_attention_softmax_f32",
      "llama_causal_attention_context_f16",
      "llama_o_projection_f16",
      "llama_gated_mlp_f16",
      "llama_gate_up_projection_f16",
      "llama_mlp_down_f16",
  };
  for (const char* name : kLlamaNames) {
    const native_r9700::LlamaKernelAsset* reviewed =
        native_r9700::find_llama_kernel_asset(name);
    const native_r9700::KernelAssetPackAttestation* attestation =
        native_r9700::find_kernel_pack_attestation(name);
    if (reviewed == nullptr || attestation == nullptr ||
        reviewed->descriptor.name != name ||
        std::string(attestation->image_path) != std::string(name) + ".image" ||
        std::string(attestation->image_sha256) != reviewed->descriptor.sha256 ||
        attestation->kernarg_bytes == 0 || attestation->kernarg_field_count == 0) {
      return 29;
    }
  }
  if (native_r9700::find_kernel_pack_attestation("not-a-llama-kernel") != nullptr) {
    return 30;
  }

  if (native_r9700::find_llama_kernel_asset("not-a-llama-kernel") != nullptr) return 6;
  if (native_r9700::find_kernel("future-llama-kernel") != nullptr ||
      native_r9700::find_kernel("c0-add-one") != nullptr ||
      native_r9700::find_llama_kernel_asset("c0-add-one") != nullptr) {
    return 7;
  }

  native_r9700::LlamaKernelAsset wrong_target = valid;
  wrong_target.location.target = "gfx1100";
  if (!rejects_without_output_mutation(wrong_target, root, kSchema)) return 9;

  if (!rejects_without_output_mutation(valid, root, "other-kernarg-schema")) return 10;
  if (!rejects_without_output_mutation(valid, root, "")) return 21;
  native_r9700::LlamaKernelAsset empty_schema = valid;
  empty_schema.kernarg_schema.clear();
  if (!rejects_without_output_mutation(empty_schema, root, kSchema)) return 11;

  native_r9700::LlamaKernelAsset wrong_provenance = valid;
  wrong_provenance.location.resource_metadata_provenance = "generated";
  if (!rejects_without_output_mutation(wrong_provenance, root, kSchema)) return 12;

  native_r9700::LlamaKernelAsset mismatch_location_digest = valid;
  mismatch_location_digest.location.sha256 =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  if (!rejects_without_output_mutation(mismatch_location_digest, root, kSchema)) return 13;

  native_r9700::LlamaKernelAsset mismatch_descriptor_digest = valid;
  mismatch_descriptor_digest.descriptor.sha256 =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  if (!rejects_without_output_mutation(mismatch_descriptor_digest, root, kSchema)) return 14;
  native_r9700::LlamaKernelAsset uppercase_digest = valid;
  uppercase_digest.location.sha256 =
      "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD";
  uppercase_digest.descriptor.sha256 = uppercase_digest.location.sha256;
  if (!rejects_without_output_mutation(uppercase_digest, root, kSchema)) return 22;

  native_r9700::LlamaKernelAsset embedded_code = valid;
  embedded_code.descriptor.code = {'a', 'b', 'c'};
  if (!rejects_without_output_mutation(embedded_code, root, kSchema)) return 15;

  if (!rejects_without_output_mutation(asset("assets"), root, kSchema)) return 16;
  if (!rejects_without_output_mutation(asset("../outside.code"), root, kSchema)) return 17;
  if (!rejects_without_output_mutation(asset("assets/../../outside.code"), root, kSchema)) return 18;

  const std::filesystem::path outside = root.parent_path() / "native-r9700-outside.code";
  if (!write_code(outside)) return 19;
  std::filesystem::create_symlink(outside, root / "external-link.code", error);
  if (error || !rejects_without_output_mutation(asset("external-link.code"), root, kSchema)) {
    return 20;
  }

  std::filesystem::remove_all(root, error);
  std::filesystem::remove(outside, error);
  return 0;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "kernel_assets_probe"
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
            str(KERNEL_ASSETS_SOURCE),
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


def test_file_backed_llama_kernel_assets_fail_closed_without_hardware(tmp_path: Path) -> None:
    """Only verified manifest-relative code may materialize a dispatchable descriptor."""
    completed = subprocess.run(
        [str(compile_kernel_assets_probe(tmp_path))], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

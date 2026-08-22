#include "kernel_assets.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <system_error>
#include <utility>
#include <vector>

namespace native_r9700 {
namespace {

// Stage assets are added here only after their code and metadata are reviewed
// together. This intentionally excludes generic probes and archived blobs.
const std::array<LlamaKernelAsset, 9> kLlamaKernelManifest = {{
    {
        {"llama_k_projection_f16",
         "9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b",
         {}, 3222208513U, 132U, 48U, 64U, 1U, 1U, 64U, 1U, 1U, 32U},
        {"llama_k_projection_f16.image",
         "9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-k-projection-f16-v1",
    },
    {
        {"llama_v_projection_f16",
         "cf200d937d6068ce1b48fdbaa6650d80abe9b4433bdeb13389e800ad3011cb6d",
         {}, 3222208513U, 132U, 48U, 64U, 1U, 1U, 64U, 1U, 1U, 32U},
        {"llama_v_projection_f16.image",
         "cf200d937d6068ce1b48fdbaa6650d80abe9b4433bdeb13389e800ad3011cb6d",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-v-projection-f16-v1",
    },
    {
        {"llama_rmsnorm_f16",
         "0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0",
         {}, 3222208513U, 132U, 160U, 64U, 1U, 1U, 64U, 1U, 1U, 32U},
        {"llama_rmsnorm_f16.image",
         "0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-rmsnorm-f16-v1",
    },
    {
        {"llama_rope_kv_f16",
         "6731222d478581cbbda7bfa539bdbcc97906f7fea255a49438ece1453564de91",
         {}, 3222208513U, 132U, 128U, 64U, 1U, 1U, 64U, 1U, 1U, 48U},
        {"llama_rope_kv_f16.image",
         "6731222d478581cbbda7bfa539bdbcc97906f7fea255a49438ece1453564de91",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-rope-kv-f16-v1",
    },
    {
        {"llama_causal_attention_score_f16",
         "08b174eb596d4f1f190bb41f6dbed419aec3432cae41400b3d3ec7caf0f704fa",
         {}, 3222208513U, 132U, 96U, 64U, 1U, 1U, 64U, 1U, 1U, 48U},
        {"llama_causal_attention_score_f16.image",
         "08b174eb596d4f1f190bb41f6dbed419aec3432cae41400b3d3ec7caf0f704fa",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-causal-attention-score-f16-v1",
    },
    {
        {"llama_causal_attention_softmax_f32",
         "e1ba09cf08e053d9ef2419b35eef7f01abba6ba62f7899b9754c28c952d6ee78",
         {}, 3222208512U, 132U, 112U, 64U, 1U, 1U, 64U, 1U, 1U, 32U},
        {"llama_causal_attention_softmax_f32.image",
         "e1ba09cf08e053d9ef2419b35eef7f01abba6ba62f7899b9754c28c952d6ee78",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-causal-attention-softmax-f32-v1",
    },
    {
        {"llama_causal_attention_context_f16",
         "34e3b1ee910a66ddb07cdd5c8e37a90e0e509abf777657a551c3b4720fa0c9fb",
         {}, 3222208512U, 132U, 80U, 64U, 1U, 1U, 64U, 1U, 1U, 40U},
        {"llama_causal_attention_context_f16.image",
         "34e3b1ee910a66ddb07cdd5c8e37a90e0e509abf777657a551c3b4720fa0c9fb",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-causal-attention-context-f16-v1",
    },
    {
        {"llama_o_projection_f16",
         "683c31d0f63e120fc8806d83d7115e5e6d3fd7c46525226ef140da3f939df557",
         {}, 3222208513U, 132U, 48U, 64U, 1U, 1U, 64U, 1U, 1U, 40U},
        {"llama_o_projection_f16.image",
         "683c31d0f63e120fc8806d83d7115e5e6d3fd7c46525226ef140da3f939df557",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-o-projection-f16-v1",
    },
    {
        {"llama_gated_mlp_f16",
         "2bca687f148135665fcf5afce4782bcb996a0a52d1fc96986e15e261a60c55fe",
         {}, 3222208513U, 132U, 176U, 64U, 1U, 1U, 64U, 1U, 1U, 56U},
        {"llama_gated_mlp_f16.image",
         "2bca687f148135665fcf5afce4782bcb996a0a52d1fc96986e15e261a60c55fe",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-gated-mlp-f16-v1",
    },
}};

bool fail(std::string* error_text, const std::string& message) {
  if (error_text != nullptr) *error_text = message;
  return false;
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}
  ~FileDescriptor() {
    if (fd_ >= 0) ::close(fd_);
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  int get() const { return fd_; }

 private:
  int fd_;
};

bool is_safe_direct_child_path(const std::filesystem::path& code_path) {
  if (code_path.empty() || code_path.is_absolute() || code_path.has_root_name() ||
      code_path.has_root_directory() || code_path.native().find('\0') != std::string::npos) {
    return false;
  }

  auto component = code_path.begin();
  if (component == code_path.end() || *component == "." || *component == "..") return false;
  return ++component == code_path.end();
}

}  // namespace

const LlamaKernelAsset* find_llama_kernel_asset(std::string_view name) {
  for (const LlamaKernelAsset& asset : kLlamaKernelManifest) {
    if (asset.descriptor.name == name) return &asset;
  }
  return nullptr;
}

bool load_verified_kernel_code(const LlamaKernelAsset& asset,
                               const std::filesystem::path& asset_root,
                               std::string_view expected_kernarg_schema,
                               KernelDescriptor* out_descriptor,
                               std::string* error_text) {
  if (out_descriptor == nullptr) return fail(error_text, "output descriptor is required");
  if (error_text != nullptr) error_text->clear();

  if (expected_kernarg_schema.empty() || asset.kernarg_schema.empty() ||
      asset.kernarg_schema != expected_kernarg_schema) {
    return fail(error_text, "kernel kernarg schema does not match the expected schema");
  }
  if (asset.location.target != "gfx1201") {
    return fail(error_text, "kernel asset target must be gfx1201");
  }
  if (asset.location.resource_metadata_provenance != "source_amdgpu_metadata") {
    return fail(error_text,
                "kernel resource metadata must come from source_amdgpu_metadata");
  }
  if (asset.location.sgpr_count < 0 || asset.location.vgpr_count < 0 ||
      asset.location.lds_bytes < 0) {
    return fail(error_text, "kernel resource metadata counts must be nonnegative");
  }
  if (asset.location.sha256 != asset.descriptor.sha256) {
    return fail(error_text, "kernel asset and descriptor digests must match");
  }
  if (!asset.descriptor.code.empty()) {
    return fail(error_text, "kernel descriptor must not embed code in the manifest");
  }

  const std::filesystem::path code_path(asset.location.code_path);
  if (!is_safe_direct_child_path(code_path)) {
    return fail(error_text,
                "kernel asset code path must name one safe direct child of the asset root");
  }

  std::error_code filesystem_error;
  const std::filesystem::file_status root_status =
      std::filesystem::symlink_status(asset_root, filesystem_error);
  if (filesystem_error) {
    return fail(error_text, "cannot inspect kernel asset root: " + filesystem_error.message());
  }
  if (std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    return fail(error_text, "kernel asset root must be a non-symlink directory");
  }

  std::filesystem::canonical(asset_root, filesystem_error);
  if (filesystem_error) {
    return fail(error_text, "cannot canonicalize kernel asset root: " + filesystem_error.message());
  }

  FileDescriptor root_fd(
      ::open(asset_root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW));
  if (root_fd.get() < 0) return fail(error_text, "cannot open kernel asset root");

  struct stat root_info {};
  if (::fstat(root_fd.get(), &root_info) != 0 || !S_ISDIR(root_info.st_mode)) {
    return fail(error_text, "kernel asset root must be an existing directory");
  }

  FileDescriptor code_fd(
      ::openat(root_fd.get(), code_path.c_str(), O_RDONLY | O_NONBLOCK | O_NOFOLLOW));
  if (code_fd.get() < 0) return fail(error_text, "cannot open kernel code file");

  struct stat code_info {};
  if (::fstat(code_fd.get(), &code_info) != 0) {
    return fail(error_text, "cannot inspect kernel code file");
  }
  if (!S_ISREG(code_info.st_mode)) {
    return fail(error_text, "kernel code path must name a regular non-symlink file");
  }
  constexpr off_t kMaxKernelCodeBytes = 4 * 1024 * 1024;
  if (code_info.st_size < 0 || code_info.st_size > kMaxKernelCodeBytes) {
    return fail(error_text, "kernel code file exceeds the 4 MiB size limit");
  }

  std::vector<uint8_t> code(static_cast<std::size_t>(code_info.st_size));
  std::size_t bytes_read = 0;
  while (bytes_read < code.size()) {
    const ssize_t read_count =
        ::read(code_fd.get(), code.data() + bytes_read, code.size() - bytes_read);
    if (read_count > 0) {
      bytes_read += static_cast<std::size_t>(read_count);
      continue;
    }
    if (read_count < 0 && errno == EINTR) continue;
    return fail(error_text, "cannot read the complete kernel code file");
  }

  KernelDescriptor loaded = asset.descriptor;
  loaded.code = std::move(code);
  std::vector<KernelDescriptor> descriptors;
  descriptors.reserve(1);
  descriptors.push_back(std::move(loaded));
  if (!validate_kernel_descriptors(descriptors, error_text)) return false;

  *out_descriptor = std::move(descriptors.front());
  return true;
}

}  // namespace native_r9700

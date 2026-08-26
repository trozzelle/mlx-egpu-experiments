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

// Stage assets are admitted here only after their source, image, ABI, and
// source-AMD metadata have been reviewed together.  The pack bridge compares
// every generated field with this table before loading any image bytes.
constexpr KernelAssetKernargField kLlamaKProjectionPackFields[] = {
    {"normalized", "uint64", 0U, 8U, 8U},
    {"k_projection_weight", "uint64", 8U, 8U, 8U},
    {"fresh_k", "uint64", 16U, 8U, 8U},
    {"sequence_length", "uint32", 24U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaVProjectionPackFields[] = {
    {"normalized", "uint64", 0U, 8U, 8U},
    {"v_projection_weight", "uint64", 8U, 8U, 8U},
    {"fresh_v", "uint64", 16U, 8U, 8U},
    {"sequence_length", "uint32", 24U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaRmsNormPackFields[] = {
    {"hidden_input", "uint64", 0U, 8U, 8U},
    {"scale", "uint64", 8U, 8U, 8U},
    {"hidden_output", "uint64", 16U, 8U, 8U},
    {"epsilon", "float32", 24U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaRopePackFields[] = {
    {"fresh_k", "uint64", 0U, 8U, 8U},
    {"fresh_v", "uint64", 8U, 8U, 8U},
    {"k_cache", "uint64", 16U, 8U, 8U},
    {"v_cache", "uint64", 24U, 8U, 8U},
    {"sequence_length", "uint32", 32U, 4U, 4U},
    {"position", "uint32", 36U, 4U, 4U},
    {"cache_capacity_tokens", "uint32", 40U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaScorePackFields[] = {
    {"normalized", "uint64", 0U, 8U, 8U},
    {"q_projection_weight", "uint64", 8U, 8U, 8U},
    {"k_cache", "uint64", 16U, 8U, 8U},
    {"attention_scores", "uint64", 24U, 8U, 8U},
    {"sequence_length", "uint32", 32U, 4U, 4U},
    {"position", "uint32", 36U, 4U, 4U},
    {"cache_capacity_tokens", "uint32", 40U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaSoftmaxPackFields[] = {
    {"attention_scores", "uint64", 0U, 8U, 8U},
    {"attention_probabilities", "uint64", 8U, 8U, 8U},
    {"sequence_length", "uint32", 16U, 4U, 4U},
    {"position", "uint32", 20U, 4U, 4U},
    {"cache_capacity_tokens", "uint32", 24U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaContextPackFields[] = {
    {"attention_probabilities", "uint64", 0U, 8U, 8U},
    {"v_cache", "uint64", 8U, 8U, 8U},
    {"context", "uint64", 16U, 8U, 8U},
    {"sequence_length", "uint32", 24U, 4U, 4U},
    {"position", "uint32", 28U, 4U, 4U},
    {"cache_capacity_tokens", "uint32", 32U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaOPackFields[] = {
    {"context", "uint64", 0U, 8U, 8U},
    {"o_projection_weight", "uint64", 8U, 8U, 8U},
    {"residual", "uint64", 16U, 8U, 8U},
    {"post_attention_hidden", "uint64", 24U, 8U, 8U},
    {"sequence_length", "uint32", 32U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaGatedMlpPackFields[] = {
    {"post_attention_hidden", "uint64", 0U, 8U, 8U},
    {"post_attention_layernorm_weight", "uint64", 8U, 8U, 8U},
    {"gate_projection_weight", "uint64", 16U, 8U, 8U},
    {"up_projection_weight", "uint64", 24U, 8U, 8U},
    {"down_projection_weight", "uint64", 32U, 8U, 8U},
    {"hidden", "uint64", 40U, 8U, 8U},
    {"sequence_length", "uint32", 48U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaGateUpPackFields[] = {
    {"post_attention_hidden", "uint64", 0U, 8U, 8U},
    {"post_attention_layernorm_weight", "uint64", 8U, 8U, 8U},
    {"gate_projection_weight", "uint64", 16U, 8U, 8U},
    {"up_projection_weight", "uint64", 24U, 8U, 8U},
    {"gate_output", "uint64", 32U, 8U, 8U},
    {"up_output", "uint64", 40U, 8U, 8U},
    {"sequence_length", "uint32", 48U, 4U, 4U},
};
constexpr KernelAssetKernargField kLlamaMlpDownPackFields[] = {
    {"gate_input", "uint64", 0U, 8U, 8U},
    {"up_input", "uint64", 8U, 8U, 8U},
    {"down_projection_weight", "uint64", 16U, 8U, 8U},
    {"residual", "uint64", 24U, 8U, 8U},
    {"hidden", "uint64", 32U, 8U, 8U},
    {"sequence_length", "uint32", 40U, 4U, 4U},
};

constexpr KernelAssetPackAttestation kLlamaKProjectionPackAttestation = {
    "gfx1201", "llama_k_projection_f16.image",
    "9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b", 14961U,
    "amdhsa-v6", 1600U, 5888U, "llama-k-projection-f16-v1", 32U, 4U,
    kLlamaKProjectionPackFields, 4U, 3222208513U, 132U, 48U, 32U, 8U, 8U, 0U,
    0U, "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U,
    1U, 0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaVProjectionPackAttestation = {
    "gfx1201", "llama_v_projection_f16.image",
    "cf200d937d6068ce1b48fdbaa6650d80abe9b4433bdeb13389e800ad3011cb6d", 14961U,
    "amdhsa-v6", 1600U, 5888U, "llama-v-projection-f16-v1", 32U, 4U,
    kLlamaVProjectionPackFields, 4U, 3222208513U, 132U, 48U, 32U, 8U, 8U, 0U,
    0U, "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U,
    1U, 0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaRmsNormPackAttestation = {
    "gfx1201", "llama_rmsnorm_f16.image",
    "0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0", 15857U,
    "amdhsa-v6", 1536U, 5888U, "llama-rmsnorm-f16-v1", 32U, 4U,
    kLlamaRmsNormPackFields, 4U, 3222208513U, 132U, 160U, 32U, 18U, 10U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaRmsNormZeroStorePackAttestation = {
    "gfx1201", "llama_rmsnorm_zero_store_f16.image",
    "8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47", 14833U,
    "amdhsa-v6", 1600U, 5888U, "llama-rmsnorm-f16-v1", 32U, 4U,
    kLlamaRmsNormPackFields, 4U, 3222208512U, 132U, 32U, 32U, 8U, 5U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaRmsNormEpsilonPackAttestation = {
    "gfx1201", "llama_rmsnorm_epsilon_arithmetic_f16.image",
    "e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037", 15089U,
    "amdhsa-v6", 1664U, 5888U, "llama-rmsnorm-f16-v1", 32U, 4U,
    kLlamaRmsNormPackFields, 4U, 3222208512U, 132U, 64U, 32U, 13U, 5U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaRopePackAttestation = {
    "gfx1201", "llama_rope_kv_f16.image",
    "6731222d478581cbbda7bfa539bdbcc97906f7fea255a49438ece1453564de91", 15601U,
    "amdhsa-v6", 1728U, 5888U, "llama-rope-kv-f16-v1", 48U, 4U,
    kLlamaRopePackFields, 7U, 3222208513U, 132U, 128U, 32U, 22U, 9U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaScorePackAttestation = {
    "gfx1201", "llama_causal_attention_score_f16.image",
    "7a5a32ffc89a7f70f347555eeb8709e77ee695530e789d2f29d875ed06c2c734", 17393U,
    "amdhsa-v6", 1792U, 6144U, "llama-causal-attention-score-f16-v1", 48U, 4U,
    kLlamaScorePackFields, 7U, 3222208514U, 132U, 320U, 32U, 32U, 20U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaSoftmaxPackAttestation = {
    "gfx1201", "llama_causal_attention_softmax_f32.image",
    "e1ba09cf08e053d9ef2419b35eef7f01abba6ba62f7899b9754c28c952d6ee78", 15473U,
    "amdhsa-v6", 1664U, 5888U, "llama-causal-attention-softmax-f32-v1", 32U, 4U,
    kLlamaSoftmaxPackFields, 5U, 3222208512U, 132U, 112U, 32U, 18U, 8U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaContextPackAttestation = {
    "gfx1201", "llama_causal_attention_context_f16.image",
    "34e3b1ee910a66ddb07cdd5c8e37a90e0e509abf777657a551c3b4720fa0c9fb", 15217U,
    "amdhsa-v6", 1728U, 5888U, "llama-causal-attention-context-f16-v1", 40U, 4U,
    kLlamaContextPackFields, 6U, 3222208512U, 132U, 80U, 32U, 20U, 6U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaOPackAttestation = {
    "gfx1201", "llama_o_projection_f16.image",
    "944a5d70745f9c17b9f1da1f96720779710caf1d1357f9e4fb988663017ead36", 15089U,
    "amdhsa-v6", 1664U, 5888U, "llama-o-projection-f16-v1", 40U, 4U,
    kLlamaOPackFields, 5U, 3222208513U, 132U, 64U, 32U, 14U, 13U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaGatedMlpPackAttestation = {
    "gfx1201", "llama_gated_mlp_f16.image",
    "71f242dbddbcd058dd73cd8b24f39007326e77238eeec4ff719b576fd86e18ec", 16241U,
    "amdhsa-v6", 1792U, 6144U, "llama-gated-mlp-f16-v1", 56U, 4U,
    kLlamaGatedMlpPackFields, 7U, 3222208513U, 132U, 176U, 32U, 26U, 16U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaGateUpPackAttestation = {
    "gfx1201", "llama_gate_up_projection_f16.image",
    "b1c6b3eb34427a206f06c39c535c4862f2c183dd9ddd387efc4b03eecf5a0421", 17393U,
    "amdhsa-v6", 1792U, 6144U, "llama-gate-up-projection-f16-v1", 56U, 4U,
    kLlamaGateUpPackFields, 7U, 3222208515U, 295044U, 320U, 32U, 22U, 27U, 4100U,
    0U, "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U,
    1U, 0U, 0U, false, 0U};
constexpr KernelAssetPackAttestation kLlamaMlpDownPackAttestation = {
    "gfx1201", "llama_mlp_down_f16.image",
    "a9ad797933d1c627ff903f47aca89d33c3cf99f22d87149c52b337a3bfde236f", 15985U,
    "amdhsa-v6", 1728U, 5888U, "llama-mlp-down-f16-v1", 48U, 4U,
    kLlamaMlpDownPackFields, 6U, 3222208515U, 132U, 176U, 32U, 30U, 32U, 0U, 0U,
    "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst", 64U, 1U, 1U, 64U, 1U, 1U,
    0U, 0U, false, 0U};
constexpr std::array<const KernelAssetPackAttestation*, 13> kLlamaPackAttestations = {{
    &kLlamaKProjectionPackAttestation,
    &kLlamaVProjectionPackAttestation,
    &kLlamaRmsNormPackAttestation,
    &kLlamaRmsNormZeroStorePackAttestation,
    &kLlamaRmsNormEpsilonPackAttestation,
    &kLlamaRopePackAttestation,
    &kLlamaScorePackAttestation,
    &kLlamaSoftmaxPackAttestation,
    &kLlamaContextPackAttestation,
    &kLlamaOPackAttestation,
    &kLlamaGatedMlpPackAttestation,
    &kLlamaGateUpPackAttestation,
    &kLlamaMlpDownPackAttestation,
}};

// Stage assets are added here only after their code and metadata are reviewed
// together. This intentionally excludes generic probes and archived blobs.
const std::array<LlamaKernelAsset, 13> kLlamaKernelManifest = {{
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
        {"llama_rmsnorm_zero_store_f16",
         "8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47",
         {}, 3222208512U, 132U, 32U, 64U, 1U, 1U, 64U, 1U, 1U, 32U},
        {"llama_rmsnorm_zero_store_f16.image",
         "8be1b744e76cab295943e9a78b7cabdfd20d6e22c16f92862baf140f27b1de47",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-rmsnorm-f16-v1",
    },
    {
        {"llama_rmsnorm_epsilon_arithmetic_f16",
         "e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037",
         {}, 3222208512U, 132U, 64U, 64U, 1U, 1U, 64U, 1U, 1U, 32U},
        {"llama_rmsnorm_epsilon_arithmetic_f16.image",
         "e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037",
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
         "7a5a32ffc89a7f70f347555eeb8709e77ee695530e789d2f29d875ed06c2c734",
         {}, 3222208514U, 132U, 320U, 64U, 1U, 1U, 64U, 1U, 1U, 48U},
        {"llama_causal_attention_score_f16.image",
         "7a5a32ffc89a7f70f347555eeb8709e77ee695530e789d2f29d875ed06c2c734",
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
         "944a5d70745f9c17b9f1da1f96720779710caf1d1357f9e4fb988663017ead36",
         {}, 3222208513U, 132U, 64U, 64U, 1U, 1U, 64U, 1U, 1U, 40U},
        {"llama_o_projection_f16.image",
         "944a5d70745f9c17b9f1da1f96720779710caf1d1357f9e4fb988663017ead36",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-o-projection-f16-v1",
    },
    {
        {"llama_gated_mlp_f16",
         "71f242dbddbcd058dd73cd8b24f39007326e77238eeec4ff719b576fd86e18ec",
         {}, 3222208513U, 132U, 176U, 64U, 1U, 1U, 64U, 1U, 1U, 56U},
        {"llama_gated_mlp_f16.image",
         "71f242dbddbcd058dd73cd8b24f39007326e77238eeec4ff719b576fd86e18ec",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-gated-mlp-f16-v1",
    },
    {
        {"llama_gate_up_projection_f16",
         "b1c6b3eb34427a206f06c39c535c4862f2c183dd9ddd387efc4b03eecf5a0421",
         {}, 3222208515U, 295044U, 320U, 64U, 1U, 1U, 64U, 1U, 1U, 56U},
        {"llama_gate_up_projection_f16.image",
         "b1c6b3eb34427a206f06c39c535c4862f2c183dd9ddd387efc4b03eecf5a0421",
         "gfx1201", 0, 0, 4100, "source_amdgpu_metadata"},
        "llama-gate-up-projection-f16-v1",
    },
    {
        {"llama_mlp_down_f16",
         "a9ad797933d1c627ff903f47aca89d33c3cf99f22d87149c52b337a3bfde236f",
         {}, 3222208515U, 132U, 176U, 64U, 1U, 1U, 64U, 1U, 1U, 48U},
        {"llama_mlp_down_f16.image",
         "a9ad797933d1c627ff903f47aca89d33c3cf99f22d87149c52b337a3bfde236f",
         "gfx1201", 0, 0, 0, "source_amdgpu_metadata"},
        "llama-mlp-down-f16-v1",
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

const KernelAssetPackAttestation* find_kernel_pack_attestation(std::string_view name) {
  for (const KernelAssetPackAttestation* attestation : kLlamaPackAttestations) {
    const std::string_view image_path = attestation->image_path;
    constexpr std::string_view kImageSuffix = ".image";
    if (image_path.size() >= kImageSuffix.size() &&
        name == image_path.substr(0, image_path.size() - kImageSuffix.size())) {
      return attestation;
    }
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

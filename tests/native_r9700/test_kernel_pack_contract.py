"""RED contracts for the allocation-free native Kernel Pack runtime boundary.

These tests intentionally compile a small C++ probe against the frozen task-set-2
surface.  The probe owns no pack data: every view points at static or stack data,
and lookup always receives an explicit generated-record span.
"""

from pathlib import Path
import json
import re
import subprocess


KERNEL_PACK_HEADER = Path("native_r9700/kernel_pack.h")
KERNEL_PACK_SOURCE = Path("native_r9700/kernel_pack.cpp")
NATIVE_INCLUDE_DIR = Path("native_r9700")

# The pack implementation delegates image/code admission to these existing
# boundaries rather than growing a second ELF or descriptor parser.
RUNTIME_SOURCES = (
    KERNEL_PACK_SOURCE,
    Path("native_r9700/kernel_assets.cpp"),
    Path("native_r9700/kernel_catalog.cpp"),
    Path("native_r9700/hsa_code_image_asset.cpp"),
)
SCALAR_PACK_ROOT = Path("native_r9700/kernels")
SCALAR_GENERATED_SOURCE = Path("native_r9700/kernel_packs_generated.inc")
SCALAR_PACK_NAMES = (
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
)


def _compile_pack_probe(tmp_path: Path) -> Path:
    """Compile the public pack API and its existing image/admission boundary."""
    assert KERNEL_PACK_HEADER.is_file(), "kernel pack header is missing"
    assert KERNEL_PACK_SOURCE.is_file(), "kernel pack source is missing"
    assert all(path.is_file() for path in RUNTIME_SOURCES), "Kernel Pack link sources are missing"

    probe_source = tmp_path / "kernel_pack_contract_probe.cpp"
    probe_source.write_text(
        r'''
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "kernel_pack.h"
#include "kernel_catalog.h"
#include "kernel_assets.h"

namespace {

using native_r9700::EvidenceRef;
using native_r9700::KernelDescriptor;
using native_r9700::KernelPackCastPoint;
using native_r9700::KernelPackCompatibilityKey;
using native_r9700::KernelPackEntry;
using native_r9700::KernelPackEvidence;
using native_r9700::KernelPackErrorBuffer;
using native_r9700::KernelPackGeometryCase;
using native_r9700::KernelPackIdentity;
using native_r9700::KernelPackKernargField;
using native_r9700::KernelPackLicenseReview;
using native_r9700::KernelPackModification;
using native_r9700::KernelPackOptional;
using native_r9700::KernelPackRecord;
using native_r9700::KernelPackShapeDimension;
using native_r9700::KernelPackShapeFamily;
using native_r9700::KernelPackSource;
using native_r9700::KernelPackSpan;
using native_r9700::KernelPackRuntimeDimension;

constexpr char kB0SourceDigest[] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr char kB0ImageDigest[] =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
constexpr char kB0RecordDigest[] =
    "3333333333333333333333333333333333333333333333333333333333333333";
constexpr char kB0InputDigest[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char kB0OutputDigest[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
// SHA-256(JCS({domain, normalized complete B0 pack without evidence/pack_sha256})).
constexpr char kB0PackDigest[] =
    "8ecc252df503285126ea7874d948050681dd785478106edb749ff48c08251ba1";
constexpr char kB0WrongFiniteValueRulePackDigest[] =
    "681fdd5339b72f63f2dd7345f21bf5db964b5e60d8e0ff791ae959e4a0d06044";
constexpr char kB0ToolchainRevision[] =
    "0123456789abcdef0123456789abcdef01234567";
constexpr char kB0GeneratorRevision[] =
    "89abcdef0123456789abcdef0123456789abcdef";

constexpr char kF2SourceDigest[] =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr char kF2ImageDigest[] =
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
constexpr char kF2CommandDigest[] =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr char kF2RecordDigest[] =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr char kF2InputDigest[] =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr char kF2NumpyOutputDigest[] =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr char kF2NativeOutputDigest[] =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
// SHA-256(JCS({domain, normalized complete F2 pack without evidence/pack_sha256})).
constexpr char kF2PackDigest[] =
    "b13c4d1d4b14c423c2dbb210f8edb749505d7a0dd13f31ca5df8461a28bead51";
constexpr char kF2WrongFiniteValueRulePackDigest[] =
    "cbe4c6b6b354dfc136ceb564e08a5b802a0473ff6c8a20b16f59605a3122e8e9";
constexpr char kF2ToolchainRevision[] =
    "1023456789abcdef0123456789abcdef01234567";
constexpr char kF2GeneratorRevision[] =
    "98abcdef0123456789abcdef0123456789abcdef";

constexpr char kB0UnknownLicensePackDigest[] =
    "5818910f1d093a3c7586a7020f9472c617f4eb4fd174f7c76e84a548e5a76907";
constexpr char kB0SourceEquivalentPackDigest[] =
    "8ecc252df503285126ea7874d948050681dd785478106edb749ff48c08251ba1";
constexpr char kB0PendingLicensePackDigest[] =
    "13d4436523d2c454d32d3a4ba14cd603e8fb4d25963c052d1fb20d7a3165b667";
constexpr char kB0WhitespaceLicensePackDigest[] =
    "0ba23878fa2ea943bf58d5ee052c61d3b39ff636f376e631694b7d33a2b26fa5";

constexpr char kLlamaSourceDigest[] =
    "f0a150bcd951d2a5247187878ea832703a06da3630a2c4302bb9a296661a19c4";
constexpr char kLlamaImageDigest[] =
    "9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b";
constexpr char kLlamaPackDigest[] =
    "cbdd926d9fb3f8bd34fef954858512ec6f306b51e30d50e72a62bdf5d6aa5428";
constexpr char kLlamaMultiPackDigest[] =
    "7c1a2d9285f0de9b378ebd84bdaecdcc271e6df553792ce3e195dc760e04fc05";
constexpr char kLlamaImagePathMismatchPackDigest[] =
    "93c1044492719d66ad109fd816a6ecc236060de00481cdd20ee30e4e172e37d9";
constexpr char kLlamaImageSizeMismatchPackDigest[] =
    "aaeb73f11779153e4c6957d68c3cfec81044f45f6c48148eac98a9ee1007ce4a";
constexpr char kLlamaCodeObjectMismatchPackDigest[] =
    "0449d6f37f7b8639c5956fd71da221540013eece332fc2610f0cd525e470f8f2";
constexpr char kLlamaDescriptorMismatchPackDigest[] =
    "565b14b300bab8d2e128690f0d7d687a8174490fa717bc711706502e5e499623";
constexpr char kLlamaEntryMismatchPackDigest[] =
    "9e76f59e358f74a4ad5f8650b9ed179b7b5f3590a68a6435060602a787849537";
constexpr char kLlamaKernargMismatchPackDigest[] =
    "ebc7a2912d3acef8ad3b33f83f5a8d78b8851cd77be4f76f8a9b3ae852274544";
constexpr char kB0LookupMalformedPackDigest[] =
    "102ad8b2bac0d275f7c53582592291f9f747e6062973ba78bf679c4ed2a3b599";
constexpr char kB0DescriptorZeroPackDigest[] =
    "a9fdd657145fdb7001b702fa84f033f149deb622eefbe54ed67e46577d757b9a";
constexpr char kB0EntryZeroPackDigest[] =
    "299fcfe90f5b96bf24fd03a688f63e1b52c571022034b7ac491553bbfa3611a3";
constexpr char kB0DescriptorAtImagePackDigest[] =
    "17e019ee2ae52ff256c71aff2243896d57dcf39140422e6fbc075581b268f4ae";
constexpr char kB0DescriptorBeyondImagePackDigest[] =
    "2b0970daf325939297008b9e74a7a1fdd1c465bae72c55300d425e053eee98f5";
constexpr char kB0EntryAtImagePackDigest[] =
    "fb33d3dd1901c38baec01071680b9b2cdec537e062559f56e828f8d8957b54ba";
constexpr char kB0EntryBeyondImagePackDigest[] =
    "0e16e1f2b7367e5ea8de434443112e0d46f1470e5b7fa35489f9f7c934c784b1";
constexpr char kB0ExactGlobalXPackDigest[] =
    "8e87e72bfc85e847a21f41beedc5535671daf611c6d12426989a2b12713ac6b5";
constexpr char kB0ExactGlobalYPackDigest[] =
    "0ad7cd19df4c2746a925f86f06c24867d3a8ade0eb3d9280858e7887ffe5e6b7";
constexpr char kB0ExactGlobalZPackDigest[] =
    "584eb5dd9f0aff4b4a74a4047c1550d468840f2ae177a2a29b4eb3b834a6d096";
constexpr char kB0DynamicLdsPackDigest[] =
    "62e67fe295ed85bbdcc2b26082a681aa8ae25e4a240a9f9e55acff8753999463";
constexpr char kB0DuplicateKernargNamePackDigest[] =
    "c1b5e6462d7f0afa1570c1cef860e630b5ba5980e790fdc10da14e8b1440989f";
constexpr char kB0KernargTypePackDigest[] =
    "0529e05f49949ee435a952e4678f40eb2d3a8ad5b59400d299af943a5748766e";
constexpr char kB0KernargSizePackDigest[] =
    "3b7ce62e83c544c47dc065ecf37b23796626eeaee448c6f2620dbfb673173b9c";
constexpr char kB0KernargAlignmentPackDigest[] =
    "9923ce9f7d1488e2e401cc1852ef2cc789743e0e663f8986e105bca6fd061297";
constexpr char kLlamaVSourceDigest[] =
    "b220e09ac16e5fd6769b92308d39e3b7ec7c8383fdcd9adda42f54c4dfe40574";
constexpr char kLlamaVImageDigest[] =
    "cf200d937d6068ce1b48fdbaa6650d80abe9b4433bdeb13389e800ad3011cb6d";
constexpr char kLlamaVPackDigest[] =
    "fa2734c327148f3f9609275dfb93c52da879f27c828cba97911f3cf514578ff3";
constexpr char kPinnedUpstreamRepository[] = "https://github.com/llvm/llvm-project";
constexpr char kPinnedUpstreamRevision[] =
    "8dba93818258d95c46fa2c17e902a8256e4d91b5";
constexpr char kPinnedUpstreamPath[] = "llvm/docs/AMDGPUUsage.rst";
constexpr char kB0PinnedProvenancePackDigest[] =
    "f67b0a3b4774b0d9e1e6e5d3f13ea24f6074fada5c26fdcfad423819be154e7a";
constexpr char kB0LocalMainProvenancePackDigest[] =
    "148ca1ca297cb7479ed47365f0ed4a6a09866f0e28829ef9ea89811e5264bf35";
constexpr char kB0LocalPinnedProvenancePackDigest[] =
    "89406d4212f19d17923c72074a2bae20538b92a074b6aa08cc7348f9f3216aa2";
constexpr char kB0PinnedLocalProvenancePackDigest[] =
    "8dada30b58b8972f5cf8b97251227d27a8332c446f8e6836d6ae8e48cf1f8de6";
constexpr char kB0PinnedMainProvenancePackDigest[] =
    "9949bd9af3b3d20429ec31e6785671a2436c05139343c518a442e65e53d591ca";
constexpr char kB0WrongRepositoryProvenancePackDigest[] =
    "d11804c0625890999abd260234148fb9c8f3eeacbe2308ffb7e6854d858f958b";
constexpr char kB0SourceDrivePathPackDigest[] =
    "47f1ca5f04a97fc3ca2980891406fc72845d699084a09902b0f6785c4b736596";
constexpr char kB0ModificationLicensePackDigest[] =
    "e69acc49babf1a2e5bdce7633ca748864c91db6ce019856d27d14ef8a3d43b7a";
constexpr std::string_view kB0SourceControlPackDigests[] = {
    "d873664acd12d26b689753939914e2cf75e29d7e8db682e7e51a153e475c2703",
    "68ddda5f2920b296c55cb900fbafb30f2e1ff06819c1f92410f13d29d1352daf",
    "78fac937ccaacc8beaa4e372c2fda25ddcdb46aff201317f0fd3936db02ebbde",
    "1bf95bd372be8549873b6e74b780f7273a56b981217888502913565d5b914117",
    "de28101a0e640f62f802278fc0fd51392154e11a1344bea6fad62c696241b2f6",
    "a6c652854ea65819629944dc0ec7238d459f2bba1067bd0f5b74f50c8a671730",
    "2883c574113b82488b89d9c52dd08b8e9d2910200f8a93e78fa258bc462a9d33",
    "90f6c5d553a59fcf3685a082a041fa461c6e1f56d820b2c454e5b6d6bf6ca9ba",
    "2d1ebf305bb5ad117156563fbb8bd5a2c1fde326fe6e2ce3dd6740d4438255bf",
    "2e240b6cf6fa7b934d04fd275f8faab308c27b59f74d0341f814346f2c32907b",
    "50ac5b1f04cb19331046ef394d9930f6b139e5a3500bcbdc945b612322a15e80",
    "2b99d6428173f6f221757f423ba3062b6443e9827353967a57f03480b4adcc73",
    "cef853e5a6a4c5d711f24f5c336b91d97edeb4c75d028f44159d96732c568faa",
    "539623344b4e98eb445544235e24358ea13cf3e9d368768bfb6c3a431ceb2cc6",
    "2a11ba034f919de419251397cc6553ca06da9c82b591b1b00296e650e3c130c2",
    "df59795bf5cc3d848f42e520eb7e3b4f3aacc1e9ea2afa7b0cfbfa8356dcb14f",
    "10a0e60e55d5499d3eeac67c26679cb6a79bb7ad586a73bfda3af6c043855538",
    "068f41830966213465cc00b9ac6b5d60e5e62122773f873650ea17d20d682e1a",
    "8ef3fdef1181efd59f7aab22612bf5d8baee08677c1b2c19d65a3e8d83e8a8e0",
    "a76b001fbff70ccbedbdfedf3d695ff0cf314c07f0fafcf6559038e439cd98cb",
    "5460eef16d12830fcd6430cadee7b553731cf420390980cc65c9b355ba273e6e",
    "7240708025d0dfa648867033e871d083adcd2f6c42cde344a65e0c8badfe6e3a",
    "ff64302fb727ac2b38a89ccdbd0f99790e11f3ab3f4e5b21b2e26607af3dc4b4",
    "fe0d87c88f4e5edee26a8aa9537dc0e46117e9ebac9672a66a52e20b62453435",
    "31e1675b307147f078ce1731278d2cfa0eba85e4fb886c335b8200b5f10b959e",
    "b94bc6e5dc891fba18d021c5afadf225852ea70af753c67958e89c36e948c53c",
    "ace427b90837448076a6523c96811c88f7285f27d1b470f3c1d55d7539632efb",
    "405fdf16ef025ff4053208aae2831dd7d6963d63e71e42139e9fc3e3543aa6d8",
    "96d8385eb61c743e5d920e7f3274f487e7c74625c0071bb50d5e9e55c14245fb",
    "4268da367871a35db77c818cbfa77b8c7038b5a0942c58045ef66a88f92e4e88",
    "7c904035d728c74f251e0d81feeed7ea6bfb096b8dd5306dfc924331fda89ca0",
    "2554f93512ee4ea0c8b7ceaa5d2aa157aefef14f8dac6370c6be6be78f45741c",
    "f9b6b85b02fc311b63954ae118c1869435f2d67a28490a7bf9fda1f8ae0bacda",
};


constexpr KernelPackSource kB0Sources[] = {{
    "native_r9700/kernels/test_pack.cpp", kB0SourceDigest,
}};
constexpr KernelPackLicenseReview kB0Licenses[] = {
    {"native_r9700/kernels/test_pack.cpp", "MIT", "review-source-1", "accepted"},
    {"test_pack.image", "MIT", "review-image-1", "accepted"},
};
constexpr KernelPackKernargField kB0KernargFields[] = {
    {"activation", "uint64", 0, 8, 8},
    {"weight", "uint64", 8, 8, 8},
    {"output", "uint64", 16, 8, 8},
    {"m", "uint32", 24, 4, 4},
};
constexpr KernelPackGeometryCase kB0GeometryCases[] = {{
    "fixed-test-family", "exact-global-v1", 1, 1, 1, 1, 1, 1, 0, 0, false, 0,
}};
constexpr KernelPackShapeDimension kB0FixedDimensions[] = {
    {"K", 1}, {"N", 1},
};
constexpr KernelPackCastPoint kB0CastPoints[] = {{"accumulate", "fp16", "fp32"}};

constexpr KernelPackSource kF2Sources[] = {{
    "native_r9700/kernels/f2_pack.cpp", kF2SourceDigest,
}};
constexpr KernelPackLicenseReview kF2Licenses[] = {
    {"native_r9700/kernels/f2_pack.cpp", "MIT", "f2-source-review", "accepted"},
    {"f2_pack.image", "MIT", "f2-image-review", "accepted"},
};
constexpr KernelPackKernargField kF2KernargFields[] = {
    {"activation", "uint64", 0, 8, 8},
    {"weight_nk", "uint64", 8, 8, 8},
    {"output", "uint64", 16, 8, 8},
    {"m", "uint32", 24, 4, 4},
};
constexpr KernelPackGeometryCase kF2GeometryCases[] = {{
    "f2-linear-gate-up-f16-v1", "f2-wmma-64x64-m-tail-v1", 128, 4, 1, 0, 0, 0,
    64, 64, false, 0,
}};
constexpr KernelPackShapeDimension kF2FixedDimensions[] = {
    {"K", 2048}, {"N", 8192},
};
constexpr KernelPackRuntimeDimension kF2RuntimeDimension = {"M", 1, 128, 128};
constexpr KernelPackCastPoint kF2CastPoints[] = {{"wmma-accumulate", "fp16", "fp32"}};
constexpr std::string_view kF2RequiredFeatures[] = {"wave32"};

EvidenceRef make_ref(std::string_view record_path,
                     std::string_view record_kind,
                     std::string_view evidence_slot,
                     std::string_view record_id,
                     std::string_view record_sha256,
                     std::string_view subject_target,
                     std::string_view image_sha256,
                     std::string_view pack_sha256,
                     std::string_view producer_kind,
                     std::string_view tool_digest,
                     std::string_view input_digest,
                     std::string_view output_digest) {
  EvidenceRef value{};
  value.record_path = record_path;
  value.record_kind = record_kind;
  value.evidence_slot = evidence_slot;
  value.record_id = record_id;
  value.record_sha256 = record_sha256;
  value.subject_target = subject_target;
  value.image_sha256 = image_sha256;
  value.pack_sha256 = pack_sha256;
  value.producer_kind = producer_kind;
  value.tool_digest = tool_digest;
  value.input_digest = input_digest;
  value.output_digest = output_digest;
  return value;
}

void set_source_review(KernelPackRecord& record, const EvidenceRef& source_review) {
  record.evidence.source_review = source_review;
}

void bind_pack_digest(KernelPackRecord& record, std::string_view pack_digest) {
  record.evidence.conformance.pack_sha256 = pack_digest;
  record.evidence.native_run.pack_sha256 = pack_digest;
  record.evidence.resource_review.pack_sha256 = pack_digest;
  record.evidence.isa_review.pack_sha256 = pack_digest;
  record.evidence.source_review.pack_sha256 = pack_digest;
  if (record.evidence.layout_proof.present) {
    record.evidence.layout_proof.value.pack_sha256 = pack_digest;
  }
  if (record.evidence.benchmark_record.present) {
    record.evidence.benchmark_record.value.pack_sha256 = pack_digest;
  }
  if (record.numerics.scalar_native_projection.present) {
    record.numerics.scalar_native_projection.value.pack_sha256 = pack_digest;
  }
}

constexpr KernelPackSource kLlamaSources[] = {{
    "native_r9700/kernels/llama_k_projection_f16.cpp", kLlamaSourceDigest,
}};
constexpr KernelPackLicenseReview kLlamaLicenses[] = {
    {"native_r9700/kernels/llama_k_projection_f16.cpp", "MIT", "llama-source-license-1",
     "accepted"},
    {"llama_k_projection_f16.image", "MIT", "llama-image-license-1", "accepted"},
};
constexpr KernelPackKernargField kLlamaKernargFields[] = {
    {"normalized", "uint64", 0, 8, 8},
    {"k_projection_weight", "uint64", 8, 8, 8},
    {"fresh_k", "uint64", 16, 8, 8},
    {"sequence_length", "uint32", 24, 4, 4},
};
constexpr KernelPackShapeDimension kLlamaFixedDimensions[] = {
    {"K", 1}, {"N", 1},
};
constexpr KernelPackShapeFamily kLlamaFamilies[] = {{
    "llama-fixed-family", {kLlamaFixedDimensions, 2}, {false, {}}, "none", "exact-global-v1",
}};
constexpr KernelPackGeometryCase kLlamaGeometryCases[] = {{
    "llama-fixed-family", "exact-global-v1", 64, 1, 1, 64, 1, 1, 0, 0, false, 0,
}};
constexpr KernelPackCastPoint kLlamaCastPoints[] = {{"accumulate", "fp16", "fp32"}};
const KernelPackEntry kLlamaEntry = [] {
  KernelPackEntry value{};
  value.symbol = "llama_k_projection_f16";
  value.descriptor_offset = 1600;
  value.entry_offset = 5888;
  value.kernargs.bytes = 32;
  value.kernargs.fields = {kLlamaKernargFields, 4};
  value.kernargs.tail_padding_bytes = 4;
  value.resources.rsrc1 = 3222208513U;
  value.resources.rsrc2 = 132U;
  value.resources.rsrc3 = 48U;
  value.resources.wave_size = 32;
  value.resources.sgpr_count = 8;
  value.resources.vgpr_count = 8;
  value.resources.lds_bytes = 0;
  value.resources.private_segment_bytes = 0;
  value.resources.metadata_provenance = "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst";
  value.geometry.cases = {kLlamaGeometryCases, 1};
  return value;
}();
constexpr KernelPackShapeFamily kLlamaMultiFamilies[] = {
    {"llama-alternate-family", {kLlamaFixedDimensions, 2}, {false, {}}, "none",
     "exact-global-v1"},
    {"llama-selected-family", {kLlamaFixedDimensions, 2}, {false, {}}, "none",
     "exact-global-v1"},
};
constexpr KernelPackGeometryCase kLlamaMultiGeometryCases[] = {
    {"llama-alternate-family", "exact-global-v1", 32, 1, 1, 32, 1, 1, 0, 0, false, 0},
    {"llama-selected-family", "exact-global-v1", 64, 1, 1, 64, 1, 1, 0, 0, false, 0},
};

const KernelPackEntry kB0Entry = [] {
  KernelPackEntry value{};
  value.symbol = "test_pack";
  value.descriptor_offset = 256;
  value.entry_offset = 512;
  value.kernargs.bytes = 32;
  value.kernargs.fields = {kB0KernargFields, 4};
  value.kernargs.tail_padding_bytes = 4;
  value.resources.rsrc1 = 1;
  value.resources.rsrc2 = 2;
  value.resources.rsrc3 = 3;
  value.resources.wave_size = 32;
  value.resources.sgpr_count = 8;
  value.resources.vgpr_count = 8;
  value.resources.lds_bytes = 0;
  value.resources.private_segment_bytes = 0;
  value.resources.metadata_provenance = "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst";
  value.geometry.cases = {kB0GeometryCases, 1};
  return value;
}();

const KernelPackEntry kF2Entry = [] {
  KernelPackEntry value{};
  value.symbol = "linear_wmma_f16";
  value.descriptor_offset = 256;
  value.entry_offset = 512;
  value.kernargs.bytes = 32;
  value.kernargs.fields = {kF2KernargFields, 4};
  value.kernargs.tail_padding_bytes = 4;
  value.resources.rsrc1 = 11;
  value.resources.rsrc2 = 22;
  value.resources.rsrc3 = 33;
  value.resources.wave_size = 32;
  value.resources.sgpr_count = 64;
  value.resources.vgpr_count = 32;
  value.resources.lds_bytes = 256;
  value.resources.private_segment_bytes = 0;
  value.resources.metadata_provenance = "source AMDGPU metadata: llvm/docs/AMDGPUUsage.rst";
  value.geometry.cases = {kF2GeometryCases, 1};
  return value;
}();

KernelPackRecord b0_pack() {
  KernelPackRecord record{};
  record.identity.schema_version = 1;
  record.identity.name = "test-pack";
  record.identity.version = "1.0.0";
  record.identity.target = "gfx1201";
  record.identity.required_features = {nullptr, 0};

  record.provenance.upstream_repository = "local";
  record.provenance.upstream_revision = "local";
  record.provenance.upstream_paths = {nullptr, 0};
  record.provenance.local_sources = {kB0Sources, 1};
  record.provenance.license_reviews = {kB0Licenses, 2};
  record.provenance.modifications = {nullptr, 0};

  record.image.image_path = "test_pack.image";
  record.image.image_sha256 = kB0ImageDigest;
  record.image.image_size = 1024;
  record.image.code_object_version = "amdhsa-v6";
  record.image.build.toolchain_id = "clang-amdgpu";
  record.image.build.toolchain_revision = kB0ToolchainRevision;
  record.image.build.generator_id = "pack-generator";
  record.image.build.generator_revision = kB0GeneratorRevision;
  record.image.build.command_sha256 = kB0SourceDigest;

  record.entries = {&kB0Entry, 1};

  record.compatibility.input_dtype = "fp16";
  record.compatibility.weight_dtype = "fp16";
  record.compatibility.output_dtype = "fp16";
  record.compatibility.source_tensor_layout_version = "f16-row-major-nk-source-v1";
  KernelPackShapeFamily b0_family{};
  b0_family.name = "fixed-test-family";
  b0_family.fixed_dimensions = {kB0FixedDimensions, 2};
  b0_family.runtime_dimension = {false, {}};
  b0_family.tail_policy = "none";
  b0_family.geometry_rule = "exact-global-v1";
  static const KernelPackShapeFamily family = b0_family;
  record.compatibility.shape_families = {&family, 1};
  record.compatibility.weight_packing_version = "source-equivalent-v1";

  record.numerics.input_dtype = "fp16";
  record.numerics.accumulation_dtype = "fp32";
  record.numerics.output_dtype = "fp16";
  record.numerics.cast_points = {kB0CastPoints, 1};
  record.numerics.finite_value_rule = "finite-input-output-v1";
  record.numerics.tolerance_policy = "exact-v1";
  record.numerics.reference_set_kind = "b0_scalar_control";
  record.numerics.retained_reference.present = true;
  record.numerics.retained_reference.value = make_ref(
      "logs/test/numpy.json", "offline_oracle", "numpy_oracle", "oracle-1",
      kB0RecordDigest, "", "", "", "cpu_reference", "", kB0InputDigest,
      kB0OutputDigest);
  record.numerics.numpy_oracle.present = false;
  record.numerics.scalar_native_projection.present = false;

  record.evidence.conformance = make_ref(
      "logs/test/conformance.json", "target_conformance", "conformance", "conformance-1",
      kB0RecordDigest, "gfx1201", kB0ImageDigest, kB0PackDigest, "r9700_native", "",
      kB0InputDigest, kB0OutputDigest);
  record.evidence.native_run = make_ref(
      "logs/test/native-run.json", "native_run", "native_run", "native-run-1",
      kB0RecordDigest, "gfx1201", kB0ImageDigest, kB0PackDigest, "r9700_native", "",
      kB0InputDigest, kB0OutputDigest);
  record.evidence.resource_review = make_ref(
      "logs/test/resource-review.json", "offline_review", "resource_review", "resource-1",
      kB0RecordDigest, "gfx1201", kB0ImageDigest, kB0PackDigest, "", kB0SourceDigest,
      kB0InputDigest, kB0OutputDigest);
  record.evidence.isa_review = make_ref(
      "logs/test/isa-review.json", "offline_review", "isa_review", "isa-1",
      kB0RecordDigest, "gfx1201", kB0ImageDigest, kB0PackDigest, "", kB0SourceDigest,
      kB0InputDigest, kB0OutputDigest);
  set_source_review(
      record,
      make_ref("logs/test/source-review.json", "offline_review", "source_review",
               "source-review-1", kB0RecordDigest, "gfx1201", kB0ImageDigest, kB0PackDigest, "",
               kB0SourceDigest, kB0InputDigest, kB0OutputDigest));
  record.evidence.layout_proof.present = false;
  record.evidence.benchmark_record.present = false;
  record.evidence.benchmark_not_applicable_reason = "correctness-control-only";
  return record;
}

KernelPackRecord f2_pack() {
  KernelPackRecord record{};
  record.identity.schema_version = 1;
  record.identity.name = "f2-pack";
  record.identity.version = "1.0.0";
  record.identity.target = "gfx1201";
  record.identity.required_features = {kF2RequiredFeatures, 1};

  record.provenance.upstream_repository = "local";
  record.provenance.upstream_revision = "local";
  record.provenance.upstream_paths = {nullptr, 0};
  record.provenance.local_sources = {kF2Sources, 1};
  record.provenance.license_reviews = {kF2Licenses, 2};
  record.provenance.modifications = {nullptr, 0};

  record.image.image_path = "f2_pack.image";
  record.image.image_sha256 = kF2ImageDigest;
  record.image.image_size = 1024;
  record.image.code_object_version = "amdhsa-v6";
  record.image.build.toolchain_id = "clang-amdgpu";
  record.image.build.toolchain_revision = kF2ToolchainRevision;
  record.image.build.generator_id = "f2-pack-generator";
  record.image.build.generator_revision = kF2GeneratorRevision;
  record.image.build.command_sha256 = kF2CommandDigest;

  record.entries = {&kF2Entry, 1};

  record.compatibility.input_dtype = "fp16";
  record.compatibility.weight_dtype = "fp16";
  record.compatibility.output_dtype = "fp16";
  record.compatibility.source_tensor_layout_version = "f16-row-major-nk-source-v1";
  KernelPackShapeFamily f2_family{};
  f2_family.name = "f2-linear-gate-up-f16-v1";
  f2_family.fixed_dimensions = {kF2FixedDimensions, 2};
  f2_family.runtime_dimension = {true, kF2RuntimeDimension};
  f2_family.tail_policy = "masked/padded";
  f2_family.geometry_rule = "f2-wmma-64x64-m-tail-v1";
  static const KernelPackShapeFamily family = f2_family;
  record.compatibility.shape_families = {&family, 1};
  record.compatibility.weight_packing_version = "f2-wmma-physical-tile-v1";

  record.numerics.input_dtype = "fp16";
  record.numerics.accumulation_dtype = "fp32";
  record.numerics.output_dtype = "fp16";
  record.numerics.cast_points = {kF2CastPoints, 1};
  record.numerics.finite_value_rule = "finite-input-output-v1";
  record.numerics.tolerance_policy = "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1";
  record.numerics.reference_set_kind = "f2_wmma_dual";
  record.numerics.retained_reference.present = false;
  record.numerics.numpy_oracle.present = true;
  record.numerics.numpy_oracle.value = make_ref(
      "logs/f2/numpy-oracle.json", "offline_oracle", "numpy_oracle", "f2-numpy-1",
      kF2RecordDigest, "", "", "", "cpu_reference", "", kF2InputDigest,
      kF2NumpyOutputDigest);
  record.numerics.scalar_native_projection.present = true;
  record.numerics.scalar_native_projection.value = make_ref(
      "logs/f2/native-projection.json", "target_conformance", "scalar_native_projection",
      "f2-native-projection-1", kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest,
      "r9700_native", "", kF2InputDigest, kF2NativeOutputDigest);

  record.evidence.conformance = make_ref(
      "logs/f2/conformance.json", "target_conformance", "conformance", "f2-conformance-1",
      kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "r9700_native", "",
      kF2InputDigest, kF2NativeOutputDigest);
  record.evidence.native_run = make_ref(
      "logs/f2/native-run.json", "native_run", "native_run", "f2-native-run-1",
      kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "r9700_native", "",
      kF2InputDigest, kF2NativeOutputDigest);
  record.evidence.resource_review = make_ref(
      "logs/f2/resource-review.json", "offline_review", "resource_review", "f2-resource-1",
      kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "", kF2CommandDigest,
      kF2InputDigest, kF2NativeOutputDigest);
  record.evidence.isa_review = make_ref(
      "logs/f2/isa-review.json", "offline_review", "isa_review", "f2-isa-1",
      kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "", kF2CommandDigest,
      kF2InputDigest, kF2NativeOutputDigest);
  set_source_review(
      record,
      make_ref("logs/f2/source-review.json", "offline_review", "source_review", "f2-source-1",
               kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "", kF2CommandDigest,
               kF2InputDigest, kF2NativeOutputDigest));
  record.evidence.layout_proof.present = true;
  record.evidence.layout_proof.value = make_ref(
      "logs/f2/layout-proof.json", "offline_review", "layout_proof", "f2-layout-1",
      kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "", kF2CommandDigest,

      kF2InputDigest, kF2NativeOutputDigest);
  record.evidence.benchmark_record.present = true;
  record.evidence.benchmark_record.value = make_ref(
      "logs/f2/benchmark.json", "benchmark", "benchmark", "f2-benchmark-1",
      kF2RecordDigest, "gfx1201", kF2ImageDigest, kF2PackDigest, "r9700_native",
      kF2CommandDigest, kF2InputDigest, kF2NativeOutputDigest);
  record.evidence.benchmark_not_applicable_reason = "";
  return record;
}
KernelPackRecord llama_pack() {
  KernelPackRecord record{};
  record.identity.schema_version = 1;
  record.identity.name = "llama-pack";
  record.identity.version = "1.0.0";
  record.identity.target = "gfx1201";
  record.identity.required_features = {nullptr, 0};

  record.provenance.upstream_repository = "local";
  record.provenance.upstream_revision = "local";
  record.provenance.upstream_paths = {nullptr, 0};
  record.provenance.local_sources = {kLlamaSources, 1};
  record.provenance.license_reviews = {kLlamaLicenses, 2};
  record.provenance.modifications = {nullptr, 0};

  record.image.image_path = "llama_k_projection_f16.image";
  record.image.image_sha256 = kLlamaImageDigest;
  record.image.image_size = 14961;
  record.image.code_object_version = "amdhsa-v6";
  record.image.build.toolchain_id = "clang-amdgpu";
  record.image.build.toolchain_revision = kB0ToolchainRevision;
  record.image.build.generator_id = "pack-generator";
  record.image.build.generator_revision = kB0GeneratorRevision;
  record.image.build.command_sha256 = kLlamaSourceDigest;

  record.entries = {&kLlamaEntry, 1};

  record.compatibility.input_dtype = "fp16";
  record.compatibility.weight_dtype = "fp16";
  record.compatibility.output_dtype = "fp16";
  record.compatibility.source_tensor_layout_version = "f16-row-major-nk-source-v1";
  record.compatibility.shape_families = {kLlamaFamilies, 1};
  record.compatibility.weight_packing_version = "source-equivalent-v1";

  record.numerics.input_dtype = "fp16";
  record.numerics.accumulation_dtype = "fp32";
  record.numerics.output_dtype = "fp16";
  record.numerics.cast_points = {kLlamaCastPoints, 1};
  record.numerics.finite_value_rule = "finite-input-output-v1";
  record.numerics.tolerance_policy = "exact-v1";
  record.numerics.reference_set_kind = "b0_scalar_control";
  record.numerics.retained_reference.present = true;
  record.numerics.retained_reference.value = make_ref(
      "logs/llama/numpy.json", "offline_oracle", "numpy_oracle", "llama-oracle-1",
      kB0RecordDigest, "", "", "", "cpu_reference", "", kB0InputDigest, kB0OutputDigest);
  record.numerics.numpy_oracle.present = false;
  record.numerics.scalar_native_projection.present = false;

  record.evidence.conformance = make_ref(
      "logs/llama/conformance.json", "target_conformance", "conformance", "llama-conformance-1",
      kB0RecordDigest, "gfx1201", kLlamaImageDigest, kLlamaPackDigest, "r9700_native", "",
      kB0InputDigest, kB0OutputDigest);
  record.evidence.native_run = make_ref(
      "logs/llama/native-run.json", "native_run", "native_run", "llama-native-run-1",
      kB0RecordDigest, "gfx1201", kLlamaImageDigest, kLlamaPackDigest, "r9700_native", "",
      kB0InputDigest, kB0OutputDigest);
  record.evidence.resource_review = make_ref(
      "logs/llama/resource-review.json", "offline_review", "resource_review", "llama-resource-1",
      kB0RecordDigest, "gfx1201", kLlamaImageDigest, kLlamaPackDigest, "", kLlamaSourceDigest,
      kB0InputDigest, kB0OutputDigest);
  record.evidence.isa_review = make_ref(
      "logs/llama/isa-review.json", "offline_review", "isa_review", "llama-isa-1", kB0RecordDigest,
      "gfx1201", kLlamaImageDigest, kLlamaPackDigest, "", kLlamaSourceDigest, kB0InputDigest,
      kB0OutputDigest);
  set_source_review(
      record,
      make_ref("logs/llama/source-review.json", "offline_review", "source_review",
               "llama-source-1", kB0RecordDigest, "gfx1201", kLlamaImageDigest, kLlamaPackDigest,
               "", kLlamaSourceDigest, kB0InputDigest, kB0OutputDigest));
  record.evidence.layout_proof.present = false;
  record.evidence.benchmark_record.present = false;
  record.evidence.benchmark_not_applicable_reason = "correctness-control-only";
  return record;
}

bool error_is_written(const char* error) {
  return error != nullptr && error[0] != '\0';
}

bool rejects(const KernelPackRecord& record) {
  char error[512] = {};
  if (native_r9700::validate_kernel_pack(record, {error, sizeof(error)})) return false;
  return error_is_written(error);
}

bool rejects_with_error_fragment(const KernelPackRecord& record,
                                 std::string_view expected_fragment) {
  char error[512] = {};
  if (native_r9700::validate_kernel_pack(record, {error, sizeof(error)})) return false;
  return std::string_view(error).find(expected_fragment) != std::string_view::npos;
}

KernelPackCompatibilityKey b0_key() {
  KernelPackCompatibilityKey key{};
  key.target = "gfx1201";
  key.required_features = {nullptr, 0};
  key.input_dtype = "fp16";
  key.weight_dtype = "fp16";
  key.output_dtype = "fp16";
  key.source_tensor_layout_version = "f16-row-major-nk-source-v1";
  key.shape_family_name = "fixed-test-family";
  key.fixed_dimensions = {kB0FixedDimensions, 2};
  key.runtime_value = {false, {}};
  key.weight_packing_version = "source-equivalent-v1";
  key.tolerance_policy = "exact-v1";
  return key;
}

KernelPackCompatibilityKey f2_key(std::uint32_t runtime_value) {
  KernelPackCompatibilityKey key{};
  key.target = "gfx1201";
  key.required_features = {kF2RequiredFeatures, 1};
  key.input_dtype = "fp16";
  key.weight_dtype = "fp16";
  key.output_dtype = "fp16";
  key.source_tensor_layout_version = "f16-row-major-nk-source-v1";
  key.shape_family_name = "f2-linear-gate-up-f16-v1";
  key.fixed_dimensions = {kF2FixedDimensions, 2};
  key.runtime_value = {true, {"M", runtime_value}};
  key.weight_packing_version = "f2-wmma-physical-tile-v1";
  key.tolerance_policy = "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1";
  return key;
}

KernelPackCompatibilityKey llama_key() {
  KernelPackCompatibilityKey key{};
  key.target = "gfx1201";
  key.required_features = {nullptr, 0};
  key.input_dtype = "fp16";
  key.weight_dtype = "fp16";
  key.output_dtype = "fp16";
  key.source_tensor_layout_version = "f16-row-major-nk-source-v1";
  key.shape_family_name = "llama-fixed-family";
  key.fixed_dimensions = {kLlamaFixedDimensions, 2};
  key.runtime_value = {false, {}};
  key.weight_packing_version = "source-equivalent-v1";
  key.tolerance_policy = "exact-v1";
  return key;
}

KernelPackCompatibilityKey llama_multi_key() {
  KernelPackCompatibilityKey key = llama_key();
  key.shape_family_name = "llama-selected-family";
  return key;
}

using AdmitKernelPackFunction = decltype(&native_r9700::admit_kernel_pack);

template <typename Function, typename = void>
struct supports_selected_admission : std::false_type {};

template <typename Function>
struct supports_selected_admission<
    Function,
    std::void_t<decltype(std::declval<Function>()(
        std::declval<const KernelPackRecord&>(),
        std::declval<const KernelPackCompatibilityKey&>(),
        std::declval<std::string_view>(),
        std::declval<std::string_view>(),
        std::declval<KernelDescriptor*>(),
        std::declval<KernelPackErrorBuffer>()))>> : std::true_type {};

constexpr bool kSupportsSelectedAdmission =
    supports_selected_admission<AdmitKernelPackFunction>::value;

template <typename Function>
bool call_admission(Function function,
                    const KernelPackRecord& record,
                    const KernelPackCompatibilityKey& key,
                    std::string_view entry_symbol,
                    std::string_view asset_root,
                    KernelDescriptor* out_descriptor,
                    KernelPackErrorBuffer error_text) {
  if constexpr (supports_selected_admission<Function>::value) {
    return function(record, key, entry_symbol, asset_root, out_descriptor, error_text);
  } else {
    return function(record, entry_symbol, asset_root, out_descriptor, error_text);
  }
}

bool admit_selected(const KernelPackRecord& record,
                    const KernelPackCompatibilityKey& key,
                    std::string_view entry_symbol,
                    std::string_view asset_root,
                    KernelDescriptor* out_descriptor,
                    KernelPackErrorBuffer error_text) {
  return call_admission(static_cast<AdmitKernelPackFunction>(&native_r9700::admit_kernel_pack),
                        record, key, entry_symbol, asset_root, out_descriptor, error_text);
}

bool valid_views_and_records() {
  static_assert(std::is_trivially_copyable_v<KernelPackSpan<int>>);
  static_assert(std::is_trivially_copyable_v<KernelPackOptional<int>>);
  static_assert(std::is_trivially_copyable_v<EvidenceRef>);
  static_assert(std::is_trivially_copyable_v<KernelPackIdentity>);
  static_assert(std::is_trivially_copyable_v<KernelPackRecord>);
  static_assert(std::is_trivially_copyable_v<KernelPackCompatibilityKey>);
  static_assert(std::is_trivially_copyable_v<KernelPackErrorBuffer>);
  static_assert(std::is_same_v<decltype(KernelPackSpan<int>::data), const int*>);
  static_assert(std::is_same_v<decltype(KernelPackSpan<int>::size), std::size_t>);
  static_assert(std::is_same_v<decltype(KernelPackErrorBuffer::data), char*>);
  static_assert(std::is_same_v<decltype(KernelPackErrorBuffer::size), std::size_t>);
  static_assert(std::is_same_v<decltype(KernelPackIdentity::name), std::string_view>);
  static_assert(std::is_same_v<decltype(KernelPackIdentity::required_features),
                               KernelPackSpan<std::string_view>>);

  const KernelPackRecord b0 = b0_pack();
  const KernelPackRecord f2 = f2_pack();
  char error[512] = {};
  if (!native_r9700::validate_kernel_pack(b0, {error, sizeof(error)})) return false;
  if (error[0] != '\0') return false;
  error[0] = '\0';
  if (!native_r9700::validate_kernel_pack(f2, {error, sizeof(error)})) return false;
  if (error[0] != '\0') return false;
  return b0.identity.schema_version == 1 && b0.identity.name == "test-pack" &&
         b0.identity.version == "1.0.0" && b0.identity.target == "gfx1201" &&
         b0.provenance.upstream_repository == "local" &&
         b0.image.image_path == "test_pack.image" && b0.image.image_size == 1024 &&
         b0.entries.size == 1 && b0.entries.data[0].symbol == "test_pack" &&
         b0.entries.data[0].kernargs.bytes == 32 &&
         b0.entries.data[0].kernargs.tail_padding_bytes == 4 &&
         b0.entries.data[0].resources.wave_size == 32 &&
         b0.entries.data[0].geometry.cases.data[0].geometry_rule == "exact-global-v1" &&
         b0.compatibility.shape_families.data[0].runtime_dimension.present == false &&
         b0.numerics.reference_set_kind == "b0_scalar_control" &&
         b0.evidence.native_run.record_kind == "native_run" &&
         b0.evidence.native_run.evidence_slot == "native_run" &&
         f2.compatibility.shape_families.data[0].runtime_dimension.present &&
         f2.compatibility.shape_families.data[0].runtime_dimension.value.max_value == 128 &&
         f2.evidence.layout_proof.present && f2.evidence.benchmark_record.present;
}

bool exact_key_matching() {
  const KernelPackRecord b0 = b0_pack();
  const KernelPackCompatibilityKey key = b0_key();
  if (!native_r9700::kernel_pack_matches_key(b0, key)) return false;

  KernelPackCompatibilityKey changed = key;
  changed.target = "gfx1202";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  const std::string_view wrong_feature[] = {"wave64"};
  changed.required_features = {wrong_feature, 1};
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.input_dtype = "bf16";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.weight_dtype = "bf16";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.output_dtype = "bf16";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.source_tensor_layout_version = "other-layout-v1";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.shape_family_name = "other-family";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  const KernelPackShapeDimension wrong_dimensions[] = {{"K", 2}, {"N", 1}};
  changed.fixed_dimensions = {wrong_dimensions, 2};
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.runtime_value = {true, {"M", 1}};
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.weight_packing_version = "other-pack-v1";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;
  changed = key;
  changed.tolerance_policy = "other-tolerance-v1";
  if (native_r9700::kernel_pack_matches_key(b0, changed)) return false;

  const KernelPackRecord f2 = f2_pack();
  if (!native_r9700::kernel_pack_matches_key(f2, f2_key(1))) return false;
  if (!native_r9700::kernel_pack_matches_key(f2, f2_key(128))) return false;
  if (native_r9700::kernel_pack_matches_key(f2, f2_key(0))) return false;
  if (native_r9700::kernel_pack_matches_key(f2, f2_key(129))) return false;
  KernelPackCompatibilityKey wrong_runtime_name = f2_key(13);
  wrong_runtime_name.runtime_value.value.name = "N";
  if (native_r9700::kernel_pack_matches_key(f2, wrong_runtime_name)) return false;
  return true;
}

bool explicit_span_lookup() {
  KernelPackRecord one[] = {b0_pack()};
  KernelPackSpan<KernelPackRecord> one_span{one, 1};
  char error[512] = {};
  if (native_r9700::find_kernel_pack(one_span, "test-pack", "1.0.0",
                                     {error, sizeof(error)}) != &one[0]) return false;
  if (native_r9700::find_kernel_pack(one_span, "test-pack", "",
                                     {error, sizeof(error)}) != nullptr || !error_is_written(error)) {
    return false;
  }
  error[0] = '\0';
  if (native_r9700::find_kernel_pack(one_span, "test-pack", "2.0.0",
                                     {error, sizeof(error)}) != nullptr || !error_is_written(error)) {
    return false;
  }
  error[0] = '\0';
  if (native_r9700::find_kernel_pack(one_span, "unknown-pack", "1.0.0",
                                     {error, sizeof(error)}) != nullptr || !error_is_written(error)) {
    return false;
  }

  const KernelPackCompatibilityKey key = b0_key();
  error[0] = '\0';
  if (native_r9700::find_kernel_pack_for_key(one_span, key, {error, sizeof(error)}) != &one[0]) {
    return false;
  }

  KernelPackRecord two[] = {b0_pack(), b0_pack()};
  error[0] = '\0';
  if (native_r9700::find_kernel_pack({two, 2}, "test-pack", "1.0.0",
                                     {error, sizeof(error)}) != nullptr || !error_is_written(error)) {
    return false;
  }
  error[0] = '\0';
  if (native_r9700::find_kernel_pack_for_key({two, 2}, key, {error, sizeof(error)}) != nullptr ||
      !error_is_written(error)) {
    return false;
  }

  KernelPackCompatibilityKey wrong_key = key;
  wrong_key.target = "gfx1202";
  error[0] = '\0';
  if (native_r9700::find_kernel_pack_for_key(one_span, wrong_key, {error, sizeof(error)}) != nullptr ||
      !error_is_written(error)) {
    return false;
  }
  return true;
}

bool malformed_records_reject() {
  KernelPackRecord candidate = b0_pack();

  candidate.identity.schema_version = 2;
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.identity.name = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.identity.version = "1.0";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.identity.target = "";
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  KernelPackSource missing_source = kB0Sources[0];
  missing_source.path = "";
  candidate.provenance.local_sources = {&missing_source, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackLicenseReview unknown_license = kB0Licenses[0];
  unknown_license.status = "unknown";
  candidate.provenance.license_reviews = {&unknown_license, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackLicenseReview missing_expression[2] = {kB0Licenses[0], kB0Licenses[1]};
  missing_expression[0].spdx_expression = "";
  candidate.provenance.license_reviews = {missing_expression, 2};
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  candidate.image.image_path = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.image.image_sha256 = "ABC";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.image.image_size = 0;
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.image.build.toolchain_id = "";
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  KernelPackEntry missing_symbol = kB0Entry;
  missing_symbol.symbol = "";
  candidate.entries = {&missing_symbol, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackEntry duplicate_entries[2] = {kB0Entry, kB0Entry};
  candidate.entries = {duplicate_entries, 2};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackEntry bad_kernarg_bytes = kB0Entry;
  bad_kernarg_bytes.kernargs.bytes = 31;
  candidate.entries = {&bad_kernarg_bytes, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackKernargField bad_alignment_fields[4] = {
      kB0KernargFields[0], kB0KernargFields[1], kB0KernargFields[2], kB0KernargFields[3]};
  bad_alignment_fields[0].offset = 1;
  KernelPackEntry bad_alignment = kB0Entry;
  bad_alignment.kernargs.fields = {bad_alignment_fields, 4};
  candidate.entries = {&bad_alignment, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackKernargField overlapping_fields[4] = {
      kB0KernargFields[0], kB0KernargFields[1], kB0KernargFields[2], kB0KernargFields[3]};
  overlapping_fields[1].offset = 4;
  KernelPackEntry overlapping = kB0Entry;
  overlapping.kernargs.fields = {overlapping_fields, 4};
  candidate.entries = {&overlapping, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackEntry bad_tail = kB0Entry;
  bad_tail.kernargs.tail_padding_bytes = 3;
  candidate.entries = {&bad_tail, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackEntry bad_wave = kB0Entry;
  bad_wave.resources.wave_size = 64;
  candidate.entries = {&bad_wave, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackEntry missing_resource_provenance = kB0Entry;
  missing_resource_provenance.resources.metadata_provenance = "";
  candidate.entries = {&missing_resource_provenance, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackGeometryCase bad_geometry = kB0GeometryCases[0];
  bad_geometry.geometry_rule = "arbitrary-formula";
  KernelPackEntry bad_geometry_entry = kB0Entry;
  bad_geometry_entry.geometry.cases = {&bad_geometry, 1};
  candidate.entries = {&bad_geometry_entry, 1};
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  candidate.compatibility.input_dtype = "float16";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.compatibility.source_tensor_layout_version = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackShapeDimension duplicate_dimensions[2] = {{"K", 1}, {"K", 1}};
  KernelPackShapeFamily duplicate_family = *candidate.compatibility.shape_families.data;
  duplicate_family.fixed_dimensions = {duplicate_dimensions, 2};
  candidate.compatibility.shape_families = {&duplicate_family, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  KernelPackShapeFamily unexpected_runtime = *candidate.compatibility.shape_families.data;
  unexpected_runtime.runtime_dimension.present = true;
  unexpected_runtime.runtime_dimension.value = {"M", 1, 1, 1};
  candidate.compatibility.shape_families = {&unexpected_runtime, 1};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.compatibility.weight_packing_version = "";
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  candidate.numerics.input_dtype = "bf16";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.accumulation_dtype = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.cast_points = {nullptr, 0};
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.finite_value_rule = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.tolerance_policy = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.reference_set_kind = "unknown";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.retained_reference.value.record_kind = "target_conformance";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.retained_reference.value.evidence_slot = "conformance";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.numerics.retained_reference.value.producer_kind = "r9700_native";
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  candidate.evidence.conformance.subject_target = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.conformance.image_sha256 = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.conformance.pack_sha256 = kB0SourceDigest;
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.conformance.producer_kind = "cpu_reference";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.conformance.tool_digest = kB0SourceDigest;
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.native_run.record_kind = "target_conformance";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.resource_review.producer_kind = "cpu_reference";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.resource_review.tool_digest = "";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.layout_proof.present = true;
  candidate.evidence.layout_proof.value = candidate.evidence.resource_review;
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.benchmark_record.present = true;
  candidate.evidence.benchmark_record.value = make_ref(
      "logs/test/benchmark.json", "benchmark", "benchmark", "benchmark-1", kB0RecordDigest,
      "gfx1201", kB0ImageDigest, kB0PackDigest, "r9700_native", kB0SourceDigest,
      kB0InputDigest, kB0OutputDigest);
  if (!rejects(candidate)) return false;

  // Mutating a preimage field while retaining the bound evidence digest must
  // reject; top-level evidence itself is excluded from the nonrecursive hash.
  candidate = b0_pack();
  candidate.identity.name = "different-pack";
  if (!rejects(candidate)) return false;
  candidate = b0_pack();
  candidate.evidence.conformance.pack_sha256 = kB0SourceDigest;
  if (!rejects(candidate)) return false;

  // The dual references bind the same request input but keep output evidence
  // separate; a disagreement is a malformed F2 pack.
  candidate = f2_pack();
  candidate.numerics.scalar_native_projection.value.input_digest = kF2SourceDigest;
  if (!rejects(candidate)) return false;
  return true;
}

bool source_review_matrix_is_representable() {
  const KernelPackRecord record = b0_pack();
  char error[512] = {};
  if (!native_r9700::validate_kernel_pack(record, {error, sizeof(error)})) return false;
  const EvidenceRef& source_review = record.evidence.source_review;
  return source_review.record_kind == "offline_review" &&
         source_review.evidence_slot == "source_review" &&
         source_review.producer_kind.empty() &&
         source_review.subject_target == "gfx1201" &&
         source_review.image_sha256 == kB0ImageDigest &&
         source_review.pack_sha256 == kB0PackDigest &&
         !record.evidence.layout_proof.present;
}

bool unresolved_license_is_rejected() {
  KernelPackLicenseReview licenses[2] = {kB0Licenses[0], kB0Licenses[1]};
  licenses[0].spdx_expression = "unknown";
  KernelPackRecord candidate = b0_pack();
  candidate.provenance.license_reviews = {licenses, 2};
  bind_pack_digest(candidate, kB0UnknownLicensePackDigest);
  if (!rejects(candidate)) return false;

  licenses[0].spdx_expression = "pending";
  candidate = b0_pack();
  candidate.provenance.license_reviews = {licenses, 2};
  bind_pack_digest(candidate, kB0PendingLicensePackDigest);
  if (!rejects(candidate)) return false;

  licenses[0].spdx_expression = "   \t";
  candidate = b0_pack();
  candidate.provenance.license_reviews = {licenses, 2};
  bind_pack_digest(candidate, kB0WhitespaceLicensePackDigest);
  return rejects(candidate);
}

bool source_equivalent_packing_needs_no_layout_proof() {
  KernelPackRecord candidate = b0_pack();
  candidate.compatibility.weight_packing_version = "source-equivalent-v1";
  candidate.evidence.layout_proof.present = false;
  bind_pack_digest(candidate, kB0SourceEquivalentPackDigest);
  char error[512] = {};
  return native_r9700::validate_kernel_pack(candidate, {error, sizeof(error)});
}

bool admission_binds_full_image_and_abi() {
  if constexpr (!kSupportsSelectedAdmission) {
    return false;
  } else {
    constexpr std::string_view kAssetRoot =
        "native_r9700/kernels/llama-k-projection-hsa-assets";
    const KernelPackCompatibilityKey key = llama_key();
    auto rejects_candidate = [&](KernelPackRecord candidate, std::string_view digest) {
      bind_pack_digest(candidate, digest);
      KernelDescriptor sentinel{};
      sentinel.name = "sentinel";
      sentinel.sha256 = "sentinel-sha";
      sentinel.code = {0xde, 0xad, 0xbe, 0xef};
      const KernelDescriptor before = sentinel;
      char error[512] = {};
      if (admit_selected(candidate, key, "llama_k_projection_f16", kAssetRoot, &sentinel,
                         {error, sizeof(error)})) {
        return false;
      }
      return sentinel.name == before.name && sentinel.sha256 == before.sha256 &&
             sentinel.code == before.code && error[0] != '\0';
    };

    KernelPackLicenseReview image_path_licenses[3] = {
        kLlamaLicenses[0], kLlamaLicenses[1],
        {"replacement.image", "MIT", "llama-replacement-license-1", "accepted"}};
    KernelPackRecord candidate = llama_pack();
    candidate.image.image_path = "replacement.image";
    candidate.provenance.license_reviews = {image_path_licenses, 3};
    if (!rejects_candidate(candidate, kLlamaImagePathMismatchPackDigest)) return false;

    candidate = llama_pack();
    candidate.image.image_size = 14960;
    if (!rejects_candidate(candidate, kLlamaImageSizeMismatchPackDigest)) return false;

    candidate = llama_pack();
    candidate.image.code_object_version = "amdhsa-v5";
    if (!rejects_candidate(candidate, kLlamaCodeObjectMismatchPackDigest)) return false;

    candidate = llama_pack();
    KernelPackEntry descriptor_mismatch = kLlamaEntry;
    descriptor_mismatch.descriptor_offset = 1601;
    candidate.entries = {&descriptor_mismatch, 1};
    if (!rejects_candidate(candidate, kLlamaDescriptorMismatchPackDigest)) return false;

    candidate = llama_pack();
    KernelPackEntry entry_mismatch = kLlamaEntry;
    entry_mismatch.entry_offset = 5889;
    candidate.entries = {&entry_mismatch, 1};
    if (!rejects_candidate(candidate, kLlamaEntryMismatchPackDigest)) return false;

    candidate = llama_pack();
    KernelPackKernargField kernarg_fields[4] = {
        kLlamaKernargFields[0], kLlamaKernargFields[1], kLlamaKernargFields[2],
        kLlamaKernargFields[3]};
    kernarg_fields[1].name = "different_weight";
    KernelPackEntry kernarg_mismatch = kLlamaEntry;
    kernarg_mismatch.kernargs.fields = {kernarg_fields, 4};
    candidate.entries = {&kernarg_mismatch, 1};
    if (!rejects_candidate(candidate, kLlamaKernargMismatchPackDigest)) return false;
    return true;
  }
}

bool admission_selects_requested_geometry_family() {
  if constexpr (!kSupportsSelectedAdmission) {
    return false;
  } else {
    constexpr std::string_view kAssetRoot =
        "native_r9700/kernels/llama-k-projection-hsa-assets";
    KernelPackRecord candidate = llama_pack();
    KernelPackEntry entry = kLlamaEntry;
    entry.geometry.cases = {kLlamaMultiGeometryCases, 2};
    candidate.entries = {&entry, 1};
    candidate.compatibility.shape_families = {kLlamaMultiFamilies, 2};
    bind_pack_digest(candidate, kLlamaMultiPackDigest);

    const KernelPackCompatibilityKey key = llama_multi_key();
    char error[512] = {};
    const KernelPackRecord* selected = native_r9700::find_kernel_pack_for_key(
        {&candidate, 1}, key, {error, sizeof(error)});
    if (selected != &candidate) {
      std::fprintf(stderr, "%s\n", error);
      return false;
    }

    KernelDescriptor descriptor{};
    error[0] = '\0';
    if (!admit_selected(candidate, key, "llama_k_projection_f16", kAssetRoot, &descriptor,
                        {error, sizeof(error)})) {
      std::fprintf(stderr, "%s\n", error);
      return false;
    }
    return descriptor.name == "llama_k_projection_f16" && descriptor.global_x == 64 &&
           descriptor.global_y == 1 && descriptor.global_z == 1;
  }
}

bool admission_preserves_output_and_fails_closed() {
  const KernelPackRecord valid = b0_pack();
  const KernelPackCompatibilityKey key = b0_key();
  KernelDescriptor sentinel{};
  sentinel.name = "sentinel";
  sentinel.sha256 = "sentinel-sha";
  sentinel.code = {0xde, 0xad, 0xbe, 0xef};
  const KernelDescriptor before = sentinel;
  char error[512] = {};

  KernelPackRecord malformed = valid;
  malformed.identity.name = "different-pack";
  if (admit_selected(
          malformed, key, "test_pack", "/definitely-not-an-asset-root", &sentinel,
          {error, sizeof(error)})) {
    return false;
  }
  if (sentinel.name != before.name || sentinel.sha256 != before.sha256 ||
      sentinel.code != before.code || error[0] == '\0') {
    return false;
  }

  error[0] = '\0';
  sentinel = before;
  if (admit_selected(
          valid, key, "not-an-entry", "/definitely-not-an-asset-root", &sentinel,
          {error, sizeof(error)})) {
    return false;
  }
  // A valid record reaches the existing asset/HSA boundary, which rejects the
  // absent asset without changing the caller-owned descriptor.
  return sentinel.name == before.name && sentinel.sha256 == before.sha256 &&
         sentinel.code == before.code && error[0] != '\0';
}

// Current-review regression: identity filtering must not hide a malformed
// generated record that does not match the requested name/version.
bool lookup_validates_malformed_nonmatching_record() {
  KernelPackRecord records[2] = {b0_pack(), b0_pack()};
  records[0].identity.name = "other-pack";
  records[0].identity.schema_version = 2;
  bind_pack_digest(records[0], kB0LookupMalformedPackDigest);
  bind_pack_digest(records[1], kB0PackDigest);

  char error[512] = {};
  const KernelPackRecord* selected = native_r9700::find_kernel_pack(
      {records, 2}, "test-pack", "1.0.0", {error, sizeof(error)});
  return selected == nullptr && error_is_written(error);
}

// Current-review regression: both runtime fixture families use the one closed
// finite-value rule, and every wrong rule mutation is independently resealed.
bool finite_value_rule_is_closed() {
  KernelPackRecord candidate = b0_pack();
  candidate.numerics.finite_value_rule = "finite-output-v1";
  bind_pack_digest(candidate, kB0WrongFiniteValueRulePackDigest);
  if (!rejects_with_error_fragment(candidate, "finite-value rule")) return false;

  candidate = f2_pack();
  candidate.numerics.finite_value_rule = "finite-output-v1";
  bind_pack_digest(candidate, kF2WrongFiniteValueRulePackDigest);
  return rejects_with_error_fragment(candidate, "finite-value rule");
}

// Current-review regression: provenance admits only local/local or the exact
// pinned LLVM source pair; repository/revision combinations are not mutable.
bool provenance_pairs_are_closed() {
  constexpr std::string_view pinned_paths[] = {kPinnedUpstreamPath};
  constexpr KernelPackLicenseReview pinned_licenses[] = {
      kB0Licenses[0],
      kB0Licenses[1],
      {kPinnedUpstreamPath, "Apache-2.0 WITH LLVM-exception", "llvm-amdgpu-usage-license-1",
       "accepted"},
  };

  KernelPackRecord candidate = b0_pack();
  candidate.provenance.upstream_repository = kPinnedUpstreamRepository;
  candidate.provenance.upstream_revision = kPinnedUpstreamRevision;
  candidate.provenance.upstream_paths = {pinned_paths, 1};
  candidate.provenance.license_reviews = {pinned_licenses, 3};
  bind_pack_digest(candidate, kB0PinnedProvenancePackDigest);
  char error[512] = {};
  if (!native_r9700::validate_kernel_pack(candidate, {error, sizeof(error)}) ||
      error[0] != '\0') {
    return false;
  }

  const auto rejects_pair = [](std::string_view repository,
                               std::string_view revision,
                               KernelPackSpan<std::string_view> upstream_paths,
                               KernelPackSpan<KernelPackLicenseReview> license_reviews,
                               std::string_view digest) {
    KernelPackRecord malformed = b0_pack();
    malformed.provenance.upstream_repository = repository;
    malformed.provenance.upstream_revision = revision;
    malformed.provenance.upstream_paths = upstream_paths;
    malformed.provenance.license_reviews = license_reviews;
    bind_pack_digest(malformed, digest);
    return rejects_with_error_fragment(malformed, "upstream provenance");
  };

  if (!rejects_pair("local", "main", {nullptr, 0}, {kB0Licenses, 2},
                    kB0LocalMainProvenancePackDigest)) {
    return false;
  }
  if (!rejects_pair("local", kPinnedUpstreamRevision, {nullptr, 0}, {kB0Licenses, 2},
                    kB0LocalPinnedProvenancePackDigest)) {
    return false;
  }
  if (!rejects_pair(kPinnedUpstreamRepository, "local", {pinned_paths, 1},
                    {pinned_licenses, 3}, kB0PinnedLocalProvenancePackDigest)) {
    return false;
  }
  if (!rejects_pair(kPinnedUpstreamRepository, "main", {pinned_paths, 1},
                    {pinned_licenses, 3}, kB0PinnedMainProvenancePackDigest)) {
    return false;
  }
  return rejects_pair("https://github.com/llvm/llvm-project-fork", kPinnedUpstreamRevision,
                      {pinned_paths, 1}, {pinned_licenses, 3},
                      kB0WrongRepositoryProvenancePackDigest);
}

// Current-review regression: runtime path safety matches the offline
// canonical relative-path boundary for source and evidence record paths.
bool unsafe_source_and_evidence_paths_reject() {
  const auto rejects_source = [](std::string_view path, std::string_view digest) {
    KernelPackSource source = kB0Sources[0];
    source.path = path;
    KernelPackLicenseReview licenses[2] = {kB0Licenses[0], kB0Licenses[1]};
    licenses[0].component = path;
    KernelPackRecord candidate = b0_pack();
    candidate.provenance.local_sources = {&source, 1};
    candidate.provenance.license_reviews = {licenses, 2};
    bind_pack_digest(candidate, digest);
    return rejects_with_error_fragment(candidate, "path is not canonical");
  };
  const auto rejects_evidence = [](std::string_view path) {
    KernelPackRecord candidate = b0_pack();
    candidate.evidence.conformance.record_path = path;
    bind_pack_digest(candidate, kB0PackDigest);
    return rejects_with_error_fragment(candidate, "path is not canonical");
  };

  std::string source_path = "C:/native_r9700/kernels/test_pack.cpp";
  if (!rejects_source(source_path, kB0SourceDrivePathPackDigest)) return false;
  std::string evidence_path = "C:/logs/test/conformance.json";
  if (!rejects_evidence(evidence_path)) return false;

  source_path = "native_r9700/kernels/test_pack.cpp";
  evidence_path = "logs/test/conformance.json";
  for (unsigned code = 0; code <= 0x1fU; ++code) {
    source_path.insert(source_path.size() - 4U, 1U, static_cast<char>(code));
    if (!rejects_source(source_path, kB0SourceControlPackDigests[code])) return false;
    source_path.erase(source_path.size() - 5U, 1U);

    evidence_path.insert(evidence_path.size() - 5U, 1U, static_cast<char>(code));
    if (!rejects_evidence(evidence_path)) return false;
    evidence_path.erase(evidence_path.size() - 6U, 1U);
  }

  source_path.insert(source_path.size() - 4U, 1U, static_cast<char>(0x7fU));
  if (!rejects_source(source_path, kB0SourceControlPackDigests[32])) return false;
  evidence_path.insert(evidence_path.size() - 5U, 1U, static_cast<char>(0x7fU));
  return rejects_evidence(evidence_path);
}

// Current-review regression: every modification component must have exactly
// one accepted component-level license review.
bool modification_license_coverage_is_required() {
  constexpr KernelPackModification modifications[] = {{
      "generated/test_pack.cpp", "deterministic generated runtime view",
  }};
  KernelPackRecord candidate = b0_pack();
  candidate.provenance.modifications = {modifications, 1};
  bind_pack_digest(candidate, kB0ModificationLicensePackDigest);
  return rejects_with_error_fragment(candidate, "modification license coverage");
}

// Corrective-green regression: zero descriptor and entry offsets are valid
// when each complete pack preimage and all evidence bindings are resealed.
bool zero_offsets_are_valid_when_resealed() {
  KernelPackRecord candidate = b0_pack();
  KernelPackEntry entry = kB0Entry;
  entry.descriptor_offset = 0;
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0DescriptorZeroPackDigest);

  char error[512] = {};
  if (!native_r9700::validate_kernel_pack(candidate, {error, sizeof(error)}) ||
      error[0] != '\0') {
    return false;
  }

  candidate = b0_pack();
  entry = kB0Entry;
  entry.entry_offset = 0;
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0EntryZeroPackDigest);
  error[0] = '\0';
  return native_r9700::validate_kernel_pack(candidate, {error, sizeof(error)}) &&
         error[0] == '\0';
}

// Current-review regression: descriptor and entry offsets are byte ranges
// inside the admitted image; the image-size boundary itself is out of range.
bool offsets_at_or_beyond_image_reject() {
  KernelPackRecord candidate = b0_pack();
  KernelPackEntry entry = kB0Entry;
  entry.descriptor_offset = candidate.image.image_size;
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0DescriptorAtImagePackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  entry = kB0Entry;
  entry.descriptor_offset = candidate.image.image_size + 1;
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0DescriptorBeyondImagePackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  entry = kB0Entry;
  entry.entry_offset = candidate.image.image_size;
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0EntryAtImagePackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  entry = kB0Entry;
  entry.entry_offset = candidate.image.image_size + 1;
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0EntryBeyondImagePackDigest);
  return rejects(candidate);
}

// Current-review regression: exact-global axes must divide their corresponding
// global dimensions, and dynamic LDS cannot exceed the entry resource limit.
bool geometry_axes_and_dynamic_lds_are_bounded() {
  KernelPackRecord candidate = b0_pack();
  KernelPackGeometryCase geometry = kB0GeometryCases[0];
  geometry.workgroup_x = 2;
  geometry.global_x = 3;
  KernelPackEntry entry = kB0Entry;
  entry.geometry.cases = {&geometry, 1};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0ExactGlobalXPackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  geometry = kB0GeometryCases[0];
  geometry.workgroup_y = 2;
  geometry.global_y = 3;
  entry = kB0Entry;
  entry.geometry.cases = {&geometry, 1};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0ExactGlobalYPackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  geometry = kB0GeometryCases[0];
  geometry.workgroup_z = 2;
  geometry.global_z = 3;
  entry = kB0Entry;
  entry.geometry.cases = {&geometry, 1};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0ExactGlobalZPackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  geometry = kB0GeometryCases[0];
  geometry.dynamic_lds_allowed = true;
  geometry.dynamic_lds_max_bytes = 1;
  entry = kB0Entry;
  entry.geometry.cases = {&geometry, 1};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0DynamicLdsPackDigest);
  return rejects(candidate);
}

// Current-review regression: a nonzero span with a null data pointer must be
// rejected before the canonical serializer dereferences it.
bool malformed_cast_point_span_rejects_without_crash() {
  KernelPackRecord candidate = b0_pack();
  candidate.numerics.cast_points = {nullptr, 1};
  char error[512] = {};
  return !native_r9700::validate_kernel_pack(candidate, {error, sizeof(error)}) &&
         error_is_written(error);
}

// Current-review regression: kernarg names are unique and each closed type
// has its declared size and alignment; every mutation below is independently
// resealed so a digest mismatch cannot explain the rejection.
bool kernarg_names_types_sizes_and_alignments_are_closed() {
  KernelPackRecord candidate = b0_pack();
  KernelPackKernargField fields[4] = {
      kB0KernargFields[0], kB0KernargFields[1], kB0KernargFields[2], kB0KernargFields[3]};
  fields[1].name = "activation";
  KernelPackEntry entry = kB0Entry;
  entry.kernargs.fields = {fields, 4};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0DuplicateKernargNamePackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  fields[0] = kB0KernargFields[0];
  fields[1] = kB0KernargFields[1];
  fields[2] = kB0KernargFields[2];
  fields[3] = kB0KernargFields[3];
  fields[3].type = "uint64";
  entry = kB0Entry;
  entry.kernargs.fields = {fields, 4};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0KernargTypePackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  fields[0] = kB0KernargFields[0];
  fields[1] = kB0KernargFields[1];
  fields[2] = kB0KernargFields[2];
  fields[3] = kB0KernargFields[3];
  fields[3].size = 8;
  entry = kB0Entry;
  entry.kernargs.tail_padding_bytes = 0;
  entry.kernargs.fields = {fields, 4};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0KernargSizePackDigest);
  if (!rejects(candidate)) return false;

  candidate = b0_pack();
  fields[0] = kB0KernargFields[0];
  fields[1] = kB0KernargFields[1];
  fields[2] = kB0KernargFields[2];
  fields[3] = kB0KernargFields[3];
  fields[3].alignment = 8;
  entry = kB0Entry;
  entry.kernargs.fields = {fields, 4};
  candidate.entries = {&entry, 1};
  bind_pack_digest(candidate, kB0KernargAlignmentPackDigest);
  return rejects(candidate);
}

// Current-review regression: admission derives the selected asset through the
// reviewed loader boundary and does not bake the K-projection ABI into the
// generic pack path. This uses a real reviewed V-projection HSA asset.
bool generic_admission_accepts_reviewed_v_projection() {
  if constexpr (!kSupportsSelectedAdmission) return false;

  const native_r9700::LlamaKernelAsset* asset =
      native_r9700::find_llama_kernel_asset("llama_v_projection_f16");
  if (asset == nullptr || asset->descriptor.name != "llama_v_projection_f16" ||
      asset->descriptor.sha256 != kLlamaVImageDigest ||
      asset->descriptor.rsrc1 != 3222208513U || asset->descriptor.rsrc2 != 132U ||
      asset->descriptor.rsrc3 != 48U || asset->descriptor.workgroup_x != 64U ||
      asset->descriptor.workgroup_y != 1U || asset->descriptor.workgroup_z != 1U ||
      asset->descriptor.global_x != 64U || asset->descriptor.global_y != 1U ||
      asset->descriptor.global_z != 1U || asset->descriptor.kernarg_bytes != 32U ||
      asset->location.code_path != "llama_v_projection_f16.image" ||
      asset->location.sha256 != kLlamaVImageDigest || asset->location.target != "gfx1201" ||
      asset->kernarg_schema != "llama-v-projection-f16-v1") {
    return false;
  }

  constexpr std::string_view kAssetRoot =
      "native_r9700/kernels/llama-v-projection-hsa-assets";
  constexpr KernelPackSource kSources[] = {{
      "native_r9700/kernels/llama_v_projection_f16.cpp", kLlamaVSourceDigest,
  }};
  constexpr KernelPackLicenseReview kLicenses[] = {
      {"native_r9700/kernels/llama_v_projection_f16.cpp", "MIT", "llama-v-source-1",
       "accepted"},
      {"llama_v_projection_f16.image", "MIT", "llama-v-image-1", "accepted"},
  };
  constexpr KernelPackKernargField kFields[] = {
      {"normalized", "uint64", 0, 8, 8},
      {"v_projection_weight", "uint64", 8, 8, 8},
      {"fresh_v", "uint64", 16, 8, 8},
      {"sequence_length", "uint32", 24, 4, 4},
  };
  KernelPackEntry entry = kLlamaEntry;
  entry.symbol = "llama_v_projection_f16";
  entry.kernargs.fields = {kFields, 4};

  KernelPackRecord candidate = llama_pack();
  candidate.provenance.local_sources = {kSources, 1};
  candidate.provenance.license_reviews = {kLicenses, 2};
  candidate.image.image_path = "llama_v_projection_f16.image";
  candidate.image.image_sha256 = kLlamaVImageDigest;
  candidate.image.build.command_sha256 = kLlamaVSourceDigest;
  candidate.entries = {&entry, 1};
  candidate.evidence.conformance.image_sha256 = kLlamaVImageDigest;
  candidate.evidence.native_run.image_sha256 = kLlamaVImageDigest;
  candidate.evidence.resource_review.image_sha256 = kLlamaVImageDigest;
  candidate.evidence.isa_review.image_sha256 = kLlamaVImageDigest;
  candidate.evidence.source_review.image_sha256 = kLlamaVImageDigest;
  bind_pack_digest(candidate, kLlamaVPackDigest);

  KernelDescriptor loaded{};
  char error[512] = {};
  if (!admit_selected(candidate, llama_key(), "llama_v_projection_f16", kAssetRoot, &loaded,
                      {error, sizeof(error)})) {
    std::fprintf(stderr, "%s\n", error);
    return false;
  }
  return error[0] == '\0' && loaded.name == "llama_v_projection_f16" &&
         loaded.sha256 == kLlamaVImageDigest && loaded.code.size() == 14961U &&
         loaded.rsrc1 == 3222208513U && loaded.rsrc2 == 132U && loaded.rsrc3 == 48U &&
         loaded.workgroup_x == 64U && loaded.workgroup_y == 1U && loaded.workgroup_z == 1U &&
         loaded.global_x == 64U && loaded.global_y == 1U && loaded.global_z == 1U &&
         loaded.kernarg_bytes == 32U;
}


}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 64;
  const std::string_view mode(argv[1]);
  if (mode == "valid") return valid_views_and_records() ? 0 : 1;
  if (mode == "key") return exact_key_matching() ? 0 : 2;
  if (mode == "lookup") return explicit_span_lookup() ? 0 : 3;
  if (mode == "reject") return malformed_records_reject() ? 0 : 4;
  if (mode == "admit") return admission_preserves_output_and_fails_closed() ? 0 : 5;
  if (mode == "source-review") return source_review_matrix_is_representable() ? 0 : 6;
  if (mode == "license") return unresolved_license_is_rejected() ? 0 : 7;
  if (mode == "source-equivalent") {
    return source_equivalent_packing_needs_no_layout_proof() ? 0 : 8;
  }
  if (mode == "admission-binding") return admission_binds_full_image_and_abi() ? 0 : 9;
  if (mode == "geometry-family") {
    return admission_selects_requested_geometry_family() ? 0 : 10;
  }
  if (mode == "lookup-malformed") {
    return lookup_validates_malformed_nonmatching_record() ? 0 : 11;
  }
  if (mode == "zero-offsets") {
    return zero_offsets_are_valid_when_resealed() ? 0 : 12;
  }
  if (mode == "finite-rule") return finite_value_rule_is_closed() ? 0 : 18;
  if (mode == "provenance") return provenance_pairs_are_closed() ? 0 : 19;
  if (mode == "paths") return unsafe_source_and_evidence_paths_reject() ? 0 : 20;
  if (mode == "modification-license") {
    return modification_license_coverage_is_required() ? 0 : 21;
  }
  if (mode == "offset-bounds") return offsets_at_or_beyond_image_reject() ? 0 : 13;
  if (mode == "geometry-lds") return geometry_axes_and_dynamic_lds_are_bounded() ? 0 : 14;
  if (mode == "cast-span") {
    return malformed_cast_point_span_rejects_without_crash() ? 0 : 15;
  }
  if (mode == "kernarg-schema") {
    return kernarg_names_types_sizes_and_alignments_are_closed() ? 0 : 16;
  }
  if (mode == "generic-admission") {
    return generic_admission_accepts_reviewed_v_projection() ? 0 : 17;
  }
  return 65;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "kernel_pack_contract_probe"
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
            *[str(path) for path in RUNTIME_SOURCES],
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


def _run_probe(executable: Path, mode: str) -> None:
    completed = subprocess.run(
        [str(executable), mode], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_kernel_pack_runtime_views_compile_and_validate(tmp_path: Path) -> None:
    """All identity/provenance/image/ABI/compatibility/numerical fields are concrete views."""
    _run_probe(_compile_pack_probe(tmp_path), "valid")


def test_kernel_pack_exact_key_matching_rejects_out_of_range_runtime_values(tmp_path: Path) -> None:
    """Key equality is complete and bounded M tails never create a new family."""
    _run_probe(_compile_pack_probe(tmp_path), "key")


def test_kernel_pack_lookup_uses_explicit_span_and_rejects_zero_or_multiple(tmp_path: Path) -> None:
    """Identity lookup is exact name/version; zero and ambiguous spans fail closed."""
    _run_probe(_compile_pack_probe(tmp_path), "lookup")




def test_kernel_pack_rejects_malformed_identity_abi_numerics_and_evidence(tmp_path: Path) -> None:
    """Malformed fields and every invalid EvidenceRef role leave no selectable record."""
    _run_probe(_compile_pack_probe(tmp_path), "reject")


def test_kernel_pack_admission_reuses_hsa_boundary_and_preserves_output(tmp_path: Path) -> None:
    """Pack validation precedes existing HSA/KernelAsset admission and output mutation."""
    _run_probe(_compile_pack_probe(tmp_path), "admit")


def test_kernel_pack_source_review_is_a_required_evidence_matrix_member(tmp_path: Path) -> None:
    """A B0 pack carries offline_review/source_review separately from layout_proof."""
    _run_probe(_compile_pack_probe(tmp_path), "source-review")


def test_kernel_pack_rejects_unresolved_spdx_even_when_status_is_accepted(tmp_path: Path) -> None:
    """Runtime license admission rejects unknown/pending SPDX expressions."""
    _run_probe(_compile_pack_probe(tmp_path), "license")


def test_kernel_pack_accepts_source_equivalent_packing_without_layout_proof(tmp_path: Path) -> None:
    """The source-equivalent marker is not physical repacking requiring a proof."""
    _run_probe(_compile_pack_probe(tmp_path), "source-equivalent")


def test_kernel_pack_admission_binds_full_image_and_abi_metadata(tmp_path: Path) -> None:
    """Image identity and same-sized ABI layout mismatches fail before output mutation."""
    _run_probe(_compile_pack_probe(tmp_path), "admission-binding")


def test_kernel_pack_admission_uses_the_key_selected_later_geometry_family(tmp_path: Path) -> None:
    """Admission resolves the exact family selected by the compatibility key."""
    _run_probe(_compile_pack_probe(tmp_path), "geometry-family")


def test_kernel_pack_current_review_lookup_validates_malformed_nonmatching_record(
    tmp_path: Path,
) -> None:
    """Production change: validate every generated record before name/version filtering."""
    _run_probe(_compile_pack_probe(tmp_path), "lookup-malformed")


def test_kernel_pack_current_review_accepts_resealed_zero_descriptor_and_entry_offsets(
    tmp_path: Path,
) -> None:
    """Corrective-green: in-image descriptor and entry offsets may both be zero."""
    _run_probe(_compile_pack_probe(tmp_path), "zero-offsets")


def test_kernel_pack_current_review_rejects_wrong_finite_value_rule_at_both_boundaries(
    tmp_path: Path,
) -> None:
    """Production change: both runtime pack families require finite-input-output-v1."""
    _run_probe(_compile_pack_probe(tmp_path), "finite-rule")


def test_kernel_pack_current_review_rejects_mutable_or_contradictory_provenance(
    tmp_path: Path,
) -> None:
    """Production change: provenance is local/local or the exact pinned LLVM pair."""
    _run_probe(_compile_pack_probe(tmp_path), "provenance")


def test_kernel_pack_current_review_rejects_noncanonical_source_and_evidence_paths(
    tmp_path: Path,
) -> None:
    """Production change: runtime safe paths reject drive prefixes and C0/DEL."""
    _run_probe(_compile_pack_probe(tmp_path), "paths")


def test_kernel_pack_current_review_rejects_unlicensed_modification_component(
    tmp_path: Path,
) -> None:
    """Production change: every modification component needs one accepted review."""
    _run_probe(_compile_pack_probe(tmp_path), "modification-license")


def test_kernel_pack_current_review_rejects_offsets_at_or_beyond_image_size(
    tmp_path: Path,
) -> None:
    """Production change: pass image_size into entry validation and reject out-of-image offsets."""
    _run_probe(_compile_pack_probe(tmp_path), "offset-bounds")


def test_kernel_pack_current_review_rejects_nondivisible_exact_geometry_and_lds_overflow(
    tmp_path: Path,
) -> None:
    """Production change: enforce exact-global divisibility and entry-resource LDS bounds."""
    _run_probe(_compile_pack_probe(tmp_path), "geometry-lds")


def test_kernel_pack_current_review_rejects_null_nonzero_cast_point_span(
    tmp_path: Path,
) -> None:
    """Production change: validate nested spans before canonical digest serialization."""
    _run_probe(_compile_pack_probe(tmp_path), "cast-span")


def test_kernel_pack_current_review_rejects_duplicate_and_mismatched_kernarg_fields(
    tmp_path: Path,
) -> None:
    """Production change: close kernarg type/size/alignment rules and reject duplicate names."""
    _run_probe(_compile_pack_probe(tmp_path), "kernarg-schema")


def test_kernel_pack_current_review_admits_reviewed_non_k_projection_symbol(
    tmp_path: Path,
) -> None:
    """Production change: derive admission from the reviewed asset, not K-projection constants."""
    _run_probe(_compile_pack_probe(tmp_path), "generic-admission")


def test_kernel_pack_runtime_has_no_owning_records_or_manifest_parser() -> None:
    """Runtime source contains views only and never parses offline JSON/YAML policy."""
    assert KERNEL_PACK_HEADER.is_file(), "kernel pack header is missing"
    assert KERNEL_PACK_SOURCE.is_file(), "kernel pack source is missing"
    header = KERNEL_PACK_HEADER.read_text(encoding="utf-8")
    source = KERNEL_PACK_SOURCE.read_text(encoding="utf-8")
    runtime_text = header + "\n" + source

    assert "std::string_view" in header
    assert "KernelPackSpan" in header
    assert "KernelPackOptional" in header
    assert "KernelPackSpan<KernelPackRecord>" in header
    for forbidden in (
        r"\bstd::string\b",
        r"\bstd::vector\b",
        r"\bstd::map\b",
        r"\bstd::unordered_map\b",
        r"\bstd::unique_ptr\b",
        r"\bstd::shared_ptr\b",
        r"operator new",
        r"operator delete",
        r"malloc\s*\(",
        r"free\s*\(",
    ):
        assert not re.search(forbidden, runtime_text), f"forbidden runtime construct: {forbidden}"

    assert "load_verified_kernel_code" in source or "load_llama_embed_hsa_image" in source
    assert "find_llama_kernel_asset" in source or "hsa_code_image_asset" in source
    assert not re.search(r"(?m)^\s*(?:static|constexpr)\s+(?:const\s+)?KernelPackRecord\b", source)
    lowered = source.lower()
    assert all(
        state in lowered for state in ("unseen", "validating", "admitted", "rejected", "loaded", "retired")
    )

    kinds = {
        "offline_oracle",
        "offline_review",
        "target_conformance",
        "native_run",
        "benchmark",
    }
    slots = {
        "numpy_oracle",
        "source_review",
        "isa_review",
        "resource_review",
        "layout_proof",
        "scalar_native_projection",
        "conformance",
        "native_run",
        "benchmark",
    }
    for value in kinds | slots:
        assert value in runtime_text, f"EvidenceRef matrix value missing: {value}"


def _compile_scalar_selection_probe(tmp_path: Path) -> Path:
    """Compile the generated scalar span and its evidence-gated selection path."""
    assert SCALAR_GENERATED_SOURCE.is_file(), "generated scalar pack source is missing"
    probe_source = tmp_path / "scalar_pack_selection_probe.cpp"
    probe_source.write_text(
        r'''
#include <filesystem>
#include <string>
#include <string_view>

#include "kernel_pack.h"

int main() {
  const native_r9700::KernelPackSpan<native_r9700::KernelPackRecord> records =
      native_r9700::llama_kernel_pack_records();
  constexpr const char* kNames[] = {
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
  if (records.data == nullptr ||
      records.size != sizeof(kNames) / sizeof(kNames[0])) {
    return 1;
  }
  for (std::size_t index = 0; index < records.size; ++index) {
    if (records.data[index].identity.name != kNames[index] ||
        records.data[index].identity.version != "1.0.0" ||
        records.data[index].entries.size != 1 ||
        records.data[index].entries.data[0].symbol != kNames[index]) {
      return 2;
    }
  }

  char error[512] = {};
  for (const char* name : kNames) {
    error[0] = '\0';
    if (native_r9700::find_llama_kernel_pack(
            name, "1.0.0", {error, sizeof(error)}) == nullptr ||
        error[0] != '\0') {
      return 3;
    }
  }
  error[0] = '\0';
  const native_r9700::KernelPackRecord* selected =
      native_r9700::find_llama_kernel_pack(
          "llama_k_projection_f16", "1.0.0", {error, sizeof(error)});
  if (selected == nullptr) return 4;

  const native_r9700::KernelPackShapeFamily& family =
      selected->compatibility.shape_families.data[0];
  native_r9700::KernelPackCompatibilityKey key{};
  key.target = selected->identity.target;
  key.required_features = selected->identity.required_features;
  key.input_dtype = selected->compatibility.input_dtype;
  key.weight_dtype = selected->compatibility.weight_dtype;
  key.output_dtype = selected->compatibility.output_dtype;
  key.source_tensor_layout_version =
      selected->compatibility.source_tensor_layout_version;
  key.shape_family_name = family.name;
  key.fixed_dimensions = family.fixed_dimensions;
  key.runtime_value.present = false;
  key.weight_packing_version = selected->compatibility.weight_packing_version;
  key.tolerance_policy = selected->numerics.tolerance_policy;

  native_r9700::KernelDescriptor descriptor{};
  error[0] = '\0';
  const std::string asset_root =
      std::filesystem::current_path().string();
  if (!native_r9700::admit_llama_kernel_pack(
          *selected, key, "llama_k_projection_f16", asset_root,
          &descriptor, {error, sizeof(error)})) {
    return 5;
  }
  return descriptor.name == "llama_k_projection_f16" &&
                 descriptor.code.size() == 14961 &&
             descriptor.sha256 ==
                 "9c2f584f4bd4c918f8c2a95a0a1f29a7102c19e8080b0d538b36f26e6e8fcc9b"
         ? 0
         : 6;
}
'''.lstrip(),
        encoding="utf-8",
    )
    executable = tmp_path / "scalar_pack_selection_probe"
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
            *(str(path) for path in RUNTIME_SOURCES),
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


def test_scalar_pack_manifests_and_generated_span_cover_current_asset_order() -> None:
    """Each reviewed Llama asset has one canonical pack and bound evidence files."""
    manifests_by_name = {}
    for path in SCALAR_PACK_ROOT.glob("*-hsa-assets/*.pack.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests_by_name[payload["name"]] = payload
        assert payload["schema_version"] == 1
        assert len(payload["entries"]) == 1
        assert payload["entries"][0]["symbol"] == payload["name"]
        assert payload["image"]["image_path"].endswith(
            f"{path.parent.name}/{payload['image']['image_path'].split('/')[-1]}"
        )
        for reference in (
            payload["numerics"]["retained_reference"],
            payload["evidence"]["conformance"],
            payload["evidence"]["source_review"],
            payload["evidence"]["native_run"],
            payload["evidence"]["resource_review"],
            payload["evidence"]["isa_review"],
        ):
            assert reference["record_path"]
            assert Path(reference["record_path"]).is_file()
    assert set(manifests_by_name) == set(SCALAR_PACK_NAMES)
    generated = SCALAR_GENERATED_SOURCE.read_text(encoding="utf-8")
    assert "llama_scalar_pack_records" in generated
    assert "llama_selectable_scalar_pack_records" in generated
    assert generated.count("constexpr KernelPackRecord") >= 15


def test_scalar_pack_selection_reuses_verified_loader_for_all_native_passes(
    tmp_path: Path,
) -> None:
    """All 13 native-pass scalar packs select through the legacy verifier."""
    completed = subprocess.run(
        [str(_compile_scalar_selection_probe(tmp_path))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
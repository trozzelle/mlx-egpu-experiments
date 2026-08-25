// native_r9700/runtime.h — C1 native runner runtime shell (task set 4).
//
// Reusable lifecycle shell for the macOS TinyGPU.app/APLRemotePCIDevice/PCIIface
// AMDev substrate, refactored from the proven C0 probe
// `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp` (C0A25
// `--kernel-proof` PASS). The C0 probe stays byte-stable and untouched; this
// header/source pair adapt its proven packet encodings and lifecycle ordering
// into a narrow harness-callable shell for C1 task sets 5-8. `--kernel-proof`
// wraps the frozen C0A25 hardware proof through RuntimeSession; `--transfer-proof`
// reuses the C0 AMDev memory/SDMA path for layer-sized byte round trips.
//
// Provenance of the encoding constants below mirrors the C0 probe, which carries
// the small mechanics rederived/ported from tinygrad (see the MIT notice in the
// C0 probe): this shell does not import, call, or vendor tinygrad runtime code.
//
// Lifecycle (all stages idempotent-ordered; out-of-order calls fail loudly).
// The no-hardware stages in this shell validate ordering and record their
// intended effect in the log. `kernel_proof()` is the hardware bridge for the
// known C0A25 `out[i]=in[i]+1` proof; later C1R tasks replace the wrapper with
// direct reusable buffer/kernel primitives:
//   initialize          -> record substrate identity; connect is a deferred gate.
//   allocate_buffers    -> record staging/readback intent; mapping is a deferred gate.
//   copy_input          -> record input digest; SDMA submit is a deferred gate.
//   load_kernel         -> kernel load status is a deferred gate (BAR0 write).
//   write_kernargs      -> 24-byte kernarg layout {output_va@0, input_va@8,
//                          scalar_va@16, scalar:u32@24}; CPU-side self-checked.
//   dispatch_and_poll   -> PM4 dispatch encoding checked; doorbell is a deferred gate.
//   readback_and_compare-> CPU-side digest check; device readback is a deferred gate.
//   cleanup             -> finalize standardized log.
//
// Every public stage writes into RuntimeLog; the runner writes a timestamped
// log file under logs/. Hardware-free contract coverage is provided by
// RuntimeSession::dry_run (see runner.cpp), which exercises ordering, kernarg
// layout, packet encodings, and log writing without a TinyGPU socket.

#ifndef NATIVE_R9700_RUNTIME_H_
#define NATIVE_R9700_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "amdev_session.h"

namespace native_r9700 {

// Frozen C1 substrate identity.
inline constexpr const char* kRuntimeSubstrate = "TinyGPU.app/APLRemotePCIDevice/PCIIface";
inline constexpr const char* kTinyGpuAppPath = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";
inline constexpr uint32_t kTargetVendor = 0x1002U;   // AMD
inline constexpr uint32_t kTargetDevice = 0x7551U;   // gfx1201 R9700 (RDNA4)
inline constexpr uint32_t kRemoteDevId = 0U;         // APLRemotePCIDevice uses pcibus "usb4".

// Frozen C1 minimal kernel identity (per-u32 add-one kernel, C0A25 PASS).
inline constexpr const char* kKernelSourceId = "c0a-minimal-u32-add-one-v3";
inline constexpr const char* kKernelArch = "gfx1201";
inline constexpr const char* kKernelInputValuesU32 = "1,2,3,4,5,6,7,8";
inline constexpr const char* kKernelExpectedOutputValuesU32 = "2,3,4,5,6,7,8,9";
inline constexpr uint32_t kElementCount = 8U;
inline constexpr uint32_t kTransferByteCount = 8U * sizeof(uint32_t);  // 32 bytes

inline constexpr uint64_t kLayerSliceTransferByteCount = 5ULL * 2048ULL * sizeof(uint16_t);
inline constexpr uint64_t kTransferProofChunkByteCount = 0x1000ULL;
inline constexpr uint64_t kMaxTransferProofByteCount = 64ULL * 1024ULL * 1024ULL;
inline constexpr const char* kFirstPrimitiveName = "fp32_add_scalar";
inline constexpr const char* kFirstPrimitiveSourceId = "c1r5-fp32-add-scalar-v1";
inline constexpr const char* kFirstPrimitiveBackend = "hardware";
inline constexpr const char* kFirstPrimitiveKernelSha256 =
    "697ba0c938e34d6f8db6498a803fb1d82181b111b28fe8c60acaac6a8d6011fd";
inline constexpr const char* kFirstPrimitiveKernelTextByteCount = "64";
inline constexpr const char* kFirstPrimitiveElementType = "fp32";
inline constexpr const char* kFirstPrimitiveElementCount = "8";
inline constexpr const char* kFirstPrimitiveByteCount = "32";
inline constexpr const char* kFirstPrimitiveScalarBits = "0x3f800000";
inline constexpr const char* kFirstPrimitiveTolerance = "exact_bytes";
inline constexpr const char* kFirstPrimitiveMaxAbsDiff = "0";
inline constexpr const char* kFirstPrimitiveMaxUlpDiff = "0";
inline constexpr const char* kFirstPrimitiveMismatchCount = "0";
inline constexpr const char* kFp16ToFp32PrimitiveName = "fp16_to_fp32_cast";
inline constexpr const char* kFp16ToFp32PrimitiveSourceId = "c1r6-fp16-to-fp32-cast-v1";
inline constexpr const char* kFp16ToFp32PrimitiveKernelSha256 =
    "d18f462a15fc21d48eedf32bcbcf24a0b6eb270a41707f8cdf95c1b27653ead0";
inline constexpr const char* kFp16ToFp32PrimitiveElementType = "fp16_to_fp32";
inline constexpr const char* kFp16ToFp32PrimitiveElementCount = "8";
inline constexpr const char* kFp16ToFp32PrimitiveInputByteCount = "16";
inline constexpr const char* kFp16ToFp32PrimitiveOutputByteCount = "32";
inline constexpr const char* kFp16ToFp32PrimitiveScalarBits = "unused";
inline constexpr const char* kFp32ToFp16PrimitiveName = "fp32_to_fp16_cast";
inline constexpr const char* kFp32ToFp16PrimitiveSourceId = "c1r6d-fp32-to-fp16-cast-v1";
inline constexpr const char* kFp32ToFp16PrimitiveKernelSha256 =
    "dc5dd58390142a22d249986d015be589ea62732d36303b68b8528e09a010735d";
inline constexpr const char* kFp32ToFp16PrimitiveElementType = "fp32_to_fp16";
inline constexpr const char* kFp32ToFp16PrimitiveElementCount = "8";
inline constexpr const char* kFp32ToFp16PrimitiveInputByteCount = "32";
inline constexpr const char* kFp32ToFp16PrimitiveOutputByteCount = "16";
inline constexpr const char* kFp32ToFp16PrimitiveScalarBits = "unused";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveName = "fp16_matmul_8x16x8";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveSourceId =
    "c1r6b-fp16-matmul-8x16x8-v1";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveKernelSha256 =
    "56e4faa6c8fa01ca6d9ea97ac5857ee9fc074d1cd51a883313c97c2fbb6cb28f";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveKernelTextByteCount = "2508";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveElementType = "fp16_matmul_fp32";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveElementCount = "64";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveInputByteCount = "512";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveOutputByteCount = "256";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveScalarBits = "unused";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveInputShape = "a=8x16,b=16x8";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveOutputShape = "8x8";
inline constexpr const char* kFp16Matmul8x16x8PrimitiveInputLayout =
    "a_row_major_then_b_kpair_col_packed";

inline constexpr const char* kFp16Matmul8x16x8Layer0KTilePrimitiveName =
    "fp16_matmul_8x16x8_layer0_k_tile";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTilePrimitiveSourceId =
    "c1r6c-layer0-k-proj-tile-v1";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileAcceptanceScope =
    "hardware_primitive_tile_only";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileModelForwardScope =
    "layer0_k_proj_partial_tile";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileNativePrefillAcceptance = "open";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileRowsValid = "5";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileRows = "8";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileInner = "16";
inline constexpr const char* kFp16Matmul8x16x8Layer0KTileCols = "8";

inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveName =
    "fp16_residual_add_layer0_attention_slice8";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveSourceId =
    "c1r6f-layer0-attention-residual-add-slice8-v1";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveKernelSha256 =
    "57309c2e2441d96284b716ad71e5612e4b689055fc4e6d8a9be8aebb76764122";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveKernelTextByteCount =
    "128";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveElementType =
    "fp16_add";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveElementCount = "8";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveInputShape =
    "lhs=8,rhs=8";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveOutputShape = "8";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveInputLayout =
    "layer0_hidden_in_fp16[0,0:8]_then_layer0_o_proj_output_fp16[0,0:8]";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveInputByteCount = "32";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveOutputByteCount = "16";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8PrimitiveScalarBits = "unused";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8AcceptanceScope =
    "hardware_primitive_slice_only";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8ModelForwardScope =
    "layer0_attention_residual_partial_slice";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8NativePrefillAcceptance = "open";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8FixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8SourceArrays =
    "layer0_hidden_in_fp16,layer0_o_proj_output_fp16,layer0_attention_residual_fp16";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8FixtureSlice =
    "token=0,hidden_dim=0:8";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8FullFixtureShape = "2x16";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8CoveredElementCount = "8";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8FullElementCount = "32";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8Tolerance = "exact_fp16_bytes";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8MaxAbsDiff = "0";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8MaxUlpDiff = "0";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8MismatchCount = "0";
inline constexpr const char* kFp16ResidualAddLayer0AttentionSlice8ByteMismatchCount = "0";

inline constexpr const char* kFp16RmsNorm1x64PrimitiveName = "fp16_rms_norm_1x64";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveSourceId =
    "c1r6g-fp16-rms-norm-1x64-v1";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveKernelSha256 =
    "d3872040a6a22820bf3a0380511baa6e3cd408f10df8ad98d72fc25b3d6bc803";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveKernelTextByteCount = "2368";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveElementType =
    "fp16_rms_norm_fp32";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveElementCount = "64";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveInputShape =
    "x=1x64,weight=64";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveOutputShape = "1x64";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveInputLayout =
    "rms_x_fp16_then_rms_weight_fp16";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveInputByteCount = "256";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveOutputByteCount = "128";
inline constexpr const char* kFp16RmsNorm1x64PrimitiveScalarBits = "0x3727c5ac";
inline constexpr const char* kFp16RmsNorm1x64AcceptanceScope =
    "hardware_primitive_tile_only";
inline constexpr const char* kFp16RmsNorm1x64ModelForwardScope =
    "rms_norm_synthetic_1x64_math_only";
inline constexpr const char* kFp16RmsNorm1x64NativePrefillAcceptance = "open";
inline constexpr const char* kFp16RmsNorm1x64SourceFixture =
    "tests/native_r9700/fixtures/primitives_fixtures.npz";
inline constexpr const char* kFp16RmsNorm1x64FixtureSha256 =
    "d52e352e2450b521a5e81dbc8672673b94a28524dbb6bf66314ba3d882924faf";
inline constexpr const char* kFp16RmsNorm1x64SourceArrays =
    "rms_x_fp16,rms_weight_fp16,rms_eps,rms_expected_fp16";
inline constexpr const char* kFp16RmsNorm1x64FixtureSlice = "row=0,hidden_dim=0:64";
inline constexpr const char* kFp16RmsNorm1x64FullFixtureShape = "1x64";
inline constexpr const char* kFp16RmsNorm1x64CoveredElementCount = "64";
inline constexpr const char* kFp16RmsNorm1x64FullElementCount = "64";
inline constexpr const char* kFp16RmsNorm1x64Tolerance = "fp16_ulp<=1";
inline constexpr const char* kFp16RmsNorm1x64MaxAbsDiff = "0";
inline constexpr const char* kFp16RmsNorm1x64MaxUlpDiff = "0";
inline constexpr const char* kFp16RmsNorm1x64MismatchCount = "0";
inline constexpr const char* kFp16RmsNorm1x64ByteMismatchCount = "0";


inline constexpr const char* kFp16Silu8x8PrimitiveName = "fp16_silu_8x8";
inline constexpr const char* kFp16Silu8x8PrimitiveSourceId = "c1r6h-fp16-silu-8x8-v1";
inline constexpr const char* kFp16Silu8x8PrimitiveKernelSha256 =
    "1d6066da60ea89369867d3a820664833b490467903aadbdd0126bccc588555cc";
inline constexpr const char* kFp16Silu8x8PrimitiveKernelTextByteCount = "512";
inline constexpr const char* kFp16Silu8x8PrimitiveElementType = "fp16_silu_fp32";
inline constexpr const char* kFp16Silu8x8PrimitiveElementCount = "64";
inline constexpr const char* kFp16Silu8x8PrimitiveInputShape = "8x8";
inline constexpr const char* kFp16Silu8x8PrimitiveOutputShape = "8x8";
inline constexpr const char* kFp16Silu8x8PrimitiveInputLayout = "silu_x_fp16_row_major";
inline constexpr const char* kFp16Silu8x8PrimitiveInputByteCount = "128";
inline constexpr const char* kFp16Silu8x8PrimitiveOutputByteCount = "128";
inline constexpr const char* kFp16Silu8x8PrimitiveScalarBits = "0xbfb8aa3b";
inline constexpr const char* kFp16Silu8x8AcceptanceScope = "hardware_primitive_tile_only";
inline constexpr const char* kFp16Silu8x8ModelForwardScope = "silu_synthetic_8x8_math_only";
inline constexpr const char* kFp16Silu8x8NativePrefillAcceptance = "open";
inline constexpr const char* kFp16Silu8x8SourceFixture =
    "tests/native_r9700/fixtures/primitives_fixtures.npz";
inline constexpr const char* kFp16Silu8x8FixtureSha256 =
    "d52e352e2450b521a5e81dbc8672673b94a28524dbb6bf66314ba3d882924faf";
inline constexpr const char* kFp16Silu8x8SourceArrays = "silu_x_fp16,silu_expected_fp16";
inline constexpr const char* kFp16Silu8x8FixtureSlice = "rows=0:8,cols=0:8";
inline constexpr const char* kFp16Silu8x8FullFixtureShape = "8x8";
inline constexpr const char* kFp16Silu8x8CoveredElementCount = "64";
inline constexpr const char* kFp16Silu8x8FullElementCount = "64";
inline constexpr const char* kFp16Silu8x8Tolerance = "fp16_ulp<=1";
inline constexpr const char* kFp16Silu8x8MaxAbsDiff = "0";
inline constexpr const char* kFp16Silu8x8MaxUlpDiff = "0";
inline constexpr const char* kFp16Silu8x8MismatchCount = "0";
inline constexpr const char* kFp16Silu8x8ByteMismatchCount = "0";

inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveName =
    "fp16_rope_split_half_layer0_k_pairs8";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveSourceId =
    "c1r6j-fp16-rope-split-half-layer0-k-pairs8-v1";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveKernelSha256 =
    "5e0f39471f8f0beadeffc5f043c94cd15fa926b873eb764282a9fed12c1693d8";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveKernelTextByteCount =
    "512";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveElementType =
    "fp16_rope_split_half_fp32";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveElementCount = "16";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputShape =
    "left=8,right=8,cos=8,sin=8";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveOutputShape =
    "left=8,right=8";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputLayout =
    "left_pre_rope_fp16_then_right_pre_rope_fp16_then_cos_fp32_then_sin_fp32";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveInputByteCount =
    "96";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveOutputByteCount =
    "32";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8PrimitiveScalarBits =
    "unused";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8AcceptanceScope =
    "hardware_primitive_slice_only";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8ModelForwardScope =
    "layer0_k_rope_partial_pair_slice";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8NativePrefillAcceptance =
    "open";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8FixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8SourceArrays =
    "layer0_k_rope_pairs12_20_input_fp16,layer0_k_rope_pairs12_20_cos_fp32,layer0_k_rope_pairs12_20_sin_fp32,layer0_k_rope_pairs12_20_expected_fp16";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8FixtureSlice =
    "layer=0,token=1,head=0,pairs=12:20,dims=12:20|44:52";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8FullFixtureShape =
    "1x8x5x64";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8CoveredElementCount = "16";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8FullElementCount = "2560";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8Tolerance = "fp16_ulp<=1";
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8PrimitiveName =
    "fp16_rope_split_half_layer0_q_pairs8";
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8PrimitiveSourceId =
    "c1r6o-fp16-rope-split-half-layer0-q-pairs8-v1";
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8PrimitiveKernelSha256 =
    kFp16RopeSplitHalfLayer0KPairs8PrimitiveKernelSha256;
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8ModelForwardScope =
    "layer0_q_rope_partial_pair_slice";
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8SourceArrays =
    "layer0_q_rope_pairs12_20_input_fp16,layer0_q_rope_pairs12_20_cos_fp32,layer0_q_rope_pairs12_20_sin_fp32,layer0_q_rope_pairs12_20_expected_fp16";
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8FullFixtureShape =
    "1x32x5x64";
inline constexpr const char* kFp16RopeSplitHalfLayer0QPairs8FullElementCount = "10240";

inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainName =
    "layer0_k_rope_token1_head0_full_head_chain";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainModelForwardScope =
    "layer0_k_rope_token1_head0_full_head";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainSourceArrays =
    "layer0_k_rope_token1_head0_full_head_input_fp16,layer0_k_rope_token1_head0_full_head_cos_fp32,layer0_k_rope_token1_head0_full_head_sin_fp32,layer0_k_rope_token1_head0_full_head_expected_fp16";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainFixtureSlice =
    "layer=0,token=1,head=0,pairs=0:32,dims=0:32|32:64";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainStageCount = "4";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainReadbackBetweenStages = "no";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainDataRegionCount = "2";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainInputRegion =
    "layer0_k_rope_token1_head0_full_head_input_chunks";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputRegion =
    "layer0_k_rope_token1_head0_full_head_output";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainInputRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainInputRegionPtbIndex = "1";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputRegionPtbIndex = "17";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainSupplementalPteCount = "1";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputRegionPteStatus = "pass";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainKernargRewriteCount = "4";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainComputeDispatchCount = "4";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainPairChunkCount = "4";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainPairChunkSize = "8";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainPairCount = "32";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainInputByteCount = "384";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputByteCount = "128";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainUploadTotalBytes = "384";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainDownloadTotalBytes = "128";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputDtype = "fp16";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainOutputShape = "4x2x8";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainFullFixtureShape = "1x8x5x64";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainCoveredElementCount = "64";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainFullElementCount = "2560";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainInputLayout =
    "left8_right8_cos8_sin8_chunks";
inline constexpr const char* kLayer0KRopeToken1Head0FullHeadChainExpectedChunkedFp16Sha256 =
    "057a20f9462451dd1009e94b441f9e18a1b23154113067f8c5a2aa60e668ea75";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainName =
    "layer0_k_rope_tokens0_5_head0_full_head_chain";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainModelForwardScope =
    "layer0_k_rope_tokens0_5_head0_full_head";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainSourceArrays =
    "layer0_k_rope_tokens0_5_head0_full_head_input_fp16,layer0_k_rope_tokens0_5_head0_full_head_cos_fp32,layer0_k_rope_tokens0_5_head0_full_head_sin_fp32,layer0_k_rope_tokens0_5_head0_full_head_expected_fp16";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,pairs=0:32,dims=0:32|32:64";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainStageCount = "20";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainReadbackBetweenStages = "no";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainDataRegionCount = "2";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainInputRegion =
    "layer0_k_rope_tokens0_5_head0_full_head_input_chunks";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputRegion =
    "layer0_k_rope_tokens0_5_head0_full_head_output";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainInputRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainInputRegionPtbIndex = "1";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputRegionPtbIndex = "17";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainSupplementalPteCount = "1";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputRegionPteStatus = "pass";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainKernargRewriteCount = "20";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainComputeDispatchCount = "20";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainPairChunkCount = "4";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainPairChunkSize = "8";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainTokenCount = "5";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainPairCount = "32";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainInputByteCount = "1920";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputByteCount = "640";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainUploadTotalBytes = "1920";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainDownloadTotalBytes = "640";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputDtype = "fp16";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainOutputShape = "5x4x2x8";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainFullFixtureShape = "1x8x5x64";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainCoveredElementCount = "320";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainFullElementCount = "2560";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainInputLayout =
    "token_major_left8_right8_cos8_sin8_chunks";
inline constexpr const char* kLayer0KRopeTokens05Head0FullHeadChainExpectedChunkedFp16Sha256 =
    "b9fc5432069f94804b047e6015226995940b2281cd03ff5897bc43e9a7a28717";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainName =
    "layer0_q_rope_token1_head0_full_head_chain";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainModelForwardScope =
    "layer0_q_rope_token1_head0_full_head";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainSourceArrays =
    "layer0_q_rope_token1_head0_full_head_input_fp16,layer0_q_rope_token1_head0_full_head_cos_fp32,layer0_q_rope_token1_head0_full_head_sin_fp32,layer0_q_rope_token1_head0_full_head_expected_fp16";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainFixtureSlice =
    "layer=0,token=1,head=0,pairs=0:32,dims=0:32|32:64";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainStageCount = "4";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainReadbackBetweenStages = "no";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainDataRegionCount = "2";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainInputRegion =
    "layer0_q_rope_token1_head0_full_head_input_chunks";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputRegion =
    "layer0_q_rope_token1_head0_full_head_output";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainInputRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainInputRegionPtbIndex = "1";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputRegionPtbIndex = "17";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainSupplementalPteCount = "1";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputRegionPteStatus = "pass";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainKernargRewriteCount = "4";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainComputeDispatchCount = "4";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainPairChunkCount = "4";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainPairChunkSize = "8";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainPairCount = "32";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainInputByteCount = "384";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputByteCount = "128";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainUploadTotalBytes = "384";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainDownloadTotalBytes = "128";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputDtype = "fp16";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainOutputShape = "4x2x8";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainFullFixtureShape = "1x32x5x64";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainCoveredElementCount = "64";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainFullElementCount = "10240";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainInputLayout =
    "left8_right8_cos8_sin8_chunks";
inline constexpr const char* kLayer0QRopeToken1Head0FullHeadChainExpectedChunkedFp16Sha256 =
    "3e4df13399c58a98201dc37dfe61dd86155a6480efa823b0e0de9ae3440fa1d0";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainName =
    "layer0_q_rope_tokens0_5_head0_full_head_chain";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainModelForwardScope =
    "layer0_q_rope_tokens0_5_head0_full_head";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainSourceArrays =
    "layer0_q_rope_tokens0_5_head0_full_head_input_fp16,layer0_q_rope_tokens0_5_head0_full_head_cos_fp32,layer0_q_rope_tokens0_5_head0_full_head_sin_fp32,layer0_q_rope_tokens0_5_head0_full_head_expected_fp16";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,pairs=0:32,dims=0:32|32:64";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainStageCount = "20";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainReadbackBetweenStages = "no";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainDataRegionCount = "2";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainInputRegion =
    "layer0_q_rope_tokens0_5_head0_full_head_input_chunks";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputRegion =
    "layer0_q_rope_tokens0_5_head0_full_head_output";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainInputRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainInputRegionPtbIndex = "1";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputRegionPtbIndex = "17";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainSupplementalPteCount = "1";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputRegionPteStatus = "pass";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainKernargRewriteCount = "20";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainComputeDispatchCount = "20";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainPairChunkCount = "4";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainPairChunkSize = "8";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainTokenCount = "5";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainPairCount = "32";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainInputByteCount = "1920";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputByteCount = "640";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainUploadTotalBytes = "1920";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainDownloadTotalBytes = "640";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputDtype = "fp16";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainOutputShape = "5x4x2x8";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainFullFixtureShape = "1x32x5x64";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainCoveredElementCount = "320";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainFullElementCount = "10240";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainInputLayout =
    "token_major_left8_right8_cos8_sin8_chunks";
inline constexpr const char* kLayer0QRopeTokens05Head0FullHeadChainExpectedChunkedFp16Sha256 =
    "b8d6efd4a75a399602ab4ca6e69d3192c066e24fceb153852699064a513436d8";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainName =
    "layer0_attention_score_raw_head0_tokens0_5_chain";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainModelForwardScope =
    "layer0_attention_score_raw_head0_tokens0_5";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainSourceArrays =
    "layer0_attention_score_raw_head0_tokens0_5_q_fp16,layer0_attention_score_raw_head0_tokens0_5_k_as_b_fp16,layer0_attention_score_raw_head0_tokens0_5_expected_fp32";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainStageCount = "4";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainDataRegionCount = "2";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainActivationRegion =
    "layer0_attention_score_raw_head0_tokens0_5_q_chunks";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainModelWeightRegion =
    "layer0_attention_score_raw_head0_tokens0_5_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputRegion =
    "layer0_attention_score_raw_head0_tokens0_5_fp32_output";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainResidentDataPageCount =
    "2";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainActivationRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainModelWeightRegionGpuVa =
    "0x0000200000019000";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputRegionGpuVa =
    "0x0000200000021000";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputRegionPtbIndex =
    "33";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainActivationRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainModelWeightRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainSupplementalPteCount =
    "2";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainKernargRewriteCount =
    "4";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainComputeDispatchCount =
    "4";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainInnerChunkCount = "4";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainInnerChunkSize = "16";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainScoreTokenCount = "5";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainTileRows = "8";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainTileCols = "8";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainTileInner = "64";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainActivationByteCount =
    "1024";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainModelWeightByteCount =
    "1024";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputByteCount = "256";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainUploadTotalBytes =
    "2048";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainDownloadTotalBytes =
    "256";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputDtype = "fp32";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainOutputShape = "8x8";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainValidScoreShape = "5x5";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainScaleStatus =
    "not_applied_raw_qk";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainCausalMaskStatus =
    "not_applied_raw_qk";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainInputLayout =
    "q_chunks8x16_row_major_then_k_as_b_chunks16x8_dot2_pair_packed";
inline constexpr const char* kLayer0AttentionScoreRawHead0Tokens05ChainExpectedFp32Sha256 =
    "befe0cbbe3f577cf2cfe4ea257069b88bbc06f47ad14a312aceeae18b2a7cb0d";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainName =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_chain";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainModelForwardScope =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainSourceArrays =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head0_tokens0_5_scaled_masked_seed_fp32,layer0_attention_scores_head0_tokens0_5_scaled_masked_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainStageCount = "4";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainDataRegionCount =
    "2";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainActivationRegion =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainModelWeightRegion =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputRegion =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainResidentDataPageCount =
    "2";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainActivationRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainModelWeightRegionGpuVa =
    "0x0000200000019000";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputRegionGpuVa =
    "0x0000200000021000";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputRegionPtbIndex =
    "33";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainActivationRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainModelWeightRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainSupplementalPteCount =
    "2";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainKernargRewriteCount =
    "4";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainComputeDispatchCount =
    "4";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainInnerChunkCount =
    "4";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainInnerChunkSize =
    "16";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainScoreTokenCount =
    "5";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainTileRows = "8";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainTileCols = "8";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainTileInner =
    "64";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainActivationByteCount =
    "1024";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainModelWeightByteCount =
    "1024";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainMaskSeedByteCount =
    "256";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputByteCount =
    "256";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainUploadTotalBytes =
    "2304";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainDownloadTotalBytes =
    "256";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputDtype =
    "fp32";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainOutputShape =
    "8x8";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainValidScoreShape =
    "5x5";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainFiniteCausalScoreCount =
    "15";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainScaleStatus =
    "applied_by_scaled_q_operand_fp16_power_of_two_0p125";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainCausalMaskStatus =
    "applied_by_fp32_output_seed_before_accum";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainPaddingMaskStatus =
    "applied_by_fp32_output_seed_before_accum";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainFinalOutputSeedStatus =
    "pass";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainInputLayout =
    "q_scaled_chunks8x16_row_major_then_k_as_b_chunks16x8_dot2_pair_packed_with_fp32_output_seed";
inline constexpr const char* kLayer0AttentionScoresHead0Tokens05ScaledMaskedChainExpectedFp32Sha256 =
    "b6a2aba120caf6d4755c283e037c772799409e1b0454189988101029eb4bd34b";

inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainName =
    "layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked_chain";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainModelForwardScope =
    "layer0_attention_probs_head0_tokens0_5_softmax_from_scaled_masked";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainNativePrefillAcceptance = "open";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainSourceArrays =
    "layer0_attention_probs_head0_tokens0_5_softmax_input_fp32,layer0_attention_probs_head0_tokens0_5_softmax_expected_fp32";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,query_rows=0:5,padded_query_rows=5:8,key_cols=0:5,padded_key_cols=5:8";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainStageCount = "1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainDataRegionCount = "2";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainScoreRegion =
    "layer0_attention_probs_head0_tokens0_5_softmax_scores_fp32";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputRegion =
    "layer0_attention_probs_head0_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainScoreRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputRegionGpuVa =
    "0x0000200000004000";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainScoreRegionPtbIndex = "1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputRegionPtbIndex = "4";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainScoreRegionPageCount = "1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputRegionPageCount = "1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainSupplementalPteCount = "2";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxKernelSourceId =
    "c1r6x-layer0-attention-softmax-head0-tokens0-5-v1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxKernelSha256 =
    "5b771f363c8de0d7081c2777d2544839ae8a51763981e4eef9655f4b7f52f436";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainKernargRewriteCount = "1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainComputeDispatchCount = "1";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainScoreTokenCount = "5";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainTileRows = "8";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainTileCols = "8";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainFiniteCausalScoreCount = "15";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainMaskedScoreStatus =
    "consumed_from_scaled_masked_fixture";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputByteCount = "256";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainUploadTotalBytes = "256";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainDownloadTotalBytes = "256";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputDtype = "fp32";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainOutputShape = "8x8";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainValidProbabilityShape = "5x5";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainExpectedFp32Sha256 =
    "8ccf332cee4ac5f2748cfb4d67170453d4cd1de3ae97874de153cd3026008665";
inline constexpr const char* kLayer0AttentionProbsHead0Tokens05SoftmaxChainInputLayout =
    "scores8x8_fp32_row_major";

inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainName =
    "layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum_chain";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainModelForwardScope =
    "layer0_attention_context_head0_tokens0_5_cols0_64_weighted_sum";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainSourceArrays =
    "layer0_attention_context_head0_tokens0_5_cols0_64_probs_fp16,layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b_fp16,layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp32";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,prob_cols=0:5,padded_prob_cols=5:16,context_cols=0:64";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainStageCount =
    "8";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainDataRegionCount =
    "2";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainActivationRegion =
    "layer0_attention_context_head0_tokens0_5_cols0_64_probs";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainModelWeightRegion =
    "layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputRegion =
    "layer0_attention_context_head0_tokens0_5_cols0_64_fp32_output";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainResidentDataPageCount =
    "2";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainActivationRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainModelWeightRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputRegionGpuVa =
    "0x0000200000012000";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainActivationRegionPtbIndex =
    "1";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainModelWeightRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputRegionPtbIndex =
    "18";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainActivationRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainModelWeightRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainSupplementalPteCount =
    "2";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainKernargRewriteCount =
    "8";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainComputeDispatchCount =
    "8";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainProbTokenCount =
    "5";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainQueryTokenCount =
    "5";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputTileRows =
    "8";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputTileCols =
    "64";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainTileInner =
    "16";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainActivationByteCount =
    "256";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainModelWeightByteCount =
    "2048";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainUploadTotalBytes =
    "2304";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputDtype =
    "fp32";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainValidContextShape =
    "5x64";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainFullElementCount =
    "512";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputTile0Cols =
    "0:8";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputTile1Cols =
    "8:16";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainOutputTile7Cols =
    "56:64";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainInputLayout =
    "probs8x16_row_major_then_v_as_b16x8_dot2_pair_packed";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainProbsSource =
    "fixture_attention_probs_fp32_cast_to_fp16";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainSoftmaxStatus =
    "not_implemented_fixture_probs";
inline constexpr const char* kLayer0AttentionContextHead0Tokens05Cols064WeightedSumChainExpectedFp32Sha256 =
    "1aea10ec6d6a1c5da8f9e76b2d4eb6af0aa38802da4667fe1d6e5c177bb37b92";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainName =
    "layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSourceArrays =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head0_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b_fp16,layer0_attention_context_head0_tokens0_5_cols0_64_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainFixtureSlice =
    "layer=0,tokens=0:5,head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=0:64";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount =
    "21";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount =
    "7";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegion =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegion =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegion =
    "layer0_attention_scores_head0_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32Region =
    "layer0_attention_probs_head0_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16Region =
    "layer0_attention_probs_head0_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegion =
    "layer0_attention_context_head0_tokens0_5_cols0_64_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head0_tokens0_5_cols0_64_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount =
    "7";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency =
    "seven_distinct_vram_pages";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa =
    "0x0000200000019000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa =
    "0x0000200000021000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa =
    "0x0000200000022000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa =
    "0x0000200000023000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa =
    "0x0000200000024000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa =
    "0x0000200000025000";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex = "17";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex = "25";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex = "33";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex = "34";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex = "35";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex = "36";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex = "37";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount = "1";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount = "7";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount = "21";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount = "21";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount = "4";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount = "1";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount = "8";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount = "8";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes = "4352";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes = "2048";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount = "256";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount = "256";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount = "2048";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount = "2048";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner = "64";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner = "16";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout =
    "q_scaled_chunks8x16_row_major_then_k_as_b_chunks16x8_dot2_pair_packed_with_fp32_output_seed_then_native_softmax_fp32_then_fp32_to_fp16_probs8x16_then_v_as_b16x8_dot2_pair_packed";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource =
    "native_softmax_fp32_cast_to_fp16";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus =
    "pass";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus =
    "produced_by_scaled_masked_stage";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainName =
    "layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainSourceArrays =
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head1_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head1_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head1_tokens0_5_cols64_128_v_as_b_fp16,layer0_attention_context_head1_tokens0_5_cols64_128_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainFixtureSlice =
    "layer=0,tokens=0:5,head=1,kv_head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=64:128";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainQRegion =
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainKRegion =
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainScoreRegion =
    "layer0_attention_scores_head1_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp32Region =
    "layer0_attention_probs_head1_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp16Region =
    "layer0_attention_probs_head1_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainModelWeightRegion =
    "layer0_attention_context_head1_tokens0_5_cols64_128_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head1_tokens0_5_cols64_128_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead1Tokens05Cols64128ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead1Tokens05Cols64128WeightedSumChainExpectedFp32Sha256 =
    "f5ff8853de675347683280bdfaf5ee4e94f7ad5a0d3c3967a19b67365011fdbe";





inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainName =
    "layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainSourceArrays =
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head2_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head2_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head2_tokens0_5_cols128_192_v_as_b_fp16,layer0_attention_context_head2_tokens0_5_cols128_192_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainFixtureSlice =
    "layer=0,tokens=0:5,head=2,kv_head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=128:192";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainQRegion =
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainKRegion =
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainScoreRegion =
    "layer0_attention_scores_head2_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp32Region =
    "layer0_attention_probs_head2_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp16Region =
    "layer0_attention_probs_head2_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainModelWeightRegion =
    "layer0_attention_context_head2_tokens0_5_cols128_192_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head2_tokens0_5_cols128_192_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead2Tokens05Cols128192ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead2Tokens05Cols128192WeightedSumChainExpectedFp32Sha256 =
    "fad72fe32aa7bab8faa0c812b43a565acdea84b117d4810dcbdaac268fc1e6da";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainName =
    "layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainSourceArrays =
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head3_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head3_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head3_tokens0_5_cols192_256_v_as_b_fp16,layer0_attention_context_head3_tokens0_5_cols192_256_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainFixtureSlice =
    "layer=0,tokens=0:5,head=3,kv_head=0,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=192:256";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainQRegion =
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainKRegion =
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainScoreRegion =
    "layer0_attention_scores_head3_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp32Region =
    "layer0_attention_probs_head3_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp16Region =
    "layer0_attention_probs_head3_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainModelWeightRegion =
    "layer0_attention_context_head3_tokens0_5_cols192_256_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head3_tokens0_5_cols192_256_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead3Tokens05Cols192256ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead3Tokens05Cols192256WeightedSumChainExpectedFp32Sha256 =
    "1ba39659595e6c57cf82c0b2624893d2a95278f9062679d350ddc165c29d8604";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainName =
    "layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainSourceArrays =
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head4_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head4_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head4_tokens0_5_cols256_320_v_as_b_fp16,layer0_attention_context_head4_tokens0_5_cols256_320_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainFixtureSlice =
    "layer=0,tokens=0:5,head=4,kv_head=1,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=256:320";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainQRegion =
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainKRegion =
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainScoreRegion =
    "layer0_attention_scores_head4_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp32Region =
    "layer0_attention_probs_head4_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp16Region =
    "layer0_attention_probs_head4_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainModelWeightRegion =
    "layer0_attention_context_head4_tokens0_5_cols256_320_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head4_tokens0_5_cols256_320_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead4Tokens05Cols256320ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead4Tokens05Cols256320WeightedSumChainExpectedFp32Sha256 =
    "987fbcc6024d557bfb41d2d9f304decae0f7ef4f02ced9432eba79471bdf7556";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainName =
    "layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainSourceArrays =
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head5_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head5_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head5_tokens0_5_cols320_384_v_as_b_fp16,layer0_attention_context_head5_tokens0_5_cols320_384_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainFixtureSlice =
    "layer=0,tokens=0:5,head=5,kv_head=1,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=320:384";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainQRegion =
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainKRegion =
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainScoreRegion =
    "layer0_attention_scores_head5_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp32Region =
    "layer0_attention_probs_head5_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp16Region =
    "layer0_attention_probs_head5_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainModelWeightRegion =
    "layer0_attention_context_head5_tokens0_5_cols320_384_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head5_tokens0_5_cols320_384_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead5Tokens05Cols320384ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead5Tokens05Cols320384WeightedSumChainExpectedFp32Sha256 =
    "0765d235cff27d0f9ee6e31e4a85a4bfefefdcb7f3ec089ace41d8301c138152";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainName =
    "layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainSourceArrays =
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head6_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head6_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head6_tokens0_5_cols384_448_v_as_b_fp16,layer0_attention_context_head6_tokens0_5_cols384_448_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainFixtureSlice =
    "layer=0,tokens=0:5,head=6,kv_head=1,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=384:448";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainQRegion =
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainKRegion =
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainScoreRegion =
    "layer0_attention_scores_head6_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp32Region =
    "layer0_attention_probs_head6_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp16Region =
    "layer0_attention_probs_head6_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainModelWeightRegion =
    "layer0_attention_context_head6_tokens0_5_cols384_448_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head6_tokens0_5_cols384_448_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead6Tokens05Cols384448ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead6Tokens05Cols384448WeightedSumChainExpectedFp32Sha256 =
    "acf5766dadc4b3e28d41b37f1e13c2e33673652dce4a61b0fad44f709535d27f";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainName =
    "layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainSourceArrays =
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head7_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head7_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head7_tokens0_5_cols448_512_v_as_b_fp16,layer0_attention_context_head7_tokens0_5_cols448_512_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainFixtureSlice =
    "layer=0,tokens=0:5,head=7,kv_head=1,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=448:512";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainQRegion =
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainKRegion =
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainScoreRegion =
    "layer0_attention_scores_head7_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp32Region =
    "layer0_attention_probs_head7_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp16Region =
    "layer0_attention_probs_head7_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainModelWeightRegion =
    "layer0_attention_context_head7_tokens0_5_cols448_512_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head7_tokens0_5_cols448_512_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead7Tokens05Cols448512ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead7Tokens05Cols448512WeightedSumChainExpectedFp32Sha256 =
    "cbf9385a70f389b62e8f3089f82674899c5f16adb01074130fc59edb3ffa45f8";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainName =
    "layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainSourceArrays =
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head8_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head8_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head8_tokens0_5_cols512_576_v_as_b_fp16,layer0_attention_context_head8_tokens0_5_cols512_576_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainFixtureSlice =
    "layer=0,tokens=0:5,head=8,kv_head=2,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=512:576";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainQRegion =
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainKRegion =
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainScoreRegion =
    "layer0_attention_scores_head8_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp32Region =
    "layer0_attention_probs_head8_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp16Region =
    "layer0_attention_probs_head8_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainModelWeightRegion =
    "layer0_attention_context_head8_tokens0_5_cols512_576_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head8_tokens0_5_cols512_576_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead8Tokens05Cols512576ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead8Tokens05Cols512576WeightedSumChainExpectedFp32Sha256 =
    "8ccd833ff75ea72dc3a2b5dd2c246523f07e3ea0a16217f6522f00a235af3629";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainName =
    "layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainSourceArrays =
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head9_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head9_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head9_tokens0_5_cols576_640_v_as_b_fp16,layer0_attention_context_head9_tokens0_5_cols576_640_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainFixtureSlice =
    "layer=0,tokens=0:5,head=9,kv_head=2,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=576:640";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainQRegion =
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainKRegion =
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainScoreRegion =
    "layer0_attention_scores_head9_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp32Region =
    "layer0_attention_probs_head9_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp16Region =
    "layer0_attention_probs_head9_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainModelWeightRegion =
    "layer0_attention_context_head9_tokens0_5_cols576_640_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head9_tokens0_5_cols576_640_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead9Tokens05Cols576640ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead9Tokens05Cols576640WeightedSumChainExpectedFp32Sha256 =
    "9106cd654e2de4ae68b962c755c22d80ba7cba867c81aac4a3ef5f4476e44fab";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainName =
    "layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainSourceArrays =
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head10_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head10_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head10_tokens0_5_cols640_704_v_as_b_fp16,layer0_attention_context_head10_tokens0_5_cols640_704_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainFixtureSlice =
    "layer=0,tokens=0:5,head=10,kv_head=2,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=640:704";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainQRegion =
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainKRegion =
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainScoreRegion =
    "layer0_attention_scores_head10_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp32Region =
    "layer0_attention_probs_head10_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp16Region =
    "layer0_attention_probs_head10_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainModelWeightRegion =
    "layer0_attention_context_head10_tokens0_5_cols640_704_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head10_tokens0_5_cols640_704_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead10Tokens05Cols640704ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead10Tokens05Cols640704WeightedSumChainExpectedFp32Sha256 =
    "d31ee73e25bd620a9bb20cc7b8e1435eb96346fe0632badb227715eb36c2e09e";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainName =
    "layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainSourceArrays =
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head11_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head11_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head11_tokens0_5_cols704_768_v_as_b_fp16,layer0_attention_context_head11_tokens0_5_cols704_768_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainFixtureSlice =
    "layer=0,tokens=0:5,head=11,kv_head=2,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=704:768";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainQRegion =
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainKRegion =
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainScoreRegion =
    "layer0_attention_scores_head11_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp32Region =
    "layer0_attention_probs_head11_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp16Region =
    "layer0_attention_probs_head11_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainModelWeightRegion =
    "layer0_attention_context_head11_tokens0_5_cols704_768_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head11_tokens0_5_cols704_768_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead11Tokens05Cols704768ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead11Tokens05Cols704768WeightedSumChainExpectedFp32Sha256 =
    "92beebcdea2b3b85d49c6d7bf7dfa4a750177a9f034f9bb33bb1c2c546c6f415";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainName =
    "layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainSourceArrays =
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head12_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head12_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head12_tokens0_5_cols768_832_v_as_b_fp16,layer0_attention_context_head12_tokens0_5_cols768_832_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainFixtureSlice =
    "layer=0,tokens=0:5,head=12,kv_head=3,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=768:832";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainQRegion =
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainKRegion =
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainScoreRegion =
    "layer0_attention_scores_head12_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp32Region =
    "layer0_attention_probs_head12_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp16Region =
    "layer0_attention_probs_head12_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainModelWeightRegion =
    "layer0_attention_context_head12_tokens0_5_cols768_832_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head12_tokens0_5_cols768_832_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead12Tokens05Cols768832ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead12Tokens05Cols768832WeightedSumChainExpectedFp32Sha256 =
    "fa1461505142860d118fc6b14180267f72f25463db76c15272162a258e597b15";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainName =
    "layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainSourceArrays =
    "layer0_attention_scores_head13_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head13_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head13_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head13_tokens0_5_cols832_896_v_as_b_fp16,layer0_attention_context_head13_tokens0_5_cols832_896_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainFixtureSlice =
    "layer=0,tokens=0:5,head=13,kv_head=3,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=832:896";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainQRegion =
    "layer0_attention_scores_head13_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainKRegion =
    "layer0_attention_scores_head13_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainScoreRegion =
    "layer0_attention_scores_head13_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp32Region =
    "layer0_attention_probs_head13_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp16Region =
    "layer0_attention_probs_head13_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainModelWeightRegion =
    "layer0_attention_context_head13_tokens0_5_cols832_896_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head13_tokens0_5_cols832_896_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead13Tokens05Cols832896ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead13Tokens05Cols832896WeightedSumChainExpectedFp32Sha256 =
    "e5085ffdf75aa048dec60d74efa637667159694077e803480463d1e11a627696";



inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainName =
    "layer0_attention_scores_softmax_context_head14_tokens0_5_cols896_960_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head14_tokens0_5_cols896_960";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainSourceArrays =
    "layer0_attention_scores_head14_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head14_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head14_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head14_tokens0_5_cols896_960_v_as_b_fp16,layer0_attention_context_head14_tokens0_5_cols896_960_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainFixtureSlice =
    "layer=0,tokens=0:5,head=14,kv_head=3,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=896:960";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainQRegion =
    "layer0_attention_scores_head14_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainKRegion =
    "layer0_attention_scores_head14_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainScoreRegion =
    "layer0_attention_scores_head14_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp32Region =
    "layer0_attention_probs_head14_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp16Region =
    "layer0_attention_probs_head14_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainModelWeightRegion =
    "layer0_attention_context_head14_tokens0_5_cols896_960_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head14_tokens0_5_cols896_960_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead14Tokens05Cols896960ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead14Tokens05Cols896960WeightedSumChainExpectedFp32Sha256 =
    "ffafcdef4ad0597794f33559efda6c0c59d2af696038a413299e478b56886495";



inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainName =
    "layer0_attention_scores_softmax_context_head15_tokens0_5_cols960_1024_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head15_tokens0_5_cols960_1024";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainSourceArrays =
    "layer0_attention_scores_head15_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head15_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head15_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head15_tokens0_5_cols960_1024_v_as_b_fp16,layer0_attention_context_head15_tokens0_5_cols960_1024_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainFixtureSlice =
    "layer=0,tokens=0:5,head=15,kv_head=3,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=960:1024";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainQRegion =
    "layer0_attention_scores_head15_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainKRegion =
    "layer0_attention_scores_head15_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainScoreRegion =
    "layer0_attention_scores_head15_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp32Region =
    "layer0_attention_probs_head15_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp16Region =
    "layer0_attention_probs_head15_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainModelWeightRegion =
    "layer0_attention_context_head15_tokens0_5_cols960_1024_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head15_tokens0_5_cols960_1024_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead15Tokens05Cols9601024ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead15Tokens05Cols9601024WeightedSumChainExpectedFp32Sha256 =
    "3a6f7ac107763b79960ebc4fb16cf237c71da880d3fa20c2d9e2b9a46c4d48cd";



inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainName =
    "layer0_attention_scores_softmax_context_head16_tokens0_5_cols1024_1088_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head16_tokens0_5_cols1024_1088";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainSourceArrays =
    "layer0_attention_scores_head16_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head16_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head16_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head16_tokens0_5_cols1024_1088_v_as_b_fp16,layer0_attention_context_head16_tokens0_5_cols1024_1088_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainFixtureSlice =
    "layer=0,tokens=0:5,head=16,kv_head=4,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1024:1088";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainQRegion =
    "layer0_attention_scores_head16_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainKRegion =
    "layer0_attention_scores_head16_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainScoreRegion =
    "layer0_attention_scores_head16_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp32Region =
    "layer0_attention_probs_head16_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp16Region =
    "layer0_attention_probs_head16_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainModelWeightRegion =
    "layer0_attention_context_head16_tokens0_5_cols1024_1088_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head16_tokens0_5_cols1024_1088_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead16Tokens05Cols10241088ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead16Tokens05Cols10241088WeightedSumChainExpectedFp32Sha256 =
    "93c54b7d855414ed559cb3b3368e76f82021c590889fabb961efec7652b99055";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainName =
    "layer0_attention_scores_softmax_context_head21_tokens0_5_cols1344_1408_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head21_tokens0_5_cols1344_1408";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainSourceArrays =
    "layer0_attention_scores_head21_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head21_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head21_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head21_tokens0_5_cols1344_1408_v_as_b_fp16,layer0_attention_context_head21_tokens0_5_cols1344_1408_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainFixtureSlice =
    "layer=0,tokens=0:5,head=21,kv_head=5,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1344:1408";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainQRegion =
    "layer0_attention_scores_head21_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainKRegion =
    "layer0_attention_scores_head21_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainScoreRegion =
    "layer0_attention_scores_head21_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp32Region =
    "layer0_attention_probs_head21_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp16Region =
    "layer0_attention_probs_head21_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainModelWeightRegion =
    "layer0_attention_context_head21_tokens0_5_cols1344_1408_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head21_tokens0_5_cols1344_1408_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead21Tokens05Cols13441408ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead21Tokens05Cols13441408WeightedSumChainExpectedFp32Sha256 =
    "4cb30ac3bfaa558cabaf4200af55f646039408ce3c8c2d97d8379fa7d8d0eebf";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainName =
    "layer0_attention_scores_softmax_context_head22_tokens0_5_cols1408_1472_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head22_tokens0_5_cols1408_1472";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainSourceArrays =
    "layer0_attention_scores_head22_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head22_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head22_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head22_tokens0_5_cols1408_1472_v_as_b_fp16,layer0_attention_context_head22_tokens0_5_cols1408_1472_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainFixtureSlice =
    "layer=0,tokens=0:5,head=22,kv_head=5,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1408:1472";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainQRegion =
    "layer0_attention_scores_head22_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainKRegion =
    "layer0_attention_scores_head22_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainScoreRegion =
    "layer0_attention_scores_head22_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp32Region =
    "layer0_attention_probs_head22_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp16Region =
    "layer0_attention_probs_head22_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainModelWeightRegion =
    "layer0_attention_context_head22_tokens0_5_cols1408_1472_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head22_tokens0_5_cols1408_1472_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead22Tokens05Cols14081472ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead22Tokens05Cols14081472WeightedSumChainExpectedFp32Sha256 =
    "fc594a856cc9a46f4246bcdf4104070e164fb28b2d39670faaf9ea3230f31229";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainName =
    "layer0_attention_scores_softmax_context_head23_tokens0_5_cols1472_1536_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head23_tokens0_5_cols1472_1536";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainSourceArrays =
    "layer0_attention_scores_head23_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head23_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head23_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head23_tokens0_5_cols1472_1536_v_as_b_fp16,layer0_attention_context_head23_tokens0_5_cols1472_1536_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainFixtureSlice =
    "layer=0,tokens=0:5,head=23,kv_head=5,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1472:1536";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainQRegion =
    "layer0_attention_scores_head23_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainKRegion =
    "layer0_attention_scores_head23_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainScoreRegion =
    "layer0_attention_scores_head23_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp32Region =
    "layer0_attention_probs_head23_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp16Region =
    "layer0_attention_probs_head23_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainModelWeightRegion =
    "layer0_attention_context_head23_tokens0_5_cols1472_1536_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head23_tokens0_5_cols1472_1536_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead23Tokens05Cols14721536ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead23Tokens05Cols14721536WeightedSumChainExpectedFp32Sha256 =
    "a7317d6e8a908c6ce037f49258d5ba457aa55464f59bbccf4bae2f0e672e1b39";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainName =
    "layer0_attention_scores_softmax_context_head24_tokens0_5_cols1536_1600_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head24_tokens0_5_cols1536_1600";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainSourceArrays =
    "layer0_attention_scores_head24_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head24_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head24_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head24_tokens0_5_cols1536_1600_v_as_b_fp16,layer0_attention_context_head24_tokens0_5_cols1536_1600_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainFixtureSlice =
    "layer=0,tokens=0:5,head=24,kv_head=6,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1536:1600";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainQRegion =
    "layer0_attention_scores_head24_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainKRegion =
    "layer0_attention_scores_head24_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainScoreRegion =
    "layer0_attention_scores_head24_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp32Region =
    "layer0_attention_probs_head24_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp16Region =
    "layer0_attention_probs_head24_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainModelWeightRegion =
    "layer0_attention_context_head24_tokens0_5_cols1536_1600_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head24_tokens0_5_cols1536_1600_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead24Tokens05Cols15361600ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead24Tokens05Cols15361600WeightedSumChainExpectedFp32Sha256 =
    "c20f9002dda2b421d0625622561822e94b3932ff4a9750f77b5e62ec60302bf3";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainName =
    "layer0_attention_scores_softmax_context_head25_tokens0_5_cols1600_1664_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head25_tokens0_5_cols1600_1664";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainSourceArrays =
    "layer0_attention_scores_head25_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head25_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head25_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head25_tokens0_5_cols1600_1664_v_as_b_fp16,layer0_attention_context_head25_tokens0_5_cols1600_1664_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainFixtureSlice =
    "layer=0,tokens=0:5,head=25,kv_head=6,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1600:1664";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainQRegion =
    "layer0_attention_scores_head25_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainKRegion =
    "layer0_attention_scores_head25_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainScoreRegion =
    "layer0_attention_scores_head25_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp32Region =
    "layer0_attention_probs_head25_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp16Region =
    "layer0_attention_probs_head25_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainModelWeightRegion =
    "layer0_attention_context_head25_tokens0_5_cols1600_1664_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head25_tokens0_5_cols1600_1664_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead25Tokens05Cols16001664ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead25Tokens05Cols16001664WeightedSumChainExpectedFp32Sha256 =
    "819511dbc030873e273171e8e58d6f230ddf5306555a245ab2743c3521c83134";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainName =
    "layer0_attention_scores_softmax_context_head26_tokens0_5_cols1664_1728_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head26_tokens0_5_cols1664_1728";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainSourceArrays =
    "layer0_attention_scores_head26_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head26_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head26_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head26_tokens0_5_cols1664_1728_v_as_b_fp16,layer0_attention_context_head26_tokens0_5_cols1664_1728_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainFixtureSlice =
    "layer=0,tokens=0:5,head=26,kv_head=6,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1664:1728";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainQRegion =
    "layer0_attention_scores_head26_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainKRegion =
    "layer0_attention_scores_head26_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainScoreRegion =
    "layer0_attention_scores_head26_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp32Region =
    "layer0_attention_probs_head26_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp16Region =
    "layer0_attention_probs_head26_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainModelWeightRegion =
    "layer0_attention_context_head26_tokens0_5_cols1664_1728_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head26_tokens0_5_cols1664_1728_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead26Tokens05Cols16641728ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead26Tokens05Cols16641728WeightedSumChainExpectedFp32Sha256 =
    "698285791924c05b280626d0d5e81b0b850f1fa3e8f8df883766fb6b4166c402";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainName =
    "layer0_attention_scores_softmax_context_head27_tokens0_5_cols1728_1792_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head27_tokens0_5_cols1728_1792";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainSourceArrays =
    "layer0_attention_scores_head27_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head27_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head27_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head27_tokens0_5_cols1728_1792_v_as_b_fp16,layer0_attention_context_head27_tokens0_5_cols1728_1792_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainFixtureSlice =
    "layer=0,tokens=0:5,head=27,kv_head=6,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1728:1792";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainQRegion =
    "layer0_attention_scores_head27_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainKRegion =
    "layer0_attention_scores_head27_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainScoreRegion =
    "layer0_attention_scores_head27_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp32Region =
    "layer0_attention_probs_head27_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp16Region =
    "layer0_attention_probs_head27_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainModelWeightRegion =
    "layer0_attention_context_head27_tokens0_5_cols1728_1792_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head27_tokens0_5_cols1728_1792_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead27Tokens05Cols17281792ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead27Tokens05Cols17281792WeightedSumChainExpectedFp32Sha256 =
    "315adaecbbeb502fb25c9ea08c8a12a15d4965dac8e9742a24d9133f7a12fc5b";


inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainName =
    "layer0_attention_scores_softmax_context_head28_tokens0_5_cols1792_1856_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head28_tokens0_5_cols1792_1856";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainSourceArrays =
    "layer0_attention_scores_head28_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head28_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head28_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head28_tokens0_5_cols1792_1856_v_as_b_fp16,layer0_attention_context_head28_tokens0_5_cols1792_1856_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainFixtureSlice =
    "layer=0,tokens=0:5,head=28,kv_head=7,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1792:1856";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainQRegion =
    "layer0_attention_scores_head28_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainKRegion =
    "layer0_attention_scores_head28_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainScoreRegion =
    "layer0_attention_scores_head28_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp32Region =
    "layer0_attention_probs_head28_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp16Region =
    "layer0_attention_probs_head28_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainModelWeightRegion =
    "layer0_attention_context_head28_tokens0_5_cols1792_1856_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head28_tokens0_5_cols1792_1856_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead28Tokens05Cols17921856ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead28Tokens05Cols17921856WeightedSumChainExpectedFp32Sha256 =
    "5054d4acbfa556380b23aafce53c49b8885b0305fa0d777ab22de68d270458d8";




inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainName =
    "layer0_attention_scores_softmax_context_head29_tokens0_5_cols1856_1920_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head29_tokens0_5_cols1856_1920";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainSourceArrays =
    "layer0_attention_scores_head29_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head29_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head29_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head29_tokens0_5_cols1856_1920_v_as_b_fp16,layer0_attention_context_head29_tokens0_5_cols1856_1920_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainFixtureSlice =
    "layer=0,tokens=0:5,head=29,kv_head=7,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1856:1920";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainQRegion =
    "layer0_attention_scores_head29_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainKRegion =
    "layer0_attention_scores_head29_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainScoreRegion =
    "layer0_attention_scores_head29_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp32Region =
    "layer0_attention_probs_head29_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp16Region =
    "layer0_attention_probs_head29_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainModelWeightRegion =
    "layer0_attention_context_head29_tokens0_5_cols1856_1920_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head29_tokens0_5_cols1856_1920_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead29Tokens05Cols18561920ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead29Tokens05Cols18561920WeightedSumChainExpectedFp32Sha256 =
    "dde7f4fd96c062870afa690520eb97cf172b9d5b44fb6e8633d1d4726223f03d";




inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainName =
    "layer0_attention_scores_softmax_context_head30_tokens0_5_cols1920_1984_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head30_tokens0_5_cols1920_1984";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainSourceArrays =
    "layer0_attention_scores_head30_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head30_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head30_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head30_tokens0_5_cols1920_1984_v_as_b_fp16,layer0_attention_context_head30_tokens0_5_cols1920_1984_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainFixtureSlice =
    "layer=0,tokens=0:5,head=30,kv_head=7,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1920:1984";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainQRegion =
    "layer0_attention_scores_head30_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainKRegion =
    "layer0_attention_scores_head30_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainScoreRegion =
    "layer0_attention_scores_head30_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp32Region =
    "layer0_attention_probs_head30_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp16Region =
    "layer0_attention_probs_head30_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainModelWeightRegion =
    "layer0_attention_context_head30_tokens0_5_cols1920_1984_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head30_tokens0_5_cols1920_1984_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead30Tokens05Cols19201984ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead30Tokens05Cols19201984WeightedSumChainExpectedFp32Sha256 =
    "df145a149aaf751f8dff3fedafaa7f29fb0dfdef6a7458ead26c5c8fe14a5d68";




inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainName =
    "layer0_attention_scores_softmax_context_head31_tokens0_5_cols1984_2048_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head31_tokens0_5_cols1984_2048";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainSourceArrays =
    "layer0_attention_scores_head31_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head31_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head31_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head31_tokens0_5_cols1984_2048_v_as_b_fp16,layer0_attention_context_head31_tokens0_5_cols1984_2048_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainFixtureSlice =
    "layer=0,tokens=0:5,head=31,kv_head=7,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1984:2048";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainQRegion =
    "layer0_attention_scores_head31_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainKRegion =
    "layer0_attention_scores_head31_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainScoreRegion =
    "layer0_attention_scores_head31_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp32Region =
    "layer0_attention_probs_head31_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp16Region =
    "layer0_attention_probs_head31_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainModelWeightRegion =
    "layer0_attention_context_head31_tokens0_5_cols1984_2048_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head31_tokens0_5_cols1984_2048_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead31Tokens05Cols19842048ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead31Tokens05Cols19842048WeightedSumChainExpectedFp32Sha256 =
    "56708c4457efda23aba6c383ebb1504c1b58b34ac3ccf586aee1c859907b0f2b";


inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainName =
    "layer0_q_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainModelForwardScope =
    "layer0_q_proj_full_inner_cols0_64";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainSourceArrays =
    "layer0_q_proj_full_inner_cols0_64_a_fp16,layer0_q_proj_full_inner_cols0_64_b_fp16,layer0_q_proj_full_inner_cols0_64_expected_fp32";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_q_full_inner_projection_fixtures.npz";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainFixtureSha256 =
    "63fee1efc814355717f47278d0cf2b2f617b66d948e1621ea6b8e46057ad1ce8";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64,inner=0:2048";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainActivationRegion =
    "layer0_q_proj_full_inner_cols0_64_activation_chunks";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainModelWeightRegion =
    "layer0_q_proj_full_inner_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainExpectedFp32Sha256 =
    "7f0c653f568bfef8dc662765dd40c22f67195488ebdadfdcdb4c0c9f7edc1e36";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainMaxAbsDiff =
    "4.291534423828125e-05";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainMaxUlpDiff = "2240";
inline constexpr const char* kLayer0QProjFullInnerCols0To64TiledAccumChainByteMismatchCount =
    "343";

inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainName =
    "layer0_o_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainModelForwardScope =
    "layer0_o_proj_full_inner_cols0_64";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainSourceArrays =
    "layer0_o_proj_full_inner_cols0_64_a_fp16,layer0_o_proj_full_inner_cols0_64_b_fp16,layer0_o_proj_full_inner_cols0_64_expected_fp32";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_fixtures.npz";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainFixtureSha256 =
    "543b67c42c774db932b02dacc01222ab354c2ff7c95366c658894acc01e51edd";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64,inner=0:2048";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainActivationRegion =
    "layer0_o_proj_full_inner_cols0_64_activation_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainModelWeightRegion =
    "layer0_o_proj_full_inner_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainExpectedFp32Sha256 =
    "4a12e1c74eef9e9f3a6b83143168604d2633ad9f1e247e8ed9a3073cdc3cbc34";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainMaxAbsDiff =
    "0";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainMaxUlpDiff = "0";
inline constexpr const char* kLayer0OProjFullInnerCols0To64TiledAccumChainByteMismatchCount =
    "0";

inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainName =
    "layer0_o_proj_full_inner_cols64_128_tiled_accum_chain";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainModelForwardScope =
    "layer0_o_proj_full_inner_cols64_128";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainSourceArrays =
    "layer0_o_proj_full_inner_cols64_128_a_fp16,layer0_o_proj_full_inner_cols64_128_b_fp16,layer0_o_proj_full_inner_cols64_128_expected_fp32";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_fixtures.npz";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainFixtureSha256 =
    "543b67c42c774db932b02dacc01222ab354c2ff7c95366c658894acc01e51edd";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=64:128,inner=0:2048";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainActivationRegion =
    "layer0_o_proj_full_inner_cols64_128_activation_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainModelWeightRegion =
    "layer0_o_proj_full_inner_cols64_128_model_weight_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainExpectedFp32Sha256 =
    "51155e8be7bc8608a31d9e35ea6439113208690ce747e00efa3b24c0fdfa8b0c";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainMaxAbsDiff =
    "3.3527612686157227e-07";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainMaxUlpDiff = "6018";
inline constexpr const char* kLayer0OProjFullInnerCols64To128TiledAccumChainByteMismatchCount =
    "350";

inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainName =
    "layer0_o_proj_full_inner_cols128_192_tiled_accum_chain";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainModelForwardScope =
    "layer0_o_proj_full_inner_cols128_192";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainSourceArrays =
    "layer0_o_proj_full_inner_cols128_192_a_fp16,layer0_o_proj_full_inner_cols128_192_b_fp16,layer0_o_proj_full_inner_cols128_192_expected_fp32";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols128_256_fixtures.npz";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainFixtureSha256 =
    "c66d07c51a1a9e212b250728f05039e3351d4e38bd7a36c21925073476756beb";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=128:192,inner=0:2048";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainActivationRegion =
    "layer0_o_proj_full_inner_cols128_192_activation_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainModelWeightRegion =
    "layer0_o_proj_full_inner_cols128_192_model_weight_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainExpectedFp32Sha256 =
    "49e23c9c01f31a5d21df7c351c202877616a98afc1c9b7088e3a4dca8f774f58";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainMaxAbsDiff =
    "2.5331974029541016e-07";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainMaxUlpDiff = "3824";
inline constexpr const char* kLayer0OProjFullInnerCols128To192TiledAccumChainByteMismatchCount =
    "352";

inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainName =
    "layer0_o_proj_full_inner_cols192_256_tiled_accum_chain";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainModelForwardScope =
    "layer0_o_proj_full_inner_cols192_256";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainSourceArrays =
    "layer0_o_proj_full_inner_cols192_256_a_fp16,layer0_o_proj_full_inner_cols192_256_b_fp16,layer0_o_proj_full_inner_cols192_256_expected_fp32";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols128_256_fixtures.npz";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainFixtureSha256 =
    "c66d07c51a1a9e212b250728f05039e3351d4e38bd7a36c21925073476756beb";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=192:256,inner=0:2048";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainActivationRegion =
    "layer0_o_proj_full_inner_cols192_256_activation_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainModelWeightRegion =
    "layer0_o_proj_full_inner_cols192_256_model_weight_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainExpectedFp32Sha256 =
    "5ff39b8fc6762e3edfe7e88923a5b3416320a93782b4b5de4aba624e4d03ef99";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainMaxAbsDiff =
    "2.0116567611694336e-07";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainMaxUlpDiff = "959";
inline constexpr const char* kLayer0OProjFullInnerCols192To256TiledAccumChainByteMismatchCount =
    "340";


inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainName =
    "layer0_o_proj_full_inner_cols256_320_tiled_accum_chain";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainModelForwardScope =
    "layer0_o_proj_full_inner_cols256_320";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainSourceArrays =
    "layer0_o_proj_full_inner_cols256_320_a_fp16,layer0_o_proj_full_inner_cols256_320_b_fp16,layer0_o_proj_full_inner_cols256_320_expected_fp32";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainFixtureSha256 =
    "805c989f5fe84c45877ace562fb31a7e4340e6f1aa88184ebefd3f8c8d111c87";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=256:320,inner=0:2048";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainActivationRegion =
    "layer0_o_proj_full_inner_cols256_320_activation_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainModelWeightRegion =
    "layer0_o_proj_full_inner_cols256_320_model_weight_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainExpectedFp32Sha256 =
    "7abde59ff9430703ea2c137017bcf76e237d1c6357d731e9748667d43850025c";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainMaxAbsDiff =
    "1.4156103134155273e-07";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainMaxUlpDiff = "2560";
inline constexpr const char* kLayer0OProjFullInnerCols256To320TiledAccumChainByteMismatchCount =
    "335";


inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainName =
    "layer0_o_proj_full_inner_cols320_384_tiled_accum_chain";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainModelForwardScope =
    "layer0_o_proj_full_inner_cols320_384";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainSourceArrays =
    "layer0_o_proj_full_inner_cols320_384_a_fp16,layer0_o_proj_full_inner_cols320_384_b_fp16,layer0_o_proj_full_inner_cols320_384_expected_fp32";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_o_full_inner_projection_cols256_384_fixtures.npz";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainFixtureSha256 =
    "805c989f5fe84c45877ace562fb31a7e4340e6f1aa88184ebefd3f8c8d111c87";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=320:384,inner=0:2048";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainActivationRegion =
    "layer0_o_proj_full_inner_cols320_384_activation_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainModelWeightRegion =
    "layer0_o_proj_full_inner_cols320_384_model_weight_chunks";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainExpectedFp32Sha256 =
    "cb480317ce8fa8f16efcd2d7cc34c275dc2c99875358f6e58a18a432ee7c6d6a";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainMaxAbsDiff =
    "1.6763806343078613e-07";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainMaxUlpDiff = "10240";
inline constexpr const char* kLayer0OProjFullInnerCols320To384TiledAccumChainByteMismatchCount =
    "339";


inline constexpr const char* kLayer0MlpFullInnerProjectionCols0To64TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_full_inner_projection_fixtures.npz";
inline constexpr const char* kLayer0MlpFullInnerProjectionCols0To64TiledAccumChainFixtureSha256 =
    "b5a6a11d98cae23d1836366ec3584de516d66c169c277367f8adc74966ad10c1";
inline constexpr const char* kLayer0MlpFullInnerProjectionCols0To64TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64,inner=0:2048";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainName =
    "layer0_mlp_gate_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainModelForwardScope =
    "layer0_mlp_gate_proj_full_inner_cols0_64";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainSourceArrays =
    "layer0_mlp_gate_proj_full_inner_cols0_64_a_fp16,layer0_mlp_gate_proj_full_inner_cols0_64_b_fp16,layer0_mlp_gate_proj_full_inner_cols0_64_expected_fp32";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainActivationRegion =
    "layer0_mlp_gate_proj_full_inner_cols0_64_activation_chunks";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainModelWeightRegion =
    "layer0_mlp_gate_proj_full_inner_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainExpectedFp32Sha256 =
    "3539b4bfa87707559d2a79c5301225c5a3153ea83d6515d022cedbd59ca3bf11";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainName =
    "layer0_mlp_up_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainModelForwardScope =
    "layer0_mlp_up_proj_full_inner_cols0_64";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainSourceArrays =
    "layer0_mlp_up_proj_full_inner_cols0_64_a_fp16,layer0_mlp_up_proj_full_inner_cols0_64_b_fp16,layer0_mlp_up_proj_full_inner_cols0_64_expected_fp32";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainActivationRegion =
    "layer0_mlp_up_proj_full_inner_cols0_64_activation_chunks";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainModelWeightRegion =
    "layer0_mlp_up_proj_full_inner_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainExpectedFp32Sha256 =
    "1d9be5465813acdf77f2144d2e8c564cc65eaceb87c369114d91df939a1de344";
inline constexpr const char* kLayer0MlpProjFullInnerCols0To64TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0MlpProjFullInnerCols0To64TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0MlpProjFullInnerCols0To64TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainMaxAbsDiff =
    "1.0728836059570312e-06";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainMaxUlpDiff =
    "11776";
inline constexpr const char* kLayer0MlpGateProjFullInnerCols0To64TiledAccumChainByteMismatchCount =
    "329";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainMaxAbsDiff =
    "8.6426734924316406e-07";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainMaxUlpDiff =
    "62464";
inline constexpr const char* kLayer0MlpUpProjFullInnerCols0To64TiledAccumChainByteMismatchCount =
    "331";

inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainName =
    "layer0_v_proj_full_inner_cols8_accum_chain";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainModelForwardScope =
    "layer0_v_proj_full_inner_cols8";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainSourceArrays =
    "layer0_v_proj_full_inner_cols8_a_fp16,layer0_v_proj_full_inner_cols8_b_fp16,layer0_v_proj_full_inner_cols8_expected_fp32";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:8,inner=0:2048";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainExpectedFp32Sha256 =
    "60eaa262244d3587aced4dcab0a267843ac727135dd0ef585f2a54b0f556e156";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainTolerance =
    "fp32_abs<=2e-6_or_ulp<=64";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainMaxAbsDiff =
    "1.3113021850585938e-06";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainMaxUlpDiff = "832";
inline constexpr const char* kLayer0VProjFullInnerCols8AccumChainByteMismatchCount =
    "44";

inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainName =
    "layer0_q_proj_full_inner_cols8_accum_chain";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainModelForwardScope =
    "layer0_q_proj_full_inner_cols8";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainSourceArrays =
    "layer0_q_proj_full_inner_cols8_a_fp16,layer0_q_proj_full_inner_cols8_b_fp16,layer0_q_proj_full_inner_cols8_expected_fp32";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:8,inner=0:2048";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainExpectedFp32Sha256 =
    "af0ac8143d62beee4eac7d493ab903de185f48d1c8ee726c51f7ca7b9f70a3a3";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainTolerance =
    "fp32_abs<=2e-6_or_ulp<=64";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainMaxAbsDiff =
    "1.621246337890625e-05";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainMaxUlpDiff = "34";
inline constexpr const char* kLayer0QProjFullInnerCols8AccumChainByteMismatchCount =
    "44";


inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8MaxAbsDiff = "0";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8MaxUlpDiff = "0";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8MismatchCount = "0";
inline constexpr const char* kFp16RopeSplitHalfLayer0KPairs8ByteMismatchCount = "0";

inline constexpr const char* kLayer0KTileMatmulToFp16ChainName =
    "layer0_k_tile_matmul_to_fp16_chain";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainProducerKind =
    "hardware_primitive_chain";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainModelForwardScope =
    "layer0_k_proj_partial_chain";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainNativePrefillAcceptance = "open";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainStageCount = "9";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainReadbackBetweenStages = "no";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainDataRegionCount = "2";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainInputRegion = "layer0_k_tile_input";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainIntermediateRegion =
    "fp32_intermediate";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputRegion = "fp16_output";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainInputRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainIntermediateRegionGpuVa =
    "0x0000200000004000";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainInputRegionPtbIndex = "1";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainIntermediateRegionPtbIndex = "4";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputRegionPtbIndex = "17";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputRegionPteStatus = "pass";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainInputByteCount = "512";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainIntermediateByteCount = "256";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputByteCount = "128";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputDtype = "fp16";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainOutputShape = "8x8";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainFinalFp16Sha256 =
    "7d8818f895f3e51bce24da8580fb10d76bffa457cba2c061ef2c7c1c0f5ee027";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainTolerance = "exact_fp16_bytes";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainMismatchCount = "0";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainByteMismatchCount = "0";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainFinalOutputClearStatus = "pass";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainKernargRewriteCount = "9";
inline constexpr const char* kLayer0KTileMatmulToFp16ChainComputeDispatchCount = "9";

inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainName =
    "layer0_k_tile_split_ab_resident_gemm";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelForwardScope =
    "layer0_k_tile_split_ab_resident_gemm";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmKernelSourceId =
    "c1r6k-layer0-k-tile-split-ab-gemm-v1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmKernelSha256 =
    "6b4817caab578ff53dfad102f70aa9fa38e7975905851c2f345e0755d7673fe6";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmKernelTextByteCount =
    "2516";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainStageCount = "1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainDataRegionCount = "2";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainActivationRegion =
    "layer0_k_tile_activation_a";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelWeightRegion =
    "layer0_k_tile_model_weight_b";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputRegion =
    "fp32_output";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainResidentDataPageCount =
    "2";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainActivationRegionGpuVa =
    "0x0000200000001000";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelWeightRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputRegionGpuVa =
    "0x0000200000012000";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainActivationRegionPtbIndex =
    "1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelWeightRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputRegionPtbIndex =
    "18";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainSupplementalPteCount =
    "2";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelWeightRegionPteStatus =
    "pass";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputRegionPteStatus =
    "pass";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainKernargScalarVaSource =
    "model_weight_region";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainKernelReadsModelWeightRegion =
    "yes";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainStage0Name =
    "layer0_k_tile_split_ab_matmul_fp32";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainKernargRewriteCount =
    "1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainComputeDispatchCount =
    "1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainActivationByteCount =
    "256";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelWeightByteCount =
    "256";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputByteCount =
    "256";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainUploadTotalBytes =
    "512";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainDownloadTotalBytes =
    "256";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputDtype = "fp32";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainOutputShape = "8x8";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainTolerance = "fp32_ulp<=1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainMaxAbsDiff =
    "1.862645149230957e-09";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainMaxUlpDiff = "1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainMismatchCount = "0";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainByteMismatchCount = "1";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainActivationUploadStatus =
    "pass";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainModelWeightUploadStatus =
    "pass";
inline constexpr const char* kLayer0KTileSplitAbResidentGemmChainFinalOutputClearStatus =
    "pass";

inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainName =
    "layer0_k_proj_full_inner_cols8_accum_chain";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelForwardScope =
    "layer0_k_proj_full_inner_cols8";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumKernelSourceId =
    "c1r6l-layer0-k-proj-full-inner-cols8-accum-v1";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumKernelSha256 =
    "e8aa56bb65c64da9862f2534219a0b2970dc95bf935ca871ca27f4fa79066853";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumKernelTextByteCount =
    "2576";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainStageCount = "128";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainDataRegionCount = "2";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationRegion =
    "layer0_k_proj_full_inner_cols8_activation_chunks";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightRegion =
    "layer0_k_proj_full_inner_cols8_model_weight_chunks";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputRegion =
    "fp32_output_accumulator";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainResidentDataPageCount =
    "17";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainDataRegionResidency =
    "seventeen_distinct_vram_pages";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightRegionGpuVa =
    "0x0000200000019000";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputRegionGpuVa =
    "0x0000200000021000";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationRegionPtbIndex =
    "17";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightRegionPtbIndex =
    "25";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputRegionPtbIndex =
    "33";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationRegionPageCount =
    "8";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightRegionPageCount =
    "8";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainSupplementalPteCount =
    "17";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationRegionPteStatus =
    "pass";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightRegionPteStatus =
    "pass";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputRegionPteStatus =
    "pass";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainKernargScalarVaSource =
    "model_weight_region";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainKernelReadsModelWeightRegion =
    "yes";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainStage0Name =
    "layer0_k_proj_full_inner_cols8_accum_chunk0";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainStage127Name =
    "layer0_k_proj_full_inner_cols8_accum_chunk127";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainKernargRewriteCount =
    "128";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainComputeDispatchCount =
    "128";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationByteCount =
    "32768";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightByteCount =
    "32768";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputByteCount =
    "256";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainUploadTotalBytes =
    "65536";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainDownloadTotalBytes =
    "256";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputDtype = "fp32";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainOutputShape = "8x8";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainSourceArrays =
    "layer0_k_proj_full_inner_cols8_a_fp16,layer0_k_proj_full_inner_cols8_b_fp16,layer0_k_proj_full_inner_cols8_expected_fp32";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:8,inner=0:2048";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainInputLayout =
    "activation_chunks8x16_row_major_then_model_weight_chunks_dot2_pair_packed";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainExpectedFp32Sha256 =
    "f231c8c34397196acd38260784294ea37e9fecea10b1309f4c2ec0348619860f";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainRowsValid = "5";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainTileRows = "8";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainTileInner = "2048";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainTileCols = "8";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainCoveredElementCount =
    "40";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainFullElementCount =
    "2560";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainTolerance =
    "fp32_ulp<=64";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainMaxAbsDiff =
    "1.621246337890625e-05";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainMaxUlpDiff = "34";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainMismatchCount = "0";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainByteMismatchCount =
    "44";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainActivationUploadStatus =
    "pass";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainModelWeightUploadStatus =
    "pass";
inline constexpr const char* kLayer0KProjFullInnerCols8AccumChainFinalOutputClearStatus =
    "pass";


inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainName =
    "layer0_k_proj_full_inner_cols0_16_tiled_accum_chain";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainModelForwardScope =
    "layer0_k_proj_full_inner_cols0_16";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainSourceArrays =
    "layer0_k_proj_full_inner_cols0_16_a_fp16,layer0_k_proj_full_inner_cols0_16_b_fp16,layer0_k_proj_full_inner_cols0_16_expected_fp32";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:16,inner=0:2048";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainStageCount =
    "256";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainActivationRegion =
    "layer0_k_proj_full_inner_cols0_16_activation_chunks";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainModelWeightRegion =
    "layer0_k_proj_full_inner_cols0_16_model_weight_chunks";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainResidentDataPageCount =
    "25";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainDataRegionResidency =
    "twenty_five_distinct_vram_pages";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputRegionGpuVa =
    "0x0000200000029000";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputRegionPtbIndex =
    "41";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainModelWeightRegionPageCount =
    "16";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainSupplementalPteCount =
    "25";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainKernargRewriteCount =
    "256";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainComputeDispatchCount =
    "256";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainModelWeightByteCount =
    "65536";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputByteCount =
    "512";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainUploadTotalBytes =
    "98304";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainDownloadTotalBytes =
    "512";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputShape =
    "8x16";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainExpectedFp32Sha256 =
    "5a749198fa05eca89df88c4d0754b9e45af874009b5f46c46957198b7d6ec7a3";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainTileCols = "16";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainCoveredElementCount =
    "80";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainFullElementCount =
    "5120";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputTileCount =
    "2";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputTile0Cols =
    "0:8";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainOutputTile1Cols =
    "8:16";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainReadbackLayout =
    "row_major_8x16_from_two_8x8_output_tiles";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_two_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainMaxAbsDiff =
    "1.621246337890625e-05";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainMaxUlpDiff = "35";
inline constexpr const char* kLayer0KProjFullInnerCols0To16TiledAccumChainByteMismatchCount =
    "88";

inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainName =
    "layer0_k_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainModelForwardScope =
    "layer0_k_proj_full_inner_cols0_64";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainSourceArrays =
    "layer0_k_proj_full_inner_cols0_64_a_fp16,layer0_k_proj_full_inner_cols0_64_b_fp16,layer0_k_proj_full_inner_cols0_64_expected_fp32";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_full_inner_projection_fixtures.npz";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainFixtureSha256 =
    "b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64,inner=0:2048";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainActivationRegion =
    "layer0_k_proj_full_inner_cols0_64_activation_chunks";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainModelWeightRegion =
    "layer0_k_proj_full_inner_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainExpectedFp32Sha256 =
    "f1387d0c28aae9aec3450fa384ac1ab178786decb9e5158250e14071dd99b047";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainMaxAbsDiff =
    "1.621246337890625e-05";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainMaxUlpDiff = "288";
inline constexpr const char* kLayer0KProjFullInnerCols0To64TiledAccumChainByteMismatchCount =
    "339";

inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainName =
    "layer0_v_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainModelForwardScope =
    "layer0_v_proj_full_inner_cols0_64";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainSourceArrays =
    "layer0_v_proj_full_inner_cols0_64_a_fp16,layer0_v_proj_full_inner_cols0_64_b_fp16,layer0_v_proj_full_inner_cols0_64_expected_fp32";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_full_inner_projection_fixtures.npz";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainFixtureSha256 =
    "b4a535f43caa33d4a9dc3d146098973ec2c66133ea63feb5233535a7ba4d038c";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64,inner=0:2048";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainStageCount =
    "1024";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainActivationRegion =
    "layer0_v_proj_full_inner_cols0_64_activation_chunks";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainModelWeightRegion =
    "layer0_v_proj_full_inner_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainResidentDataPageCount =
    "73";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainDataRegionResidency =
    "seventy_three_distinct_vram_pages";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputRegionPtbIndex =
    "89";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainModelWeightRegionPageCount =
    "64";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputRegionPageCount =
    "1";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainSupplementalPteCount =
    "73";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainKernargRewriteCount =
    "1024";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainComputeDispatchCount =
    "1024";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainModelWeightByteCount =
    "262144";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputByteCount =
    "2048";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainUploadTotalBytes =
    "294912";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainDownloadTotalBytes =
    "2048";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputShape =
    "8x64";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainExpectedFp32Sha256 =
    "28496084d43f9c0e257095edf97ea77885d6e9a762657e1dfd8e431a0e938927";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainTileCols = "64";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainCoveredElementCount =
    "320";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainFullElementCount =
    "20480";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputTileCount =
    "8";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainOutputTileCols =
    "8";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainInputLayout =
    "activation_chunks8x16_row_major_then_eight_model_weight_tile_streams_dot2_pair_packed";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainMaxAbsDiff =
    "1.3113021850585938e-06";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainMaxUlpDiff = "4352";
inline constexpr const char* kLayer0VProjFullInnerCols0To64TiledAccumChainByteMismatchCount =
    "347";

inline constexpr const char* kLayer0MlpActivationCols064ChainName =
    "layer0_mlp_activation_cols0_64_silu_mul_chain";
inline constexpr const char* kLayer0MlpActivationCols064ModelForwardScope =
    "layer0_mlp_activation_cols0_64";
inline constexpr const char* kLayer0MlpActivationCols064SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_activation_cols0_64_fixtures.npz";
inline constexpr const char* kLayer0MlpActivationCols064FixtureSha256 =
    "cb193cabaf06912806641fb058fd29bf0e1689e9eb90f642854366c5e5e3fe65";
inline constexpr const char* kLayer0MlpActivationCols064SourceArrays =
    "layer0_mlp_activation_cols0_64_gate_fp16,layer0_mlp_activation_cols0_64_up_fp16,layer0_mlp_activation_cols0_64_expected_fp16";
inline constexpr const char* kLayer0MlpActivationCols064FixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64";
inline constexpr const char* kLayer0MlpActivationCols064StageCount = "64";
inline constexpr const char* kLayer0MlpActivationCols064DataRegionCount = "2";
inline constexpr const char* kLayer0MlpActivationCols064InputRegion =
    "layer0_mlp_activation_cols0_64_gate_up_pairs";
inline constexpr const char* kLayer0MlpActivationCols064OutputRegion =
    "layer0_mlp_activation_cols0_64_output";
inline constexpr const char* kLayer0MlpActivationCols064ResidentDataPageCount = "2";
inline constexpr const char* kLayer0MlpActivationCols064DataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0MlpActivationCols064InputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0MlpActivationCols064OutputRegionGpuVa =
    "0x0000200000059000";
inline constexpr const char* kLayer0MlpActivationCols064InputRegionPtbIndex = "17";
inline constexpr const char* kLayer0MlpActivationCols064OutputRegionPtbIndex = "89";
inline constexpr const char* kLayer0MlpActivationCols064InputRegionPageCount = "1";
inline constexpr const char* kLayer0MlpActivationCols064OutputRegionPageCount = "1";
inline constexpr const char* kLayer0MlpActivationCols064KernargRewriteCount = "64";
inline constexpr const char* kLayer0MlpActivationCols064ComputeDispatchCount = "64";
inline constexpr const char* kLayer0MlpActivationCols064ActivationByteCount = "1024";
inline constexpr const char* kLayer0MlpActivationCols064OutputByteCount = "1024";
inline constexpr const char* kLayer0MlpActivationCols064UploadTotalBytes = "1024";
inline constexpr const char* kLayer0MlpActivationCols064DownloadTotalBytes = "1024";
inline constexpr const char* kLayer0MlpActivationCols064OutputShape = "8x64";
inline constexpr const char* kLayer0MlpActivationCols064ExpectedFp16Sha256 =
    "343350605cc2f3469145c18978fc2b0942f373547e47020c10a9c3806237430c";
inline constexpr const char* kLayer0MlpActivationCols064KernelSourceId =
    "c1r7d-layer0-mlp-silu-mul-slice8-v1";
inline constexpr const char* kLayer0MlpActivationCols064KernelSha256 = "7b1a31ea7c2150c813d09f85eab4a35db925dd1fc38d644dbfe2ff726722afc1";
inline constexpr const char* kLayer0MlpActivationCols064RowsValid = "5";
inline constexpr const char* kLayer0MlpActivationCols064TileRows = "8";
inline constexpr const char* kLayer0MlpActivationCols064TileCols = "64";
inline constexpr const char* kLayer0MlpActivationCols064CoveredElementCount = "320";
inline constexpr const char* kLayer0MlpActivationCols064FullElementCount = "512";
inline constexpr const char* kLayer0MlpActivationCols064ReadbackLayout =
    "row_major_8x64_from_eight_8x8_output_tiles";
inline constexpr const char* kLayer0MlpActivationCols064InputLayout =
    "tile_major_row_gate8_then_up8";
inline constexpr const char* kLayer0MlpActivationCols064Tolerance = "fp16_ulp<=1";
inline constexpr const char* kLayer0MlpActivationCols064MaxAbsDiff =
    "3.0517578125e-05";
inline constexpr const char* kLayer0MlpActivationCols064MaxUlpDiff = "1";
inline constexpr const char* kLayer0MlpActivationCols064MismatchCount = "0";
inline constexpr const char* kLayer0MlpActivationCols064ByteMismatchCount = "91";

inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ChainName =
    "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ModelForwardScope =
    "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_partial";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_inner_cols0_64_to_cols0_64_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064FixtureSha256 =
    "62ba57e858a723ea0e326e6d3773d915e423c777f84b39bd6b322a39a98f30ad";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064SourceArrays =
    "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_activation_fp16,layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_weight_fp16,layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_expected_fp32";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064FixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,activation_inner=0:64,output_cols=0:64";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064StageCount = "32";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ActivationRegion =
    "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_activation_chunks";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ModelWeightRegion =
    "layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_model_weight_chunks";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064OutputRegion =
    "fp32_output_accumulator_tiles";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ResidentDataPageCount = "4";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064DataRegionResidency =
    "four_distinct_vram_pages";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ActivationRegionPageCount = "1";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ModelWeightRegionPageCount = "2";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064OutputRegionPageCount = "1";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064SupplementalPteCount = "4";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064KernargRewriteCount = "32";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ComputeDispatchCount = "32";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064InnerChunkCount = "4";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064InnerChunkSize = "16";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ActivationByteCount = "1024";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ModelWeightByteCount = "8192";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064OutputByteCount = "2048";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064UploadTotalBytes = "9216";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064DownloadTotalBytes = "2048";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064OutputShape = "8x64";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ExpectedFp32Sha256 =
    "64559386ab500f4807074afadb1878c50f14069a9c1ff4bd48c1931658ade390";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064RowsValid = "5";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064TileRows = "8";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064TileInner = "64";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064TileCols = "64";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064CoveredElementCount = "320";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064FullElementCount = "512";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064Tolerance =
    "fp32_abs<=2e-6_or_ulp<=64";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064MaxAbsDiff =
    "9.3132257461547852e-10";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064MaxUlpDiff = "64";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064MismatchCount = "0";
inline constexpr const char* kLayer0MlpDownProjInnerCols064ToCols064ByteMismatchCount = "125";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols064ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols064ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols0_64";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols064SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols0_64_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols064FixtureSha256 =
    "e3aab29d893f849fc4627e4781ca36fef1574ccf4d5dda562fcdacf3438bb338";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols064ExpectedFp32Sha256 =
    "84f9ddf66e1e71849b928caa061b6abcca81d00bea59081635592ca7d58f4d7e";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols64128ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols64_128_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols64128ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols64_128";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols64128SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols64_128_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols64128FixtureSha256 =
    "d1242c9add185957e7c5cf8273d6f26d9eb4103786e0496bf1a0d5e29d9929f6";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols64128ExpectedFp32Sha256 =
    "cd7f1e930959cf668116142d14ad8374ddbf732d6b71134aa9550de9c277d21a";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols128192ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols128_192_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols128192ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols128_192";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols128192SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols128_192_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols128192FixtureSha256 =
    "ba75e101395b1682c92585b8030ea7d78431f15b3c58f40ea47564c28aac9b4d";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols128192ExpectedFp32Sha256 =
    "f70e9db966a3e63ca22a38f06c68b60b4910c2522055670404f4eb24405f89b4";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols192256ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols192_256_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols192256ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols192_256";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols192256SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols192_256_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols192256FixtureSha256 =
    "691e6c216090c5569a39177c532f3eca6b8e4792ef30656dbdeb4529495378f6";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols192256ExpectedFp32Sha256 =
    "f5ca75c595cfebb249605cb43788a2e64f6bd508f3cd5dd262a24b3101fa3533";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols256320ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols256_320_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols256320ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols256_320";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols256320SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols256_320_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols256320FixtureSha256 =
    "252fdad991788ef0caf826450e0e35058ad5913b943569cbaeeca2a606c264e2";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols256320ExpectedFp32Sha256 =
    "d9c9e4bf8a22f7c23842c9e7bc45eaf4160d02ffe853bf213f2137e4650ac3ea";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols320384ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols320_384_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols320384ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols320_384";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols320384SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols320_384_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols320384FixtureSha256 =
    "c80392d1613bffe36e0d910a17b909100f5a1ab3443a4b7d9d12fe9abd42ae35";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols320384ExpectedFp32Sha256 =
    "d778614a9ca5543a9b399379f6e9161af6e14722e74d1a204047ab8e0e17bc94";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols384448ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols384_448_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols384448ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols384_448";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols384448SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols384_448_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols384448FixtureSha256 =
    "ee04edb8003f6b7d90e6febb1493aeec40a5c44b5693f436fbf0f746c33c855d";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols384448ExpectedFp32Sha256 =
    "009dfb2599ca19db39614c27b269db20b4d29408a16e78ca3ff89037818fb4e6";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols448512ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols448_512_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols448512ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols448_512";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols448512SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols448_512_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols448512FixtureSha256 =
    "d617b8f69e5d484db6ec7abe96e888b990ba091b4e7b44c43318533e5035c069";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols448512ExpectedFp32Sha256 =
    "9460250f2aa90b73b75b74e904c3e049ab2e0d734001faa772b205debe279c26";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols512576ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols512_576_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols512576ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols512_576";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols512576SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols512_576_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols512576FixtureSha256 =
    "dba88e1d9feb9454c0cc9a510d705d133640a6e823467ea9c7ada4fab07ae12b";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols512576ExpectedFp32Sha256 =
    "1dbf59efedf86d3b58e67ebe5914c907720b91b36d358594e807813ce9b08e46";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols576640ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols576_640_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols576640ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols576_640";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols576640SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols576_640_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols576640FixtureSha256 =
    "649fd23988fe8f7a4c40f3ca09b2e3c04bdffa25f807e0ca81892effac815e77";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols576640ExpectedFp32Sha256 =
    "811bfd494dd3f5fbbe89281e90a85a31f03623fe3837d7bb6a12cb5d9dd7df55";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols640704ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols640_704_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols640704ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols640_704";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols640704SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols640_704_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols640704FixtureSha256 =
    "4f87494495e5d07765f76504711c810ee9bf20e680c83f26d9656e3ac4f7ba9c";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols640704ExpectedFp32Sha256 =
    "98d6b80871779d4c01cb6205d4d6b95e9a3a7d2ee66fdf5f8d864feacc3088f8";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols704768ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols704_768_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols704768ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols704_768";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols704768SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols704_768_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols704768FixtureSha256 =
    "bd14dc7b032e7c2fe2340743cbf360339f514c7ca0f938ea89665c473098f22e";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols704768ExpectedFp32Sha256 =
    "db8c4b0630d95561e8cc3dfebfbe2a3d0aa13b14116dbaaeb4fcf24efc45de0a";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols768832ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols768_832_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols768832ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols768_832";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols768832SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols768_832_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols768832FixtureSha256 =
    "9492bbaac58440e9495cf9d452c51f7889bc2035fca18dea8c4644001fe4178d";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols768832ExpectedFp32Sha256 =
    "efec5949b3122b9bb0e50ffd16d70619060448ad613e8808b117d4420d03d0d7";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols832896ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols832_896_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols832896ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols832_896";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols832896SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols832_896_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols832896FixtureSha256 =
    "ce7e699faee42b53bbd4b20f3517615113436af86296d39487a274b38cd1cca3";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols832896ExpectedFp32Sha256 =
    "371bd55cba32daf9d363aa8b1df52bda6db006ce82b333846829f143b9e14750";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols896960ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols896_960_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols896960ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols896_960";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols896960SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols896_960_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols896960FixtureSha256 =
    "0191316e76c2770adc3da5bdd3e6d67fd27199051b97ab3d6f62ed8c1ba228ff";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols896960ExpectedFp32Sha256 =
    "141ee577a5ec10d7fb2529f7f6f7aa9c398b39bbba25356c254138e1aa4d30b3";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols9601024ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols960_1024_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols9601024ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols960_1024";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols9601024SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols960_1024_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols9601024FixtureSha256 =
    "3d5ec83e98bd07500b1e40af66a1d117ef0d2b71901a1a2ecc0679fd5b1b51df";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols9601024ExpectedFp32Sha256 =
    "1ce368e6195745aa49d0569e17a28847c8086d59d8449cc9382bc56edf4a0830";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10241088ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1024_1088_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10241088ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1024_1088";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10241088SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1024_1088_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10241088FixtureSha256 =
    "1437af73e83249565ca7c4205d4bcae23a52c6cdd3a4fbb89bf5d7777fca4153";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10241088ExpectedFp32Sha256 =
    "b477fecfa61b2e402af9e4fe4be4a3ee562defe89f7089d8daf0de37574f3f43";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10881152ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1088_1152_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10881152ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1088_1152";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10881152SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1088_1152_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10881152FixtureSha256 =
    "ec8b3887d7d19cc2b83260384e67eaef9fe38cfa6ad548eb0d94265ace88ed72";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols10881152ExpectedFp32Sha256 =
    "96a4426d900daec45deb30b32c628c2b70ea51972b2b841984f62500f0b1cb28";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols11521216ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1152_1216_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols11521216ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1152_1216";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols11521216SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1152_1216_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols11521216FixtureSha256 =
    "9185f0abdb25976573b1481a08dbe809bddee2e46fc44b9fb57fbc93ed669e5a";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols11521216ExpectedFp32Sha256 =
    "9d32308facc313a99452dc1266c2ec2d0a0bec9b0f371cbf72f93119a8a3eaea";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12161280ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1216_1280_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12161280ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1216_1280";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12161280SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1216_1280_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12161280FixtureSha256 =
    "610ded7944f4a92930cf6d610e9c3f5d4a857bb17d40b029be723cef96a8d84e";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12161280ExpectedFp32Sha256 =
    "0474bf43af83683b747b2944216bbe77885ccd29f275bf6f1fc35fd8f25ae3aa";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12801344ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1280_1344_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12801344ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1280_1344";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12801344SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1280_1344_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12801344FixtureSha256 =
    "021dd5f080ccc96e73c05747d0c70d215687ee6a60cf5a2983ed0a692b897c68";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols12801344ExpectedFp32Sha256 =
    "f0129a3371ee92b6cc844bb3654e859e19f503e87c2d8ad8cbefea4880280112";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols13441408ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1344_1408_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols13441408ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1344_1408";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols13441408SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1344_1408_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols13441408FixtureSha256 =
    "30ea05c720ff1004a8c76c2db6ef869cebca003c63bb472457f9d5c19012cd13";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols13441408ExpectedFp32Sha256 =
    "c33442e0720192d8485a2d3d267bde9e1c391a1da741eae0b4e7ed9ee62284d7";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14081472ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1408_1472_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14081472ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1408_1472";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14081472SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1408_1472_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14081472FixtureSha256 =
    "c734eec72fd4784ab699e4f9654253130b3775f3240c42cd3dc857f3452acffc";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14081472ExpectedFp32Sha256 =
    "d83fbf7899a07c74e9c064f799baa86eacbfbc25447047a62e966a31642660a0";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14721536ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1472_1536_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14721536ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1472_1536";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14721536SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1472_1536_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14721536FixtureSha256 =
    "1c71e4cc882f89cfb62882d3ff04e4380d36b88fa6890df024c276e30d1f85d8";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols14721536ExpectedFp32Sha256 =
    "23a1ab9ad45b36d9c6d46a2463e62a23f71cb1183ec2a91f1c4226f93557fba3";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols15361600ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1536_1600_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols15361600ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1536_1600";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols15361600SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1536_1600_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols15361600FixtureSha256 =
    "65df5bacd3d3fe1c26f76f86bb745fac4f83a98347e063a0760d8aed4a4b3cae";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols15361600ExpectedFp32Sha256 =
    "e63fa909ecc36084bb4fad28144f25d583898a9a5cfad55e310c90cd7da27b3e";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16001664ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1600_1664_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16001664ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1600_1664";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16001664SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1600_1664_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16001664FixtureSha256 =
    "ef2858ea16d1651e7bee0f40e70ec493f5cd02857f06c404e3867c9c1b96fe20";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16001664ExpectedFp32Sha256 =
    "798d5abe872dc3a2ab90d0ecf8d3e726a099490d41beeb7664b175b36c082c67";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16641728ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1664_1728_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16641728ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1664_1728";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16641728SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1664_1728_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16641728FixtureSha256 =
    "038a847c96b6657fe529b3b25fc48bf73b0b0328c1c139b4463392656945d437";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols16641728ExpectedFp32Sha256 =
    "b3dac35c0a94c400326c4065504781ca4ee51a5ee9a2cad421fd5d5cbebc7995";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17281792ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1728_1792_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17281792ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1728_1792";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17281792SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1728_1792_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17281792FixtureSha256 =
    "1b621038e1fd86431ad4e71ec5ae0e596cad4b12cdbe1aa3cfea495d774073b4";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17281792ExpectedFp32Sha256 =
    "8f56445ef4b2f1a53ac290a34fbe786a19667b9622443e78fe3e97a9fa912e00";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17921856ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1792_1856_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17921856ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1792_1856";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17921856SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1792_1856_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17921856FixtureSha256 =
    "381a14fcf4e909735f238cc0a346cd496644dc60f3ddcb6fc826a95e94341c9b";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols17921856ExpectedFp32Sha256 =
    "f43a8b5ce52c594f6270ec62fe7e26e0e7e4a169573550280b4a0ff3cf69e66c";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols18561920ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1856_1920_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols18561920ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1856_1920";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols18561920SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1856_1920_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols18561920FixtureSha256 =
    "7ea12733c622bfd2ed0dc3293c73c4c61a264e7730eb60e4768dd89b7b18f206";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols18561920ExpectedFp32Sha256 =
    "5ca70f7d6aee8c712acedef671c1eb8514cce0754318411afe90906eda956ea7";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19201984ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1920_1984_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19201984ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1920_1984";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19201984SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1920_1984_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19201984FixtureSha256 =
    "efabe98f441524e20aeed7810fa56c853f577b72ce1c531e1bfc403c11fa2cb2";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19201984ExpectedFp32Sha256 =
    "7a78d5e6e416906132a279dcfe7a065c73594849361d2af7a9ea5cee37798bc0";

inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19842048ChainName =
    "layer0_mlp_down_proj_full_inner_to_cols1984_2048_tiled_accum_chain";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19842048ModelForwardScope =
    "layer0_mlp_down_proj_full_inner_to_cols1984_2048";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19842048SourceFixture =
    "tests/native_r9700/fixtures/layer_trace_mlp_down_projection_full_inner_to_cols1984_2048_fixtures.npz";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19842048FixtureSha256 =
    "4e431df60dae178163293afe2efa74f0ab9867e85e6bf68b60cee1efea4d186f";
inline constexpr const char* kLayer0MlpDownProjFullInnerToCols19842048ExpectedFp32Sha256 =
    "a020ed331d7ba0e6c3b63a46c991fbe201e042e51ea36c641922b82111f31f79";








inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainName =
    "layer0_attention_scores_softmax_context_head17_tokens0_5_cols1088_1152_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head17_tokens0_5_cols1088_1152";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainSourceArrays =
    "layer0_attention_scores_head17_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head17_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head17_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head17_tokens0_5_cols1088_1152_v_as_b_fp16,layer0_attention_context_head17_tokens0_5_cols1088_1152_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainFixtureSlice =
    "layer=0,tokens=0:5,head=17,kv_head=4,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1088:1152";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainQRegion =
    "layer0_attention_scores_head17_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainKRegion =
    "layer0_attention_scores_head17_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainScoreRegion =
    "layer0_attention_scores_head17_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp32Region =
    "layer0_attention_probs_head17_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp16Region =
    "layer0_attention_probs_head17_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainModelWeightRegion =
    "layer0_attention_context_head17_tokens0_5_cols1088_1152_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head17_tokens0_5_cols1088_1152_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead17Tokens05Cols10881152ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead17Tokens05Cols10881152WeightedSumChainExpectedFp32Sha256 =
    "4764560bde6630e3ee97d1779a832d741891fadefee2a8f7a196f297273ef697";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainName =
    "layer0_attention_scores_softmax_context_head18_tokens0_5_cols1152_1216_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head18_tokens0_5_cols1152_1216";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainSourceArrays =
    "layer0_attention_scores_head18_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head18_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head18_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head18_tokens0_5_cols1152_1216_v_as_b_fp16,layer0_attention_context_head18_tokens0_5_cols1152_1216_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainFixtureSlice =
    "layer=0,tokens=0:5,head=18,kv_head=4,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1152:1216";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainQRegion =
    "layer0_attention_scores_head18_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainKRegion =
    "layer0_attention_scores_head18_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainScoreRegion =
    "layer0_attention_scores_head18_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp32Region =
    "layer0_attention_probs_head18_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp16Region =
    "layer0_attention_probs_head18_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainModelWeightRegion =
    "layer0_attention_context_head18_tokens0_5_cols1152_1216_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head18_tokens0_5_cols1152_1216_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead18Tokens05Cols11521216ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead18Tokens05Cols11521216WeightedSumChainExpectedFp32Sha256 =
    "1064eb8a60ffe4472d4864530cf464701593d216cf56c31cb62e23bbb16171d4";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainName =
    "layer0_attention_scores_softmax_context_head19_tokens0_5_cols1216_1280_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head19_tokens0_5_cols1216_1280";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainSourceArrays =
    "layer0_attention_scores_head19_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head19_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head19_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head19_tokens0_5_cols1216_1280_v_as_b_fp16,layer0_attention_context_head19_tokens0_5_cols1216_1280_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainFixtureSlice =
    "layer=0,tokens=0:5,head=19,kv_head=4,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1216:1280";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainQRegion =
    "layer0_attention_scores_head19_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainKRegion =
    "layer0_attention_scores_head19_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainScoreRegion =
    "layer0_attention_scores_head19_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp32Region =
    "layer0_attention_probs_head19_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp16Region =
    "layer0_attention_probs_head19_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainModelWeightRegion =
    "layer0_attention_context_head19_tokens0_5_cols1216_1280_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head19_tokens0_5_cols1216_1280_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead19Tokens05Cols12161280ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead19Tokens05Cols12161280WeightedSumChainExpectedFp32Sha256 =
    "751b2ba3203029935e2f46d25421219db5416a89728e1087e1bfbfc313bb17e5";

inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainName =
    "layer0_attention_scores_softmax_context_head20_tokens0_5_cols1280_1344_chain";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainAcceptanceScope =
    "hardware_primitive_chain_only_partial";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainModelForwardScope =
    "layer0_attention_scores_softmax_context_head20_tokens0_5_cols1280_1344";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_fixtures.npz";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainFixtureSha256 =
    "a28fca99ccc4b9eaf25226258496f21167b76b0c208dad7fdb6aa34bf794ca96";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainSourceArrays =
    "layer0_attention_scores_head20_tokens0_5_scaled_masked_q_scaled_fp16,layer0_attention_scores_head20_tokens0_5_scaled_masked_k_as_b_fp16,layer0_attention_scores_head20_tokens0_5_scaled_masked_seed_fp32,layer0_attention_context_head20_tokens0_5_cols1280_1344_v_as_b_fp16,layer0_attention_context_head20_tokens0_5_cols1280_1344_expected_fp32";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainFixtureSlice =
    "layer=0,tokens=0:5,head=20,kv_head=5,q_rows=0:5,padded_rows=5:8,k_cols=0:5,padded_cols=5:8,head_dim=0:64,context_cols=1280:1344";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainReadbackBetweenStages =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenStages;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainReadbackBetweenOutputTiles =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainReadbackBetweenOutputTiles;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainDataRegionCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainQRegion =
    "layer0_attention_scores_head20_tokens0_5_scaled_masked_q_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainKRegion =
    "layer0_attention_scores_head20_tokens0_5_scaled_masked_k_as_b_chunks";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainScoreRegion =
    "layer0_attention_scores_head20_tokens0_5_scaled_masked_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp32Region =
    "layer0_attention_probs_head20_tokens0_5_softmax_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp16Region =
    "layer0_attention_probs_head20_tokens0_5_softmax_fp16_cast_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainModelWeightRegion =
    "layer0_attention_context_head20_tokens0_5_cols1280_1344_v_as_b";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainOutputRegion =
    "layer0_attention_scores_softmax_context_head20_tokens0_5_cols1280_1344_fp32_output";
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainResidentDataPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainResidentDataPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainDataRegionResidency =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDataRegionResidency;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainQRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainKRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainScoreRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp32RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp16RegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainModelWeightRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainOutputRegionGpuVa =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionGpuVa;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainQRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainQRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainKRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainScoreRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp32RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp32RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp16RegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16RegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainModelWeightRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainModelWeightRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainOutputRegionPtbIndex =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputRegionPtbIndex;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainRegionPageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainRegionPageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainSupplementalPteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSupplementalPteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainKernargRewriteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainKernargRewriteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainComputeDispatchCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainComputeDispatchCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainScaledMaskedStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScaledMaskedStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainSoftmaxStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbabilityCastStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbabilityCastStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainContextStageCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextStageCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainUploadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainUploadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainDownloadTotalBytes =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainDownloadTotalBytes;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainSoftmaxOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsFp16ByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsFp16ByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainContextModelWeightByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextModelWeightByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainOutputByteCount =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainOutputByteCount;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainScoreTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainScoreTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainContextTileInner =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainContextTileInner;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainInputLayout =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainInputLayout;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainProbsSource =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainProbsSource;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainSoftmaxStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainSoftmaxStatus;
inline constexpr const char* kLayer0AttentionScoresSoftmaxContextHead20Tokens05Cols12801344ChainMaskedScoreStatus =
    kLayer0AttentionScoresSoftmaxContextHead0Tokens05Cols064ChainMaskedScoreStatus;
inline constexpr const char* kLayer0AttentionContextHead20Tokens05Cols12801344WeightedSumChainExpectedFp32Sha256 =
    "e9ec619bd7ebb08106173f8b6baa99f46f35fe75f8ea80d3e7dae711aa9d2620";


inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainName =
    "layer0_attention_residual_cols0_64_after_o_proj_chain";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainModelForwardScope =
    "layer0_attention_residual_cols0_64_after_o_proj";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainNativePrefillAcceptance =
    "open";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_attention_residual_cols0_64_fixtures.npz";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainFixtureSha256 =
    "2a76c96c4c6920eacb6938744d64955707bf10cbc4efb9dd0939d75fe48a79f8";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainSourceArrays =
    "layer0_attention_residual_cols0_64_hidden_in_fp16,layer0_attention_residual_cols0_64_o_proj_output_fp16,layer0_attention_residual_cols0_64_expected_fp16";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,cols=0:64";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainStageCount = "64";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainReadbackBetweenStages = "no";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainDataRegionCount = "2";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainInputRegion =
    "layer0_attention_residual_cols0_64_lhs_rhs_pairs";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainOutputRegion =
    "layer0_attention_residual_cols0_64_output";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainResidentDataPageCount = "2";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainDataRegionResidency =
    "two_distinct_vram_pages";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainInputRegionGpuVa =
    "0x0000200000011000";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainOutputRegionGpuVa =
    "0x0000200000021000";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainInputRegionPtbIndex = "17";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainOutputRegionPtbIndex = "33";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainRegionPageCount = "1";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainSupplementalPteCount = "2";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainKernargRewriteCount = "64";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainComputeDispatchCount = "64";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainElementType = "fp16_add";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainElementCount = "512";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainStageElementCount = "8";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainValidRows = "5";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainTileRows = "8";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainTileCols = "64";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainOutputDtype = "fp16";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainOutputShape = "8x64";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainCoveredElementCount = "320";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainFullElementCount = "512";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainReadbackLayout =
    "row_major_8x64_from_eight_8_col_slices";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainInputLayout =
    "hidden_in_rows8x64_then_o_proj_rows8x64_sliced_by_8_columns";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainOProjSource =
    "layer0_o_proj_full_inner_cols0_64_tiled_accum_chain";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainTolerance = "exact_fp16_bytes";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainExpectedFp16Sha256 =
    "a28d2b3e2e3d8dfd28e788ec21286099e713f6e3056190870029668b0a602de4";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainMaxAbsDiff = "0";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainMaxUlpDiff = "0";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainMismatchCount = "0";
inline constexpr const char* kLayer0AttentionResidualCols064AfterOProjChainByteMismatchCount = "0";

inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainName =
    "layer0_post_attention_rmsnorm_cols0_64_chain";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainAcceptanceScope =
    "hardware_primitive_chain_only";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainModelForwardScope =
    "layer0_post_attention_rmsnorm_cols0_64";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNativePrefillAcceptance = "open";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSourceFixture =
    "tests/native_r9700/fixtures/layer_trace_post_attention_rmsnorm_cols0_64_fixtures.npz";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainFixtureSha256 =
    "58e213bed698fcf00584ab7e7a653f9a51d0c6cde4cfdde133ab995c863c6c59";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSourceArrays =
    "layer0_post_attention_rmsnorm_cols0_64_residual_in_fp16,layer0_post_attention_rmsnorm_cols0_64_weight_fp16,layer0_post_attention_rmsnorm_cols0_64_expected_fp16";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainFixtureSlice =
    "layer=0,rows=0:5,padded_rows=5:8,input_cols=0:2048,output_cols=0:64";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainStageCount = "136";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainReadbackBetweenStages =
    "no";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainReadbackBetweenOutputTiles =
    "no";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainDataRegionCount = "5";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainInputRegion =
    "layer0_post_attention_rmsnorm_cols0_64_residual_full";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainWeightRegion =
    "layer0_post_attention_rmsnorm_cols0_64_weight_full";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqRegion =
    "layer0_post_attention_rmsnorm_cols0_64_sumsq_accumulator_tile";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqRhsRegion =
    "layer0_post_attention_rmsnorm_cols0_64_sumsq_diagonal_chunks";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainOutputRegion =
    "layer0_post_attention_rmsnorm_cols0_64_output";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainResidentDataPageCount = "27";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainDataRegionResidency =
    "twenty_seven_distinct_vram_pages";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainInputRegionPageCount = "8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainWeightRegionPageCount = "1";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqRegionPageCount = "1";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainOutputRegionPageCount = "1";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSupplementalPteCount = "27";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainKernargRewriteCount = "136";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainComputeDispatchCount = "136";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqStageCount = "128";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNormalizeStageCount = "8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqChunkCount = "128";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqChunkSize = "16";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqAccumulatorDtype =
    "fp32";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqAccumulatorShape =
    "8x8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNormalizeTileCount = "8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNormalizeTileRows = "8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNormalizeTileCols = "8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNormalizePointerKernargs =
    "output_va,input_va,weight_va,sum_va";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainNormalizeScalarBits =
    "inv_hidden=0x3a000000,eps=0x3727c5ac";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainWeightReadCols = "0:64";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064FinalizeKernelSourceId =
    "c1r7b-post-attention-rmsnorm64-finalize-r4-b32-unrolled-store16-v1";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064FinalizeKernelSha256 =
    "f1cba3381de4eefbcda357bf84724fc85e3a2c8699686d1ba99a8b3473106943";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainElementType =
    "fp16_rms_norm_fp32";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainElementCount = "512";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainStageElementCount = "64";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainReductionElementCount =
    "2048";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainValidRows = "5";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainTileRows = "8";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainTileCols = "64";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainInputShape = "8x2048";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainWeightShape = "2048";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainOutputDtype = "fp16";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainOutputShape = "8x64";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainCoveredElementCount = "320";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainFullElementCount = "512";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainInputByteCount = "32768";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainWeightByteCount = "4096";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainSumsqByteCount = "256";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainOutputByteCount = "1024";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainUploadTotalBytes = "102400";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainDownloadTotalBytes =
    "1024";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainInputLayout =
    "residual_full_rows8x2048_for_normalize_plus_packed_8x16_sumsq_chunks";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainReadbackLayout = "row_major_8x64_stitched_from_eight_8x8_normalize_tiles";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainResidualSource =
    "layer0_attention_residual_cols0_64_after_o_proj_chain_requires_full_width";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainTolerance = "exact_fp16_bytes";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainExpectedFp16Sha256 =
    "66d70f967e30b3ddad71dc8fadf7e9157d7badc2fd0d654e26177a908fddd903";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainMaxAbsDiff = "0";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainMaxUlpDiff = "0";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainMismatchCount = "0";
inline constexpr const char* kLayer0PostAttentionRmsNormCols064ChainByteMismatchCount = "0";


// Frozen 24-byte kernarg layout: {output_va@0, input_va@8, scalar_va@16, scalar:u32@24}.
inline constexpr size_t kKernargOutputVaOffset = 0;
inline constexpr size_t kKernargInputVaOffset = 8;
inline constexpr size_t kKernargScalarVaOffset = 16;
inline constexpr size_t kKernargScalarOffset = 24;  // u32 scalar value
inline constexpr size_t kKernargByteSize = 24;      // 3 x u64 VA (24 bytes)
static_assert(kKernargOutputVaOffset + sizeof(uint64_t) <= kKernargScalarOffset,
              "output_va must fit before the scalar");
static_assert(kKernargInputVaOffset + sizeof(uint64_t) <= kKernargScalarOffset,
              "input_va must fit before the scalar");
static_assert(kKernargScalarVaOffset + sizeof(uint64_t) <= kKernargScalarOffset,
              "scalar_va must fit before the scalar");


// Lifecycle stage identifiers, in required execution order.
enum class LifecycleStage {
  Created = 0,
  Initialized,
  BuffersAllocated,
  InputCopied,
  KernelLoaded,
  KernargsWritten,
  Dispatched,
  ReadbackCompared,
  CleanedUp,
  Failed,
};

// Human-readable name for a lifecycle stage.
const char* lifecycle_stage_name(LifecycleStage stage);

// Standardized runtime status tokens (C0 print_*_log conventions).
inline constexpr const char* kStatusNotRun = "not_run";
inline constexpr const char* kStatusPass = "pass";
inline constexpr const char* kStatusFail = "fail";
inline constexpr const char* kStatusBlocked = "blocked";
inline constexpr const char* kFailureStageNone = "none";
inline constexpr const char* kFailureTextNone = "none";

// Serialized (CPU-side) 24-byte kernarg block. This is the frozen layout C1
// task sets 5-8 write to the compute-control buffer; it is pure host data and
// fully testable without hardware.
struct Kernargs {
  uint64_t output_va = 0;
  uint64_t input_va = 0;
  uint64_t scalar_va = 0;
  uint32_t scalar = 1U;  // actual scalar value placed at kernarg+24

  // Serialize to the exact LE layout {output_va@0, input_va@8, scalar_va@16,
  // scalar:u32@24}: `out[0..8)=output_va`, `out[8..16)=input_va`,
  // `out[16..24)=scalar_va`, `out[24..28)=scalar`. `out_capacity` must be at
  // least kKernargScalarOffset + sizeof(uint32_t).
  void encode(uint8_t* out, size_t out_capacity) const;

  // Verify a previously-encoded block matches this layout. Returns false with
  // a message on mismatch (CPU-side self-check, no hardware).
  bool verify(const uint8_t* data, size_t size, std::string* error_text) const;
};

// Standardized run-log fields (C1 contract: timestamped path under logs/,
// command line, substrate, device identity, build metadata, input/output
// digest, exit status, failure text).
struct RuntimeLog {
  std::string timestamp_utc = "not_run";
  std::string command_line = "not_run";
  std::string log_path = "not_run";
  std::string socket_path = "not_run";
  std::string runtime_substrate = kRuntimeSubstrate;
  std::string pci_id = "unknown";
  std::string arch = "not_discovered";
  std::string arch_discovery_status = kStatusNotRun;
  std::string build_metadata = "not_run";
  std::string input_digest = "not_run";
  std::string output_digest = "not_run";

  // Per-stage status (C0 *_status conventions).
  std::string connect_status = kStatusNotRun;
  std::string bar_map_status = kStatusNotRun;
  std::string sdma_h2d_status = kStatusNotRun;
  std::string kernel_blob_load_status = kStatusNotRun;
  std::string kernarg_write_status = kStatusNotRun;
  std::string kernel_launch_status = kStatusNotRun;
  std::string sdma_d2h_status = kStatusNotRun;
  std::string cpu_comparison_status = kStatusNotRun;
  std::string host_device_transfer_status = kStatusNotRun;

  std::string failure_stage = kFailureStageNone;

  std::string failure_text = kFailureTextNone;
  int exit_status = 1;

  // Current lifecycle position.
  LifecycleStage stage = LifecycleStage::Created;
};

struct TransferRoundTripResult {
  uint64_t byte_count = 0;
  uint64_t chunk_size_bytes = kTransferProofChunkByteCount;
  uint64_t chunk_count = 0;
  std::string bridge_source;
  std::string bridge_build_command;
  std::string bridge_build_output;
  std::string bridge_command;
  std::string bridge_output;
};

// Narrow native-worker boundary. The runner validates command-line JSON before
// constructing this request; the worker revalidates it before any device work.
struct NativePrefillRequest {
  std::string model_dir;
  std::vector<uint32_t> token_ids;
  std::string out_npz_path;
  std::string log_path;
  bool gpu_stage_profile = false;
  ComputeCompletionPolicy compute_completion_policy =
      ComputeCompletionPolicy::PerStageTimeline;
  ComputeBarrierPolicy compute_barrier_policy = ComputeBarrierPolicy::Full;
  uint32_t block_tokens = 1;
};

struct NativePrefillResult {
  std::string producer_kind = "r9700_native";
  std::string native_prefill_acceptance = "open";
  std::string prefill_npz_path;
  std::string hardware_log_path;
  uint64_t kernel_count = 0;
  uint32_t block_tokens = 1;
  uint32_t block_count = 0;
  uint64_t transfer_bytes = 0;
  // These fields make a non-accepting result actionable without attributing
  // fixture or retired-diagnostic work to the native prefill producer.
  std::string native_prefill_full_layer_loop_status = "blocked";
  std::string native_prefill_blocker_source;
  std::string failure_stage;
  std::string failure_text;
  // Exclusive leaf timing from the resident session plus inclusive top-level
  // host phases; the runner finalizes wall-time attribution before rendering.
  PhaseTimers phase_timers;
  // End-to-end native prefill wall time measured by the runner (usec).
  uint64_t wall_usec = 0;
  // Number of prompt tokens prefilled (S-1 cache prefix length).
  uint32_t n_prefix = 0;
  std::array<uint64_t, 10> gpu_stage_tick_total{};
  std::array<uint64_t, 10> gpu_stage_tick_min{};
  std::array<uint64_t, 10> gpu_stage_tick_max{};
  uint64_t gpu_stage_profile_sample_count = 0;
  int exit_status = 1;
};

// Runs the fail-closed native prefill worker. It validates request and output
// ownership before reporting the first unimplemented production layer-execution
// prerequisite. It never creates an NPZ until all 16 layers are dispatched
// against model weights and read back into the atomic fp16 output.
int run_native_prefill(const NativePrefillRequest& request, NativePrefillResult* result,
                       std::string* error_text);

// Request-scoped numerical diagnostic for one layer-0/token-0 resident Llama
// boundary. This is intentionally separate from NativePrefillRequest: no trace
// invocation can select an NPZ path or become an accepted cache producer.
struct LlamaStageTraceRequest {
  std::string model_dir;
  uint32_t token_id = 0;
  uint32_t layer_index = 0;
  uint32_t position = 0;
  std::string stage;
  std::string trace_dir;
  // Diagnostic-only override: replace the resident RMSNorm scale upload with
  // 2048 F16 1.0 values. It is accepted only at the normalized trace boundary.
  bool rmsnorm_unit_scale = false;
  // Diagnostic-only override: replace the resident RMSNorm input upload with
  // 2048 F16 0.0 values. It requires the normalized unit-scale trace probe.
  bool rmsnorm_zero_input = false;
  // Diagnostic-only output-store probe: initialize the normalized output with
  // 2048 F16 1.0 values before dispatch. It requires the zero-input,
  // unit-scale normalized RMSNorm trace probe.
  bool rmsnorm_output_sentinel = false;
  // Diagnostic-only code override: replace only trace RMSNorm stage 0 with
  // the ABI-compatible zero-store asset. It requires the fully constrained
  // zero-input, unit-scale, sentinel normalized trace probe.
  bool rmsnorm_zero_store = false;
  // Diagnostic-only code override: replace only trace RMSNorm stage 0 with
  // the ABI-compatible epsilon/sqrt/reciprocal arithmetic probe. It requires
  // the fully constrained zero-input, unit-scale, sentinel normalized trace
  // probe and is mutually exclusive with the zero-store diagnostic.
  bool rmsnorm_epsilon_arithmetic = false;

};

struct LlamaStageTraceResult {
  uint32_t token_index = 0;
  uint32_t layer_index = 0;
  std::string stage;
  std::string buffer;
  std::string shape_json;
  std::string dtype;
  uint64_t byte_count = 0;
  std::string sha256;
  uint64_t finite_count = 0;
  std::string raw_path;
  std::string json_path;
  std::string kernarg_hex;
  std::string hsa_image_sha256;
  uint64_t gpu_va = 0;
  std::string rmsnorm_kernel = "llama_rmsnorm_f16";
  std::string scalars_json;
  std::string scale_source = "model_f16";
  std::string input_source = "model_f16";
  std::string output_initialization = "none";
  // Metadata for the epsilon arithmetic probe's repeated expected F16 value.
  std::string rmsnorm_expected_output = "none";
  std::string failure_stage = "not_run";
  std::string failure_text = "not_run";
  int exit_status = 1;
};

// Dispatches only the prefix ending at the requested shared boundary, reads
// back that boundary's sole declared resident buffer, and atomically emits raw
// bytes plus JSON beneath trace_dir only for finite output. A non-finite
// dispatched output atomically emits metadata-only failure JSON and never
// invokes prefill serialization.
int run_llama_stage_trace(const LlamaStageTraceRequest& request,
                          LlamaStageTraceResult* result,
                          std::string* error_text);

// Selected-row hardware slice: it proves exactly one model-sourced F16
// embedding row transfers through the generated HSA image. It is intentionally
// not a prefill or model-forward acceptance path.
struct LlamaEmbedSmokeRequest {
  std::string model_dir;
  uint32_t token_id = 0;
};

struct LlamaEmbedSmokeResult {
  std::string model_identity;
  uint32_t token_id = 0;
  uint64_t model_token_count = 0;
  std::string binder_span_path;
  uint64_t binder_span_offset_bytes = 0;
  uint64_t binder_span_byte_count = 0;
  std::string hsa_image_sha256;
  uint64_t hsa_image_gpu_va = 0;
  uint64_t hsa_image_physical_offset = 0;
  uint64_t hsa_image_entry_offset = 0;
  uint64_t hsa_image_descriptor_offset = 0;
  uint64_t hsa_image_size = 0;
  uint64_t embedding_row_gpu_va = 0;
  uint64_t embedding_row_physical_offset = 0;
  uint64_t hidden_output_gpu_va = 0;
  uint64_t hidden_output_physical_offset = 0;
  uint64_t selected_row_gpu_va = 0;
  uint64_t selected_row_physical_offset = 0;
  uint64_t dynamic_ptb_count = 0;
  uint64_t dynamic_ptb_physical_offset = 0;
  uint64_t page_table_pool_base = 0;
  uint64_t page_table_pool_bytes = 0;
  uint64_t payload_allocation_range_start = 0;
  uint64_t payload_allocation_range_end = 0;
  std::string hardware_identity;
  std::string kernarg_hex = "not_run";
  uint64_t pm4_dispatch_word_count = 0;
  std::string pm4_dispatch_digest = "not_run";
  uint64_t pm4_dispatch_count = 0;
  std::string bar0_hsa_image_readback_status = "not_run";
  std::string resident_buffer_zero_status = "not_run";
  std::string sdma_h2d_status = "not_run";
  std::string sdma_d2h_status = "not_run";
  std::string fp16_row_hidden_byte_equality = "not_run";
  std::string failure_stage = "not_run";
  std::string failure_text = "not_run";
  int exit_status = 1;
};

int run_llama_embed_smoke(const LlamaEmbedSmokeRequest& request,
                          LlamaEmbedSmokeResult* result,
                          std::string* out_text,
                          std::string* log_path,
                          std::string* error_text);


// Runtime session owning the lifecycle + log. Each public stage advances the
// lifecycle state machine; calling out of order or twice fails loudly.
// `dry_run()` exercises the full lifecycle contract without a TinyGPU socket
// (no hardware) and writes a log under logs/. The session holds no host
// resources here — TinyGPU connect/BAR/SDMA/doorbell are deferred hardware gates.
class RuntimeSession {
 public:
  RuntimeSession();
  ~RuntimeSession();

  RuntimeSession(const RuntimeSession&) = delete;
  RuntimeSession& operator=(const RuntimeSession&) = delete;

  RuntimeLog& log() { return log_; }
  const RuntimeLog& log() const { return log_; }
  LifecycleStage stage() const { return stage_; }

  // --- Lifecycle stages (hardware-free state-machine advances; each records
  // its intended effect and validates ordering. Real hardware mechanics are
  // deferred gates for C1 task sets 5-8.) ---
  bool initialize(const std::string& socket_path, std::string* error_text);
  bool allocate_buffers(std::string* error_text);
  bool copy_input(const std::vector<uint8_t>& input, std::string* error_text);
  bool load_kernel(std::string* error_text);
  bool write_kernargs(const Kernargs& kernargs, std::string* error_text);
  bool dispatch_and_poll(const std::vector<uint32_t>& dispatch_words, std::string* error_text);
  bool readback_and_compare(const std::vector<uint8_t>& expected, std::string* error_text);
  void cleanup();

  // --- Hardware-free lifecycle contract exercise ---
  // Runs every stage in order against the in-memory contract (kernarg layout,
  // packet encodings, lifecycle ordering, log writing) with no TinyGPU socket.
  // Writes a timestamped log under logs/ and returns the log path.
  int dry_run(std::string* out_text, std::string* log_path);
  // --- Hardware proof bridge ---
  // Builds/runs the frozen C0A25 proof, or runs the executable named by
  // NATIVE_R9700_C0_PROBE for tests. The output is wrapped with
  // `producer_kind: hardware_probe` and written under logs/.
  int kernel_proof(std::string* out_text, std::string* log_path);
  // Builds/runs the C1R-4 streaming transfer bridge, or runs the executable
  // named by NATIVE_R9700_C1_TRANSFER_BRIDGE for tests. The wrapper validates
  // the full transfer pass marker set before reporting success.
  int transfer_proof(uint64_t byte_count, std::string* out_text, std::string* log_path);
  // Reusable C1R-4 transfer manager entry point for caller-owned bytes. The
  // current implementation builds/runs the same C0-backed bridge as
  // `transfer_proof`, but the API accepts input bytes and returns downloaded
  // bytes so later primitive/layer code does not depend on the proof pattern.
  int transfer_round_trip_bytes(const std::vector<uint8_t>& input, std::vector<uint8_t>* output,
                                TransferRoundTripResult* result, std::string* error_text);

  // Executes one direct resident-VRAM vector-add smoke through this session.
  // It is intentionally not reachable from any native-prefill producer route.
  int vram_smoke(std::string* out_text, std::string* log_path);
  // Runs only an explicitly injected historical primitive executable for
  // diagnostic comparison. This route is not a product proof and is unavailable
  // without NATIVE_R9700_C1_PRIMITIVE_BRIDGE.
  int legacy_primitive_diagnostic(const std::string& primitive_name, std::string* out_text,
                                  std::string* log_path);


 private:
  bool transition_to(LifecycleStage expected, LifecycleStage next, std::string* error_text);
  void fail_log(const std::string& stage, const std::string& text);

  RuntimeLog log_;
  LifecycleStage stage_ = LifecycleStage::Created;
};

// Writes a timestamped standardized log to `logs/<name>` and returns the path.
// `name` must not contain a path separator. All log fields are emitted in the
// C0 `key: value` convention (runtime_substrate, pci_id, arch, *_status,
// failure_stage, exit_status, ...).
std::string write_run_log(const RuntimeLog& log, const std::string& name);

}  // namespace native_r9700

#endif  // NATIVE_R9700_RUNTIME_H_

// native_r9700/amdev_packets.cpp — pure C0-compatible AMDev packet encoders.
#include "amdev_packets.h"

#include <array>
#include <initializer_list>

namespace native_r9700 {
namespace {

// SDMA0 queue-0 linear-copy packet + trailing fence, byte-faithfully ported
// from the C0 probe build_sdma_linear_copy_packet / build_sdma_fence_packet
// (probe L855-886, namespace am_sdma).
constexpr uint32_t kSdmaOpCopy = 1U;
constexpr uint32_t kSdmaSubopCopyLinear = 0U;
constexpr uint32_t kSdmaFenceHeader = 0x00030005U;  // am_sdma::kFenceHeader
constexpr uint32_t kSdmaLinearCopyPacketDwords = 7U;  // am_sdma::kSdmaLinearCopyPacketDwords
constexpr uint32_t kFencePacketDwords = 4U;           // am_sdma::kFencePacketDwords

// PM4 dispatch packet constants, byte-faithfully ported from the C0 probe
// namespace am_compute (probe L336-395, 571-605, 623-660).
constexpr uint32_t kPm4DispatchDwordCount = 59U;   // am_compute::kPm4DispatchDwordCount
constexpr uint32_t kPacketType3 = 3U;              // am_compute::kPacketType3
constexpr uint32_t kPacket3DispatchDirect = 0x15U;
constexpr uint32_t kPacket3EventWrite = 0x46U;
constexpr uint32_t kPacket3ReleaseMem = 0x49U;
constexpr uint32_t kPacket3AcquireMem = 0x58U;
constexpr uint32_t kPacket3SetShReg = 0x76U;
constexpr uint32_t kEventTypeCsPartialFlush = 7U;
constexpr uint32_t kEventIndexPartialFlush = 4U;
constexpr uint32_t kEventTypeCacheFlushAndInvTs = 20U;
constexpr uint32_t kReleaseMemEventIndexEndOfPipe = 5U;
constexpr uint32_t kReleaseMemDataSelSend32BitLow = 1U;
constexpr uint32_t kReleaseMemIntSelNone = 0U;
constexpr uint32_t kAcquireMemGcrCntlGliInvShift = 0U;   // instruction-cache invalidate
constexpr uint32_t kAcquireMemGcrCntlGlmWbShift = 4U;
constexpr uint32_t kAcquireMemGcrCntlGlmInvShift = 5U;
constexpr uint32_t kAcquireMemGcrCntlGlkWbShift = 6U;
constexpr uint32_t kAcquireMemGcrCntlGlkInvShift = 7U;
constexpr uint32_t kAcquireMemGcrCntlGlvInvShift = 8U;
constexpr uint32_t kAcquireMemGcrCntlGl1InvShift = 9U;
constexpr uint32_t kAcquireMemGcrCntlGl2InvShift = 14U;  // L2 invalidate
constexpr uint32_t kAcquireMemGcrCntlGl2WbShift = 15U;   // L2 writeback
constexpr uint32_t kReleaseMemGcrGlmWb = 1U << 12;
constexpr uint32_t kReleaseMemGcrGlmInv = 1U << 13;
constexpr uint32_t kReleaseMemGcrGlvInv = 1U << 14;
constexpr uint32_t kReleaseMemGcrGl1Inv = 1U << 15;
constexpr uint32_t kReleaseMemGcrGl2Inv = 1U << 20;
constexpr uint32_t kReleaseMemGcrGl2Wb = 1U << 21;
constexpr uint32_t kReleaseMemGcrSeq = 1U << 22;
constexpr uint32_t kReleaseMemTimelineValue = 1U;  // am_compute::kReleaseMemTimelineValue

// Compute SET_SH_REG register offsets (probe L336-344); note the C0 segment
// base kComputeSetShBase = 0x00002c00U is not needed here because the encoder
// emits the absolute offsets directly.
constexpr uint32_t kComputeStartXSetShOffset = 0x00000204U;
constexpr uint32_t kComputePgmLoSetShOffset = 0x0000020cU;
constexpr uint32_t kComputePgmRsrc1SetShOffset = 0x00000212U;
constexpr uint32_t kComputeResourceLimitsSetShOffset = 0x00000215U;
constexpr uint32_t kComputeTmpringSizeSetShOffset = 0x00000218U;
constexpr uint32_t kComputeRestartXSetShOffset = 0x0000021bU;
constexpr uint32_t kComputePgmRsrc3SetShOffset = 0x00000228U;
constexpr uint32_t kComputeUserData0SetShOffset = 0x00000240U;

// Kernel descriptor resource granules (probe L171-174).
constexpr uint32_t kKernelReferenceRsrc1 = 0xc00c0040U;
constexpr uint32_t kKernelReferenceRsrc2 = 0x00000084U;
constexpr uint32_t kKernelReferenceRsrc3 = 0x00000010U;

// Dispatch: one workgroup of 8 lanes so v[0] = 0..7 (C0A24/C0A25 contract).
constexpr uint32_t kDispatchGlobalSizeX = 1U;
constexpr uint32_t kDispatchGlobalSizeY = 1U;
constexpr uint32_t kDispatchGlobalSizeZ = 1U;
constexpr uint32_t kDispatchLocalSizeX = 8U;
constexpr uint32_t kDispatchLocalSizeY = 1U;
constexpr uint32_t kDispatchLocalSizeZ = 1U;

uint32_t lo32_impl(uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffULL); }
uint32_t hi32_impl(uint64_t v) { return static_cast<uint32_t>((v >> 32) & 0xffffffffULL); }

// Byte-faithful ports of the C0 probe packet builders (probe L855-886, 571-611).
std::array<uint32_t, 7> build_sdma_linear_copy_packet(uint64_t src_va, uint64_t dst_va,
                                                      uint32_t byte_count) {
  return {{kSdmaOpCopy | (kSdmaSubopCopyLinear << 8), byte_count - 1U, 0U,
           static_cast<uint32_t>(src_va & 0xffffffffULL),
           static_cast<uint32_t>(src_va >> 32),
           static_cast<uint32_t>(dst_va & 0xffffffffULL),
           static_cast<uint32_t>(dst_va >> 32)}};
}

std::array<uint32_t, 4> build_sdma_fence_packet(uint64_t fence_va, uint32_t value) {
  return {{kSdmaFenceHeader, static_cast<uint32_t>(fence_va & 0xffffffffULL),
           static_cast<uint32_t>(fence_va >> 32), value}};
}

uint32_t pm4_packet3(uint32_t opcode, uint32_t count) {
  return (kPacketType3 << 30) | ((opcode & 0xffU) << 8) | ((count & 0x3fffU) << 16);
}

uint32_t encode_dispatch_initiator(bool wave32) {
  return (1U << 0) | (1U << 2) | (wave32 ? (1U << 15) : 0U);
}

uint32_t encode_acquire_mem_gcr_cntl_for_dispatch() {
  return (1U << kAcquireMemGcrCntlGliInvShift) | (1U << kAcquireMemGcrCntlGlmWbShift) |
         (1U << kAcquireMemGcrCntlGlmInvShift) | (1U << kAcquireMemGcrCntlGlkWbShift) |
         (1U << kAcquireMemGcrCntlGlkInvShift) | (1U << kAcquireMemGcrCntlGlvInvShift) |
         (1U << kAcquireMemGcrCntlGl1InvShift) | (1U << kAcquireMemGcrCntlGl2InvShift) |
         (1U << kAcquireMemGcrCntlGl2WbShift);
}

uint32_t encode_event_write_cs_partial_flush() {
  return kEventTypeCsPartialFlush | (kEventIndexPartialFlush << 8);
}

uint32_t encode_release_mem_event() {
  return kEventTypeCacheFlushAndInvTs | (kReleaseMemEventIndexEndOfPipe << 8) |
         kReleaseMemGcrGlmWb | kReleaseMemGcrGlmInv | kReleaseMemGcrGlvInv |
         kReleaseMemGcrGl1Inv | kReleaseMemGcrGl2Inv | kReleaseMemGcrGl2Wb |
         kReleaseMemGcrSeq;
}

uint32_t encode_release_mem_data_sel() {
  return kReleaseMemDataSelSend32BitLow << 29 | kReleaseMemIntSelNone << 24;
}

void append_pm4_packet3(std::vector<uint32_t>* words, uint32_t opcode,
                        std::initializer_list<uint32_t> payload) {
  words->push_back(pm4_packet3(opcode, static_cast<uint32_t>(payload.size() - 1U)));
  words->insert(words->end(), payload.begin(), payload.end());
}

}  // namespace

std::vector<uint32_t> build_sdma_copy_words(uint64_t src_va, uint64_t dst_va,
                                            uint32_t byte_count, uint64_t fence_va,
                                            uint32_t fence_value) {
  // C0 build_sdma_copy_submit_words (probe L873-885): copy packet + fence, with
  // the fence value passed explicitly (C0 uses am_sdma::kFenceValue = 1U).
  std::vector<uint32_t> words;
  words.reserve(kSdmaLinearCopyPacketDwords + kFencePacketDwords);
  const std::array<uint32_t, 7> copy = build_sdma_linear_copy_packet(src_va, dst_va, byte_count);
  const std::array<uint32_t, 4> fence = build_sdma_fence_packet(fence_va, fence_value);
  words.insert(words.end(), copy.begin(), copy.end());
  words.insert(words.end(), fence.begin(), fence.end());
  return words;
}

std::vector<uint32_t> build_pm4_dispatch_words(const Pm4DispatchConfig& config) {
  // C0 build_compute_dispatch_words (probe L623-660), parameterized only at
  // the documented program-resource and launch-geometry fields.
  std::vector<uint32_t> words;
  words.reserve(kPm4DispatchDwordCount);
  append_pm4_packet3(&words, kPacket3AcquireMem,
                     {0U, 0xffffffffU, 0xffffffffU, 0U, 0U, 0U,
                      encode_acquire_mem_gcr_cntl_for_dispatch()});
  const uint64_t code_addr = config.code_va >> 8;
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputePgmLoSetShOffset, lo32_impl(code_addr), hi32_impl(code_addr)});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputePgmRsrc1SetShOffset, config.rsrc1, config.rsrc2});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputePgmRsrc3SetShOffset, config.rsrc3});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputeTmpringSizeSetShOffset, 0U});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputeRestartXSetShOffset, 0U, 0U, 0U});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputeUserData0SetShOffset, lo32_impl(config.kernargs_va),
                      hi32_impl(config.kernargs_va)});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputeResourceLimitsSetShOffset, 0U});
  append_pm4_packet3(&words, kPacket3SetShReg,
                     {kComputeStartXSetShOffset, 0U, 0U, 0U, config.workgroup_x,
                      config.workgroup_y, config.workgroup_z, 0U, 0U});
  append_pm4_packet3(&words, kPacket3DispatchDirect,
                     {config.global_x, config.global_y, config.global_z,
                      encode_dispatch_initiator(config.wave32)});
  append_pm4_packet3(&words, kPacket3EventWrite, {encode_event_write_cs_partial_flush()});
  append_pm4_packet3(&words, kPacket3ReleaseMem,
                     {encode_release_mem_event(), encode_release_mem_data_sel(),
                      lo32_impl(config.timeline_va), hi32_impl(config.timeline_va),
                      kReleaseMemTimelineValue, 0U, 0U});
  return words;
}

std::vector<uint32_t> build_pm4_dispatch_words(uint64_t code_va, uint64_t kernargs_va,
                                               uint64_t timeline_va) {
  return build_pm4_dispatch_words(
      {code_va, kernargs_va, timeline_va, kKernelReferenceRsrc1, kKernelReferenceRsrc2,
       kKernelReferenceRsrc3, false, kDispatchLocalSizeX, kDispatchLocalSizeY, kDispatchLocalSizeZ,
       kDispatchGlobalSizeX, kDispatchGlobalSizeY, kDispatchGlobalSizeZ});
}

}  // namespace native_r9700

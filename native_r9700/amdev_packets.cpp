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
constexpr uint32_t kReleaseMemDataSelNone = 0U;
constexpr uint32_t kReleaseMemDataSelSend32BitLow = 1U;
// tinygrad/runtime/autogen/am/pm4_soc15.py:32 generated gfx12 selector.
constexpr uint32_t kReleaseMemDataSelSendGpuClockCounter = 3U;
constexpr uint32_t kReleaseMemIntSelNone = 0U;
// tinygrad/runtime/autogen/am/pm4_soc15.py:31; used by ops_amd.py:374.
constexpr uint32_t kReleaseMemIntSelSendInterruptAfterWriteConfirm = 2U;
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

uint32_t encode_release_mem_event_without_cache_flush() {
  return kEventTypeCacheFlushAndInvTs | (kReleaseMemEventIndexEndOfPipe << 8);
}

uint32_t encode_release_mem_event() {
  return encode_release_mem_event_without_cache_flush() | kReleaseMemGcrGlmWb |
         kReleaseMemGcrGlmInv | kReleaseMemGcrGlvInv | kReleaseMemGcrGl1Inv |
         kReleaseMemGcrGl2Inv | kReleaseMemGcrGl2Wb | kReleaseMemGcrSeq;
}

uint32_t encode_release_mem_data_control(uint32_t data_sel, uint32_t int_sel) {
  return data_sel << 29 | int_sel << 24;
}

uint32_t encode_release_mem_data_sel() {
  return encode_release_mem_data_control(kReleaseMemDataSelSend32BitLow,
                                         kReleaseMemIntSelNone);
}

uint32_t encode_release_mem_data_sel_none() {
  return encode_release_mem_data_control(kReleaseMemDataSelNone, kReleaseMemIntSelNone);
}

uint32_t encode_release_mem_gpu_clock_data_sel() {
  return encode_release_mem_data_control(kReleaseMemDataSelSendGpuClockCounter,
                                         kReleaseMemIntSelNone);
}

uint32_t encode_release_mem_ordering_data_sel() {
  return encode_release_mem_data_control(
      kReleaseMemDataSelNone, kReleaseMemIntSelSendInterruptAfterWriteConfirm);
}

void append_pm4_packet3(std::vector<uint32_t>* words, uint32_t opcode,
                        std::initializer_list<uint32_t> payload) {
  words->push_back(pm4_packet3(opcode, static_cast<uint32_t>(payload.size() - 1U)));
  words->insert(words->end(), payload.begin(), payload.end());
}

void append_pm4_acquire_mem(std::vector<uint32_t>* words) {
  append_pm4_packet3(words, kPacket3AcquireMem,
                     {0U, 0xffffffffU, 0xffffffffU, 0U, 0U, 0U,
                      encode_acquire_mem_gcr_cntl_for_dispatch()});
}

void append_pm4_dispatch_body(std::vector<uint32_t>* words,
                              const Pm4DispatchConfig& config) {
  append_pm4_acquire_mem(words);
  const uint64_t code_addr = config.code_va >> 8;
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputePgmLoSetShOffset, lo32_impl(code_addr), hi32_impl(code_addr)});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputePgmRsrc1SetShOffset, config.rsrc1, config.rsrc2});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputePgmRsrc3SetShOffset, config.rsrc3});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputeTmpringSizeSetShOffset, 0U});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputeRestartXSetShOffset, 0U, 0U, 0U});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputeUserData0SetShOffset, lo32_impl(config.kernargs_va),
                      hi32_impl(config.kernargs_va)});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputeResourceLimitsSetShOffset, 0U});
  append_pm4_packet3(words, kPacket3SetShReg,
                     {kComputeStartXSetShOffset, 0U, 0U, 0U, config.workgroup_x,
                      config.workgroup_y, config.workgroup_z, 0U, 0U});
  append_pm4_packet3(words, kPacket3DispatchDirect,
                     {config.global_x, config.global_y, config.global_z,
                      encode_dispatch_initiator(config.wave32)});
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
  return build_pm4_dispatch_words(config, Pm4StageTail{});
}

std::vector<uint32_t> build_pm4_dispatch_words(const Pm4DispatchConfig& config,
                                               const Pm4StageTail& tail) {
  std::vector<uint32_t> words;
  words.reserve(kPm4DispatchDwordCount);
  append_pm4_dispatch_body(&words, config);
  if (tail.emit_cs_partial_flush) {
    append_pm4_packet3(&words, kPacket3EventWrite,
                       {encode_event_write_cs_partial_flush()});
  }
  if (tail.emit_cache_release) {
    append_pm4_packet3(&words, kPacket3ReleaseMem,
                       {encode_release_mem_event(),
                        tail.write_timeline ? encode_release_mem_data_sel()
                                            : encode_release_mem_data_sel_none(),
                        tail.write_timeline ? lo32_impl(config.timeline_va) : 0U,
                        tail.write_timeline ? hi32_impl(config.timeline_va) : 0U,
                        tail.write_timeline ? config.timeline_value : 0U,
                        0U, 0U});
  }
  return words;
}

std::vector<uint32_t> build_pm4_gpu_timestamp_words(uint64_t timestamp_va) {
  std::vector<uint32_t> words;
  words.reserve(24U);
  append_pm4_packet3(&words, kPacket3ReleaseMem,
                     {encode_release_mem_event_without_cache_flush(),
                      encode_release_mem_ordering_data_sel(),
                      0U, 0U, 0U, 0U, 0U});
  append_pm4_packet3(&words, kPacket3ReleaseMem,
                     {encode_release_mem_event_without_cache_flush(),
                      encode_release_mem_gpu_clock_data_sel(),
                      lo32_impl(timestamp_va), hi32_impl(timestamp_va),
                      0U, 0U, 0U});
  append_pm4_acquire_mem(&words);
  return words;
}

std::vector<uint32_t> build_pm4_timeline_signal_words(uint64_t timeline_va,
                                                      uint32_t timeline_value) {
  std::vector<uint32_t> words;
  words.reserve(8U);
  append_pm4_packet3(&words, kPacket3ReleaseMem,
                     {encode_release_mem_event(), encode_release_mem_data_sel(),
                      lo32_impl(timeline_va), hi32_impl(timeline_va),
                      timeline_value, 0U, 0U});
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

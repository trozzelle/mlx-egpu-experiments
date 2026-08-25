// native_r9700/amdev_packets.h — pure C0-compatible AMDev packet encoders.
#ifndef NATIVE_R9700_AMDEV_PACKETS_H_
#define NATIVE_R9700_AMDEV_PACKETS_H_

#include <cstdint>
#include <vector>

namespace native_r9700 {

// C0 SDMA linear-copy packet plus a caller-addressed fence (11 dwords total).
std::vector<uint32_t> build_sdma_copy_words(uint64_t src_va, uint64_t dst_va,
                                            uint32_t byte_count, uint64_t fence_va,
                                            uint32_t fence_value);

// PM4 values selected by a reviewed resident-kernel asset. The C0 packet
// layout is fixed; only program resources and dispatch geometry vary.
struct Pm4DispatchConfig {
  uint64_t code_va = 0;
  uint64_t kernargs_va = 0;
  uint64_t timeline_va = 0;
  uint32_t rsrc1 = 0;
  uint32_t rsrc2 = 0;
  uint32_t rsrc3 = 0;
  bool wave32 = false;  // sets COMPUTE_DISPATCH_INITIATOR.CS_W32_EN (bit 15)
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t global_x = 0;
  uint32_t global_y = 0;
  uint32_t global_z = 0;
  uint32_t timeline_value = 1;  // monotonic RELEASE_MEM write payload (was fixed kReleaseMemTimelineValue)
};

struct Pm4StageTail {
  bool emit_cs_partial_flush = true;
  bool emit_cache_release = true;
  bool write_timeline = true;
};

// C0A25 compute dispatch stream: 12 packets and 59 dwords total. This is the
// reusable form consumed by AMDevSession's physical resident-dispatch path.
std::vector<uint32_t> build_pm4_dispatch_words(const Pm4DispatchConfig& config);
std::vector<uint32_t> build_pm4_dispatch_words(const Pm4DispatchConfig& config,
                                                const Pm4StageTail& tail);

// Pure queue-stage completion encoders. The timestamp stream orders prior GPU
// work, writes the GPU clock to timestamp_va, then makes that write visible.
std::vector<uint32_t> build_pm4_gpu_timestamp_words(uint64_t timestamp_va);
std::vector<uint32_t> build_pm4_timeline_signal_words(uint64_t timeline_va,
                                                       uint32_t timeline_value);

// Frozen C0 add-one packet stream retained as the proof-packet contract.
std::vector<uint32_t> build_pm4_dispatch_words(uint64_t code_va, uint64_t kernargs_va,
                                                uint64_t timeline_va);

}  // namespace native_r9700

#endif  // NATIVE_R9700_AMDEV_PACKETS_H_

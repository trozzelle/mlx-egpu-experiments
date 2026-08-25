#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "amdev_packets.h"
#include "hsa_code_image_asset.h"
#include "kernel_catalog.h"
namespace native_r9700 {
struct VramLayout;

struct RpcOperationTiming {
  uint64_t count = 0;
  uint64_t usec = 0;
};

// Per-phase wall-clock timing (usec) + submission counters accumulated by a
// resident session. Exposed so the native-prefill result can report the
// launch/transport breakdown instead of only the aggregate wall time.
struct PhaseTimers {
  long model_load_usec = 0;
  long staging_copy_usec = 0;
  long sdma_setup_usec = 0;
  long sdma_submit_inclusive_usec = 0;
  long sdma_fence_wait_usec = 0;
  long sdma_submit_exclusive_usec = 0;
  long model_bind_inclusive_usec = 0;
  long dispatch_build_inclusive_usec = 0;
  long device_prepare_inclusive_usec = 0;
  long embedding_upload_inclusive_usec = 0;
  long weight_upload_inclusive_usec = 0;
  long compute_loop_inclusive_usec = 0;
  long kv_readback_inclusive_usec = 0;
  long session_close_inclusive_usec = 0;
  long npz_serialization_inclusive_usec = 0;
  uint64_t measured_exclusive_total_usec = 0;
  uint64_t unattributed_usec = 0;
  long pm4_build_usec = 0;
  long hdp_flush_usec = 0;
  long doorbell_usec = 0;
  long timeline_wait_usec = 0;
  uint64_t sdma_setup_count = 0;
  uint64_t compute_submit_count = 0;
  uint64_t socket_rpc_count = 0;
  std::array<RpcOperationTiming, 14> rpc_operations{};
};

void finalize_phase_accounting(uint64_t wall_usec, PhaseTimers* timers);


// A bounded C0-compatible launch: one input page, one output page, one code
// page, and one kernarg page. It becomes physical only through
// dispatch_resident_kernel; the in-memory allocation API is not used here.
//
// The fixed pages, BAR0 code write/readback, SDMA upload/readback, compute-ring
// submit, and timeline polls are ported directly from the successful C0A25
// kernel-proof sequence in native_amdev_transfer_probe.cpp:6339-6609. This
// request supplies only reviewed asset data and launch-specific byte spans.
struct ResidentKernelDispatch {
  KernelDescriptor kernel;
  std::vector<uint8_t> kernargs;
  std::vector<uint8_t> input_bytes;
  uint32_t output_byte_count = 0;
};

struct ResidentKernelDispatchResult {
  std::vector<uint8_t> output_bytes;
  uint64_t transfer_bytes = 0;
  std::string hardware_identity;
  std::string failure_stage;
};


// A raw-byte HSA dispatch through dynamically resident VRAM. Buffer uploads
// are copied verbatim; this layer neither interprets tensor data nor performs
// model math. Each binding writes the resident GPU VA of one buffer into the
// caller-supplied kernarg byte layout.
struct ResidentHsaBuffer {
  std::string name;
  std::vector<uint8_t> upload_bytes;
  uint64_t allocation_byte_count = 0;
  uint64_t readback_byte_count = 0;
  // Only the explicit layer-local weight window may opt into raw replacement
  // uploads after prepare. Hidden state and K/V cache buffers remain resident.
  bool allow_post_prepare_upload = false;
};

struct ResidentHsaKernargBinding {
  uint32_t buffer_index = 0;
  uint32_t kernarg_byte_offset = 0;
};

// A stage reuses the dispatch's live image and buffers. Kernargs are rewritten
// in the mapped C0 page before each submission; stage dispatch never performs
// a host-side payload upload or readback.
struct ResidentHsaStage {
  uint32_t hsa_image_index = 0;
  uint64_t entry_offset = 0;
  std::vector<ResidentHsaKernargBinding> kernarg_bindings;
  std::vector<uint8_t> kernargs;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t global_x = 0;
  uint32_t global_y = 0;
  uint32_t global_z = 0;
};

struct ResidentHsaDispatch {
  const HsaCodeImageAsset* hsa_image = nullptr;
  // When nonempty, stage image indices select from this resident image table.
  // The legacy hsa_image field remains the single-image form.
  std::vector<const HsaCodeImageAsset*> hsa_images;
  std::vector<ResidentHsaBuffer> buffers;
  std::vector<ResidentHsaKernargBinding> kernarg_bindings;
  std::vector<uint8_t> kernargs;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t global_x = 0;
  uint32_t global_y = 0;
  uint32_t global_z = 0;
  // When present, stages execute against the one resident image and the same
  // named buffers in order. The legacy top-level fields describe one stage.
  std::vector<ResidentHsaStage> stages;
};

struct GpuStageTickSample {
  std::array<uint64_t, 11> boundaries{};
};

inline bool gpu_stage_tick_deltas(const GpuStageTickSample& sample,
                                  std::array<uint64_t, 10>* stage_ticks,
                                  std::string* error_text) {
  if (stage_ticks == nullptr) return false;
  if (sample.boundaries[0] == 0) {
    if (error_text != nullptr)
      *error_text = "gpu timestamp boundaries are not strictly increasing";
    return false;
  }
  for (std::size_t index = 0; index < stage_ticks->size(); ++index) {
    if (sample.boundaries[index + 1] <= sample.boundaries[index]) {
      if (error_text != nullptr)
        *error_text = "gpu timestamp boundaries are not strictly increasing";
      return false;
    }
    (*stage_ticks)[index] = sample.boundaries[index + 1] - sample.boundaries[index];
  }
  if (error_text != nullptr) error_text->clear();
  return true;
}

enum class ComputeCompletionPolicy {
  PerStageTimeline,
  TerminalTimeline,
};

enum class ComputeBarrierPolicy {
  Full,
  OverlapKvProjections,
};

inline const char* compute_completion_policy_name(
    ComputeCompletionPolicy policy) {
  return policy == ComputeCompletionPolicy::PerStageTimeline ? "per-stage"
                                                              : "terminal";
}

inline const char* compute_barrier_policy_name(ComputeBarrierPolicy policy) {
  return policy == ComputeBarrierPolicy::Full ? "full" : "overlap-kv";
}

struct ResidentHsaBatchOptions {
  bool capture_gpu_timestamps = false;
  ComputeCompletionPolicy completion_policy =
      ComputeCompletionPolicy::TerminalTimeline;
  ComputeBarrierPolicy barrier_policy = ComputeBarrierPolicy::Full;
};

inline Pm4StageTail compute_stage_tail(const ResidentHsaBatchOptions& options,
                                      std::size_t stage_index,
                                      std::size_t stage_count) {
  const bool terminal_stage = stage_index + 1U == stage_count;
  return {
      options.barrier_policy != ComputeBarrierPolicy::OverlapKvProjections ||
          stage_index != 1U,
      true,
      options.completion_policy == ComputeCompletionPolicy::PerStageTimeline ||
          (!options.capture_gpu_timestamps && terminal_stage),
  };
}

inline bool compute_batch_uses_terminal_timeline_signal(
    const ResidentHsaBatchOptions& options) {
  return options.capture_gpu_timestamps;
}

inline std::size_t compute_batch_host_signal_count(
    const ResidentHsaBatchOptions& options, std::size_t stage_count) {
  if (stage_count == 0U) return 0U;
  if (options.capture_gpu_timestamps) {
    return options.completion_policy == ComputeCompletionPolicy::PerStageTimeline
               ? stage_count + 1U
               : 1U;
  }
  return options.completion_policy == ComputeCompletionPolicy::PerStageTimeline
             ? stage_count
             : 1U;
}

struct ResidentHsaDispatchResult {
  std::string hardware_identity;
  uint64_t hsa_image_gpu_va = 0;
  uint64_t hsa_image_physical_offset = 0;
  std::vector<uint64_t> hsa_image_gpu_vas;
  std::vector<uint64_t> hsa_image_physical_offsets;
  // These parallel arrays expose the actual named resident allocations created
  // by prepare; entries retain their identity until close succeeds.
  std::vector<std::string> buffer_names;
  std::vector<uint64_t> buffer_gpu_vas;
  // Captured while the image and caller buffers are still dynamically mapped.
  uint64_t dynamic_ptb_count = 0;
  uint64_t dynamic_ptb_physical_offset = 0;
  std::vector<uint64_t> buffer_physical_offsets;
  std::vector<std::vector<uint8_t>> readback_bytes;
  uint64_t sdma_upload_bytes = 0;
  uint64_t sdma_download_bytes = 0;
  uint64_t pm4_dispatch_word_count = 0;
  std::vector<GpuStageTickSample> gpu_stage_tick_samples;

  std::string pm4_dispatch_digest = "not_run";
  uint64_t pm4_dispatch_count = 0;
  std::string failure_stage = "not_run";
};

// Keeps one dynamically mapped image table, named resident buffers, C0
// control pages, queue state, and page-table ownership live across stages.
// prepare uploads raw image/buffer bytes once; dispatch does not transfer model
// payloads; upload_named may replace only an opted-in layer-local weight window;
// readback copies only explicitly requested named buffers. close must succeed
// before ownership is released.
class ResidentHsaSession {
 public:
  ResidentHsaSession();
  ~ResidentHsaSession();

  ResidentHsaSession(const ResidentHsaSession&) = delete;
  ResidentHsaSession& operator=(const ResidentHsaSession&) = delete;
  ResidentHsaSession(ResidentHsaSession&&) = delete;
  ResidentHsaSession& operator=(ResidentHsaSession&&) = delete;

  bool prepare(const ResidentHsaDispatch& request, ResidentHsaDispatchResult* result,
               std::string* error_text);
  bool dispatch(const ResidentHsaStage& stage, ResidentHsaDispatchResult* result,
                std::string* error_text);
  // Dispatches up to kKernargSlotCount stages as one ring write, one doorbell,
  // and one terminal timeline poll. Slot i binds stage i's kernargs into the
  // in-page slot i, so the single-stage path (slot 0, timeline_value 1) is
  // byte-identical to dispatch.
  bool dispatch_batch(const std::vector<ResidentHsaStage>& stages,
                      ResidentHsaDispatchResult* result, std::string* error_text,
                      const ResidentHsaBatchOptions& options = {});
  // Reads the live compute ring rptr/wptr (in dwords) from the control page.
  // Must be called before close; close resets the control mapping.
  bool compute_ring_pointers(uint64_t* rptr_dwords, uint64_t* wptr_dwords,
                             std::string* error_text);
  // Returns the session's accumulated phase timers + submission counters.
  // Valid after prepare; reflects the completed work after close.
  const PhaseTimers& phase_timers() const;
  bool upload_named(const std::string& buffer_name, const uint8_t* bytes,
                    uint64_t byte_count, ResidentHsaDispatchResult* result,
                    std::string* error_text);
  bool readback(const std::vector<std::string>& buffer_names,
                ResidentHsaDispatchResult* result, std::string* error_text);
  bool close(std::string* error_text);

 private:
  // Per-stage PM4 transform (preflight, kernarg slot bind, build). Policy
  // selection changes only the stage tail; the dispatch body remains frozen.
  bool build_stage_pm4(const ResidentHsaStage& stage, uint32_t slot,
                       std::vector<uint32_t>* words, std::string* error_text,
                       const Pm4StageTail& tail = {});
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// A no-device allocation plan produced by the same ResidentMemory order used
// by dispatch_resident_hsa. Callers use these VAs to validate stage bindings
// before any device connection; they are not independently invented addresses.
struct ResidentHsaDispatchPlan {
  std::vector<uint64_t> hsa_image_gpu_vas;
  std::vector<uint64_t> buffer_gpu_vas;
};

bool plan_resident_hsa_dispatch(const VramLayout& layout, const ResidentHsaDispatch& request,
                                ResidentHsaDispatchPlan* plan, std::string* error_text);
// One genuine model-data embedding gather through the resident C0 VM path.
// `embedding_row` is one validated 2048-element F16 model row, never a fixture
// or a whole embedding matrix. The selected row scalar is deliberately zero:
// the host transfers precisely the selected row window and the kernel gathers
// row zero from that resident window.
struct LlamaEmbedSmokeDispatch {
  const HsaCodeImageAsset* hsa_image = nullptr;
  std::vector<uint8_t> embedding_row;
};

// Hardware evidence returned by the selected-row dispatch. Status fields retain
// their default until their corresponding side effect completes.
struct LlamaEmbedSmokeDispatchResult {
  std::string hardware_identity;
  uint64_t hsa_image_gpu_va = 0;
  uint64_t hsa_image_physical_offset = 0;
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
  std::string bar0_image_readback_status = "not_run";
  std::string resident_buffer_zero_status = "not_run";
  std::string sdma_h2d_status = "not_run";
  std::string sdma_d2h_status = "not_run";
  std::string fp16_row_hidden_byte_equality = "not_run";
  std::string kernarg_hex = "not_run";
  uint64_t pm4_dispatch_word_count = 0;
  std::string pm4_dispatch_digest = "not_run";
  uint64_t pm4_dispatch_count = 0;
  uint64_t sdma_upload_bytes = 0;
  uint64_t sdma_download_bytes = 0;
  std::string failure_stage = "not_run";
};

// Evidence from one direct, session-bound resident-VRAM vector-add dispatch.
// Values remain `not_run` until their corresponding hardware side effect has
// completed. This route is never a native-prefill acceptance signal.
struct VramSmokeResult {
  std::string pci_id = "unknown";
  std::string arch = "unknown";
  std::string source_asset_path = "not_run";
  std::string asset_sha256 = "not_run";
  uint64_t code_byte_count = 0;
  std::string bar0_code_readback_status = "not_run";
  std::string vram_allocation_status = "not_run";
  uint64_t resident_mapping_count = 0;
  uint64_t bar0_aperture_bytes = 0;
  std::string large_bar = "not_run";
  uint64_t page_table_pool_base = 0;
  uint64_t page_table_pool_bytes = 0;
  uint64_t dynamic_ptb_count = 0;
  uint64_t dynamic_ptb_physical_offset = 0;
  uint64_t payload_allocation_range_start = 0;
  uint64_t payload_allocation_range_end = 0;
  std::string mapping_uncertainty_status = "not_run";
  uint64_t a_gpu_va = 0;
  uint64_t a_physical_offset = 0;
  uint64_t b_gpu_va = 0;
  uint64_t b_physical_offset = 0;
  uint64_t out_gpu_va = 0;
  uint64_t out_physical_offset = 0;
  std::string bar0_zero_status = "not_run";
  std::string pte_map_status = "not_run";
  std::string pte_write_status = "not_run";
  std::string pte_readback_status = "not_run";
  std::string mmhub_tlb_flush_status = "not_run";
  std::string gc_tlb_flush_status = "not_run";
  uint64_t compute_dispatch_count = 0;
  std::string sdma_h2d_status = "not_run";
  std::string sdma_d2h_status = "not_run";
  uint64_t sdma_upload_bytes = 0;
  uint64_t sdma_download_bytes = 0;
  uint64_t kernarg_byte_count = 0;
  std::string kernarg_hex = "not_run";
  uint64_t pm4_dispatch_word_count = 0;
  std::string pm4_dispatch_digest = "not_run";
  std::string cpu_comparison_status = "not_run";
  std::string failure_stage = "not_run";
  std::string failure_text = "not_run";
};

// Validates a reviewed asset and its C0 bounded launch inputs before any
// TinyGPU connection or MMIO operation.
bool validate_resident_kernel_dispatch(const ResidentKernelDispatch& request,
                                       std::string* error_text);

// The bounded in-memory store is a no-hardware test seam. The physical C0
// transfer and dispatch paths never use those host-vector allocations.
class AMDevSession {
 public:
  explicit AMDevSession(uint64_t memory_limit_bytes = 64ULL * 1024ULL * 1024ULL);

  bool allocate(uint64_t size_bytes, uint64_t* gpu_va, std::string* error_text);
  bool upload(uint64_t gpu_va, const uint8_t* data, uint64_t size_bytes,
              std::string* error_text);
  bool download(uint64_t gpu_va, uint8_t* data, uint64_t size_bytes,
                std::string* error_text);
  void release(uint64_t gpu_va);

  int streaming_transfer_proof(uint64_t byte_count);
  int transfer_round_trip_file(const std::string& input_path, const std::string& output_path,
                               std::string* error_text);


  // Performs one direct C0-backed VRAM vector-add dispatch. It owns neither a
  // reusable producer route nor any model buffers.
  bool vram_smoke(VramSmokeResult* result, std::string* error_text);

  // Executes the reviewed descriptor through the C0 physical path. It never
  // interprets host-memory allocations as device execution and fails before
  // connection when preflight rejects the supplied asset or spans.
  bool dispatch_resident_kernel(const ResidentKernelDispatch& request,
                                ResidentKernelDispatchResult* result,
                                std::string* error_text);

  // Dispatches an attested full HSA image from dynamically resident VRAM. All
  // payload movement is raw SDMA byte copying; optional buffer readback is
  // explicit per buffer.
  bool dispatch_resident_hsa(const ResidentHsaDispatch& request,
                             ResidentHsaDispatchResult* result,
                             std::string* error_text);

  // Discovers the active layout and returns the exact resident allocation VAs
  // without staging payloads, programming VM state, or dispatching a kernel.
  bool plan_resident_hsa_dispatch(const ResidentHsaDispatch& request,
                                  ResidentHsaDispatchPlan* plan,
                                  std::string* error_text);
  // Runs one selected-row Llama embedding gather through resident VRAM. It
  // performs no model parsing or source compilation; callers must provide the
  // attested generated HSA image and exactly one binder-read F16 row.
  bool llama_embed_smoke(const LlamaEmbedSmokeDispatch& request,
                         LlamaEmbedSmokeDispatchResult* result,
                         std::string* error_text);

 private:
  uint64_t memory_limit_bytes_;
  uint64_t allocated_bytes_ = 0;
  uint64_t next_gpu_va_ = 0x0000200000005000ULL;
  std::unordered_map<uint64_t, std::vector<uint8_t>> allocations_;

};

}  // namespace native_r9700

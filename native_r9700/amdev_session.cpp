// native_r9700/amdev_session.cpp — source-grounded C0 AMDev session lifecycle.

// The frozen C0 probe is included once in this translation unit. This session
// owns its TinyGPU connection, BAR/VM setup, and SDMA transfer mechanics; the
// C1 bridge remains only a command-line adapter.
#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>

#include "amdev_session.h"
#include "amdev_packets.h"
#include "dynamic_page_table.h"
#include "hardware_lock.h"
#include "resident_memory.h"
#include "vram_smoke_asset.h"


#define setup_fixed_vm_mapping native_r9700_c0_setup_fixed_vm_mapping_legacy
#define main native_r9700_c0_probe_unused_main
#include "../experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp"
#undef main
#undef setup_fixed_vm_mapping

namespace {

bool flush_gc_tlb_vmid0_native(const RemoteClient& client, DiscoveryLog* log,
                               std::string* error_text, long* hdp_flush_usec = nullptr) {
  timeval hdp_start{};
  if (hdp_flush_usec != nullptr) gettimeofday(&hdp_start, nullptr);
  const bool hdp_flushed = flush_hdp(client, *log, error_text);
  if (hdp_flush_usec != nullptr) {
    timeval hdp_now{};
    gettimeofday(&hdp_now, nullptr);
    *hdp_flush_usec += (hdp_now.tv_sec - hdp_start.tv_sec) * 1000000L +
                       (hdp_now.tv_usec - hdp_start.tv_usec);
  }
  if (!hdp_flushed) {
    *error_text = "GC TLB HDP flush failed: " + *error_text;
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  if (!write_register_dword(client, *log, log->ip.gc, regs_gfx1201::kGcInvalidateEng17Req,
                            encode_invalidate_req_vmid0(), error_text)) {
    *error_text = std::string("write ") + regs_gfx1201::kGcInvalidateEng17Req.name +
                  " failed: " + *error_text;
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  if (!poll_register_mask(client, *log, log->ip.gc, regs_gfx1201::kGcInvalidateEng17Ack,
                          am_vm::kInvalidateMaskVmid0, am_vm::kInvalidateMaskVmid0,
                          regs_gfx1201::kGcInvalidateEng17Ack.name, error_text)) {
    log->vm.gc_tlb_flush_status = "fail";
    return false;
  }
  log->vm.gc_tlb_flush_status = "pass";
  return true;
}

}  // namespace


bool setup_fixed_vm_mapping(const RemoteClient& client, DiscoveryLog* log,
                            const VmBufferLog& staging, const VmBufferLog& readback,
                            const VmBufferLog& sdma_control,
                            const VmBufferLog* compute_control, bool enable_gc_hub,
                            FixedVmMappingResult* result, long* hdp_flush_usec = nullptr) {
  result->tables = log->vm.tables;
  std::string error;
  if (!(enable_gc_hub ? is_supported_gfx1201_vm_ip_layout(*log, &error)
                      : is_supported_gfx1201_ip_layout(*log, &error))) {
    log->vm.page_tables_written = "fail";
    result->error_text = error;
    return false;
  }
  if (!write_fixed_page_tables(client, log, staging, readback, sdma_control, compute_control,
                               &error)) {
    log->vm.page_tables_written = "fail";
    result->error_text = error;
    return false;
  }
  result->page_tables_written = true;
  if (!program_mmhubs_vmid0(client, log, &error)) {
    log->vm.vmid0_context_status = "fail";
    result->error_text = error;
    return false;
  }
  result->vmid0_context_programmed = true;
  if (!flush_mmhubs_tlb(client, log, &error)) {
    result->error_text = error;
    return false;
  }
  result->tlb_flushed = true;
  if (!enable_gc_hub) return true;
  if (!validate_direct_pm4_topology(*log, &error)) {
    result->failure_stage = "multi_xcc_aql_required";
    result->error_text = error;
    return false;
  }
  if (!program_gc_hub_vmid0(client, log, &error)) {
    result->failure_stage = "gc_hub_init";
    result->error_text = error;
    return false;
  }
  if (!flush_gc_tlb_vmid0_native(client, log, &error, hdp_flush_usec)) {
    result->failure_stage = "gc_tlb_flush";
    result->error_text = error;
    return false;
  }
  return true;
}
namespace native_r9700 {
namespace {


constexpr uint64_t kC1TransferChunkByteCount = kPageSize;
// The resident HSA session maps a 1 MiB staging window (256 fixed PTB
// entries) so streamed model weights pay one SDMA submission per MiB instead
// of one queue round trip per 4 KiB page.
constexpr uint64_t kResidentStagingByteCount = 256ULL * kPageSize;
constexpr uint64_t kC1MaxTransferByteCount = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t kSmokePtbBoundaryGpuVa = 0x0000200000200000ULL;
constexpr uint64_t kSmokePayloadPageCount = 4;
constexpr RegDef kCpMecRs64PrgrmCntrStart{"regCP_MEC_RS64_PRGRM_CNTR_START", 10550U, 1U};


class TerminalComputeQueue0Retirement {
 public:
  TerminalComputeQueue0Retirement(const RemoteClient& client, const DiscoveryLog& log)
      : client_(client), log_(log) {}

  void arm() { state_ = State::kRequired; }

  bool retire(std::string* error_text) {
    if (state_ == State::kNotRequired || state_ == State::kRetired) return true;
    if (state_ == State::kFailed) {
      *error_text = retirement_failure_;
      return false;
    }

    // A failed source sequence must remain terminal; retrying can touch a
    // queue whose active state was not proven clear.
    state_ = State::kFailed;
    if (!reset_compute_queue0(client_, log_, &retirement_failure_)) {
      *error_text = retirement_failure_;
      return false;
    }
    state_ = State::kRetired;
    return true;
  }

 private:
  enum class State { kNotRequired, kRequired, kRetired, kFailed };

  const RemoteClient& client_;
  const DiscoveryLog& log_;
  State state_ = State::kNotRequired;
  std::string retirement_failure_;
};


struct StreamTransferStatus {
  std::string sdma_h2d_status = "not_run";
  std::string sdma_d2h_status = "not_run";
  std::string cpu_comparison_status = "not_run";
  std::string host_device_transfer_status = "fail";
  std::string failure_stage = "not_run";
  std::string failure_text = "not_run";
  uint64_t chunks_completed = 0;
  uint64_t bytes_uploaded = 0;
  uint64_t bytes_downloaded = 0;
};


uint8_t transfer_pattern_byte(uint64_t absolute_offset) {
  // Deterministic, non-constant byte stream. Uses only low bits by design.
  return static_cast<uint8_t>(((absolute_offset * 131ULL) + 17ULL) & 0xffULL);
}

void fill_transfer_chunk(uint8_t* dst, uint64_t absolute_offset, uint64_t byte_count) {
  for (uint64_t i = 0; i < byte_count; ++i) {
    dst[i] = transfer_pattern_byte(absolute_offset + i);
  }
}

bool compare_transfer_chunk(const uint8_t* observed, uint64_t absolute_offset, uint64_t byte_count,
                            std::string* error_text) {
  for (uint64_t i = 0; i < byte_count; ++i) {
    const uint8_t expected = transfer_pattern_byte(absolute_offset + i);
    if (observed[i] != expected) {
      *error_text = "streamed transfer mismatch at absolute_offset=" +
                    std::to_string(absolute_offset + i) + " expected=0x" +
                    hex_encode_bytes(&expected, 1) + " observed=0x" +
                    hex_encode_bytes(observed + i, 1);
      return false;
    }
  }
  return true;
}

bool read_binary_file(const std::string& path, std::vector<uint8_t>* data, std::string* error_text) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    *error_text = "open " + path + " for read failed: " + std::strerror(errno);
    return false;
  }
  data->clear();
  uint8_t buf[4096];
  for (;;) {
    const ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
      if (data->size() + static_cast<size_t>(n) > kC1MaxTransferByteCount) {
        *error_text = "input file exceeds max C1R-4 transfer policy";
        close(fd);
        return false;
      }
      data->insert(data->end(), buf, buf + n);
      continue;
    }
    if (n == 0) break;
    if (errno == EINTR) continue;
    *error_text = "read " + path + " failed: " + std::strerror(errno);
    close(fd);
    return false;
  }
  close(fd);
  return true;
}

bool write_binary_file(const std::string& path, const std::vector<uint8_t>& data,
                       std::string* error_text) {
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    *error_text = "open " + path + " for write failed: " + std::strerror(errno);
    return false;
  }
  size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t written = write(fd, data.data() + offset, data.size() - offset);
    if (written < 0) {
      if (errno == EINTR) continue;
      *error_text = "write " + path + " failed: " + std::strerror(errno);
      close(fd);
      return false;
    }
    if (written == 0) {
      *error_text = "write " + path + " made no progress";
      close(fd);
      return false;
    }
    offset += static_cast<size_t>(written);
  }
  close(fd);
  return true;
}

std::vector<uint32_t> build_chunk_transfer_words(uint64_t staging_va, uint64_t vram_va,
                                                 uint64_t readback_va, uint32_t byte_count) {
  std::vector<uint32_t> words;
  words.reserve((2U * kSdmaLinearCopyPacketDwords) + am_sdma::kFencePacketDwords);
  const auto h2d = build_sdma_linear_copy_packet(staging_va, vram_va, byte_count);
  const auto d2h = build_sdma_linear_copy_packet(vram_va, readback_va, byte_count);
  const auto fence = build_sdma_fence_packet(am_sdma::kFenceVa, am_sdma::kFenceValue);
  words.insert(words.end(), h2d.begin(), h2d.end());
  words.insert(words.end(), d2h.begin(), d2h.end());
  words.insert(words.end(), fence.begin(), fence.end());
  return words;
}

void print_stream_transfer_log(const DiscoveryLog& log, const VmBufferLog& staging,
                               const VmBufferLog& readback, const VmBufferLog& sdma_control,
                               uint64_t byte_count, uint64_t chunk_count,
                               const StreamTransferStatus& status, int exit_status) {
  std::printf("producer_kind: hardware_memory_transfer\n");
  std::printf("runtime_substrate: %s\n", kRuntimeSubstrate);
  std::printf("socket_path: %s\n", log.socket_path.c_str());
  std::printf("pci_id: %s\n", log.pci_id.c_str());
  std::printf("arch: %s\n", log.arch.c_str());
  std::printf("arch_discovery_status: %s\n", log.arch_discovery_status.c_str());
  std::printf("gc_ip_version: %s\n", log.gc_ip_version.c_str());
  std::printf("gc_ip_bases: %s\n", log.gc_ip_bases.c_str());
  std::printf("mmhub_ip_version: %s\n", log.mmhub_ip_version.c_str());
  std::printf("mmhub_ip_bases: %s\n", log.mmhub_ip_bases.c_str());
  std::printf("nbif_ip_version: %s\n", log.nbif_ip_version.c_str());
  std::printf("nbif_ip_bases: %s\n", log.nbif_ip_bases.c_str());
  std::printf("sdma_ip_version: %s\n", log.sdma_ip_version.c_str());
  std::printf("sdma_ip_bases: %s\n", log.sdma_ip_bases.c_str());
  std::printf("config_response_header_hex: %s\n", log.config_response_header_hex.c_str());
  std::printf("bar0_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar0.size));
  std::printf("bar2_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar2.size));
  std::printf("bar5_size_bytes: %llu\n", static_cast<unsigned long long>(log.bar5.size));
  std::printf("vram_size_bytes: %llu\n", static_cast<unsigned long long>(log.vram_size_bytes));
  std::printf("transfer_byte_count: %llu\n", static_cast<unsigned long long>(byte_count));
  std::printf("transfer_chunk_count: %llu\n", static_cast<unsigned long long>(chunk_count));
  std::printf("transfer_chunks_completed: %llu\n",
              static_cast<unsigned long long>(status.chunks_completed));
  std::printf("transfer_chunk_size_bytes: %llu\n",
              static_cast<unsigned long long>(kC1TransferChunkByteCount));
  std::printf("buffer_count: 3\n");
  std::printf("allocation_total_bytes: %llu\n",
              static_cast<unsigned long long>(3ULL * kC1TransferChunkByteCount));
  std::printf("upload_total_bytes: %llu\n", static_cast<unsigned long long>(status.bytes_uploaded));
  std::printf("download_total_bytes: %llu\n",
              static_cast<unsigned long long>(status.bytes_downloaded));
  std::printf("streaming_required: %s\n",
              byte_count > kC1TransferChunkByteCount ? "yes" : "no");
  std::printf("vm_vram_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(kTransferProofVmVramVa));
  std::printf("vm_vram_byte_count: %llu\n",
              static_cast<unsigned long long>(kC1TransferChunkByteCount));
  std::printf("vm_page_table_root_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.root_pdb2_paddr));
  std::printf("vm_pdb1_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_pdb1_paddr));
  std::printf("vm_pdb0_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_pdb0_paddr));
  std::printf("vm_ptb_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.child_ptb_paddr));
  std::printf("vm_vram_paddr: 0x%016llx\n",
              static_cast<unsigned long long>(log.vm.tables.device_buffer_paddr));
  std::printf("vm_page_tables_written: %s\n", log.vm.page_tables_written.c_str());
  std::printf("vmid0_context_status: %s\n", log.vm.vmid0_context_status.c_str());
  std::printf("vm_gc_context_status: %s\n", log.vm.vm_gc_context_status.c_str());
  std::printf("mm_tlb_flush_status: %s\n", log.vm.mm_tlb_flush_status.c_str());
  std::printf("gc_tlb_flush_status: %s\n", log.vm.gc_tlb_flush_status.c_str());
  print_vm_buffer_log("sysmem_staging", staging);
  print_vm_buffer_log("sysmem_readback", readback);
  print_vm_buffer_log("sysmem_sdma_control", sdma_control);
  std::printf("sdma_ring_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kControlVa));
  std::printf("sdma_ring_size_bytes: %llu\n", static_cast<unsigned long long>(am_sdma::kRingSize));
  std::printf("sdma_rptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kRptrVa));
  std::printf("sdma_wptr_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kWptrVa));
  std::printf("sdma_fence_gpu_va: 0x%016llx\n", static_cast<unsigned long long>(am_sdma::kFenceVa));
  std::printf("sdma_doorbell_index: %u\n", am_sdma::kDoorbellIndex);
  std::printf("sdma_doorbell_bar2_byte_offset: 0x%016llx\n",
              static_cast<unsigned long long>(am_sdma::kDoorbellBar2ByteOffset));
  std::printf("sdma_submit_dwords: %zu\n",
              (2U * kSdmaLinearCopyPacketDwords) + am_sdma::kFencePacketDwords);
  std::printf("sdma_linear_copy_packet_dwords: %zu\n", kSdmaLinearCopyPacketDwords);
  std::printf("sdma_queue_setup_status: %s\n", log.sdma.queue_setup_status.c_str());
  std::printf("sdma_submit_status: %s\n", log.sdma.submit_status.c_str());
  std::printf("sdma_timeline_status: %s\n", log.sdma.timeline_status.c_str());
  std::printf("sdma_h2d_status: %s\n", status.sdma_h2d_status.c_str());
  std::printf("sdma_d2h_status: %s\n", status.sdma_d2h_status.c_str());
  std::printf("cpu_comparison_status: %s\n", status.cpu_comparison_status.c_str());
  std::printf("host_device_transfer_status: %s\n", status.host_device_transfer_status.c_str());
  std::printf("failure_stage: %s\n", status.failure_stage.c_str());
  std::printf("failure_text: %s\n", status.failure_text.c_str());
  std::printf("exit_status: %d\n", exit_status);
}

int finish_stream_transfer(DiscoveryLog& log, const VmBufferLog& staging,
                           const VmBufferLog& readback, const VmBufferLog& sdma_control,
                           uint64_t byte_count, uint64_t chunk_count,
                           StreamTransferStatus status, const std::string& stage,
                           const std::string& text) {
  status.failure_stage = stage;
  status.failure_text = text;
  print_stream_transfer_log(log, staging, readback, sdma_control, byte_count, chunk_count, status,
                            1);
  return 1;
}

int run_streaming_transfer_proof(uint64_t byte_count, const std::vector<uint8_t>* source_bytes,
                                 const std::string* output_path) {
  const uint64_t chunk_count =
      (byte_count / kC1TransferChunkByteCount) +
      ((byte_count % kC1TransferChunkByteCount) != 0 ? 1ULL : 0ULL);
  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kC1TransferChunkByteCount, 0,
                      "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kC1TransferChunkByteCount, 0,
                       "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
  StreamTransferStatus status;

  if (byte_count == 0) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "transfer_request",
                                  "transfer byte count must be nonzero");
  }
  if (byte_count > kC1MaxTransferByteCount) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "transfer_request",
                                  "transfer byte count exceeds max C1R-4 policy");
  }
  if (source_bytes != nullptr && source_bytes->size() != byte_count) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "transfer_request",
                                  "source byte count does not match transfer byte count");
  }

  HardwareLock hardware_lock;
  std::string lock_error;
  if (!hardware_lock.acquire(&lock_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "hardware_ownership",
                                  "hardware lock acquire failed: " + lock_error);
  }

  UniqueFd socket_fd;
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  std::string connect_error;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &connect_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "tinygpu_connect", connect_error);
  }
  std::string health_error;
  if (!hardware_lock_health_check(log.socket_path, &health_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "hardware_ownership", health_error);
  }

  const RemoteClient client(socket_fd.get());
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  log.config_response_header_hex =
      config.response_header_hex.empty() ? "unavailable" : config.response_header_hex;
  if (!config.ok) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "config-read",
                                  rpc_failure_text("CFG_READ vendor_device", config));
  }
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "config-read",
                                  "expected 1002:7551, observed " +
                                      pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);

  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "map-bar0", rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "map-bar2", rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "map-bar5", rpc_failure_text("MAP_BAR bar5", bar_result));
  }

  std::string required_discovery_error;
  if (!try_discover_arch(client, &log, &required_discovery_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "arch_discovery", required_discovery_error);
  }

  std::string vm_error;
  if (!map_sysmem_buffer(client, &staging, &staging_mapping, &vm_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "vm_mapping", vm_error);
  }
  if (!map_sysmem_buffer(client, &readback, &readback_mapping, &vm_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "vm_mapping", vm_error);
  }
  if (!map_sysmem_buffer(client, &sdma_control, &sdma_control_mapping, &vm_error)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "vm_mapping", vm_error);
  }
  if (staging_mapping.size < kC1TransferChunkByteCount ||
      readback_mapping.size < kC1TransferChunkByteCount ||
      sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, "vm_mapping",
                                  "MAP_SYSMEM_FD CPU mappings are smaller than C1 streaming scratch spans");
  }

  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(client, &log, staging, readback, sdma_control, nullptr, false,
                              &vm_result)) {
    return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                  status, vm_result.failure_stage, vm_result.error_text);
  }

  std::vector<uint8_t> observed;
  if (output_path != nullptr) observed.resize(static_cast<size_t>(byte_count));

  uint64_t absolute_offset = 0;
  while (absolute_offset < byte_count) {
    const uint64_t remaining = byte_count - absolute_offset;
    const uint64_t chunk_bytes = std::min(remaining, kC1TransferChunkByteCount);
    std::memset(staging_mapping.data, 0, staging_mapping.size);
    std::memset(readback_mapping.data, 0, readback_mapping.size);
    std::memset(sdma_control_mapping.data, 0, sdma_control_mapping.size);
    if (source_bytes != nullptr) {
      std::memcpy(staging_mapping.data, source_bytes->data() + absolute_offset, chunk_bytes);
    } else {
      fill_transfer_chunk(static_cast<uint8_t*>(staging_mapping.data), absolute_offset, chunk_bytes);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);

    std::string sdma_error;
    if (!setup_sdma_queue0(client, &log, &sdma_error)) {
      return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                    status, "sdma_ring_setup", sdma_error);
    }
    const std::vector<uint32_t> words = build_chunk_transfer_words(
        staging.gpu_va, kTransferProofVmVramVa, readback.gpu_va, static_cast<uint32_t>(chunk_bytes));
    if (!submit_sdma_words(client, &log, &sdma_control_mapping, words, 0, &sdma_error)) {
      status.sdma_h2d_status = "fail";
      status.sdma_d2h_status = "fail";
      return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                    status, "sdma_submit", sdma_error);
    }
    if (!poll_sdma_fence(sdma_control_mapping, &sdma_error)) {
      log.sdma.timeline_status = "fail";
      status.sdma_h2d_status = "fail";
      status.sdma_d2h_status = "fail";
      return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                    status, "timeline_timeout", sdma_error);
    }
    log.sdma.timeline_status = "pass";
    status.sdma_h2d_status = "pass";
    status.sdma_d2h_status = "pass";
    status.bytes_uploaded += chunk_bytes;
    status.bytes_downloaded += chunk_bytes;

    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::string compare_error;
    const uint8_t* readback_bytes = static_cast<const uint8_t*>(readback_mapping.data);
    if (source_bytes != nullptr) {
      const uint8_t* expected = source_bytes->data() + absolute_offset;
      if (std::memcmp(readback_bytes, expected, static_cast<size_t>(chunk_bytes)) != 0) {
        status.cpu_comparison_status = "fail";
        return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                      status, "readback_mismatch",
                                      "streamed transfer mismatch for caller-supplied bytes");
      }
      if (output_path != nullptr) {
        std::memcpy(observed.data() + absolute_offset, readback_bytes, static_cast<size_t>(chunk_bytes));
      }
    } else if (!compare_transfer_chunk(readback_bytes, absolute_offset, chunk_bytes, &compare_error)) {
      status.cpu_comparison_status = "fail";
      return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                    status, "readback_mismatch", compare_error);
    }
    status.cpu_comparison_status = "pass";
    ++status.chunks_completed;
    absolute_offset += chunk_bytes;
  }

  if (output_path != nullptr) {
    std::string write_error;
    if (!write_binary_file(*output_path, observed, &write_error)) {
      return finish_stream_transfer(log, staging, readback, sdma_control, byte_count, chunk_count,
                                    status, "roundtrip_output_write", write_error);
    }
  }

  status.host_device_transfer_status = "pass";
  status.failure_stage = "none";
  status.failure_text = "none";
  print_stream_transfer_log(log, staging, readback, sdma_control, byte_count, chunk_count, status, 0);
  return 0;
}

// C0A25 kernel proof L3203-3245, generalized only over a validated reviewed
// descriptor. The mapped code page and BAR0 write/readback remain the C0 path.
bool load_resident_kernel_asset(const RemoteClient& client, uint64_t bar0_size,
                                const KernelDescriptor& kernel, std::string* error_text) {
  if (kernel.code.size() > kPageSize) {
    *error_text = "kernel code exceeds the C0 mapped code page";
    return false;
  }
  if (bar0_size < am_compute::kCodeVramPaddr + kernel.code.size()) {
    *error_text = "BAR0 is too small for the C0 mapped code page";
    return false;
  }
  if (!mmio_write_bar0(client, am_compute::kCodeVramPaddr, kernel.code, error_text)) {
    *error_text = "kernel code BAR0 write failed: " + *error_text;
    return false;
  }
  RemoteRpcResult readback =
      mmio_read(client, 0, am_compute::kCodeVramPaddr, kernel.code.size());
  if (!readback.ok) {
    *error_text = rpc_failure_text("MMIO_READ BAR0 kernel code", readback);
    return false;
  }
  if (readback.readout.size() != kernel.code.size() ||
      std::memcmp(readback.readout.data(), kernel.code.data(), kernel.code.size()) != 0) {
    *error_text = "kernel code BAR0 readback mismatch";
    return false;
  }
  return true;
}

// C0A25 kernel proof L3247-3281, generalized over the reviewed byte-exact
// kernarg layout. The kernarg page remains mapped by C0's fixed VM setup.
bool bind_resident_kernel_kernargs(const ResidentKernelDispatch& request,
                                   SysmemMapping* compute_control_mapping,
                                   std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr ||
      compute_control_mapping->size < am_compute::kComputeControlByteCount) {
    *error_text = "compute control mapping is smaller than the C0 fixed control span";
    return false;
  }
  uint8_t* const kernargs = static_cast<uint8_t*>(compute_control_mapping->data) +
                            am_compute::kComputeControlKernargsCpuOffset;
  std::memset(kernargs, 0, kPageSize);
  std::memcpy(kernargs, request.kernargs.data(), request.kernargs.size());
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (std::memcmp(kernargs, request.kernargs.data(), request.kernargs.size()) != 0) {
    *error_text = "kernarg CPU layout readback mismatch";
    return false;
  }
  return true;
}

// Slices the existing C0 kernarg page (compute_control.sys_pages[1] at
// kKernargsVa) into kKernargSlotCount immutable 256-byte slots. Zeroes ONLY
// the target slot so previously prepared stages' arguments survive a batch.
// The legacy bind_resident_kernel_kernargs (whole-page) stays intact for the
// Task 2.2 A/B ladder.
bool bind_resident_kernel_kernargs_slot(const ResidentKernelDispatch& request,
                                        SysmemMapping* compute_control_mapping,
                                        uint32_t slot, uint64_t* slot_va,
                                        std::string* error_text) {
  if (compute_control_mapping == nullptr || compute_control_mapping->data == nullptr ||
      compute_control_mapping->size < am_compute::kComputeControlByteCount) {
    *error_text = "compute control mapping is smaller than the C0 fixed control span";
    return false;
  }
  if (slot >= am_compute::kKernargSlotCount) {
    *error_text = "kernarg slot is out of range";
    return false;
  }
  if (request.kernargs.empty() || request.kernargs.size() > am_compute::kKernargSlotByteCount) {
    *error_text = "kernargs exceed the 256-byte slot";
    return false;
  }
  uint8_t* const destination = static_cast<uint8_t*>(compute_control_mapping->data) +
                               am_compute::kComputeControlKernargsCpuOffset +
                               static_cast<uint64_t>(slot) * am_compute::kKernargSlotByteCount;
  std::memset(destination, 0, am_compute::kKernargSlotByteCount);
  std::memcpy(destination, request.kernargs.data(), request.kernargs.size());
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (std::memcmp(destination, request.kernargs.data(), request.kernargs.size()) != 0) {
    *error_text = "kernarg slot CPU layout readback mismatch";
    return false;
  }
  *slot_va = am_compute::kKernargsVa +
             static_cast<uint64_t>(slot) * am_compute::kKernargSlotByteCount;
  return true;
}

void store_u64_le(uint8_t* destination, uint64_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    destination[index] = static_cast<uint8_t>(value >> (8U * index));
  }
}

void store_u32_le(uint8_t* destination, uint32_t value) {
  for (size_t index = 0; index < sizeof(value); ++index) {
    destination[index] = static_cast<uint8_t>(value >> (8U * index));
  }
}

uint32_t load_u32_le(const uint8_t* source) {
  uint32_t value = 0;
  for (size_t index = 0; index < sizeof(value); ++index) {
    value |= static_cast<uint32_t>(source[index]) << (8U * index);
  }
  return value;
}

std::string pm4_dispatch_digest(const std::vector<uint32_t>& words) {
  // Hash serialized little-endian dwords so the log binds the exact stream
  // submitted to the compute ring rather than host-endian memory layout.
  uint64_t digest = 14695981039346656037ULL;
  for (const uint32_t word : words) {
    for (uint32_t byte = 0; byte < sizeof(word); ++byte) {
      digest ^= static_cast<uint8_t>(word >> (byte * 8U));
      digest *= 1099511628211ULL;
    }
  }
  constexpr char kHex[] = "0123456789abcdef";
  std::string text = "fnv1a64:";
  text.reserve(text.size() + 16);
  for (uint32_t nibble = 0; nibble < 16; ++nibble) {
    text += kHex[(digest >> ((15U - nibble) * 4U)) & 0xfU];
  }
  return text;
}

void log_compute_queue_diagnostic_read(const RemoteClient& client, const DiscoveryLog& log,
                                       const RegDef& reg, const char* key) {
  uint32_t value = 0;
  std::string error;
  if (!read_register_dword(client, log, log.ip.gc, reg, &value, &error)) {
    std::printf("%s: unavailable (%s)\n", key, error.c_str());
    return;
  }
  std::printf("%s: 0x%08x\n", key, value);
}

void log_compute_queue_doorbell_hit(const RemoteClient& client, const DiscoveryLog& log,
                                    const RegDef& reg, const char* key) {
  uint32_t value = 0;
  std::string error;
  if (!read_register_dword(client, log, log.ip.gc, reg, &value, &error)) {
    std::printf("%s: unavailable (%s)\n", key, error.c_str());
    return;
  }
  std::printf("%s: 0x%08x\n", key, (value >> 31U) & 1U);
}

void log_compute_queue_diagnostics(const RemoteClient& client, const DiscoveryLog& log) {
  std::string selection_error;
  if (!select_grbm_queue0(client, log, &selection_error)) {
    for (const char* key : {"compute_queue_mec_rs64_cntl",
                            "compute_queue_mec_rs64_instruction_pointer",
                            "compute_queue_mec_rs64_program_start_lo",
                            "compute_queue_mec_rs64_program_start_hi",
                            "compute_queue_mec_doorbell_range_lower",
                            "compute_queue_mec_doorbell_range_upper",
                            "compute_queue_hqd_active",
                            "compute_queue_hqd_pq_doorbell_control"}) {
      std::printf("%s: unavailable (%s)\n", key, selection_error.c_str());
    }
    return;
  }

  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpMecRs64Cntl,
                                    "compute_queue_mec_rs64_cntl");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpMecRs64InstrPntr,
                                    "compute_queue_mec_rs64_instruction_pointer");
  log_compute_queue_diagnostic_read(client, log, kCpMecRs64PrgrmCntrStart,
                                    "compute_queue_mec_rs64_program_start_lo");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpMecRs64PrgrmCntrStartHi,
                                    "compute_queue_mec_rs64_program_start_hi");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpMecDoorbellRangeLower,
                                    "compute_queue_mec_doorbell_range_lower");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpMecDoorbellRangeUpper,
                                    "compute_queue_mec_doorbell_range_upper");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpHqdActive,
                                    "compute_queue_hqd_active");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpHqdPqDoorbellControl,
                                    "compute_queue_hqd_pq_doorbell_control");

  std::string restore_error;
  (void)restore_grbm_default_select(client, log, &restore_error);
}

void log_compute_queue_post_doorbell_diagnostics(const RemoteClient& client,
                                                 const DiscoveryLog& log) {
  std::string selection_error;
  if (!select_grbm_queue0(client, log, &selection_error)) {
    for (const char* key : {"compute_queue_post_doorbell_hit",
                            "compute_queue_post_wptr_lo"}) {
      std::printf("%s: unavailable (%s)\n", key, selection_error.c_str());
    }
    return;
  }

  log_compute_queue_doorbell_hit(client, log, regs_gfx1201::kCpHqdPqDoorbellControl,
                                  "compute_queue_post_doorbell_hit");
  log_compute_queue_diagnostic_read(client, log, regs_gfx1201::kCpHqdPqWptrLo,
                                    "compute_queue_post_wptr_lo");

  std::string restore_error;
  (void)restore_grbm_default_select(client, log, &restore_error);
}

bool submit_compute_dispatch_with_post_doorbell_diagnostics(
    const RemoteClient& client, DiscoveryLog* log, SysmemMapping* compute_control_mapping,
    const std::vector<uint32_t>& words, std::string* error_text, long* hdp_flush_usec = nullptr,
    long* doorbell_usec = nullptr) {
  if (!submit_compute_dispatch(client, log, compute_control_mapping, words, error_text, true,
                               hdp_flush_usec, doorbell_usec))
    return false;
  log_compute_queue_post_doorbell_diagnostics(client, *log);
  return true;
}

bool poll_compute_timeline_with_consumption_diagnostics(
    const RemoteClient& client, DiscoveryLog* log,
    const SysmemMapping& compute_control_mapping, long* elapsed_usec,
    std::string* error_text,
    uint32_t expected_value = am_compute::kReleaseMemTimelineValue) {
  if (poll_compute_timeline(compute_control_mapping, elapsed_usec, error_text,
                            expected_value))
    return true;

  ComputeQueueDebugSnapshot queue_snapshot;
  std::string queue_error;
  if (read_compute_queue_debug_snapshot(client, *log, false, &queue_snapshot, &queue_error)) {
    log->compute.doorbell_probe_timeout = format_compute_queue_debug_snapshot(queue_snapshot);
    log->compute.doorbell_probe_classification =
        classify_compute_doorbell_timeout(queue_snapshot);
  } else {
    log->compute.doorbell_probe_timeout = "read_failed: " + queue_error;
    log->compute.doorbell_probe_classification = "compute_doorbell_delivery_unclassified";
  }

  ComputeDoorbellConsumptionSnapshot consumption_snapshot;
  std::string consumption_error;
  if (read_compute_doorbell_consumption_snapshot(client, *log, compute_control_mapping,
                                                 &consumption_snapshot, &consumption_error)) {
    log->compute.doorbell_consumption_timeout =
        format_compute_doorbell_consumption_snapshot(consumption_snapshot);
    log->compute.doorbell_consumption_classification =
        classify_compute_doorbell_consumption_timeout(consumption_snapshot);
  } else {
    log->compute.doorbell_consumption_timeout = "read_failed: " + consumption_error;
    log->compute.doorbell_consumption_classification = "doorbell_consumption_unclassified";
  }
  *error_text += "; queue=" + log->compute.doorbell_probe_timeout +
                 "; consumption=" + log->compute.doorbell_consumption_timeout;
  return false;
}

class C0DynamicPageTableBackend final : public DynamicPageTableBackend {
 public:
  C0DynamicPageTableBackend(const RemoteClient& client, DiscoveryLog* log,
                            VramSmokeResult* result)
      : client_(client), log_(log), result_(result) {}

  void begin_deferred_flushes() { defer_flushes_ = true; }

  bool flush_deferred(std::string* error_text) {
    defer_flushes_ = false;
    if (pending_gc_flush_) {
      pending_gc_flush_ = false;
      if (!flush_gc(error_text)) return false;
    }
    if (pending_mmhub_flush_) {
      pending_mmhub_flush_ = false;
      if (!flush_mmhub(error_text)) return false;
    }
    return true;
  }

  bool zero_page(uint64_t physical_page, std::string* error_text) override {
    if (!zero_bar0_page(client_, physical_page, error_text)) {
      result_->bar0_zero_status = "fail";
      return false;
    }
    return true;
  }

  bool write_pte(uint64_t table_page, uint16_t entry_index, uint64_t pte,
                 std::string* error_text) override {
    if (!write_bar0_qword(client_, table_page + static_cast<uint64_t>(entry_index) *
                                               sizeof(uint64_t),
                          pte, error_text)) {
      result_->pte_write_status = "fail";
      return false;
    }
    if (result_->pte_write_status != "fail") result_->pte_write_status = "pass";
    return true;
  }

  bool read_pte(uint64_t table_page, uint16_t entry_index, uint64_t* pte,
                std::string* error_text) override {
    if (!read_bar0_qword(client_, table_page + static_cast<uint64_t>(entry_index) *
                                              sizeof(uint64_t),
                         pte, error_text)) {
      result_->pte_readback_status = "fail";
      return false;
    }
    if (result_->pte_readback_status != "fail") result_->pte_readback_status = "pass";
    return true;
  }

  bool flush_mmhub(std::string* error_text) override {
    if (defer_flushes_) {
      pending_mmhub_flush_ = true;
      return true;
    }
    if (!flush_mmhubs_tlb(client_, log_, error_text)) {
      result_->mmhub_tlb_flush_status = "fail";
      return false;
    }
    if (result_->mmhub_tlb_flush_status != "fail") result_->mmhub_tlb_flush_status = "pass";
    return true;
  }

  bool flush_gc(std::string* error_text) override {
    if (defer_flushes_) {
      pending_gc_flush_ = true;
      return true;
    }
    if (!flush_gc_tlb_vmid0_native(client_, log_, error_text)) {
      result_->gc_tlb_flush_status = "fail";
      return false;
    }
    if (result_->gc_tlb_flush_status != "fail") result_->gc_tlb_flush_status = "pass";
    return true;
  }

 private:
  const RemoteClient& client_;
  DiscoveryLog* log_;
  VramSmokeResult* result_;
  bool defer_flushes_ = false;
  bool pending_gc_flush_ = false;
  bool pending_mmhub_flush_ = false;
};

bool run_vram_smoke(VramSmokeResult* result, std::string* error_text) {
  auto fail = [&](const char* stage, const std::string& text) {
    if (result->failure_stage == "not_run") {
      result->failure_stage = stage;
      result->failure_text = text;
    }
    if (error_text != nullptr) *error_text = text;
    return false;
  };

  std::string detail;
  VramSmokeAsset asset;
  if (!preflight_vram_smoke_add_asset(&asset, &detail)) {
    return fail("asset_preflight", detail);
  }
  KernelDescriptor kernel = std::move(asset.descriptor);
  result->source_asset_path = asset.source_asset_path;
  result->asset_sha256 = kernel.sha256;
  result->code_byte_count = kernel.code.size();

  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kPageSize, 0, "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kPageSize, 0, "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
  VmBufferLog compute_control{"compute_control", am_compute::kRptrVa,
                              am_compute::kComputeControlByteCount, 0, "not_run", {}};
  HardwareLock hardware_lock;
  std::string lock_error;
  if (!hardware_lock.acquire(&lock_error)) {
    return fail("hardware_ownership", "hardware lock acquire failed: " + lock_error);
  }

  UniqueFd socket_fd;
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  SysmemMapping compute_control_mapping;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &detail)) {
    return fail("tinygpu_connect", detail);
  }
  std::string health_error;
  if (!hardware_lock_health_check(log.socket_path, &health_error)) {
    return fail("hardware_ownership", health_error);
  }

  const RemoteClient client(socket_fd.get());
  TerminalComputeQueue0Retirement compute_queue_retirement(client, log);
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  if (!config.ok) return fail("config_read", rpc_failure_text("CFG_READ vendor_device", config));
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return fail("config_read", "expected 1002:7551, observed " +
                                   pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);
  result->pci_id = log.pci_id;
  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return fail("map_bar0", rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return fail("map_bar2", rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return fail("map_bar5", rpc_failure_text("MAP_BAR bar5", bar_result));
  }
  if (!try_discover_arch(client, &log, &detail)) return fail("arch_discovery", detail);
  result->arch = log.arch;

  if (!map_sysmem_buffer(client, &staging, &staging_mapping, &detail) ||
      !map_sysmem_buffer(client, &readback, &readback_mapping, &detail) ||
      !map_sysmem_buffer(client, &sdma_control, &sdma_control_mapping, &detail) ||
      !map_sysmem_buffer(client, &compute_control, &compute_control_mapping, &detail)) {
    return fail("vm_mapping", detail);
  }
  if (staging_mapping.size < kPageSize || readback_mapping.size < kPageSize ||
      sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t) ||
      compute_control_mapping.size < am_compute::kComputeControlByteCount ||
      compute_control.sys_pages.size() < 10) {
    return fail("vm_mapping", "C0 fixed dispatch mappings are smaller than required");
  }
  std::memset(staging_mapping.data, 0, staging_mapping.size);
  std::memset(readback_mapping.data, 0, readback_mapping.size);
  std::memset(sdma_control_mapping.data, 0, sdma_control_mapping.size);
  std::memset(compute_control_mapping.data, 0, compute_control_mapping.size);
  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(client, &log, staging, readback, sdma_control, &compute_control,
                              true, &vm_result)) {
    return fail(vm_result.failure_stage.c_str(), vm_result.error_text);
  }

  const uint64_t vram_mib = log.vram_size_bytes >> 20U;
  if (vram_mib > std::numeric_limits<uint32_t>::max()) {
    return fail("vram_layout", "discovered VRAM MiB does not fit RCC_CONFIG_MEMSIZE");
  }
  VramLayout layout{};
  if (!derive_vram_layout(static_cast<uint32_t>(vram_mib), log.bar0.size, &layout, &detail)) {
    return fail("vram_layout", detail);
  }
  if (layout.large_bar) {
    return fail("vram_layout",
                "large BAR0 is unsupported: this smoke validates only the small-BAR page-table pool");
  }
  if (layout.allocatable_bytes == 0 ||
      layout.allocatable_base >
          std::numeric_limits<uint64_t>::max() - layout.allocatable_bytes) {
    return fail("vram_layout", "payload allocation range is invalid");
  }
  result->bar0_aperture_bytes = log.bar0.size;
  result->large_bar = layout.large_bar ? "true" : "false";
  result->page_table_pool_base = layout.page_table_pool_base;
  result->page_table_pool_bytes = layout.page_table_pool_bytes;
  result->payload_allocation_range_start = layout.allocatable_base;
  result->payload_allocation_range_end =
      layout.allocatable_base + layout.allocatable_bytes;

  if (layout.resident_gpu_va_base >= kSmokePtbBoundaryGpuVa ||
      layout.resident_gpu_va_limit <
          kSmokePtbBoundaryGpuVa + kSmokePayloadPageCount * kPageSize) {
    return fail("vram_layout",
                "resident window cannot place smoke payload in PDB0 index 1");
  }
  const uint64_t smoke_ptb_boundary_bytes =
      kSmokePtbBoundaryGpuVa - layout.resident_gpu_va_base;

  VramAllocator payload_allocator(layout);
  VramLayout page_table_layout = layout;
  if (!layout.large_bar) {
    if (layout.page_table_pool_base == 0 || layout.page_table_pool_bytes == 0 ||
        layout.page_table_pool_base % kPageSize != 0 ||
        layout.page_table_pool_bytes % kPageSize != 0 ||
        layout.page_table_pool_base > log.bar0.size ||
        layout.page_table_pool_bytes > log.bar0.size - layout.page_table_pool_base) {
      return fail("vram_layout", "small BAR0 page-table pool is missing or invalid");
    }
    page_table_layout.allocatable_base = layout.page_table_pool_base + kPageSize;
    page_table_layout.allocatable_bytes = layout.page_table_pool_bytes - kPageSize;
  }
  VramAllocator page_table_allocator(page_table_layout);
  C0DynamicPageTableBackend pte_backend(client, &log, result);
  DynamicPageTable page_table(
      layout, page_table_allocator, &pte_backend,
      FixedPageTablePages{vm_result.tables.root_pdb2_paddr, vm_result.tables.child_pdb1_paddr,
                          vm_result.tables.child_pdb0_paddr, vm_result.tables.child_ptb_paddr});
  bool mapping_uncertain = false;
  uint64_t mapping_uncertain_gpu_va = 0;
  std::string mapping_uncertainty_text;
  ResidentMemory resident(
      layout, payload_allocator,
      [&page_table, result, &mapping_uncertain, &mapping_uncertain_gpu_va,
       &mapping_uncertainty_text](ResidentPageOperation operation, uint64_t gpu_va,
                                  uint64_t physical_page, std::string* map_error) {
        if (operation == ResidentPageOperation::kUnmap) {
          // After an uncertain cleanup, ResidentMemory must not release any
          // other payload until this page has been proven unmapped.
          if (mapping_uncertain && gpu_va != mapping_uncertain_gpu_va) return false;
          const bool unmapped =
              page_table.unmap_range(gpu_va, kPageSize, map_error);
          if (unmapped && mapping_uncertain) {
            mapping_uncertain = false;
            mapping_uncertain_gpu_va = 0;
            mapping_uncertainty_text.clear();
          }
          return unmapped;
        }
        if (mapping_uncertain) {
          // This can occur only while ResidentMemory finishes recording the
          // guard allocation. Retain its pages; do not issue more PTE writes.
          return true;
        }

        if (page_table.map_range(gpu_va, physical_page, kPageSize, map_error)) {
          result->pte_map_status = "pass";
          return true;
        }

        result->pte_map_status = "fail";
        const std::string map_failure =
            map_error != nullptr && !map_error->empty()
                ? *map_error
                : std::string("page-table map failed");
        std::string cleanup_error;
        if (page_table.unmap_range(gpu_va, kPageSize, &cleanup_error)) {
          return false;
        }

        mapping_uncertain = true;
        mapping_uncertain_gpu_va = gpu_va;
        result->mapping_uncertainty_status = "uncertain";
        mapping_uncertainty_text =
            map_failure + "; failed to prove page-table cleanup: " + cleanup_error;
        if (map_error != nullptr) *map_error = mapping_uncertainty_text;
        // Returning true retains this allocation in ResidentMemory so a later
        // release_all can retry cleanup without releasing a stale-reachable page.
        return true;
      });
  auto release_resident = [&resident]() { resident.release_all(); };
  auto fail_after_compute_queue_setup = [&](const char* stage, const std::string& text) {
    std::string retirement_detail;
    if (!compute_queue_retirement.retire(&retirement_detail)) {
      return fail("compute_queue_retirement",
                  text + "; terminal queue-0 retirement failed: " + retirement_detail);
    }
    release_resident();
    return fail(stage, text);
  };
  std::string allocation_failure_stage;
  auto first_pte_failure_stage = [&]() -> const char* {
    if (result->pte_map_status == "fail") return "pte_map";
    if (result->bar0_zero_status == "fail") return "bar0_zero";
    if (result->pte_write_status == "fail") return "pte_write";
    if (result->pte_readback_status == "fail") return "pte_readback";
    if (result->mmhub_tlb_flush_status == "fail") return "mmhub_tlb_flush";
    if (result->gc_tlb_flush_status == "fail") return "gc_tlb_flush";
    return nullptr;
  };
  auto allocate_resident = [&](const char* name, uint64_t size_bytes,
                               bool zero_after_map, ResidentBuffer* buffer) {
    if (!resident.allocate(name, size_bytes, buffer, &detail)) {
      const char* pte_failure_stage = first_pte_failure_stage();
      if (pte_failure_stage != nullptr) {
        allocation_failure_stage = pte_failure_stage;
      } else {
        result->vram_allocation_status = "fail";
        allocation_failure_stage = "vram_allocation";
      }
      return false;
    }
    if (mapping_uncertain) {
      result->pte_map_status = "fail";
      allocation_failure_stage = "pte_map";
      detail = mapping_uncertainty_text;
      return false;
    }
    result->vram_allocation_status = "pass";
    ++result->resident_mapping_count;
    if (!zero_after_map) return true;
    if (!zero_bar0_page(client, buffer->allocation.physical_offset, &detail)) {
      result->bar0_zero_status = "fail";
      allocation_failure_stage = "bar0_zero";
      return false;
    }
    return true;
  };

  ResidentBuffer smoke_ptb_boundary{};
  ResidentBuffer a{};
  ResidentBuffer b{};
  ResidentBuffer out{};
  ResidentBuffer code{};
  if (!allocate_resident("smoke-ptb-boundary", smoke_ptb_boundary_bytes, false,
                         &smoke_ptb_boundary) ||
      !allocate_resident("smoke-a", kPageSize, true, &a) ||
      !allocate_resident("smoke-b", kPageSize, true, &b) ||
      !allocate_resident("smoke-out", kPageSize, true, &out) ||
      !allocate_resident("smoke-code", kPageSize, true, &code)) {
    release_resident();
    return fail(allocation_failure_stage.c_str(), detail);
  }
  result->a_gpu_va = a.gpu_va;
  result->a_physical_offset = a.allocation.physical_offset;
  result->b_gpu_va = b.gpu_va;
  result->b_physical_offset = b.allocation.physical_offset;
  result->out_gpu_va = out.gpu_va;
  result->out_physical_offset = out.allocation.physical_offset;
  result->dynamic_ptb_count = page_table.dynamic_ptb_count();
  result->dynamic_ptb_physical_offset =
      page_table.first_dynamic_ptb_physical_offset();
  if (a.gpu_va != kSmokePtbBoundaryGpuVa ||
      b.gpu_va != a.gpu_va + kPageSize ||
      out.gpu_va != b.gpu_va + kPageSize ||
      code.gpu_va != out.gpu_va + kPageSize ||
      result->dynamic_ptb_count == 0 ||
      result->dynamic_ptb_physical_offset == 0) {
    result->pte_map_status = "fail";
    release_resident();
    return fail("pte_map",
                "smoke payload did not force a live PDB0 index 1 dynamic PTB");
  }
  result->vram_allocation_status = "pass";
  result->bar0_zero_status = "pass";
  if (kernel.code.size() > code.size_bytes) {
    release_resident();
    return fail("kernel_asset_validate", "smoke code exceeds allocated page");
  }
  if (!mmio_write_bar0(client, code.allocation.physical_offset, kernel.code, &detail)) {
    release_resident();
    return fail("kernel_asset_write", detail);
  }
  RemoteRpcResult code_readback =
      mmio_read(client, 0, code.allocation.physical_offset, kernel.code.size());
  if (!code_readback.ok || code_readback.readout.size() != kernel.code.size() ||
      std::memcmp(code_readback.readout.data(), kernel.code.data(), kernel.code.size()) != 0) {
    result->bar0_code_readback_status = "fail";
    release_resident();
    return fail("bar0_code_readback",
                code_readback.ok ? "smoke code BAR0 readback mismatch"
                                 : rpc_failure_text("MMIO_READ BAR0 smoke code", code_readback));
  }
  result->bar0_code_readback_status = "pass";

  constexpr size_t kElementCount = 64;
  constexpr size_t kVectorBytes = kElementCount * sizeof(uint32_t);
  std::array<uint8_t, kVectorBytes> host_a{};
  std::array<uint8_t, kVectorBytes> host_b{};
  std::array<uint8_t, kVectorBytes> expected{};
  for (size_t index = 0; index < kElementCount; ++index) {
    const uint32_t a_value = 0x10203040U + static_cast<uint32_t>(index * 3U);
    const uint32_t b_value = 0x55667788U + static_cast<uint32_t>(index * 5U);
    store_u32_le(host_a.data() + index * sizeof(uint32_t), a_value);
    store_u32_le(host_b.data() + index * sizeof(uint32_t), b_value);
    store_u32_le(expected.data() + index * sizeof(uint32_t), a_value + b_value);
  }
  if (!setup_sdma_queue0(client, &log, &detail)) {
    release_resident();
    return fail("sdma_ring_setup", detail);
  }
  std::memcpy(staging_mapping.data, host_a.data(), host_a.size());
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, staging.gpu_va, a.gpu_va,
                        static_cast<uint32_t>(host_a.size()), am_sdma::kFenceValue, 0, &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    result->sdma_h2d_status = "fail";
    release_resident();
    return fail("sdma_h2d_a", detail);
  }
  result->sdma_upload_bytes += host_a.size();
  std::memset(static_cast<uint8_t*>(sdma_control_mapping.data) + am_sdma::kFenceOffset, 0,
              sizeof(uint32_t));
  std::memcpy(staging_mapping.data, host_b.data(), host_b.size());
  std::atomic_thread_fence(std::memory_order_seq_cst);
  constexpr uint64_t kSecondH2dSubmitByteOffset =
      static_cast<uint64_t>(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords) *
      sizeof(uint32_t);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, staging.gpu_va, b.gpu_va,
                        static_cast<uint32_t>(host_b.size()), am_sdma::kFenceValue,
                        kSecondH2dSubmitByteOffset, &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    result->sdma_h2d_status = "fail";
    release_resident();
    return fail("sdma_h2d_b", detail);
  }
  result->sdma_upload_bytes += host_b.size();
  result->sdma_h2d_status = "pass";

  if (!setup_compute_ring0(client, &log, &compute_control_mapping, &detail)) {
    release_resident();
    return fail("compute_ring_setup", detail);
  }
  compute_queue_retirement.arm();
  ResidentKernelDispatch dispatch_request;
  dispatch_request.kernel = kernel;
  dispatch_request.kernargs.resize(kernel.kernarg_bytes);
  store_u64_le(dispatch_request.kernargs.data(), a.gpu_va);
  store_u64_le(dispatch_request.kernargs.data() + sizeof(uint64_t), b.gpu_va);
  store_u64_le(dispatch_request.kernargs.data() + 2 * sizeof(uint64_t), out.gpu_va);
  result->kernarg_hex =
      hex_encode_bytes(dispatch_request.kernargs.data(), dispatch_request.kernargs.size());
  result->kernarg_byte_count = dispatch_request.kernargs.size();
  dispatch_request.input_bytes.assign(host_a.begin(), host_a.end());
  dispatch_request.output_byte_count = static_cast<uint32_t>(kVectorBytes);
  if (!bind_resident_kernel_kernargs(dispatch_request, &compute_control_mapping, &detail)) {
    return fail_after_compute_queue_setup("kernarg_bind", detail);
  }
  const Pm4DispatchConfig pm4{code.gpu_va, am_compute::kKernargsVa, am_compute::kTimelineVa,
                              kernel.rsrc1, kernel.rsrc2, kernel.rsrc3, false, kernel.workgroup_x,
                              kernel.workgroup_y, kernel.workgroup_z, kernel.global_x,
                              kernel.global_y, kernel.global_z};
  const std::vector<uint32_t> pm4_words = build_pm4_dispatch_words(pm4);
  result->pm4_dispatch_word_count = pm4_words.size();
  result->pm4_dispatch_digest = pm4_dispatch_digest(pm4_words);
  log_compute_queue_diagnostics(client, log);
  if (!submit_compute_dispatch_with_post_doorbell_diagnostics(
          client, &log, &compute_control_mapping, pm4_words, &detail)) {
    return fail_after_compute_queue_setup("pm4_submit", detail);
  }
  long elapsed_usec = 0;
  if (!poll_compute_timeline_with_consumption_diagnostics(
          client, &log, compute_control_mapping, &elapsed_usec, &detail)) {
    return fail_after_compute_queue_setup("compute_fence_poll", detail);
  }
  result->compute_dispatch_count = 1;

  std::memset(static_cast<uint8_t*>(sdma_control_mapping.data) + am_sdma::kFenceOffset, 0,
              sizeof(uint32_t));
  std::atomic_thread_fence(std::memory_order_seq_cst);
  constexpr uint64_t kD2hSubmitByteOffset =
      2ULL * static_cast<uint64_t>(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords) *
      sizeof(uint32_t);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, out.gpu_va, readback.gpu_va,
                        static_cast<uint32_t>(kVectorBytes), am_sdma::kFenceValue,
                        kD2hSubmitByteOffset, &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    result->sdma_d2h_status = "fail";
    return fail_after_compute_queue_setup("sdma_d2h", detail);
  }
  result->sdma_download_bytes = kVectorBytes;
  result->sdma_d2h_status = "pass";
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const uint8_t* observed = static_cast<const uint8_t*>(readback_mapping.data);
  for (size_t index = 0; index < kElementCount; ++index) {
    const uint32_t expected_value = load_u32_le(expected.data() + index * sizeof(uint32_t));
    const uint32_t observed_value = load_u32_le(observed + index * sizeof(uint32_t));
    if (observed_value != expected_value) {
      result->cpu_comparison_status = "fail";
      return fail_after_compute_queue_setup(
          "readback_mismatch", "vector-add mismatch at element " + std::to_string(index) +
                                   " expected=" + std::to_string(expected_value) +
                                   " observed=" + std::to_string(observed_value));
    }
  }
  result->cpu_comparison_status = "pass";
  std::string retirement_detail;
  if (!compute_queue_retirement.retire(&retirement_detail)) {
    return fail("compute_queue_retirement",
                "terminal queue-0 retirement failed: " + retirement_detail);
  }
  release_resident();
  if (const char* pte_failure_stage = first_pte_failure_stage();
      pte_failure_stage != nullptr) {
    return fail(pte_failure_stage, "resident VRAM cleanup did not complete");
  }
  result->failure_stage = "none";
  result->failure_text = "none";
  return true;
}


bool run_llama_embed_smoke(const LlamaEmbedSmokeDispatch& request,
                           LlamaEmbedSmokeDispatchResult* result,
                           std::string* error_text) {
  auto fail = [&](const char* stage, const std::string& text) {
    if (result->failure_stage == "not_run") result->failure_stage = stage;
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (request.embedding_row.size() != kPageSize) {
    return fail("row_preflight", "Llama embedding row must be exactly 4096 bytes");
  }
  const HsaCodeImageAsset* const hsa_image = request.hsa_image;
  if (hsa_image == nullptr || hsa_image->image.empty() ||
      hsa_image->entry_offset >= hsa_image->image.size()) {
    return fail("hsa_image_preflight", "HSA image entry offset is outside its image");
  }

  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kPageSize, 0, "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kPageSize, 0, "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
  VmBufferLog compute_control{"compute_control", am_compute::kRptrVa,
                              am_compute::kComputeControlByteCount, 0, "not_run", {}};
  HardwareLock hardware_lock;
  std::string lock_error;
  if (!hardware_lock.acquire(&lock_error)) {
    return fail("hardware_ownership", "hardware lock acquire failed: " + lock_error);
  }

  UniqueFd socket_fd;
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  SysmemMapping compute_control_mapping;
  std::string detail;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &detail)) {
    return fail("tinygpu_connect", detail);
  }
  std::string health_error;
  if (!hardware_lock_health_check(log.socket_path, &health_error)) {
    return fail("hardware_ownership", health_error);
  }

  const RemoteClient client(socket_fd.get());
  TerminalComputeQueue0Retirement compute_queue_retirement(client, log);
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  if (!config.ok) return fail("config_read", rpc_failure_text("CFG_READ vendor_device", config));
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return fail("config_read", "expected 1002:7551, observed " +
                                   pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);
  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return fail("map_bar0", rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return fail("map_bar2", rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return fail("map_bar5", rpc_failure_text("MAP_BAR bar5", bar_result));
  }
  if (!try_discover_arch(client, &log, &detail)) return fail("arch_discovery", detail);
  result->hardware_identity = std::string(kRuntimeSubstrate) + " pci_id=" + log.pci_id +
                              " arch=" + log.arch;
  if (!map_sysmem_buffer(client, &staging, &staging_mapping, &detail) ||
      !map_sysmem_buffer(client, &readback, &readback_mapping, &detail) ||
      !map_sysmem_buffer(client, &sdma_control, &sdma_control_mapping, &detail) ||
      !map_sysmem_buffer(client, &compute_control, &compute_control_mapping, &detail)) {
    return fail("vm_mapping", detail);
  }
  if (staging_mapping.size < kPageSize || readback_mapping.size < kPageSize ||
      sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t) ||
      compute_control_mapping.size < am_compute::kComputeControlByteCount ||
      compute_control.sys_pages.size() < 10) {
    return fail("vm_mapping", "C0 fixed dispatch mappings are smaller than required");
  }
  std::memset(staging_mapping.data, 0, staging_mapping.size);
  std::memset(readback_mapping.data, 0, readback_mapping.size);
  std::memset(sdma_control_mapping.data, 0, sdma_control_mapping.size);
  std::memset(compute_control_mapping.data, 0, compute_control_mapping.size);
  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(client, &log, staging, readback, sdma_control, &compute_control,
                              true, &vm_result)) {
    return fail(vm_result.failure_stage.c_str(), vm_result.error_text);
  }
  const uint64_t vram_mib = log.vram_size_bytes >> 20U;
  if (vram_mib > std::numeric_limits<uint32_t>::max()) {
    return fail("vram_layout", "discovered VRAM MiB does not fit RCC_CONFIG_MEMSIZE");
  }
  VramLayout layout{};
  if (!derive_vram_layout(static_cast<uint32_t>(vram_mib), log.bar0.size, &layout, &detail)) {
    return fail("vram_layout", detail);
  }
  if (layout.large_bar) {
    return fail("vram_layout",
                "large BAR0 is unsupported: selected-row smoke requires the lower-BAR pool");
  }
  if (layout.allocatable_bytes == 0 ||
      layout.allocatable_base > std::numeric_limits<uint64_t>::max() - layout.allocatable_bytes) {
    return fail("vram_layout", "payload allocation range is invalid");
  }
  if (layout.page_table_pool_base == 0 || layout.page_table_pool_bytes == 0 ||
      layout.page_table_pool_base % kPageSize != 0 || layout.page_table_pool_bytes % kPageSize != 0 ||
      layout.page_table_pool_base > log.bar0.size ||
      layout.page_table_pool_bytes > log.bar0.size - layout.page_table_pool_base) {
    return fail("vram_layout", "small BAR0 page-table pool is missing or invalid");
  }
  if (layout.resident_gpu_va_base >= kSmokePtbBoundaryGpuVa ||
      layout.resident_gpu_va_limit < kSmokePtbBoundaryGpuVa + 8 * kPageSize) {
    return fail("vram_layout", "resident window cannot force the dynamic PTB boundary");
  }
  result->page_table_pool_base = layout.page_table_pool_base;
  result->page_table_pool_bytes = layout.page_table_pool_bytes;
  result->payload_allocation_range_start = layout.allocatable_base;
  result->payload_allocation_range_end = layout.allocatable_base + layout.allocatable_bytes;

  VramAllocator payload_allocator(layout);
  VramLayout page_table_layout = layout;
  page_table_layout.allocatable_base = layout.page_table_pool_base;
  page_table_layout.allocatable_bytes = layout.page_table_pool_bytes;
  VramAllocator page_table_allocator(page_table_layout);
  VramSmokeResult pte_result;
  C0DynamicPageTableBackend pte_backend(client, &log, &pte_result);
  DynamicPageTable page_table(
      layout, page_table_allocator, &pte_backend,
      FixedPageTablePages{vm_result.tables.root_pdb2_paddr, vm_result.tables.child_pdb1_paddr,
                          vm_result.tables.child_pdb0_paddr, vm_result.tables.child_ptb_paddr});
  bool mapping_uncertain = false;
  uint64_t uncertain_gpu_va = 0;
  std::string uncertain_text;
  ResidentMemory resident(
      layout, payload_allocator,
      [&page_table, &pte_result, &mapping_uncertain, &uncertain_gpu_va,
       &uncertain_text](ResidentPageOperation operation, uint64_t gpu_va,
                        uint64_t physical_page, std::string* map_error) {
        if (operation == ResidentPageOperation::kUnmap) {
          if (mapping_uncertain && gpu_va != uncertain_gpu_va) return false;
          const bool unmapped = page_table.unmap_range(gpu_va, kPageSize, map_error);
          if (unmapped && mapping_uncertain) {
            mapping_uncertain = false;
            uncertain_gpu_va = 0;
            uncertain_text.clear();
          }
          return unmapped;
        }
        if (mapping_uncertain) return true;
        if (page_table.map_range(gpu_va, physical_page, kPageSize, map_error)) {
          pte_result.pte_map_status = "pass";
          return true;
        }
        pte_result.pte_map_status = "fail";
        const std::string map_failure =
            map_error != nullptr && !map_error->empty() ? *map_error : "page-table map failed";
        std::string cleanup_error;
        if (page_table.unmap_range(gpu_va, kPageSize, &cleanup_error)) return false;
        mapping_uncertain = true;
        uncertain_gpu_va = gpu_va;
        uncertain_text = map_failure + "; failed to prove page-table cleanup: " + cleanup_error;
        if (map_error != nullptr) *map_error = uncertain_text;
        return true;
      });
  auto first_pte_failure = [&]() -> const char* {
    if (pte_result.pte_map_status == "fail") return "pte_map";
    if (pte_result.bar0_zero_status == "fail") return "bar0_zero";
    if (pte_result.pte_write_status == "fail") return "pte_write";
    if (pte_result.pte_readback_status == "fail") return "pte_readback";
    if (pte_result.mmhub_tlb_flush_status == "fail") return "mmhub_tlb_flush";
    if (pte_result.gc_tlb_flush_status == "fail") return "gc_tlb_flush";
    return nullptr;
  };
  auto release_resident = [&]() { resident.release_all(); };
  auto fail_after_compute_queue_setup = [&](const char* stage, const std::string& text) {
    std::string retirement_detail;
    if (!compute_queue_retirement.retire(&retirement_detail)) {
      return fail("compute_queue_retirement",
                  text + "; terminal queue-0 retirement failed: " + retirement_detail);
    }
    release_resident();
    return fail(stage, text);
  };
  auto allocate_resident = [&](const char* name, uint64_t size, ResidentBuffer* buffer) {
    if (!resident.allocate(name, size, buffer, &detail)) return false;
    if (mapping_uncertain) {
      detail = uncertain_text;
      return false;
    }
    for (uint64_t offset = 0; offset < buffer->size_bytes; offset += kPageSize) {
      if (!zero_bar0_page(client, buffer->allocation.physical_offset + offset, &detail)) {
        pte_result.bar0_zero_status = "fail";
        return false;
      }
    }
    return true;
  };
  const uint64_t boundary_bytes = kSmokePtbBoundaryGpuVa - layout.resident_gpu_va_base;
  ResidentBuffer boundary_guard{};
  ResidentBuffer image{};
  ResidentBuffer row{};
  ResidentBuffer hidden{};
  ResidentBuffer selected_row{};
  if (!allocate_resident("llama-embed-ptb-boundary", boundary_bytes, &boundary_guard) ||
      !allocate_resident("llama-embed-hsa-image", hsa_image->image.size(), &image) ||
      !allocate_resident("llama-embed-row", kPageSize, &row) ||
      !allocate_resident("llama-embed-hidden", kPageSize, &hidden) ||
      !allocate_resident("llama-embed-selected-row", kPageSize, &selected_row)) {
    const char* stage = first_pte_failure();
    release_resident();
    return fail(stage != nullptr ? stage : "vram_allocation", detail);
  }
  result->resident_buffer_zero_status = "pass";
  if (image.gpu_va != kSmokePtbBoundaryGpuVa || page_table.dynamic_ptb_count() == 0 ||
      page_table.first_dynamic_ptb_physical_offset() == 0) {
    release_resident();
    return fail("pte_map", "selected-row payload did not force a live dynamic PTB");
  }
  result->hsa_image_gpu_va = image.gpu_va;
  result->hsa_image_physical_offset = image.allocation.physical_offset;
  result->embedding_row_gpu_va = row.gpu_va;
  result->embedding_row_physical_offset = row.allocation.physical_offset;
  result->hidden_output_gpu_va = hidden.gpu_va;
  result->hidden_output_physical_offset = hidden.allocation.physical_offset;
  result->selected_row_gpu_va = selected_row.gpu_va;
  result->selected_row_physical_offset = selected_row.allocation.physical_offset;
  result->dynamic_ptb_count = page_table.dynamic_ptb_count();
  result->dynamic_ptb_physical_offset = page_table.first_dynamic_ptb_physical_offset();

  if (!mmio_write_bar0(client, image.allocation.physical_offset, hsa_image->image, &detail)) {
    release_resident();
    return fail("hsa_image_write", detail);
  }
  RemoteRpcResult image_readback =
      mmio_read(client, 0, image.allocation.physical_offset, hsa_image->image.size());
  if (!image_readback.ok || image_readback.readout.size() != hsa_image->image.size() ||
      std::memcmp(image_readback.readout.data(), hsa_image->image.data(),
                  hsa_image->image.size()) != 0) {
    result->bar0_image_readback_status = "fail";
    release_resident();
    return fail("bar0_hsa_image_readback",
                image_readback.ok ? "HSA image BAR0 readback mismatch"
                                  : rpc_failure_text("MMIO_READ BAR0 HSA image", image_readback));
  }
  result->bar0_image_readback_status = "pass";
  if (!setup_sdma_queue0(client, &log, &detail)) {
    release_resident();
    return fail("sdma_ring_setup", detail);
  }
  std::memcpy(staging_mapping.data, request.embedding_row.data(), kPageSize);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, staging.gpu_va, row.gpu_va, kPageSize,
                        am_sdma::kFenceValue, 0, &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    result->sdma_h2d_status = "fail";
    release_resident();
    return fail("sdma_h2d_row", detail);
  }
  result->sdma_upload_bytes += kPageSize;
  std::memset(static_cast<uint8_t*>(sdma_control_mapping.data) + am_sdma::kFenceOffset, 0,
              sizeof(uint32_t));
  std::memset(staging_mapping.data, 0, sizeof(uint64_t));
  std::atomic_thread_fence(std::memory_order_seq_cst);
  constexpr uint64_t kSecondH2dSubmitByteOffset =
      static_cast<uint64_t>(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords) *
      sizeof(uint32_t);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, staging.gpu_va, selected_row.gpu_va,
                        sizeof(uint64_t), am_sdma::kFenceValue, kSecondH2dSubmitByteOffset,
                        &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    result->sdma_h2d_status = "fail";
    release_resident();
    return fail("sdma_h2d_selected_row", detail);
  }
  result->sdma_upload_bytes += sizeof(uint64_t);
  result->sdma_h2d_status = "pass";
  if (!setup_compute_ring0(client, &log, &compute_control_mapping, &detail)) {
    release_resident();
    return fail("compute_ring_setup", detail);
  }
  compute_queue_retirement.arm();
  std::array<uint8_t, 24> kernargs{};
  store_u64_le(kernargs.data(), row.gpu_va);
  store_u64_le(kernargs.data() + sizeof(uint64_t), hidden.gpu_va);
  store_u64_le(kernargs.data() + 2 * sizeof(uint64_t), selected_row.gpu_va);
  ResidentKernelDispatch kernarg_request;
  kernarg_request.kernargs.assign(kernargs.begin(), kernargs.end());
  result->kernarg_hex = hex_encode_bytes(kernargs.data(), kernargs.size());
  if (!bind_resident_kernel_kernargs(kernarg_request, &compute_control_mapping, &detail)) {
    return fail_after_compute_queue_setup("kernarg_bind", detail);
  }
  const Pm4DispatchConfig pm4{image.gpu_va + hsa_image->entry_offset, am_compute::kKernargsVa,
                              am_compute::kTimelineVa, hsa_image->rsrc1, hsa_image->rsrc2,
                              hsa_image->rsrc3, false, 256, 1, 1, 2048, 1, 1};
  const std::vector<uint32_t> pm4_words = build_pm4_dispatch_words(pm4);
  result->pm4_dispatch_word_count = pm4_words.size();
  result->pm4_dispatch_digest = pm4_dispatch_digest(pm4_words);
  log_compute_queue_diagnostics(client, log);
  if (!submit_compute_dispatch_with_post_doorbell_diagnostics(
          client, &log, &compute_control_mapping, pm4_words, &detail)) {
    return fail_after_compute_queue_setup("pm4_submit", detail);
  }
  long elapsed_usec = 0;
  if (!poll_compute_timeline_with_consumption_diagnostics(
          client, &log, compute_control_mapping, &elapsed_usec, &detail)) {
    return fail_after_compute_queue_setup("compute_fence_poll", detail);
  }
  result->pm4_dispatch_count = 1;
  std::memset(static_cast<uint8_t*>(sdma_control_mapping.data) + am_sdma::kFenceOffset, 0,
              sizeof(uint32_t));
  std::atomic_thread_fence(std::memory_order_seq_cst);
  constexpr uint64_t kD2hSubmitByteOffset =
      2ULL * static_cast<uint64_t>(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords) *
      sizeof(uint32_t);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, hidden.gpu_va, readback.gpu_va,
                        kPageSize, am_sdma::kFenceValue, kD2hSubmitByteOffset, &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    result->sdma_d2h_status = "fail";
    return fail_after_compute_queue_setup("sdma_d2h", detail);
  }
  result->sdma_download_bytes = kPageSize;
  result->sdma_d2h_status = "pass";
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (std::memcmp(readback_mapping.data, request.embedding_row.data(), kPageSize) != 0) {
    result->fp16_row_hidden_byte_equality = "fail";
    return fail_after_compute_queue_setup("readback_mismatch",
                                          "selected-row FP16 byte comparison failed");
  }
  result->fp16_row_hidden_byte_equality = "pass";
  std::string retirement_detail;
  if (!compute_queue_retirement.retire(&retirement_detail)) {
    return fail("compute_queue_retirement",
                "terminal queue-0 retirement failed: " + retirement_detail);
  }
  release_resident();
  if (const char* stage = first_pte_failure(); stage != nullptr) {
    return fail(stage, "resident VRAM cleanup did not complete");
  }
  result->failure_stage = "none";
  return true;
}
bool validate_resident_hsa_dispatch(const ResidentHsaDispatch& request,
                                    std::string* error_text) {
  auto fail = [&](const std::string& text) {
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  const std::vector<const HsaCodeImageAsset*> images =
      request.hsa_images.empty() ? std::vector<const HsaCodeImageAsset*>{request.hsa_image}
                                 : request.hsa_images;
  if (images.empty()) return fail("HSA image table is required");
  for (const HsaCodeImageAsset* image : images) {
    if (image == nullptr || image->image.empty() || image->image.size() > 4U * 1024U * 1024U) {
      return fail("HSA image must be nonempty and no larger than 4 MiB");
    }
    if (image->entry_offset >= image->image.size() || (image->entry_offset & 0xffU) != 0) {
      return fail("HSA image entry offset must be a 256-byte-aligned image offset");
    }
    if (image->rsrc1 == 0 || image->rsrc2 == 0 || image->rsrc3 == 0) {
      return fail("HSA image program resources must be nonzero");
    }
  }
  auto validate_stage = [&](uint32_t image_index, uint64_t entry_offset,
                            const std::vector<uint8_t>& kernargs,
                            const std::vector<ResidentHsaKernargBinding>& bindings,
                            uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z,
                            uint32_t global_x, uint32_t global_y, uint32_t global_z) {
    if (image_index >= images.size() || entry_offset >= images[image_index]->image.size() ||
        (entry_offset & 0xffU) != 0) {
      return fail("HSA stage entry offset must be a 256-byte-aligned image offset");
    }
    if (kernargs.empty() || kernargs.size() > kPageSize) {
      return fail("HSA kernargs must be nonempty and fit the C0 kernarg page");
    }
    const uint32_t dimensions[] = {workgroup_x, workgroup_y, workgroup_z,
                                   global_x, global_y, global_z};
    for (uint32_t dimension : dimensions) {
      if (dimension == 0) return fail("HSA dispatch geometry dimensions must be nonzero");
    }
    std::vector<uint32_t> occupied_kernarg_offsets;
    occupied_kernarg_offsets.reserve(bindings.size());
    for (const ResidentHsaKernargBinding& binding : bindings) {
      if (binding.buffer_index >= request.buffers.size() ||
          binding.kernarg_byte_offset > kernargs.size() ||
          kernargs.size() - binding.kernarg_byte_offset < sizeof(uint64_t)) {
        return fail("HSA kernarg binding is outside its buffer or kernarg layout");
      }
      for (uint32_t occupied : occupied_kernarg_offsets) {
        if (binding.kernarg_byte_offset < occupied + sizeof(uint64_t) &&
            occupied < binding.kernarg_byte_offset + sizeof(uint64_t)) {
          return fail("HSA kernarg bindings must not overlap");
        }
      }
      occupied_kernarg_offsets.push_back(binding.kernarg_byte_offset);
    }
    return true;
  };
  if (request.buffers.empty()) return fail("HSA dispatch requires at least one resident buffer");
  for (size_t index = 0; index < request.buffers.size(); ++index) {
    const ResidentHsaBuffer& buffer = request.buffers[index];
    if (buffer.name.empty()) return fail("HSA resident buffer names must be nonempty");
    for (size_t prior = 0; prior < index; ++prior) {
      if (request.buffers[prior].name == buffer.name) {
        return fail("HSA resident buffer names must not overlap");
      }
    }
    if (buffer.allocation_byte_count == 0 ||
        buffer.upload_bytes.size() > buffer.allocation_byte_count ||
        buffer.readback_byte_count > buffer.allocation_byte_count) {
      return fail("HSA resident buffer spans must fit their explicit allocation");
    }
  }
  if (request.stages.empty()) {
    return validate_stage(0, images[0]->entry_offset, request.kernargs, request.kernarg_bindings,
                          request.workgroup_x, request.workgroup_y, request.workgroup_z,
                          request.global_x, request.global_y, request.global_z);
  }
  for (const ResidentHsaStage& stage : request.stages) {
    if (!validate_stage(stage.hsa_image_index, stage.entry_offset, stage.kernargs,
                        stage.kernarg_bindings, stage.workgroup_x, stage.workgroup_y,
                        stage.workgroup_z, stage.global_x, stage.global_y, stage.global_z)) {
      return false;
    }
  }
  return true;
}


bool run_resident_kernel_dispatch(const ResidentKernelDispatch& request,
                                  ResidentKernelDispatchResult* result,
                                  std::string* error_text) {
  auto fail = [&](const char* stage, const std::string& text) {
    result->failure_stage = stage;
    if (error_text != nullptr) *error_text = text;
    return false;
  };

  // C0A25 kernel proof L6343-6437: exact connection, discovery, mapping, and
  // fixed-VM setup sequence. No C1 bridge or host allocation is involved.
  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kPageSize, 0, "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kPageSize, 0, "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
  VmBufferLog compute_control{"compute_control", am_compute::kRptrVa,
                              am_compute::kComputeControlByteCount, 0, "not_run", {}};
  HardwareLock hardware_lock;
  std::string lock_error;
  if (!hardware_lock.acquire(&lock_error)) {
    return fail("hardware_ownership", "hardware lock acquire failed: " + lock_error);
  }

  UniqueFd socket_fd;
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  SysmemMapping compute_control_mapping;
  std::string detail;

  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &detail)) {
    return fail("tinygpu_connect", detail);
  }
  std::string health_error;
  if (!hardware_lock_health_check(log.socket_path, &health_error)) {
    return fail("hardware_ownership", health_error);
  }

  const RemoteClient client(socket_fd.get());
  TerminalComputeQueue0Retirement compute_queue_retirement(client, log);
  auto fail_after_compute_queue_setup = [&](const char* stage, const std::string& text) {
    std::string retirement_detail;
    if (!compute_queue_retirement.retire(&retirement_detail)) {
      return fail("compute_queue_retirement",
                  text + "; terminal queue-0 retirement failed: " + retirement_detail);
    }
    return fail(stage, text);
  };
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  if (!config.ok) return fail("config_read", rpc_failure_text("CFG_READ vendor_device", config));
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    return fail("config_read", "expected 1002:7551, observed " +
                                   pci_id_text(log.config_vendor_id, log.config_device_id));
  }
  log.pci_id = pci_id_text(log.config_vendor_id, log.config_device_id);
  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result)) {
    return fail("map_bar0", rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(client, 2, &log.bar2, &bar_result)) {
    return fail("map_bar2", rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(client, 5, &log.bar5, &bar_result)) {
    return fail("map_bar5", rpc_failure_text("MAP_BAR bar5", bar_result));
  }
  if (!try_discover_arch(client, &log, &detail)) return fail("arch_discovery", detail);
  result->hardware_identity = std::string(kRuntimeSubstrate) + " pci_id=" + log.pci_id +
                              " arch=" + log.arch;
  if (!map_sysmem_buffer(client, &staging, &staging_mapping, &detail) ||
      !map_sysmem_buffer(client, &readback, &readback_mapping, &detail) ||
      !map_sysmem_buffer(client, &sdma_control, &sdma_control_mapping, &detail) ||
      !map_sysmem_buffer(client, &compute_control, &compute_control_mapping, &detail)) {
    return fail("vm_mapping", detail);
  }
  if (staging_mapping.size < kPageSize || readback_mapping.size < kPageSize ||
      sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t) ||
      compute_control_mapping.size < am_compute::kComputeControlByteCount ||
      compute_control.sys_pages.size() < 10) {
    return fail("vm_mapping", "C0 fixed dispatch mappings are smaller than required");
  }
  std::memset(staging_mapping.data, 0, staging_mapping.size);
  std::memcpy(staging_mapping.data, request.input_bytes.data(), request.input_bytes.size());
  std::memset(readback_mapping.data, 0, readback_mapping.size);
  std::memset(sdma_control_mapping.data, 0, sdma_control_mapping.size);
  std::memset(compute_control_mapping.data, 0, compute_control_mapping.size);
  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(client, &log, staging, readback, sdma_control, &compute_control,
                              true, &vm_result)) {
    return fail(vm_result.failure_stage.c_str(), vm_result.error_text);
  }

  // C0A25 L6439-6461: SDMA upload plus caller-visible fence poll.
  if (!setup_sdma_queue0(client, &log, &detail)) return fail("sdma_ring_setup", detail);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, staging.gpu_va,
                        am_compute::kInputVramVa,
                        static_cast<uint32_t>(request.input_bytes.size()), am_sdma::kFenceValue, 0,
                        &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    return fail("sdma_h2d", detail);
  }

  // C0A25 L6463-6509: compute-ring setup, reviewed asset load/kernarg bind,
  // descriptor-sourced PM4 submit, and timeline poll.
  if (!setup_compute_ring0(client, &log, &compute_control_mapping, &detail)) {
    return fail("compute_ring_setup", detail);
  }
  compute_queue_retirement.arm();
  if (!load_resident_kernel_asset(client, log.bar0.size, request.kernel, &detail)) {
    return fail_after_compute_queue_setup("kernel_asset_load", detail);
  }
  if (!bind_resident_kernel_kernargs(request, &compute_control_mapping, &detail)) {
    return fail_after_compute_queue_setup("kernarg_bind", detail);
  }
  const Pm4DispatchConfig pm4{am_compute::kCodeVramVa, am_compute::kKernargsVa,
                              am_compute::kTimelineVa, request.kernel.rsrc1,
                              request.kernel.rsrc2, request.kernel.rsrc3, false,
                              request.kernel.workgroup_x, request.kernel.workgroup_y,
                              request.kernel.workgroup_z, request.kernel.global_x,
                              request.kernel.global_y, request.kernel.global_z};
  const std::vector<uint32_t> dispatch_words = build_pm4_dispatch_words(pm4);
  log_compute_queue_diagnostics(client, log);
  long elapsed_usec = 0;
  if (!submit_compute_dispatch_with_post_doorbell_diagnostics(
          client, &log, &compute_control_mapping, dispatch_words, &detail)) {
    return fail_after_compute_queue_setup("pm4_submit", detail);
  }
  if (!poll_compute_timeline_with_consumption_diagnostics(
          client, &log, compute_control_mapping, &elapsed_usec, &detail)) {
    return fail_after_compute_queue_setup("compute_fence_poll", detail);
  }

  // C0A25 L6544-6571: D2H SDMA readback plus the same fence poll.
  std::memset(static_cast<uint8_t*>(sdma_control_mapping.data) + am_sdma::kFenceOffset, 0,
              sizeof(uint32_t));
  std::atomic_thread_fence(std::memory_order_seq_cst);
  constexpr uint64_t kD2hSubmitByteOffset =
      static_cast<uint64_t>(kSdmaLinearCopyPacketDwords + am_sdma::kFencePacketDwords) *
      sizeof(uint32_t);
  if (!submit_sdma_copy(client, &log, &sdma_control_mapping, am_compute::kOutputVramVa,
                        readback.gpu_va, request.output_byte_count, am_sdma::kFenceValue,
                        kD2hSubmitByteOffset, &detail) ||
      !poll_sdma_fence(sdma_control_mapping, &detail)) {
    return fail_after_compute_queue_setup("sdma_d2h", detail);
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const uint8_t* bytes = static_cast<const uint8_t*>(readback_mapping.data);
  result->output_bytes.assign(bytes, bytes + request.output_byte_count);
  result->transfer_bytes = static_cast<uint64_t>(request.kernel.code.size()) +
                           request.kernargs.size() + request.input_bytes.size() +
                           request.output_byte_count;
  std::string retirement_detail;
  if (!compute_queue_retirement.retire(&retirement_detail)) {
    return fail("compute_queue_retirement",
                "terminal queue-0 retirement failed: " + retirement_detail);
  }
  result->failure_stage = "none";
  return true;
}

}  // namespace



struct ScopedUsec {
  long* target;
  timeval start{};
  explicit ScopedUsec(long* out) : target(out) { gettimeofday(&start, nullptr); }
  ~ScopedUsec() {
    timeval now{}; gettimeofday(&now, nullptr);
    *target += (now.tv_sec - start.tv_sec) * 1000000L + (now.tv_usec - start.tv_usec);
  }
};

struct SdmaRingState {
  uint64_t put_bytes = 0;
  uint64_t next_fence = 1;
};

struct ResidentHsaSession::Impl {
  struct Image {
    uint32_t rsrc1 = 0;
    uint32_t rsrc2 = 0;
    uint32_t rsrc3 = 0;
    uint64_t byte_count = 0;
    bool wave32 = false;
  };

  DiscoveryLog log;
  PhaseTimers phase_timers;
  // Close-time snapshot of the phase accumulator, captured before the reset so
  // callers can read the completed breakdown after close().
  PhaseTimers final_timers;
  UniqueFd socket_fd;
  std::unique_ptr<HardwareLock> hardware_lock;

  std::unique_ptr<RemoteClient> client;
  VmBufferLog staging{"staging", kTransferProofVmStagingVa, kResidentStagingByteCount, 0,
                      "not_run", {}};
  VmBufferLog readback{"readback", kTransferProofVmReadbackVa, kPageSize, 0, "not_run", {}};
  VmBufferLog sdma_control{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
  VmBufferLog compute_control{"compute_control", am_compute::kRptrVa,
                              am_compute::kComputeControlByteCount, 0, "not_run", {}};
  SysmemMapping staging_mapping;
  SysmemMapping readback_mapping;
  SysmemMapping sdma_control_mapping;
  SysmemMapping compute_control_mapping;
  VramLayout layout{};
  std::unique_ptr<VramAllocator> payload_allocator;
  std::unique_ptr<VramAllocator> page_table_allocator;
  VramSmokeResult pte_result;
  std::unique_ptr<C0DynamicPageTableBackend> pte_backend;
  std::unique_ptr<DynamicPageTable> page_table;
  std::unique_ptr<ResidentMemory> resident;
  std::unique_ptr<TerminalComputeQueue0Retirement> compute_queue_retirement;
  std::vector<Image> images;
  std::vector<ResidentBuffer> image_buffers;
  std::vector<ResidentBuffer> buffers;
  std::vector<std::string> buffer_names;
  std::vector<uint64_t> readback_byte_counts;
  std::vector<uint8_t> post_prepare_upload_allowed;
  std::vector<uint64_t> requested_allocation_byte_counts;
  bool mapping_uncertain = false;
  uint64_t uncertain_gpu_va = 0;
  std::string uncertain_text;
  std::string release_error;
  bool prepared = false;
  // Monotonic RELEASE_MEM timeline value assigned to each stage's PM4 stream.
  // The terminal poll waits for (next_timeline_value - 1), so no per-dispatch
  // timeline reset may occur between stages of a batch.
  uint32_t next_timeline_value = 1;
  SdmaRingState sdma_ring;

  const char* first_pte_failure() const {
    if (pte_result.pte_map_status == "fail") return "pte_map";
    if (pte_result.bar0_zero_status == "fail") return "bar0_zero";
    if (pte_result.pte_write_status == "fail") return "pte_write";
    if (pte_result.pte_readback_status == "fail") return "pte_readback";
    if (pte_result.mmhub_tlb_flush_status == "fail") return "mmhub_tlb_flush";
    if (pte_result.gc_tlb_flush_status == "fail") return "gc_tlb_flush";
    return nullptr;
  }

  bool release_resident(std::string* error_text) {
    if (resident == nullptr) return true;
    release_error.clear();
    resident->release_all();
    std::string deferred_flush_error;
    if (pte_backend != nullptr && !pte_backend->flush_deferred(&deferred_flush_error)) {
      release_error = deferred_flush_error;
    }
    if (mapping_uncertain || !release_error.empty() || first_pte_failure() != nullptr) {
      if (error_text != nullptr) {
        *error_text = !release_error.empty() ? release_error
                    : !uncertain_text.empty() ? uncertain_text
                    : "resident VRAM cleanup did not complete";
      }
      return false;
    }
    return true;
  }

  void reset_after_close() {
    compute_queue_retirement.reset();
    resident.reset();
    page_table.reset();
    pte_backend.reset();
    page_table_allocator.reset();
    payload_allocator.reset();
    image_buffers.clear();
    buffers.clear();
    buffer_names.clear();
    readback_byte_counts.clear();
    post_prepare_upload_allowed.clear();
    requested_allocation_byte_counts.clear();
    images.clear();
    client.reset();
    staging_mapping.reset();
    readback_mapping.reset();
    sdma_control_mapping.reset();
    compute_control_mapping.reset();
    socket_fd.reset();
    hardware_lock.reset();
    log = DiscoveryLog{};
    phase_timers = PhaseTimers{};

    staging = VmBufferLog{"staging", kTransferProofVmStagingVa, kResidentStagingByteCount, 0,
                          "not_run", {}};
    readback = VmBufferLog{"readback", kTransferProofVmReadbackVa, kPageSize, 0, "not_run", {}};
    sdma_control = VmBufferLog{"sdma_control", am_sdma::kControlVa, kPageSize, 0, "not_run", {}};
    compute_control = VmBufferLog{"compute_control", am_compute::kRptrVa,
                                  am_compute::kComputeControlByteCount, 0, "not_run", {}};
    layout = VramLayout{};
    pte_result = VramSmokeResult{};
    mapping_uncertain = false;
    uncertain_gpu_va = 0;
    uncertain_text.clear();
    release_error.clear();
    prepared = false;
    next_timeline_value = 1;
    sdma_ring = SdmaRingState{};
  }
  bool submit_sdma_chunk_persistent(uint64_t src_va, uint64_t dst_va, uint32_t byte_count,
                                    std::string* error_text) {
    const uint64_t submit_byte_offset = sdma_ring.put_bytes % am_sdma::kRingSize;
    const uint32_t fence_value = static_cast<uint32_t>(sdma_ring.next_fence);
    const std::vector<uint32_t> words =
        build_sdma_copy_submit_words(src_va, dst_va, byte_count, am_sdma::kFenceVa, fence_value);
    if (!write_sdma_ring_words_wrap(&sdma_control_mapping, words,
                                    submit_byte_offset, error_text)) {
      log.sdma.submit_status = "fail";
      return false;
    }
    const uint64_t new_put_bytes =
        sdma_ring.put_bytes + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
    if (!write_control_u64(&sdma_control_mapping, am_sdma::kWptrOffset,
                           new_put_bytes, error_text)) {
      log.sdma.submit_status = "fail";
      return false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const std::vector<uint8_t> doorbell = u64_payload_le(new_put_bytes);
    if (!client->mmio_write_fire_and_forget(2, am_sdma::kDoorbellBar2ByteOffset,
                                            doorbell, error_text)) {
      log.sdma.submit_status = "fail";
      return false;
    }
    sdma_ring.put_bytes = new_put_bytes;
    ++sdma_ring.next_fence;
    log.sdma.submit_status = "pass";
    ScopedUsec timer(&phase_timers.sdma_fence_wait_usec);
    return poll_sdma_fence(sdma_control_mapping, fence_value, error_text);
  }
};
ResidentHsaSession::ResidentHsaSession() : impl_(std::make_unique<Impl>()) {}

ResidentHsaSession::~ResidentHsaSession() {
  std::string close_error;
  if (!close(&close_error)) std::abort();
}


bool ResidentHsaSession::prepare(const ResidentHsaDispatch& request,
                                 ResidentHsaDispatchResult* result,
                                 std::string* error_text) {
  auto fail = [&](const char* stage, const std::string& text) {
    if (result != nullptr) result->failure_stage = stage;
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (result == nullptr) return fail("preflight", "resident HSA dispatch result is required");
  *result = ResidentHsaDispatchResult{};
  if (impl_->prepared || impl_->resident != nullptr || impl_->client != nullptr) {
    return fail("prepare", "resident HSA session must close before it can be prepared again");
  }

  const std::vector<const HsaCodeImageAsset*> source_images =
      request.hsa_images.empty() ? std::vector<const HsaCodeImageAsset*>{request.hsa_image}
                                 : request.hsa_images;
  if (source_images.empty()) return fail("preflight", "HSA image table is required");
  for (const HsaCodeImageAsset* image : source_images) {
    if (image == nullptr || image->image.empty() || image->image.size() > 4U * 1024U * 1024U) {
      return fail("preflight", "HSA image must be nonempty and no larger than 4 MiB");
    }
    if (image->entry_offset >= image->image.size() || (image->entry_offset & 0xffU) != 0) {
      return fail("preflight", "HSA image entry offset must be a 256-byte-aligned image offset");
    }
    if (image->rsrc1 == 0 || image->rsrc2 == 0 || image->rsrc3 == 0) {
      return fail("preflight", "HSA image PM4 resource registers must be nonzero");
    }
  }
  if (request.buffers.empty()) return fail("preflight", "HSA dispatch requires at least one resident buffer");
  for (size_t index = 0; index < request.buffers.size(); ++index) {
    const ResidentHsaBuffer& buffer = request.buffers[index];
    if (buffer.name.empty()) return fail("preflight", "HSA resident buffer names must be nonempty");
    for (size_t prior = 0; prior < index; ++prior) {
      if (request.buffers[prior].name == buffer.name) {
        return fail("preflight", "HSA resident buffer names must not overlap");
      }
    }
    if (buffer.allocation_byte_count == 0 ||
        buffer.upload_bytes.size() > buffer.allocation_byte_count ||
        buffer.readback_byte_count > buffer.allocation_byte_count) {
      return fail("preflight", "HSA resident buffer spans must fit their explicit allocation");
    }
  }

  Impl& state = *impl_;
  state.log.socket_path = tinygpu_socket_path();
  state.hardware_lock = std::make_unique<HardwareLock>();
  std::string lock_error;
  if (!state.hardware_lock->acquire(&lock_error)) {
    return fail("hardware_ownership", "hardware lock acquire failed: " + lock_error);
  }

  std::string detail;
  if (!connect_tinygpu_server(state.log.socket_path, &state.socket_fd, &detail)) {
    return fail("tinygpu_connect", detail);
  }
  std::string health_error;
  if (!hardware_lock_health_check(state.log.socket_path, &health_error)) {
    return fail("hardware_ownership", health_error);
  }

  state.client = std::make_unique<RemoteClient>(state.socket_fd.get());
  RemoteRpcResult config = state.client->rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  if (!config.ok) return fail("config_read", rpc_failure_text("CFG_READ vendor_device", config));
  state.log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  state.log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (state.log.config_vendor_id != kTargetVendor || state.log.config_device_id != kTargetDevice) {
    return fail("config_read", "expected 1002:7551, observed " +
                               pci_id_text(state.log.config_vendor_id, state.log.config_device_id));
  }
  state.log.pci_id = pci_id_text(state.log.config_vendor_id, state.log.config_device_id);
  RemoteRpcResult bar_result;
  if (!map_bar(*state.client, 0, &state.log.bar0, &bar_result)) {
    return fail("map_bar0", rpc_failure_text("MAP_BAR bar0", bar_result));
  }
  if (!map_bar(*state.client, 2, &state.log.bar2, &bar_result)) {
    return fail("map_bar2", rpc_failure_text("MAP_BAR bar2", bar_result));
  }
  if (!map_bar(*state.client, 5, &state.log.bar5, &bar_result)) {
    return fail("map_bar5", rpc_failure_text("MAP_BAR bar5", bar_result));
  }
  if (!try_discover_arch(*state.client, &state.log, &detail)) return fail("arch_discovery", detail);
  result->hardware_identity = std::string(kRuntimeSubstrate) + " pci_id=" + state.log.pci_id +
                              " arch=" + state.log.arch;
  if (!map_sysmem_buffer(*state.client, &state.staging, &state.staging_mapping, &detail) ||
      !map_sysmem_buffer(*state.client, &state.readback, &state.readback_mapping, &detail) ||
      !map_sysmem_buffer(*state.client, &state.sdma_control, &state.sdma_control_mapping, &detail) ||
      !map_sysmem_buffer(*state.client, &state.compute_control, &state.compute_control_mapping, &detail)) {
    return fail("vm_mapping", detail);
  }

  if (state.staging_mapping.size < kResidentStagingByteCount ||
      state.readback_mapping.size < kPageSize ||
      state.sdma_control_mapping.size < am_sdma::kFenceOffset + sizeof(uint32_t) ||
      state.compute_control_mapping.size < am_compute::kComputeControlByteCount ||
      state.compute_control.sys_pages.size() < 10) {
    return fail("vm_mapping", "C0 fixed dispatch mappings are smaller than required");
  }
  std::memset(state.staging_mapping.data, 0, state.staging_mapping.size);
  std::memset(state.readback_mapping.data, 0, state.readback_mapping.size);
  std::memset(state.sdma_control_mapping.data, 0, state.sdma_control_mapping.size);
  std::memset(state.compute_control_mapping.data, 0, state.compute_control_mapping.size);
  FixedVmMappingResult vm_result;
  if (!setup_fixed_vm_mapping(*state.client, &state.log, state.staging, state.readback,
                              state.sdma_control, &state.compute_control, true, &vm_result,
                              &state.phase_timers.hdp_flush_usec)) {
    return fail(vm_result.failure_stage.c_str(), vm_result.error_text);
  }
  const uint64_t vram_mib = state.log.vram_size_bytes >> 20U;
  if (vram_mib > std::numeric_limits<uint32_t>::max()) {
    return fail("vram_layout", "discovered VRAM MiB does not fit RCC_CONFIG_MEMSIZE");
  }
  if (!derive_vram_layout(static_cast<uint32_t>(vram_mib), state.log.bar0.size, &state.layout,
                          &detail) || state.layout.large_bar) {
    return fail("vram_layout", detail.empty() ? "large BAR0 is unsupported" : detail);
  }
  if (state.layout.page_table_pool_base == 0 || state.layout.page_table_pool_bytes == 0 ||
      state.layout.resident_gpu_va_base >= kSmokePtbBoundaryGpuVa ||
      state.layout.resident_gpu_va_limit < kSmokePtbBoundaryGpuVa + kPageSize) {
    return fail("vram_layout", "resident window cannot host the dynamic HSA image table");
  }
  state.payload_allocator = std::make_unique<VramAllocator>(state.layout);
  VramLayout page_table_layout = state.layout;
  page_table_layout.allocatable_base = state.layout.page_table_pool_base + kPageSize;
  page_table_layout.allocatable_bytes = state.layout.page_table_pool_bytes - kPageSize;
  state.page_table_allocator = std::make_unique<VramAllocator>(page_table_layout);
  state.pte_backend =
      std::make_unique<C0DynamicPageTableBackend>(*state.client, &state.log, &state.pte_result);
  state.page_table = std::make_unique<DynamicPageTable>(
      state.layout, *state.page_table_allocator, state.pte_backend.get(),
      FixedPageTablePages{vm_result.tables.root_pdb2_paddr, vm_result.tables.child_pdb1_paddr,
                          vm_result.tables.child_pdb0_paddr, vm_result.tables.child_ptb_paddr});
  state.resident = std::make_unique<ResidentMemory>(
      state.layout, *state.payload_allocator,
      [&state](ResidentPageOperation operation, uint64_t gpu_va, uint64_t physical_page,
               std::string* map_error) {
        if (operation == ResidentPageOperation::kUnmap) {
          if (state.mapping_uncertain && gpu_va != state.uncertain_gpu_va) return false;
          const bool unmapped = state.page_table->unmap_range(gpu_va, kPageSize, map_error);
          if (!unmapped && map_error != nullptr && !map_error->empty()) state.release_error = *map_error;
          if (unmapped && state.mapping_uncertain) {
            state.mapping_uncertain = false;
            state.uncertain_gpu_va = 0;
            state.uncertain_text.clear();
          }
          return unmapped;
        }
        if (state.mapping_uncertain) return true;
        if (state.page_table->map_range(gpu_va, physical_page, kPageSize, map_error)) {
          state.pte_result.pte_map_status = "pass";
          return true;
        }
        state.pte_result.pte_map_status = "fail";
        const std::string map_failure =
            map_error != nullptr && !map_error->empty() ? *map_error : "page-table map failed";
        std::string cleanup_error;
        if (state.page_table->unmap_range(gpu_va, kPageSize, &cleanup_error)) return false;
        state.mapping_uncertain = true;
        state.uncertain_gpu_va = gpu_va;
        state.uncertain_text = map_failure + "; failed to prove page-table cleanup: " + cleanup_error;
        if (map_error != nullptr) *map_error = state.uncertain_text;
        return true;
      });
  state.pte_backend->begin_deferred_flushes();
  auto fail_after_resident = [&](const char* stage, const std::string& text) {
    if (state.compute_queue_retirement != nullptr) {
      std::string retirement_error;
      if (!state.compute_queue_retirement->retire(&retirement_error)) {
        return fail("compute_queue_retirement",
                    text + "; terminal queue-0 retirement failed: " + retirement_error);
      }
    }
    std::string cleanup_error;
    if (!state.release_resident(&cleanup_error)) {
      return fail(state.first_pte_failure() != nullptr ? state.first_pte_failure() : "resident_cleanup",
                  text + "; " + cleanup_error);
    }
    return fail(stage, text);
  };
  uint32_t mapped_resident_count = 0;
  auto flush_mapping_batch = [&]() {
    if (++mapped_resident_count % 4U != 0U) return true;
    if (!state.pte_backend->flush_deferred(&detail)) return false;
    state.pte_backend->begin_deferred_flushes();
    return true;
  };
  const uint64_t boundary_bytes = kSmokePtbBoundaryGpuVa - state.layout.resident_gpu_va_base;
  ResidentBuffer boundary_guard{};
  state.image_buffers.resize(source_images.size());
  state.buffers.resize(request.buffers.size());
  if (!state.resident->allocate("resident-hsa-ptb-boundary", boundary_bytes, &boundary_guard,
                                &detail)) {
    return fail_after_resident("vram_allocation", detail);
  }
  if (!flush_mapping_batch()) return fail_after_resident("gc_tlb_flush", detail);
  for (size_t index = 0; index < source_images.size(); ++index) {
    if (!state.resident->allocate("resident-hsa-image-" + std::to_string(index),
                                  source_images[index]->image.size(), &state.image_buffers[index],
                                  &detail)) {
      return fail_after_resident("vram_allocation", detail);
    }
    if (!flush_mapping_batch()) return fail_after_resident("gc_tlb_flush", detail);
  }
  for (size_t index = 0; index < request.buffers.size(); ++index) {
    if (!state.resident->allocate(request.buffers[index].name,
                                  request.buffers[index].allocation_byte_count,
                                  &state.buffers[index], &detail)) {
      return fail_after_resident("vram_allocation", detail);
    }
    if (!flush_mapping_batch()) return fail_after_resident("gc_tlb_flush", detail);
  }
  if (!state.pte_backend->flush_deferred(&detail)) {
    return fail_after_resident("gc_tlb_flush", detail);
  }
  if (state.image_buffers[0].gpu_va != kSmokePtbBoundaryGpuVa ||
      state.page_table->dynamic_ptb_count() == 0) {
    return fail_after_resident("pte_map", "resident HSA image table did not force a dynamic PTB");
  }
  result->hsa_image_gpu_va = state.image_buffers[0].gpu_va;
  result->hsa_image_physical_offset = state.image_buffers[0].allocation.physical_offset;
  result->dynamic_ptb_count = state.page_table->dynamic_ptb_count();
  result->dynamic_ptb_physical_offset = state.page_table->first_dynamic_ptb_physical_offset();
  for (size_t index = 0; index < source_images.size(); ++index) {
    const HsaCodeImageAsset& image = *source_images[index];
    state.images.push_back(
        Impl::Image{image.rsrc1, image.rsrc2, image.rsrc3, image.image.size(), image.wave32});
    result->hsa_image_gpu_vas.push_back(state.image_buffers[index].gpu_va);
    result->hsa_image_physical_offsets.push_back(state.image_buffers[index].allocation.physical_offset);
    if (!mmio_write_bar0(*state.client, state.image_buffers[index].allocation.physical_offset,
                         image.image, &detail)) {
      return fail_after_resident("hsa_image_write", detail);
    }
    RemoteRpcResult image_readback = mmio_read(*state.client, 0,
                                                state.image_buffers[index].allocation.physical_offset,
                                                image.image.size());
    if (!image_readback.ok || image_readback.readout != image.image) {
      return fail_after_resident("bar0_hsa_image_readback",
                                 image_readback.ok ? "HSA image BAR0 readback mismatch"
                                                   : rpc_failure_text("MMIO_READ BAR0 HSA image",
                                                                      image_readback));
    }
  }
  {
    ScopedUsec timer(&state.phase_timers.sdma_setup_usec);
    ++state.phase_timers.sdma_setup_count;
    if (!setup_sdma_queue0(*state.client, &state.log, &detail)) {
      return fail_after_resident("sdma_queue_setup", detail);
    }
  }
  for (size_t index = 0; index < request.buffers.size(); ++index) {
    state.buffer_names.push_back(request.buffers[index].name);
    state.readback_byte_counts.push_back(request.buffers[index].readback_byte_count);
    state.post_prepare_upload_allowed.push_back(
        request.buffers[index].allow_post_prepare_upload ? 1U : 0U);
    state.requested_allocation_byte_counts.push_back(request.buffers[index].allocation_byte_count);
    result->buffer_names.push_back(request.buffers[index].name);
    result->buffer_gpu_vas.push_back(state.buffers[index].gpu_va);
    result->buffer_physical_offsets.push_back(state.buffers[index].allocation.physical_offset);
    const std::vector<uint8_t>& upload = request.buffers[index].upload_bytes;
    uint64_t offset = 0;
    while (offset < upload.size()) {

      const uint32_t chunk =
          static_cast<uint32_t>(std::min<uint64_t>(kResidentStagingByteCount, upload.size() - offset));
      {
        ScopedUsec timer(&state.phase_timers.staging_copy_usec);
        std::memcpy(state.staging_mapping.data, upload.data() + offset, chunk);
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      {
        ScopedUsec timer(&state.phase_timers.sdma_submit_usec);
        if (!state.submit_sdma_chunk_persistent(state.staging.gpu_va,
                                               state.buffers[index].gpu_va + offset, chunk,
                                               &detail)) {
          return fail_after_resident("sdma_h2d", detail);
        }
      }
      offset += chunk;
    }
    result->sdma_upload_bytes += upload.size();
  }
  state.compute_queue_retirement =
      std::make_unique<TerminalComputeQueue0Retirement>(*state.client, state.log);
  state.compute_queue_retirement->arm();
  if (!setup_compute_ring0(*state.client, &state.log, &state.compute_control_mapping, &detail)) {
    return fail_after_resident("compute_ring_setup", detail);
  }
  state.prepared = true;
  state.next_timeline_value = 1;
  result->failure_stage = "none";
  return true;
}

bool ResidentHsaSession::dispatch(const ResidentHsaStage& stage,
                                  ResidentHsaDispatchResult* result,
                                  std::string* error_text) {
  auto fail = [&](const char* failure_stage, const std::string& text) {
    if (result != nullptr) result->failure_stage = failure_stage;
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (result == nullptr) return fail("dispatch", "resident HSA dispatch result is required");
  Impl& state = *impl_;
  if (!state.prepared) return fail("dispatch", "resident HSA session is not prepared");
  return dispatch_batch({stage}, result, error_text);
}

bool ResidentHsaSession::build_stage_pm4(const ResidentHsaStage& stage, uint32_t slot,
                                         std::vector<uint32_t>* words,
                                         std::string* error_text) {
  Impl& state = *impl_;
  // preflight: image index, entry offset (256-aligned), nonempty kernargs <= slot,
  // nonzero geometry, in-bounds non-overlapping bindings — the former dispatch()
  // checks, sized against kKernargSlotByteCount.
  if (stage.hsa_image_index >= state.images.size() ||
      stage.entry_offset >= state.images[stage.hsa_image_index].byte_count ||
      (stage.entry_offset & 0xffU) != 0 || stage.kernargs.empty() ||
      stage.kernargs.size() > am_compute::kKernargSlotByteCount) {
    *error_text = "HSA stage does not fit the prepared resident image table";
    return false;
  }
  const uint32_t dimensions[] = {stage.workgroup_x, stage.workgroup_y, stage.workgroup_z,
                                 stage.global_x, stage.global_y, stage.global_z};
  for (uint32_t dimension : dimensions) {
    if (dimension == 0) {
      *error_text = "HSA dispatch geometry dimensions must be nonzero";
      return false;
    }
  }
  ResidentKernelDispatch kernarg_request;
  kernarg_request.kernargs = stage.kernargs;
  std::vector<uint32_t> occupied_offsets;
  for (const ResidentHsaKernargBinding& binding : stage.kernarg_bindings) {
    if (binding.buffer_index >= state.buffers.size() ||
        binding.kernarg_byte_offset > kernarg_request.kernargs.size() ||
        kernarg_request.kernargs.size() - binding.kernarg_byte_offset < sizeof(uint64_t)) {
      *error_text = "HSA kernarg binding is outside its buffer or kernarg layout";
      return false;
    }
    for (uint32_t occupied : occupied_offsets) {
      if (binding.kernarg_byte_offset < occupied + sizeof(uint64_t) &&
          occupied < binding.kernarg_byte_offset + sizeof(uint64_t)) {
        *error_text = "HSA kernarg bindings must not overlap";
        return false;
      }
    }
    occupied_offsets.push_back(binding.kernarg_byte_offset);
    store_u64_le(kernarg_request.kernargs.data() + binding.kernarg_byte_offset,
                 state.buffers[binding.buffer_index].gpu_va);
  }
#ifdef NATIVE_R9700_DIAG_DISPATCH
  for (size_t i = 0; i < state.buffers.size(); ++i) {
    std::fprintf(stderr, "DIAG buffer[%zu] gpu_va=0x%016llx phys=0x%016llx\n",
                 i, static_cast<unsigned long long>(state.buffers[i].gpu_va),
                 static_cast<unsigned long long>(state.buffers[i].allocation.physical_offset));
  }
  for (size_t i = 0; i + 8 <= kernarg_request.kernargs.size(); i += 8) {
    uint64_t v = 0;
    for (size_t b = 0; b < 8; ++b)
      v |= static_cast<uint64_t>(kernarg_request.kernargs[i + b]) << (8 * b);
    std::fprintf(stderr, "DIAG kernarg[%zu] = 0x%016llx\n", i / 8,
                 static_cast<unsigned long long>(v));
  }
#endif
  uint64_t slot_va = 0;
  std::string detail;
  if (!bind_resident_kernel_kernargs_slot(kernarg_request, &state.compute_control_mapping,
                                          slot, &slot_va, &detail)) {
    *error_text = detail;
    return false;
  }
  const Impl::Image& image = state.images[stage.hsa_image_index];
  // Diagnostic: override COMPUTE_PGM_RSRC3 (INST_PREF_SIZE on GFX12) without
  // changing the image or its digest (PDF step 4, --override-rsrc3).
  const char* rsrc3_override_env = std::getenv("NATIVE_RSRC3_OVERRIDE");
  const uint32_t rsrc3 = (rsrc3_override_env != nullptr && rsrc3_override_env[0] != '\0')
                             ? static_cast<uint32_t>(std::strtoul(rsrc3_override_env, nullptr, 0))
                             : image.rsrc3;
  const Pm4DispatchConfig pm4{state.image_buffers[stage.hsa_image_index].gpu_va + stage.entry_offset,
                              slot_va, am_compute::kTimelineVa,
                              image.rsrc1, image.rsrc2, rsrc3, image.wave32, stage.workgroup_x,
                              stage.workgroup_y, stage.workgroup_z, stage.global_x,
                              stage.global_y, stage.global_z, state.next_timeline_value++};
  const std::vector<uint32_t> stage_words = build_pm4_dispatch_words(pm4);
  words->insert(words->end(), stage_words.begin(), stage_words.end());
  return true;
}

bool ResidentHsaSession::dispatch_batch(const std::vector<ResidentHsaStage>& stages,
                                        ResidentHsaDispatchResult* result,
                                        std::string* error_text) {
  auto fail = [&](const char* failure_stage, const std::string& text) {
    if (result != nullptr) result->failure_stage = failure_stage;
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (result == nullptr) return fail("dispatch", "resident HSA dispatch result is required");
  Impl& state = *impl_;
  if (!state.prepared) return fail("dispatch", "resident HSA session is not prepared");
  if (stages.empty()) return fail("preflight", "dispatch batch has no stages");
  if (stages.size() > am_compute::kKernargSlotCount)
    return fail("preflight", "dispatch batch exceeds the 10 in-page kernarg slots");

  std::vector<uint32_t> batch;
  batch.reserve(stages.size() * am_compute::kPm4DispatchDwordCount);
  {
    ScopedUsec timer(&state.phase_timers.pm4_build_usec);
    for (size_t index = 0; index < stages.size(); ++index) {
      std::string detail;
      if (!build_stage_pm4(stages[index], static_cast<uint32_t>(index), &batch, &detail))
        return fail("preflight", detail);
    }
  }

  // Single submission: read current WPTR once, write the whole batch, publish once.
  ++state.phase_timers.compute_submit_count;
  uint64_t current_wptr_dwords = 0;
  std::string detail;
  std::memcpy(&current_wptr_dwords,
              static_cast<const uint8_t*>(state.compute_control_mapping.data) +
                  am_compute::kWptrOffset,
              sizeof(uint64_t));
  if (!write_compute_ring_words(&state.compute_control_mapping, batch,
                                current_wptr_dwords, &detail))
    return fail("pm4_batch", detail);
  {
    ScopedUsec timer(&state.phase_timers.hdp_flush_usec);
    if (!flush_hdp(*state.client, state.log, &detail)) return fail("hdp_flush", detail);
  }
  const uint64_t new_wptr_dwords = current_wptr_dwords + batch.size();
  if (!write_compute_control_u64(&state.compute_control_mapping, am_compute::kWptrOffset,
                                 new_wptr_dwords, &detail))
    return fail("pm4_wptr", detail);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  {
    ScopedUsec timer(&state.phase_timers.doorbell_usec);
    if (!state.client->mmio_write_fire_and_forget(
            2, am_compute::kMecDoorbellBar2ByteOffset, u64_payload_le(new_wptr_dwords), &detail))
      return fail("pm4_doorbell", detail);
  }
  long elapsed_usec = 0;
  {
    ScopedUsec timer(&state.phase_timers.timeline_wait_usec);
    if (!poll_compute_timeline_with_consumption_diagnostics(
            *state.client, &state.log, state.compute_control_mapping, &elapsed_usec, &detail,
            state.next_timeline_value - 1))
      return fail("compute_fence_poll", detail);
  }
  result->pm4_dispatch_count += static_cast<uint32_t>(stages.size());
  result->pm4_dispatch_word_count += static_cast<uint64_t>(batch.size());
  result->failure_stage = "none";
  return true;
}

bool ResidentHsaSession::compute_ring_pointers(uint64_t* rptr_dwords, uint64_t* wptr_dwords,
                                               std::string* error_text) {
  if (rptr_dwords == nullptr || wptr_dwords == nullptr) {
    if (error_text != nullptr) *error_text = "compute ring pointer outputs are required";
    return false;
  }
  Impl& state = *impl_;
  if (state.compute_control_mapping.data == nullptr ||
      am_compute::kRptrOffset > state.compute_control_mapping.size ||
      sizeof(uint64_t) > state.compute_control_mapping.size - am_compute::kRptrOffset ||
      am_compute::kWptrOffset > state.compute_control_mapping.size ||
      sizeof(uint64_t) > state.compute_control_mapping.size - am_compute::kWptrOffset) {
    if (error_text != nullptr) *error_text = "compute control mapping cannot hold rptr/wptr";
    return false;
  }
  std::memcpy(rptr_dwords,
              static_cast<const uint8_t*>(state.compute_control_mapping.data) +
                  am_compute::kRptrOffset,
              sizeof(uint64_t));
  std::memcpy(wptr_dwords,
              static_cast<const uint8_t*>(state.compute_control_mapping.data) +
                  am_compute::kWptrOffset,
              sizeof(uint64_t));
  return true;
}

bool ResidentHsaSession::upload_named(const std::string& buffer_name, const uint8_t* bytes,
                                      uint64_t byte_count, ResidentHsaDispatchResult* result,
                                      std::string* error_text) {
  auto fail = [&](const std::string& text) {
    if (result != nullptr) result->failure_stage = "sdma_h2d";
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (result == nullptr) return fail("resident HSA dispatch result is required");
  Impl& state = *impl_;
  if (!state.prepared) return fail("resident HSA session is not prepared");
  if (bytes == nullptr) return fail("resident HSA upload bytes are required");
  size_t buffer_index = state.buffer_names.size();
  for (size_t index = 0; index < state.buffer_names.size(); ++index) {
    if (state.buffer_names[index] == buffer_name) {
      buffer_index = index;
      break;
    }
  }
  if (buffer_index == state.buffer_names.size()) {
    return fail("resident HSA upload buffer does not exist: " + buffer_name);
  }
  if (state.post_prepare_upload_allowed[buffer_index] == 0) {
    return fail("resident HSA upload is allowed only for an opted-in weight window: " + buffer_name);
  }
  if (byte_count != state.requested_allocation_byte_counts[buffer_index]) {

    return fail("resident HSA upload must exactly fill its declared weight-window allocation");
  }

  uint64_t offset = 0;
  std::string detail;

  while (offset < byte_count) {
    const uint32_t chunk =
        static_cast<uint32_t>(std::min<uint64_t>(kResidentStagingByteCount, byte_count - offset));
    {
      ScopedUsec timer(&state.phase_timers.staging_copy_usec);
      std::memcpy(state.staging_mapping.data, bytes + offset, chunk);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    {
      ScopedUsec timer(&state.phase_timers.sdma_submit_usec);
      if (!state.submit_sdma_chunk_persistent(state.staging.gpu_va,
                                              state.buffers[buffer_index].gpu_va + offset, chunk,
                                              &detail)) {
        return fail(detail);
      }
    }
    offset += chunk;
  }
  result->sdma_upload_bytes += byte_count;
  result->failure_stage = "none";
  return true;
}

bool ResidentHsaSession::readback(const std::vector<std::string>& names,
                                  ResidentHsaDispatchResult* result,
                                  std::string* error_text) {
  auto fail = [&](const std::string& text) {
    if (result != nullptr) result->failure_stage = "sdma_d2h";
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (result == nullptr) return fail("resident HSA dispatch result is required");
  Impl& state = *impl_;
  if (!state.prepared) return fail("resident HSA session is not prepared");
  result->readback_bytes.clear();
  result->readback_bytes.resize(names.size());
  for (size_t requested_index = 0; requested_index < names.size(); ++requested_index) {
    size_t buffer_index = state.buffer_names.size();
    for (size_t index = 0; index < state.buffer_names.size(); ++index) {
      if (state.buffer_names[index] == names[requested_index]) {
        buffer_index = index;
        break;
      }
    }
    if (buffer_index == state.buffer_names.size()) {
      return fail("requested resident readback buffer does not exist: " + names[requested_index]);
    }
    const uint64_t byte_count = state.readback_byte_counts[buffer_index];
    std::vector<uint8_t>& observed = result->readback_bytes[requested_index];
    observed.resize(byte_count);
    uint64_t offset = 0;
    std::string detail;
    while (offset < byte_count) {
      const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(kPageSize, byte_count - offset));
      {
        ScopedUsec timer(&state.phase_timers.sdma_submit_usec);
        if (!state.submit_sdma_chunk_persistent(state.buffers[buffer_index].gpu_va + offset,
                                               state.readback.gpu_va, chunk, &detail)) {
          return fail(detail);
        }
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      std::memcpy(observed.data() + offset, state.readback_mapping.data, chunk);
      offset += chunk;
    }
    result->sdma_download_bytes += byte_count;
  }
  result->failure_stage = "none";
  return true;
}

bool ResidentHsaSession::close(std::string* error_text) {
  Impl& state = *impl_;
  if (state.resident == nullptr) {
    state.reset_after_close();
    return true;
  }
  if (state.client != nullptr) {
    state.phase_timers.socket_rpc_count = state.client->rpc_count;
  }
  std::printf("phase_timer model_load_usec: %ld\n", state.phase_timers.model_load_usec);
  std::printf("phase_timer staging_copy_usec: %ld\n", state.phase_timers.staging_copy_usec);
  std::printf("phase_timer sdma_setup_usec: %ld\n", state.phase_timers.sdma_setup_usec);
  std::printf("phase_timer sdma_submit_usec: %ld\n", state.phase_timers.sdma_submit_usec);
  std::printf("phase_timer sdma_fence_wait_usec: %ld\n", state.phase_timers.sdma_fence_wait_usec);
  std::printf("phase_timer pm4_build_usec: %ld\n", state.phase_timers.pm4_build_usec);
  std::printf("phase_timer hdp_flush_usec: %ld\n", state.phase_timers.hdp_flush_usec);
  std::printf("phase_timer doorbell_usec: %ld\n", state.phase_timers.doorbell_usec);
  std::printf("phase_timer timeline_wait_usec: %ld\n", state.phase_timers.timeline_wait_usec);
  std::printf("phase_counter sdma_setup_count: %llu\n",
              static_cast<unsigned long long>(state.phase_timers.sdma_setup_count));
  std::printf("phase_counter compute_submit_count: %llu\n",
              static_cast<unsigned long long>(state.phase_timers.compute_submit_count));
  std::printf("phase_counter socket_rpc_count: %llu\n",
              static_cast<unsigned long long>(state.phase_timers.socket_rpc_count));

  std::string detail;
  if (state.compute_queue_retirement != nullptr &&
      !state.compute_queue_retirement->retire(&detail)) {
    if (error_text != nullptr) *error_text = "terminal queue-0 retirement failed: " + detail;
    return false;
  }
  if (!state.release_resident(&detail)) {
    if (error_text != nullptr) *error_text = "resident VRAM cleanup did not complete: " + detail;
    return false;
  }
  state.final_timers = state.phase_timers;
  state.reset_after_close();
  return true;
}

const PhaseTimers& ResidentHsaSession::phase_timers() const {
  // After close, the accumulator is reset; return the close-time snapshot.
  return impl_->resident == nullptr ? impl_->final_timers : impl_->phase_timers;
}

bool validate_resident_kernel_dispatch(const ResidentKernelDispatch& request,
                                       std::string* error_text) {
  auto fail = [&](const std::string& text) {
    if (error_text != nullptr) *error_text = text;
    return false;
  };
  if (!validate_kernel_descriptors(std::vector<KernelDescriptor>{request.kernel}, error_text)) {
    return false;
  }
  if (request.kernel.code.size() > kPageSize) {
    return fail("kernel code exceeds the C0 mapped code page");
  }
  if (request.kernargs.size() != request.kernel.kernarg_bytes) {
    return fail("kernarg byte count does not match the reviewed kernel layout");
  }
  if (request.kernargs.size() > kPageSize) {
    return fail("kernarg layout exceeds the C0 mapped kernarg page");
  }
  if (request.input_bytes.empty()) return fail("resident dispatch input bytes are required");
  if (request.input_bytes.size() > kPageSize) {
    return fail("resident dispatch input exceeds the C0 mapped input page");
  }
  if (request.output_byte_count == 0) return fail("resident dispatch output bytes are required");
  if (request.output_byte_count > kPageSize) {
    return fail("resident dispatch output exceeds the C0 mapped output page");
  }
  return true;
}

bool plan_resident_hsa_dispatch(const VramLayout& layout, const ResidentHsaDispatch& request,
                                ResidentHsaDispatchPlan* plan, std::string* error_text) {
  if (plan == nullptr) {
    if (error_text != nullptr) *error_text = "resident HSA dispatch plan is required";
    return false;
  }
  if (!validate_resident_hsa_dispatch(request, error_text)) return false;
  if (layout.resident_gpu_va_base >= kSmokePtbBoundaryGpuVa ||
      layout.resident_gpu_va_limit < kSmokePtbBoundaryGpuVa + kPageSize) {
    if (error_text != nullptr) {
      *error_text = "resident window cannot host the dynamic HSA image table";
    }
    return false;
  }
  const std::vector<const HsaCodeImageAsset*> images =
      request.hsa_images.empty() ? std::vector<const HsaCodeImageAsset*>{request.hsa_image}
                                 : request.hsa_images;
  VramAllocator allocator(layout);
  ResidentMemory resident(
      layout, allocator,
      [](ResidentPageOperation, uint64_t, uint64_t, std::string*) { return true; });
  const uint64_t boundary_bytes = kSmokePtbBoundaryGpuVa - layout.resident_gpu_va_base;
  ResidentBuffer boundary_guard{};
  std::vector<ResidentBuffer> image_buffers(images.size());
  std::vector<ResidentBuffer> buffers(request.buffers.size());
  std::string detail;
  if (!resident.allocate("resident-hsa-ptb-boundary", boundary_bytes, &boundary_guard, &detail)) {
    if (error_text != nullptr) *error_text = detail;
    return false;
  }
  for (size_t index = 0; index < images.size(); ++index) {
    if (!resident.allocate("resident-hsa-image-" + std::to_string(index),
                          images[index]->image.size(), &image_buffers[index], &detail)) {
      if (error_text != nullptr) *error_text = detail;
      return false;
    }
  }
  for (size_t index = 0; index < request.buffers.size(); ++index) {
    if (!resident.allocate(request.buffers[index].name, request.buffers[index].allocation_byte_count,
                          &buffers[index], &detail)) {
      if (error_text != nullptr) *error_text = detail;
      return false;
    }
  }
  ResidentHsaDispatchPlan candidate;
  candidate.hsa_image_gpu_vas.reserve(image_buffers.size());
  candidate.buffer_gpu_vas.reserve(buffers.size());
  for (const ResidentBuffer& image : image_buffers) candidate.hsa_image_gpu_vas.push_back(image.gpu_va);
  for (const ResidentBuffer& buffer : buffers) candidate.buffer_gpu_vas.push_back(buffer.gpu_va);
  *plan = std::move(candidate);
  return true;
}

bool AMDevSession::dispatch_resident_kernel(const ResidentKernelDispatch& request,
                                            ResidentKernelDispatchResult* result,
                                            std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "resident dispatch result is required";
    return false;
  }
  result->output_bytes.clear();
  result->transfer_bytes = 0;
  result->hardware_identity.clear();
  result->failure_stage = "preflight";
  if (!validate_resident_kernel_dispatch(request, error_text)) return false;

  return run_resident_kernel_dispatch(request, result, error_text);
}

bool AMDevSession::dispatch_resident_hsa(const ResidentHsaDispatch& request,
                                         ResidentHsaDispatchResult* result,
                                         std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "resident HSA dispatch result is required";
    return false;
  }
  *result = ResidentHsaDispatchResult{};
  if (!validate_resident_hsa_dispatch(request, error_text)) {
    result->failure_stage = "preflight";
    return false;
  }
  ResidentHsaSession session;
  if (!session.prepare(request, result, error_text)) return false;
  const std::vector<ResidentHsaStage> stages =
      request.stages.empty()
          ? std::vector<ResidentHsaStage>{ResidentHsaStage{
                0, request.hsa_images.empty() ? request.hsa_image->entry_offset
                                               : request.hsa_images[0]->entry_offset,
                request.kernarg_bindings, request.kernargs, request.workgroup_x, request.workgroup_y,
                request.workgroup_z, request.global_x, request.global_y, request.global_z}}
          : request.stages;
  for (const ResidentHsaStage& stage : stages) {
    if (!session.dispatch(stage, result, error_text)) {
      std::string dispatch_error = error_text != nullptr ? *error_text : "stage dispatch failed";
      if (!session.close(&dispatch_error)) {
        result->failure_stage = "compute_queue_retirement";
        if (error_text != nullptr) *error_text = dispatch_error;
      }
      return false;
    }
  }
  std::vector<std::string> readback_names;
  std::vector<size_t> readback_indices;
  for (size_t index = 0; index < request.buffers.size(); ++index) {
    if (request.buffers[index].readback_byte_count != 0) {
      readback_names.push_back(request.buffers[index].name);
      readback_indices.push_back(index);
    }
  }
  if (!readback_names.empty() && !session.readback(readback_names, result, error_text)) {
    std::string readback_error = error_text != nullptr ? *error_text : "resident readback failed";
    if (!session.close(&readback_error)) {
      result->failure_stage = "compute_queue_retirement";
      if (error_text != nullptr) *error_text = readback_error;
    }
    return false;
  }
  std::vector<std::vector<uint8_t>> requested_readbacks = std::move(result->readback_bytes);
  result->readback_bytes.resize(request.buffers.size());
  for (size_t index = 0; index < readback_indices.size(); ++index) {
    result->readback_bytes[readback_indices[index]] = std::move(requested_readbacks[index]);
  }
  if (!session.close(error_text)) {
    result->failure_stage = "compute_queue_retirement";
    return false;
  }
  result->failure_stage = "none";
  return true;
}

bool AMDevSession::plan_resident_hsa_dispatch(const ResidentHsaDispatch& request,
                                              ResidentHsaDispatchPlan* plan,
                                              std::string* error_text) {
  if (plan == nullptr) {
    if (error_text != nullptr) *error_text = "resident HSA dispatch plan is required";
    return false;
  }
  if (!validate_resident_hsa_dispatch(request, error_text)) return false;
  DiscoveryLog log;
  log.socket_path = tinygpu_socket_path();
  HardwareLock hardware_lock;
  std::string lock_error;
  if (!hardware_lock.acquire(&lock_error)) {
    if (error_text != nullptr) *error_text = "hardware lock acquire failed: " + lock_error;
    return false;
  }

  UniqueFd socket_fd;
  std::string detail;
  if (!connect_tinygpu_server(log.socket_path, &socket_fd, &detail)) {
    if (error_text != nullptr) *error_text = detail;
    return false;
  }
  std::string health_error;
  if (!hardware_lock_health_check(log.socket_path, &health_error)) {
    if (error_text != nullptr) *error_text = health_error;
    return false;
  }

  const RemoteClient client(socket_fd.get());
  RemoteRpcResult config = client.rpc_no_payload(RemoteCmd::CFG_READ, 0, 0, 4);
  if (!config.ok) {
    if (error_text != nullptr) *error_text = rpc_failure_text("CFG_READ vendor_device", config);
    return false;
  }
  log.config_vendor_id = static_cast<uint32_t>(config.value0 & 0xffffU);
  log.config_device_id = static_cast<uint32_t>((config.value0 >> 16) & 0xffffU);
  if (log.config_vendor_id != kTargetVendor || log.config_device_id != kTargetDevice) {
    if (error_text != nullptr) {
      *error_text = "expected 1002:7551, observed " +
                    pci_id_text(log.config_vendor_id, log.config_device_id);
    }
    return false;
  }
  RemoteRpcResult bar_result;
  if (!map_bar(client, 0, &log.bar0, &bar_result) ||
      !map_bar(client, 2, &log.bar2, &bar_result) ||
      !map_bar(client, 5, &log.bar5, &bar_result)) {
    if (error_text != nullptr) *error_text = rpc_failure_text("MAP_BAR", bar_result);
    return false;
  }
  if (!try_discover_arch(client, &log, &detail)) {
    if (error_text != nullptr) *error_text = detail;
    return false;
  }
  const uint64_t vram_mib = log.vram_size_bytes >> 20U;
  if (vram_mib > std::numeric_limits<uint32_t>::max()) {
    if (error_text != nullptr) *error_text = "discovered VRAM MiB does not fit RCC_CONFIG_MEMSIZE";
    return false;
  }
  VramLayout layout{};
  if (!derive_vram_layout(static_cast<uint32_t>(vram_mib), log.bar0.size, &layout, &detail) ||
      layout.large_bar) {
    if (error_text != nullptr) *error_text = detail.empty() ? "large BAR0 is unsupported" : detail;
    return false;
  }
  return native_r9700::plan_resident_hsa_dispatch(layout, request, plan, error_text);
}
bool AMDevSession::vram_smoke(VramSmokeResult* result, std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "VRAM smoke result is required";
    return false;
  }
  *result = VramSmokeResult{};
  return run_vram_smoke(result, error_text);
}

bool AMDevSession::llama_embed_smoke(const LlamaEmbedSmokeDispatch& request,
                                     LlamaEmbedSmokeDispatchResult* result,
                                     std::string* error_text) {
  if (result == nullptr) {
    if (error_text != nullptr) *error_text = "Llama embedding smoke result is required";
    return false;
  }
  *result = LlamaEmbedSmokeDispatchResult{};
  return run_llama_embed_smoke(request, result, error_text);
}

int AMDevSession::streaming_transfer_proof(uint64_t byte_count) {
  return run_streaming_transfer_proof(byte_count, nullptr, nullptr);
}

int AMDevSession::transfer_round_trip_file(const std::string& input_path,
                                           const std::string& output_path,
                                           std::string* error_text) {
  std::vector<uint8_t> input;
  if (!read_binary_file(input_path, &input, error_text)) return 2;
  if (input.empty()) {
    if (error_text != nullptr) *error_text = "input file is empty";
    return 2;
  }
  return run_streaming_transfer_proof(static_cast<uint64_t>(input.size()), &input, &output_path);
}

AMDevSession::AMDevSession(uint64_t memory_limit_bytes)
    : memory_limit_bytes_(memory_limit_bytes) {}

bool AMDevSession::allocate(uint64_t size_bytes, uint64_t* gpu_va, std::string* error_text) {
  if (gpu_va == nullptr) {
    if (error_text != nullptr) *error_text = "GPU virtual-address output is required";
    return false;
  }
  if (size_bytes == 0) {
    if (error_text != nullptr) *error_text = "allocation size must be nonzero";
    return false;
  }
  if (size_bytes > memory_limit_bytes_ ||
      allocated_bytes_ > memory_limit_bytes_ - size_bytes) {
    if (error_text != nullptr) *error_text = "AMDevSession allocation exceeds bounded memory";
    return false;
  }
  if (size_bytes > std::numeric_limits<uint64_t>::max() - next_gpu_va_) {
    if (error_text != nullptr) *error_text = "AMDevSession GPU virtual-address space exhausted";
    return false;
  }

  const uint64_t allocated_gpu_va = next_gpu_va_;
  allocations_.emplace(allocated_gpu_va, std::vector<uint8_t>(static_cast<size_t>(size_bytes)));
  allocated_bytes_ += size_bytes;
  next_gpu_va_ += size_bytes;
  *gpu_va = allocated_gpu_va;
  return true;
}

bool AMDevSession::upload(uint64_t gpu_va, const uint8_t* data, uint64_t size_bytes,
                          std::string* error_text) {
  if (data == nullptr) {
    if (error_text != nullptr) *error_text = "upload data is required";
    return false;
  }
  const auto found = allocations_.find(gpu_va);
  if (found == allocations_.end()) {
    if (error_text != nullptr) *error_text = "AMDevSession allocation is not live";
    return false;
  }
  if (size_bytes > found->second.size()) {
    if (error_text != nullptr) *error_text = "upload size exceeds AMDevSession allocation";
    return false;
  }
  std::memcpy(found->second.data(), data, static_cast<size_t>(size_bytes));
  return true;
}

bool AMDevSession::download(uint64_t gpu_va, uint8_t* data, uint64_t size_bytes,
                            std::string* error_text) {
  if (data == nullptr) {
    if (error_text != nullptr) *error_text = "download data is required";
    return false;
  }
  const auto found = allocations_.find(gpu_va);
  if (found == allocations_.end()) {
    if (error_text != nullptr) *error_text = "AMDevSession allocation is not live";
    return false;
  }
  if (size_bytes > found->second.size()) {
    if (error_text != nullptr) *error_text = "download size exceeds AMDevSession allocation";
    return false;
  }
  std::memcpy(data, found->second.data(), static_cast<size_t>(size_bytes));
  return true;
}

void AMDevSession::release(uint64_t gpu_va) {
  const auto found = allocations_.find(gpu_va);
  if (found == allocations_.end()) return;
  allocated_bytes_ -= static_cast<uint64_t>(found->second.size());
  allocations_.erase(found);
}

}  // namespace native_r9700

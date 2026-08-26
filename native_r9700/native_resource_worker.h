#ifndef NATIVE_R9700_NATIVE_RESOURCE_WORKER_H_
#define NATIVE_R9700_NATIVE_RESOURCE_WORKER_H_

#include <cstdint>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>
namespace native_r9700 {

inline constexpr const char* kNativeResourceProtocolVersion =
    "r9700_native_resource_v1";
inline constexpr std::size_t kNativeResourceMaxFrameBytes = 65536U;

struct NativeCacheCapacity {
  uint64_t batch = 0;
  uint64_t prefix_positions = 0;
};

struct NativeKernelPack {
  std::string name;
  std::string version;
  std::vector<std::string> digests;
};

struct NativeResourceBudget {
  uint64_t resident_bytes_max = 0;
  uint64_t scratch_bytes_max = 0;
  uint64_t total_bytes_max = 0;
};

// The fingerprint object is received only after the Python-side canonical
// identity checks.  Native keeps its canonical JCS representation opaque: it
// is identity data, never a native model/config parser boundary.
struct NativeResourceSpec {
  std::string model_uri;
  std::string model_digest;
  std::string model_fingerprint;
  NativeCacheCapacity cache_capacity;
  NativeKernelPack kernel_pack;
  NativeResourceBudget resource_budget;
};

struct NativeResourcePrefillRequest {
  uint64_t resource_generation = 0;
  std::string request_id;
  std::vector<uint32_t> token_ids;
  std::string prefill_npz_path;
  std::string hardware_log_path;
};

struct NativeResourceError {
  std::string domain;
  std::string message;
  std::string failure_stage;
};

struct NativePrepareResult {
  uint64_t resource_generation = 0;
  std::string state;
  std::string producer_fingerprint;
  // The child publishes the bytes it actually executed.  The private client
  // compares this value with the hash captured before launching the child.
  std::string runner_binary_sha256;
};
struct NativeCommitResult {
  uint64_t resource_generation = 0;
  std::string state;
  std::string producer_fingerprint;
};

struct NativeCleanupResult {
  uint64_t resource_generation = 0;
  std::string state;
  bool already_released = false;
};

struct NativeResourcePrefillResult {
  uint64_t resource_generation = 0;
  std::string producer_fingerprint;
  std::string native_prefill_acceptance;
  std::string native_prefill_full_layer_loop_status;
  std::string runtime_substrate;
  std::string hardware_log_path;
  std::string compute_completion_policy;
  std::string compute_barrier_policy;
  std::string prefill_npz_path;
  uint64_t kernel_count = 0;
  uint64_t transfer_bytes = 0;
  uint64_t block_tokens = 0;
  uint64_t block_count = 0;
  std::string failure_stage;
  int64_t exit_status = 0;
  std::string failure_text;
};

struct NativeHealthResult {
  std::string child_state;
  bool has_resource_generation = false;
  uint64_t resource_generation = 0;
  std::string resource_state;
  std::string producer_fingerprint;
  bool has_error_summary = false;
  NativeResourceError error_summary;
};

struct NativeShutdownResult {
  std::string state;
};

class NativeResourceBackend {
 public:
  virtual ~NativeResourceBackend() = default;

  virtual bool prepare(const NativeResourceSpec& spec,
                       NativePrepareResult* result,
                       NativeResourceError* error) = 0;
  virtual bool commit(uint64_t generation, NativeCommitResult* result,
                      NativeResourceError* error) = 0;
  virtual bool rollback(uint64_t generation, NativeCleanupResult* result,
                        NativeResourceError* error) = 0;
  virtual bool release(uint64_t generation, NativeCleanupResult* result,
                       NativeResourceError* error) = 0;
  virtual bool prefill(const NativeResourcePrefillRequest& request,
                       NativeResourcePrefillResult* result,
                       NativeResourceError* error) = 0;
  virtual bool health(NativeHealthResult* result,
                      NativeResourceError* error) = 0;
  virtual bool shutdown(NativeShutdownResult* result,
                        NativeResourceError* error) = 0;
};

// Drives one private JSONL child loop.  The caller owns the backend and the
// streams; this seam intentionally has no executable entrypoint of its own.
int run_native_resource_worker(std::istream& input, std::ostream& output,
                               NativeResourceBackend& backend);

struct NativeProducerIdentity {
  std::string runner_binary_sha256;
  std::vector<std::string> ordered_kernel_pack_sha256;
  std::string target = "gfx1201";
  std::string runtime_substrate = "TinyGPU.app/APLRemotePCIDevice/PCIIface";
  std::string completion_policy = "terminal";
  std::string barrier_policy = "full";
  std::string vendor_id = "1002";
  std::string device_id = "7551";
};

// Computes sha256:<lowercase-hex> over the exact JCS preimage frozen for F1.
// This helper has no filesystem/device side effects and is usable by the
// runner's concrete backend after it has selected its immutable identities.
std::string compute_native_producer_fingerprint(const NativeProducerIdentity& identity);

}  // namespace native_r9700

#endif  // NATIVE_R9700_NATIVE_RESOURCE_WORKER_H_

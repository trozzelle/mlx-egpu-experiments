#ifndef NATIVE_R9700_PREFILL_NPZ_H
#define NATIVE_R9700_PREFILL_NPZ_H

#include <cstdint>
#include <string>
#include <vector>

namespace native_r9700 {

// Raw final readback of one complete native prefill request: exactly 32
// buffers ordered layer0 K, layer0 V, ..., layer15 K, layer15 V. Each buffer
// holds cache_capacity_tokens * 8 * 64 fp16 values in the device-resident
// head-major [kv_head][capacity_token][head_dim] layout written by the
// rope_kv stage. Serialization slices the accepted (1, 8, n_prefix, 64)
// prefix from each raw buffer; it performs no model math and never invents
// K/V values.
struct NativePrefillNpzPayload {
  std::string model;
  uint32_t n_prefix = 0;
  uint32_t cache_capacity_tokens = 0;
  std::vector<std::vector<uint8_t>> kv_readback_bytes;
};

// Validates only the live fp16 prefix in each full-capacity head-major K/V
// readback buffer. Unused cache suffix bytes are intentionally ignored.
bool validate_native_prefill_kv_finite(
    const NativePrefillNpzPayload& payload, std::string* error_text);

// Writes the strict prefill NPZ consumed by
// native_worker.validate_native_prefill_npz and kv_cache.py: stored (not
// compressed) zip entries model.npy, n_prefix.npy, num_layers.npy,
// producer_kind.npy, and layer{i}_K.npy/layer{i}_V.npy fp16 arrays shaped
// (1, 8, n_prefix, 64). The write is atomic: a temp sibling is fully written
// and renamed over out_path; failure removes the temp and leaves out_path
// absent.
bool write_native_prefill_npz(const NativePrefillNpzPayload& payload,
                              const std::string& out_path, std::string* error_text);

}  // namespace native_r9700

#endif  // NATIVE_R9700_PREFILL_NPZ_H

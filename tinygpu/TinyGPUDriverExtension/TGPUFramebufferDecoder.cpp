#include "TGPUFramebufferDecoder.h"

namespace {

constexpr uint32_t kFramebufferFieldMask = 0x00ffffffU;
constexpr uint32_t kApertureShift = 18U;

}  // namespace

TGPUStatus TGPUDecodeFramebufferLocation(uint32_t raw_base,
                                         uint32_t raw_top,
                                         TGPUFramebufferDecodeResult* out) {
  if (out == nullptr) return TGPU_STATUS_INVALID_REQUEST;

  const uint64_t base_bytes =
      static_cast<uint64_t>(raw_base & kFramebufferFieldMask) << 24U;
  const uint64_t top_bytes =
      static_cast<uint64_t>(raw_top & kFramebufferFieldMask) << 24U;
  if (top_bytes < base_bytes) return TGPU_STATUS_RANGE;

  const TGPUFramebufferDecodeResult decoded{
      base_bytes,
      top_bytes,
      static_cast<uint32_t>(base_bytes >> kApertureShift),
      static_cast<uint32_t>(top_bytes >> kApertureShift),
  };
  *out = decoded;
  return TGPU_STATUS_OK;
}

#ifndef TGPU_FRAMEBUFFER_DECODER_H
#define TGPU_FRAMEBUFFER_DECODER_H

#include "TGPUABI.h"

#include <cstdint>

struct TGPUFramebufferDecodeResult {
  uint64_t base_bytes = 0;
  uint64_t top_bytes = 0;
  uint32_t base_aperture_register = 0;
  uint32_t top_aperture_register = 0;
};

TGPUStatus TGPUDecodeFramebufferLocation(uint32_t raw_base,
                                         uint32_t raw_top,
                                         TGPUFramebufferDecodeResult* out);

#endif  // TGPU_FRAMEBUFFER_DECODER_H

// RED contract for MMHUB framebuffer-location field decoding.
//
// This is a host-buildable pure seam.  It does not open DriverKit, a PCI
// device, or TinyGPU.app.  The DEXT adapter must use the same checked decode
// before programming the system-aperture registers.
#include "TGPUFramebufferDecoder.h"
#include "TGPUABI.h"

#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kStatusOk = 0;
constexpr uint32_t kStatusInvalidRequest = 1;
constexpr uint32_t kStatusRange = 6;
constexpr uint64_t kSentinel = 0xD1EAD5A5D1EAD5A5ULL;

bool expect(bool condition, const char* message) {
  if (condition) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

template <typename Status>
bool expect_status(Status observed, uint32_t expected, const char* message) {
  const uint32_t observed_value = static_cast<uint32_t>(observed);
  if (observed_value == expected) return true;
  std::fprintf(stderr, "FAIL: %s (observed=%u expected=%u)\n", message,
               observed_value, expected);
  return false;
}

bool expect_unchanged(const TGPUFramebufferDecodeResult& value,
                      const TGPUFramebufferDecodeResult& expected,
                      const char* message) {
  return expect(value.base_bytes == expected.base_bytes &&
                    value.top_bytes == expected.top_bytes &&
                    value.base_aperture_register ==
                        expected.base_aperture_register &&
                    value.top_aperture_register ==
                        expected.top_aperture_register,
                message);
}

}  // namespace

int main() {
  // The hand-derived R9700 field is a 24-bit value in 16 MiB units:
  // 0x00008000 << 24 = 0x0000008000000000 bytes.  Upper register bits are
  // ignored, then each decoded byte address is represented by address >> 18
  // in the MMHUB system-aperture register.
  TGPUFramebufferDecodeResult decoded{};
  if (!expect_status(
          TGPUDecodeFramebufferLocation(0xAB008000U, 0xCD009000U, &decoded),
          kStatusOk, "valid framebuffer fields decode") ||
      !expect(decoded.base_bytes == 0x0000008000000000ULL,
              "raw base 0x00008000 decodes to the hand-derived byte base") ||
      !expect(decoded.top_bytes == 0x0000009000000000ULL,
              "raw top 0x00009000 decodes to the hand-derived byte top") ||
      !expect(decoded.base_aperture_register ==
                  0x0000008000000000ULL >> 18,
              "base aperture register is decoded base shifted by 18") ||
      !expect(decoded.top_aperture_register ==
                  0x0000009000000000ULL >> 18,
              "top aperture register is decoded top shifted by 18")) {
    return 1;
  }

  // A field that wraps below the base after the required 24-bit mask is not
  // a valid aperture.  The output object must remain untouched on rejection.
  TGPUFramebufferDecodeResult rejected{
      kSentinel, kSentinel, static_cast<uint32_t>(kSentinel),
      static_cast<uint32_t>(kSentinel)};
  const TGPUFramebufferDecodeResult rejected_before = rejected;
  if (!expect_status(
          TGPUDecodeFramebufferLocation(0x00FFFFFFU, 0x01000000U, &rejected),
          kStatusRange,
          "masked framebuffer fields reject a descending or wrapped range") ||
      !expect_unchanged(rejected, rejected_before,
                        "invalid framebuffer range preserves output")) {
    return 1;
  }

  // A null output is an invalid request, not permission to perform unchecked
  // shifts or register programming.
  if (!expect_status(TGPUDecodeFramebufferLocation(0x00008000U, 0x00009000U,
                                                   nullptr),
                     kStatusInvalidRequest,
                     "framebuffer decode requires an output object")) {
    return 1;
  }

  return 0;
}

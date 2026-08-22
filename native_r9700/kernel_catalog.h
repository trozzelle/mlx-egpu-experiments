#ifndef NATIVE_R9700_KERNEL_CATALOG_H_
#define NATIVE_R9700_KERNEL_CATALOG_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace native_r9700 {

// Identity and launch contract for a reviewed, resident gfx1201 kernel asset.
// `code` is copied to C0's mapped code page after BAR0 readback verification;
// the resource and geometry fields are written to the C0-compatible PM4 stream.
struct KernelDescriptor {
  std::string name;
  std::string sha256;
  std::vector<uint8_t> code;
  uint32_t rsrc1 = 0;
  uint32_t rsrc2 = 0;
  uint32_t rsrc3 = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t global_x = 0;
  uint32_t global_y = 0;
  uint32_t global_z = 0;
  uint32_t kernarg_bytes = 0;
};

// Validates descriptors supplied by a caller before they are eligible for
// lookup or dispatch. On failure, error_text receives the rejected property.
bool validate_kernel_descriptors(const std::vector<KernelDescriptor>& descriptors,
                                 std::string* error_text);

// Returns the compact, reviewed descriptor for name, or nullptr when no
// descriptor exists. Unknown names are never synthesized.
const KernelDescriptor* find_kernel(std::string_view name);

}  // namespace native_r9700

#endif  // NATIVE_R9700_KERNEL_CATALOG_H_

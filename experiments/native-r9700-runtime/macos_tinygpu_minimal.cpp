#include <libusb.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

struct TinyGpuId {
  uint16_t vendor;
  uint16_t product;
  const char *label;
};

constexpr std::array<TinyGpuId, 2> kTinyGpuIds{{
    {0xADD1, 0x0001, "TinyGPU USB/DMA transport (vendor 0xADD1)"},
    {0x3801, 0x0001, "TinyGPU USB/DMA transport (vendor 0x3801)"},
}};

constexpr std::array<uint32_t, 8> kInputA{{0, 1, 2, 3, 5, 8, 13, 21}};
constexpr std::array<uint32_t, 8> kInputB{{34, 55, 89, 144, 233, 377, 610, 987}};

uint64_t fnv1a_u32(const uint32_t *values, size_t count) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    uint32_t value = values[i];
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<unsigned char>((value >> (byte * 8)) & 0xffu);
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

bool is_tinygpu(uint16_t vendor, uint16_t product, const char **label) {
  for (const TinyGpuId &id : kTinyGpuIds) {
    if (vendor == id.vendor && product == id.product) {
      *label = id.label;
      return true;
    }
  }
  return false;
}

void print_usb_string(libusb_device_handle *handle, uint8_t index, const char *field) {
  if (index == 0) {
    std::printf("device_%s: unavailable\n", field);
    return;
  }

  unsigned char buffer[256]{};
  const int rc = libusb_get_string_descriptor_ascii(handle, index, buffer, sizeof(buffer));
  if (rc < 0) {
    std::printf("device_%s: unavailable (%s)\n", field, libusb_error_name(rc));
    return;
  }
  std::printf("device_%s: %.*s\n", field, rc, buffer);
}

void print_reference_operation() {
  std::array<uint32_t, kInputA.size()> expected{};
  for (size_t i = 0; i < expected.size(); ++i) expected[i] = kInputA[i] + kInputB[i];

  std::printf("operation: uint32_vector_add_8\n");
  std::printf("input_shape: 8xu32\n");
  std::printf("input_a_sample: [%u,%u,%u,%u,...]\n", kInputA[0], kInputA[1], kInputA[2], kInputA[3]);
  std::printf("input_b_sample: [%u,%u,%u,%u,...]\n", kInputB[0], kInputB[1], kInputB[2], kInputB[3]);
  std::printf("cpu_expected_sample: [%u,%u,%u,%u,...]\n", expected[0], expected[1], expected[2], expected[3]);
  std::printf("cpu_expected_digest_fnv1a64: 0x%016llx\n", static_cast<unsigned long long>(fnv1a_u32(expected.data(), expected.size())));
  std::printf("device_output_sample: unavailable\n");
  std::printf("device_output_digest_fnv1a64: unavailable\n");
  std::printf("cpu_comparison_status: not_run_missing_native_tinygpu_dma_and_kernel_launch_abi\n");
}

}  // namespace

int main() {
  const auto start = std::chrono::steady_clock::now();

  std::printf("source_name: experiments/native-r9700-runtime/macos_tinygpu_minimal.cpp\n");
  std::printf("command_placeholder: see docs/tasks/native-r9700-producer/validation-commands.md C0 macOS eGPU minimal runtime probe\n");
  std::printf("runtime_substrate: macOS TinyGPU USB/libusb tinygrad-free probe\n");
  std::printf("tinygrad_execution_path: false\n");
  std::printf("model: none\n");
  std::printf("target_device_hint: AMD Radeon AI PRO R9700 RDNA4/gfx12-class via TinyGPU USB/DMA\n");
  std::printf("known_usb_ids: 0xADD1:0x0001,0x3801:0x0001\n");

  print_reference_operation();

  libusb_context *ctx = nullptr;
  int rc = libusb_init(&ctx);
  if (rc != LIBUSB_SUCCESS) {
    std::printf("device_discovery_status: failed\n");
    std::printf("failure_text: libusb_init failed: %s\n", libusb_error_name(rc));
    std::printf("exit_status_semantics: 1=libusb setup/list failure,2=transport_or_kernel_launch_unsupported_after_discovery,3=no_tinygpu_usb_device_detected\n");
    std::printf("exit_status: 1\n");
    return 1;
  }

  libusb_device **devices = nullptr;
  const ssize_t count = libusb_get_device_list(ctx, &devices);
  if (count < 0) {
    std::printf("device_discovery_status: failed\n");
    std::printf("failure_text: libusb_get_device_list failed: %s\n", libusb_error_name(static_cast<int>(count)));
    std::printf("exit_status_semantics: 1=libusb setup/list failure,2=transport_or_kernel_launch_unsupported_after_discovery,3=no_tinygpu_usb_device_detected\n");
    std::printf("exit_status: 1\n");
    libusb_exit(ctx);
    return 1;
  }

  int matches = 0;
  for (ssize_t i = 0; i < count; ++i) {
    libusb_device_descriptor desc{};
    rc = libusb_get_device_descriptor(devices[i], &desc);
    if (rc != LIBUSB_SUCCESS) continue;

    const char *label = nullptr;
    if (!is_tinygpu(desc.idVendor, desc.idProduct, &label)) continue;

    std::printf("device_%d_match: true\n", matches);
    std::printf("device_%d_runtime_substrate: %s\n", matches, label);
    std::printf("device_%d_usb_vid_pid: 0x%04X:0x%04X\n", matches, desc.idVendor, desc.idProduct);
    std::printf("device_%d_bus_address: bus=%u address=%u\n", matches,
                libusb_get_bus_number(devices[i]), libusb_get_device_address(devices[i]));
    std::printf("device_%d_usb_class_subclass_protocol: 0x%02X:0x%02X:0x%02X\n", matches,
                desc.bDeviceClass, desc.bDeviceSubClass, desc.bDeviceProtocol);

    libusb_device_handle *handle = nullptr;
    rc = libusb_open(devices[i], &handle);
    if (rc == LIBUSB_SUCCESS && handle != nullptr) {
      char field[64]{};
      std::snprintf(field, sizeof(field), "%d_manufacturer", matches);
      print_usb_string(handle, desc.iManufacturer, field);
      std::snprintf(field, sizeof(field), "%d_product", matches);
      print_usb_string(handle, desc.iProduct, field);
      std::snprintf(field, sizeof(field), "%d_serial", matches);
      print_usb_string(handle, desc.iSerialNumber, field);
      libusb_close(handle);
    } else {
      std::printf("device_%d_open_status: failed (%s)\n", matches, libusb_error_name(rc));
      std::printf("device_%d_identity_note: USB VID/PID and bus/address are available; strings require opening the device.\n", matches);
    }

    ++matches;
  }

  std::printf("tinygpu_device_count: %d\n", matches);
  libusb_free_device_list(devices, 1);
  libusb_exit(ctx);

  const auto end = std::chrono::steady_clock::now();
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::printf("elapsed_us: %lld\n", static_cast<long long>(elapsed_us));

  std::printf("host_device_transfer_status: not_run_missing_native_tinygpu_dma_protocol\n");
  std::printf("kernel_launch_status: not_run_missing_native_tinygpu_command_queue_and_kernel_dispatch_abi\n");
  if (matches == 0) {
    std::printf("failure_text: no TinyGPU USB device matched pinned IDs 0xADD1:0x0001 or 0x3801:0x0001; native kernel launch also remains unsupported without the TinyGPU DMA/queue ABI outside tinygrad.\n");
    std::printf("exit_status_semantics: 1=libusb setup/list failure,2=transport_or_kernel_launch_unsupported_after_discovery,3=no_tinygpu_usb_device_detected\n");
    std::printf("exit_status: 3\n");
    return 3;
  }

  std::printf("failure_text: TinyGPU USB device discovery succeeded, but this probe intentionally stops before host/device transfer or kernel launch because the safe native TinyGPU DMA mapping, command queue, and kernel dispatch ABI are not pinned for tinygrad-free use in this repo.\n");
  std::printf("exit_status_semantics: 1=libusb setup/list failure,2=transport_or_kernel_launch_unsupported_after_discovery,3=no_tinygpu_usb_device_detected\n");
  std::printf("exit_status: 2\n");
  return 2;
}

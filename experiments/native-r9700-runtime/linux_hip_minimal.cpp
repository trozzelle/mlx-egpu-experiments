#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <chrono>

namespace {

constexpr int kOk = 0;
constexpr int kHipFailure = 2;
constexpr int kNoDevice = 3;
constexpr int kMismatch = 4;
constexpr std::size_t kElementCount = 1 << 20;

__global__ void vector_add_kernel(const float* a, const float* b, float* out, std::size_t count) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = a[idx] + b[idx];
  }
}

long long elapsed_us(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point finish) {
  return std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
}

int fail_with_hip(const char* step, hipError_t status) {
  std::printf("failure_step: %s\n", step);
  std::printf("hip_error_code: %d\n", static_cast<int>(status));
  std::printf("hip_error_name: %s\n", hipGetErrorName(status));
  std::printf("hip_error_text: %s\n", hipGetErrorString(status));
  std::printf("probe_status: fail\n");
  std::printf("probe_exit_status: %d\n", kHipFailure);
  return kHipFailure;
}

int check_hip(const char* step, hipError_t status) {
  if (status != hipSuccess) {
    return fail_with_hip(step, status);
  }
  return kOk;
}

}  // namespace

int main() {
  std::printf("probe_name: c0-linux-hip-minimal\n");
  std::printf("probe_operation: vector_add\n");
  std::printf("model: none\n");
  std::printf("element_count: %zu\n", kElementCount);
  std::printf("exit_status_semantics: 0=pass,2=hip_runtime_failure,3=no_hip_device,4=cpu_compare_mismatch\n");

  int runtime_version = 0;
  hipError_t status = hipRuntimeGetVersion(&runtime_version);
  if (status == hipSuccess) {
    std::printf("hip_runtime_version: %d\n", runtime_version);
  } else {
    std::printf("hip_runtime_version: unavailable\n");
    return fail_with_hip("hipRuntimeGetVersion", status);
  }

  int driver_version = 0;
  status = hipDriverGetVersion(&driver_version);
  if (status == hipSuccess) {
    std::printf("hip_driver_version: %d\n", driver_version);
  } else {
    std::printf("hip_driver_version: unavailable\n");
    std::printf("hip_driver_version_error: %s\n", hipGetErrorString(status));
  }

  int device_count = 0;
  status = hipGetDeviceCount(&device_count);
  if (status != hipSuccess) {
    return fail_with_hip("hipGetDeviceCount", status);
  }
  std::printf("hip_device_count: %d\n", device_count);
  if (device_count <= 0) {
    std::printf("failure_step: hipGetDeviceCount\n");
    std::printf("failure_text: no HIP devices reported by runtime\n");
    std::printf("probe_status: fail\n");
    std::printf("probe_exit_status: %d\n", kNoDevice);
    return kNoDevice;
  }

  const int device = 0;
  if (int rc = check_hip("hipSetDevice", hipSetDevice(device)); rc != kOk) {
    return rc;
  }

  hipDeviceProp_t props{};
  if (int rc = check_hip("hipGetDeviceProperties", hipGetDeviceProperties(&props, device)); rc != kOk) {
    return rc;
  }
  std::printf("hip_device_index: %d\n", device);
  std::printf("hip_device_name: %s\n", props.name);
  std::printf("hip_device_gcn_arch_name: %s\n", props.gcnArchName);
  std::printf("hip_device_compute_capability: %d.%d\n", props.major, props.minor);
  std::printf("hip_device_multiprocessors: %d\n", props.multiProcessorCount);
  std::printf("hip_device_global_mem_bytes: %zu\n", static_cast<std::size_t>(props.totalGlobalMem));
  std::printf("hip_device_warp_size: %d\n", props.warpSize);

  std::vector<float> host_a(kElementCount);
  std::vector<float> host_b(kElementCount);
  std::vector<float> host_out(kElementCount, 0.0f);
  std::vector<float> cpu_out(kElementCount);
  for (std::size_t i = 0; i < kElementCount; ++i) {
    host_a[i] = static_cast<float>(static_cast<int>(i % 251) - 125) * 0.25f;
    host_b[i] = static_cast<float>(static_cast<int>(i % 127) - 63) * -0.5f;
    cpu_out[i] = host_a[i] + host_b[i];
  }

  const std::size_t bytes = kElementCount * sizeof(float);
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_out = nullptr;
  if (int rc = check_hip("hipMalloc(device_a)", hipMalloc(&device_a, bytes)); rc != kOk) {
    return rc;
  }
  if (int rc = check_hip("hipMalloc(device_b)", hipMalloc(&device_b, bytes)); rc != kOk) {
    hipFree(device_a);
    return rc;
  }
  if (int rc = check_hip("hipMalloc(device_out)", hipMalloc(&device_out, bytes)); rc != kOk) {
    hipFree(device_b);
    hipFree(device_a);
    return rc;
  }

  auto h2d_start = std::chrono::steady_clock::now();
  status = hipMemcpy(device_a, host_a.data(), bytes, hipMemcpyHostToDevice);
  if (status == hipSuccess) {
    status = hipMemcpy(device_b, host_b.data(), bytes, hipMemcpyHostToDevice);
  }
  auto h2d_finish = std::chrono::steady_clock::now();
  if (status != hipSuccess) {
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return fail_with_hip("hipMemcpyHostToDevice", status);
  }
  std::printf("host_to_device_bytes: %zu\n", bytes * 2);
  std::printf("host_to_device_us: %lld\n", elapsed_us(h2d_start, h2d_finish));

  hipEvent_t kernel_start = nullptr;
  hipEvent_t kernel_stop = nullptr;
  if (int rc = check_hip("hipEventCreate(start)", hipEventCreate(&kernel_start)); rc != kOk) {
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return rc;
  }
  if (int rc = check_hip("hipEventCreate(stop)", hipEventCreate(&kernel_stop)); rc != kOk) {
    hipEventDestroy(kernel_start);
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return rc;
  }

  const int threads_per_block = 256;
  const int block_count = static_cast<int>((kElementCount + threads_per_block - 1) / threads_per_block);
  if (int rc = check_hip("hipEventRecord(start)", hipEventRecord(kernel_start)); rc != kOk) {
    hipEventDestroy(kernel_stop);
    hipEventDestroy(kernel_start);
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return rc;
  }
  hipLaunchKernelGGL(vector_add_kernel, dim3(block_count), dim3(threads_per_block), 0, 0,
                     device_a, device_b, device_out, kElementCount);
  status = hipGetLastError();
  if (status == hipSuccess) {
    status = hipEventRecord(kernel_stop);
  }
  if (status == hipSuccess) {
    status = hipEventSynchronize(kernel_stop);
  }
  if (status != hipSuccess) {
    hipEventDestroy(kernel_stop);
    hipEventDestroy(kernel_start);
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return fail_with_hip("kernel_launch_or_sync", status);
  }
  float kernel_ms = 0.0f;
  if (int rc = check_hip("hipEventElapsedTime", hipEventElapsedTime(&kernel_ms, kernel_start, kernel_stop)); rc != kOk) {
    hipEventDestroy(kernel_stop);
    hipEventDestroy(kernel_start);
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return rc;
  }
  std::printf("kernel_launch_grid_blocks: %d\n", block_count);
  std::printf("kernel_launch_threads_per_block: %d\n", threads_per_block);
  std::printf("kernel_elapsed_ms: %.6f\n", kernel_ms);

  auto d2h_start = std::chrono::steady_clock::now();
  status = hipMemcpy(host_out.data(), device_out, bytes, hipMemcpyDeviceToHost);
  auto d2h_finish = std::chrono::steady_clock::now();
  if (status != hipSuccess) {
    hipEventDestroy(kernel_stop);
    hipEventDestroy(kernel_start);
    hipFree(device_out);
    hipFree(device_b);
    hipFree(device_a);
    return fail_with_hip("hipMemcpyDeviceToHost", status);
  }
  std::printf("device_to_host_bytes: %zu\n", bytes);
  std::printf("device_to_host_us: %lld\n", elapsed_us(d2h_start, d2h_finish));

  double max_abs_error = 0.0;
  std::size_t mismatch_count = 0;
  std::size_t first_mismatch = 0;
  for (std::size_t i = 0; i < kElementCount; ++i) {
    const double err = std::fabs(static_cast<double>(host_out[i]) - static_cast<double>(cpu_out[i]));
    if (err > max_abs_error) {
      max_abs_error = err;
    }
    if (err > 0.0) {
      if (mismatch_count == 0) {
        first_mismatch = i;
      }
      ++mismatch_count;
    }
  }

  std::printf("sample_out_0: %.6f\n", host_out[0]);
  std::printf("sample_cpu_0: %.6f\n", cpu_out[0]);
  std::printf("sample_out_last: %.6f\n", host_out[kElementCount - 1]);
  std::printf("sample_cpu_last: %.6f\n", cpu_out[kElementCount - 1]);
  std::printf("cpu_compare_mismatches: %zu\n", mismatch_count);
  std::printf("cpu_compare_max_abs_error: %.9f\n", max_abs_error);

  hipEventDestroy(kernel_stop);
  hipEventDestroy(kernel_start);
  hipFree(device_out);
  hipFree(device_b);
  hipFree(device_a);

  if (mismatch_count != 0) {
    std::printf("failure_step: cpu_compare\n");
    std::printf("first_mismatch_index: %zu\n", first_mismatch);
    std::printf("first_mismatch_gpu: %.9f\n", host_out[first_mismatch]);
    std::printf("first_mismatch_cpu: %.9f\n", cpu_out[first_mismatch]);
    std::printf("probe_status: fail\n");
    std::printf("probe_exit_status: %d\n", kMismatch);
    return kMismatch;
  }

  std::printf("probe_status: pass\n");
  std::printf("probe_exit_status: %d\n", kOk);
  return kOk;
}

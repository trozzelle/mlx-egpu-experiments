#ifndef TGPU_COLD_LIFECYCLE_H
#define TGPU_COLD_LIFECYCLE_H

#include <cstdint>

// The coordinator deliberately has no DriverKit or PCI dependency.  This
// keeps stage ordering/failure attribution testable without hardware while
// the DEXT adapter owns all device side effects.
enum class TGPUColdStage : uint32_t {
  PspSosTmr = 1,
  Smu = 2,
  Imu = 3,
  Rlc = 4,
  CpMesGfxSdma = 5,
  GmcGartVm = 6,
  None = 0,
};

class TGPUColdStageExecutor {
 public:
  virtual ~TGPUColdStageExecutor() = default;
  virtual bool execute(TGPUColdStage stage) = 0;
};

struct TGPUColdLifecycleResult {
  bool ready = false;
  TGPUColdStage failure_stage = TGPUColdStage::None;
};

class TGPUColdLifecycle final {
 public:
  explicit TGPUColdLifecycle(TGPUColdStageExecutor& executor)
      : executor_(executor) {}

  TGPUColdLifecycleResult initialize();

 private:
  TGPUColdStageExecutor& executor_;
};

#endif  // TGPU_COLD_LIFECYCLE_H

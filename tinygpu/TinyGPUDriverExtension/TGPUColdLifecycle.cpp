#include "TGPUColdLifecycle.h"

#include <array>

namespace {

constexpr std::array<TGPUColdStage, 6> kColdStages = {
    TGPUColdStage::PspSosTmr,
    TGPUColdStage::Smu,
    TGPUColdStage::Imu,
    TGPUColdStage::Rlc,
    TGPUColdStage::CpMesGfxSdma,
    TGPUColdStage::GmcGartVm,
};

}  // namespace

TGPUColdLifecycleResult TGPUColdLifecycle::initialize() {
  TGPUColdLifecycleResult result{};
  for (const TGPUColdStage stage : kColdStages) {
    if (!executor_.execute(stage)) {
      result.failure_stage = stage;
      return result;
    }
  }

  result.ready = true;
  return result;
}

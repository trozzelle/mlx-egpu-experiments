#include "TGPUColdLifecycle.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace {

constexpr std::array<TGPUColdStage, 6> kExpectedStages = {
  TGPUColdStage::PspSosTmr,
  TGPUColdStage::Smu,
  TGPUColdStage::Imu,
  TGPUColdStage::Rlc,
  TGPUColdStage::CpMesGfxSdma,
  TGPUColdStage::GmcGartVm,
};

class FakeStageExecutor final : public TGPUColdStageExecutor {
 public:
  explicit FakeStageExecutor(TGPUColdStage fail_at = TGPUColdStage::None) : fail_at_(fail_at) {}

  bool execute(TGPUColdStage stage) override {
    if (call_count_ == calls_.size()) std::abort();
    calls_[call_count_++] = stage;
    return stage != fail_at_;
  }

  size_t call_count() const { return call_count_; }
  TGPUColdStage stage_at(size_t index) const { return calls_[index]; }

 private:
  TGPUColdStage fail_at_;
  std::array<TGPUColdStage, kExpectedStages.size()> calls_{};
  size_t call_count_ = 0;
};

void expect(bool condition, const char *message) {
  if (condition) return;
  std::cerr << "TGPU cold lifecycle contract failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

// Production mutation caught: changing the frozen stage order, marking the
// device ready before the final memory stage, or losing the terminal stage.
void test_success_runs_frozen_order_before_ready() {
  FakeStageExecutor executor;
  TGPUColdLifecycle lifecycle(executor);
  const TGPUColdLifecycleResult result = lifecycle.initialize();

  expect(result.ready, "all cold stages must be required before ready");
  expect(result.failure_stage == TGPUColdStage::None,
         "successful cold initialization must not report a failure stage");
  expect(executor.call_count() == kExpectedStages.size(),
         "successful cold initialization must execute every stage exactly once");
  for (size_t index = 0; index < kExpectedStages.size(); ++index) {
    expect(executor.stage_at(index) == kExpectedStages[index],
           "cold stage families must execute in the frozen order");
  }
}

// Production mutation caught: continuing after an executor failure, reporting
// a later/generic stage instead of the first failure, or exposing ready state
// after a partially initialized device.
void test_first_failure_stops_and_is_attributed_exactly() {
  for (size_t failure_index = 0; failure_index < kExpectedStages.size(); ++failure_index) {
    FakeStageExecutor executor(kExpectedStages[failure_index]);
    TGPUColdLifecycle lifecycle(executor);
    const TGPUColdLifecycleResult result = lifecycle.initialize();

    expect(!result.ready, "a failed cold stage must fail closed, never ready");
    expect(result.failure_stage == kExpectedStages[failure_index],
           "failure attribution must identify the first failed stage family");
    expect(executor.call_count() == failure_index + 1,
           "cold initialization must not execute stages after the first failure");
    for (size_t index = 0; index <= failure_index; ++index) {
      expect(executor.stage_at(index) == kExpectedStages[index],
             "failed cold initialization must preserve the prefix stage order");
    }
  }
}

}  // namespace

int main() {
  test_success_runs_frozen_order_before_ready();
  test_first_failure_stops_and_is_attributed_exactly();
  return EXIT_SUCCESS;
}

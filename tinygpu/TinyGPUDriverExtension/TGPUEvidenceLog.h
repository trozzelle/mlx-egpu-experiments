#ifndef TGPU_EVIDENCE_LOG_H
#define TGPU_EVIDENCE_LOG_H

#include "TGPUABI.h"

#include <cstdint>

struct TGPUEvidenceRecord {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t selector;
  uint32_t status;
  uint32_t failure_stage;
  uint64_t device_epoch;
  uint32_t exit_status;
  uint8_t failure_text[TGPU_MAX_FAULT_TEXT_BYTES];
};

class TGPUEvidenceLog final {
 public:
  static bool Write(const char* path, const TGPUEvidenceRecord& record);
};

#endif  // TGPU_EVIDENCE_LOG_H

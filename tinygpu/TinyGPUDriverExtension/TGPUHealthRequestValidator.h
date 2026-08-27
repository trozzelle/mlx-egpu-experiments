#ifndef TGPU_HEALTH_REQUEST_VALIDATOR_H
#define TGPU_HEALTH_REQUEST_VALIDATOR_H

#include "TGPUABI.h"

// Validate the complete typed health request for the inference role. Common
// descriptor bounds and ABI-prefix checks remain part of the user-client
// boundary, but this pure seam repeats the fixed v1.0 header checks so callers
// cannot accidentally bypass them when testing a typed request.
TGPUStatus TGPUValidateInferenceHealthRequest(
    const TGPUHealthFaultQueryRequest& request);

#endif  // TGPU_HEALTH_REQUEST_VALIDATOR_H

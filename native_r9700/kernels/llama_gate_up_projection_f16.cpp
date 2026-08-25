extern "C" __attribute__((global)) void llama_gate_up_projection_f16(
    const unsigned short* post_attention_hidden,
    const unsigned short* post_attention_layernorm_weight,
    const unsigned short* gate_projection_weight,
    const unsigned short* up_projection_weight,
    unsigned short* gate_output,
    unsigned short* up_output,
    unsigned int sequence_length) {
  constexpr unsigned int kHiddenSize = 2048U;
  constexpr unsigned int kIntermediateSize = 8192U;
  constexpr unsigned int kLanesPerWorkgroup = 64U;
  constexpr unsigned int kWorkgroupsPerToken = kIntermediateSize / kLanesPerWorkgroup;
  constexpr float kEpsilon = 1.0e-5f;
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  const unsigned int token = workgroup / kWorkgroupsPerToken;
  const unsigned int intermediate =
      (workgroup % kWorkgroupsPerToken) * kLanesPerWorkgroup + lane;
  if (token >= sequence_length || intermediate >= kIntermediateSize) return;

  __attribute__((shared)) unsigned short normalized_tile[2048];
  __attribute__((shared)) float shared_inverse_rms;
  if (lane == 0U) {
    float sum_of_squares = 0.0f;
    for (unsigned int column = 0U; column < kHiddenSize; ++column) {
      const unsigned short raw_bits =
          post_attention_hidden[(unsigned long long)token * kHiddenSize + column];
      normalized_tile[column] = raw_bits;
      const float value = (float)__builtin_bit_cast(_Float16, raw_bits);
      sum_of_squares += value * value;
    }
    shared_inverse_rms = 1.0f / __builtin_sqrtf(
        sum_of_squares * (1.0f / (float)kHiddenSize) + kEpsilon);
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

  for (unsigned int column = lane; column < kHiddenSize;
       column += kLanesPerWorkgroup) {
    const float input =
        (float)__builtin_bit_cast(_Float16, normalized_tile[column]);
    const float norm = (float)__builtin_bit_cast(
        _Float16, post_attention_layernorm_weight[column]);
    const _Float16 normalized =
        (_Float16)(input * norm * shared_inverse_rms);
    normalized_tile[column] =
        __builtin_bit_cast(unsigned short, normalized);
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

  float gate = 0.0f;
  float up = 0.0f;
  for (unsigned int column = 0U; column < kHiddenSize; ++column) {
    const _Float16 normalized =
        __builtin_bit_cast(_Float16, normalized_tile[column]);
    gate += (float)normalized * (float)__builtin_bit_cast(
        _Float16, gate_projection_weight[(unsigned long long)intermediate * kHiddenSize + column]);
    up += (float)normalized * (float)__builtin_bit_cast(
        _Float16, up_projection_weight[(unsigned long long)intermediate * kHiddenSize + column]);
  }
  gate_output[(unsigned long long)token * kIntermediateSize + intermediate] =
      __builtin_bit_cast(unsigned short, (_Float16)gate);
  up_output[(unsigned long long)token * kIntermediateSize + intermediate] =
      __builtin_bit_cast(unsigned short, (_Float16)up);
}

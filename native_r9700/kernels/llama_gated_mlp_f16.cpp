extern "C" __attribute__((global)) void llama_gated_mlp_f16(
    const unsigned short* post_attention_hidden,
    const unsigned short* post_attention_layernorm_weight,
    const unsigned short* gate_projection_weight,
    const unsigned short* up_projection_weight,
    const unsigned short* down_projection_weight,
    unsigned short* hidden,
    unsigned int sequence_length) {
  constexpr unsigned int kHiddenSize = 2048U;
  constexpr unsigned int kIntermediateSize = 8192U;
  constexpr float kEpsilon = 1.0e-5f;
  const unsigned int token = __builtin_amdgcn_workgroup_id_x();
  const unsigned int output_column = __builtin_amdgcn_workitem_id_x();
  if (token >= sequence_length || output_column >= kHiddenSize) return;

  float sum_of_squares = 0.0f;
  for (unsigned int column = 0U; column < kHiddenSize; ++column) {
    const float value = (float)__builtin_bit_cast(
        _Float16, post_attention_hidden[(unsigned long long)token * kHiddenSize + column]);
    sum_of_squares += value * value;
  }
  const float inverse_rms = 1.0f / __builtin_sqrtf(
      sum_of_squares * (1.0f / (float)kHiddenSize) + kEpsilon);
  float accumulator = 0.0f;
  for (unsigned int intermediate = 0U; intermediate < kIntermediateSize; ++intermediate) {
    float gate = 0.0f;
    float up = 0.0f;
    for (unsigned int column = 0U; column < kHiddenSize; ++column) {
      const float input = (float)__builtin_bit_cast(
          _Float16, post_attention_hidden[(unsigned long long)token * kHiddenSize + column]);
      const float norm = (float)__builtin_bit_cast(_Float16, post_attention_layernorm_weight[column]);
      const float normalized = input * norm * inverse_rms;
      gate += normalized * (float)__builtin_bit_cast(
          _Float16, gate_projection_weight[(unsigned long long)intermediate * kHiddenSize + column]);
      up += normalized * (float)__builtin_bit_cast(
          _Float16, up_projection_weight[(unsigned long long)intermediate * kHiddenSize + column]);
    }
    const float silu_gate = gate / (1.0f + __builtin_expf(-gate));
    const float down = (float)__builtin_bit_cast(
        _Float16, down_projection_weight[(unsigned long long)output_column * kIntermediateSize + intermediate]);
    accumulator += silu_gate * up * down;
  }
  const float residual = (float)__builtin_bit_cast(
      _Float16, post_attention_hidden[(unsigned long long)token * kHiddenSize + output_column]);
  hidden[(unsigned long long)token * kHiddenSize + output_column] =
      __builtin_bit_cast(unsigned short, (_Float16)(accumulator + residual));
}

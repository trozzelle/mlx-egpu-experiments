extern "C" __attribute__((global)) void llama_mlp_down_f16(
    const unsigned short* gate_input,
    const unsigned short* up_input,
    const unsigned short* down_projection_weight,
    const unsigned short* residual,
    unsigned short* hidden,
    unsigned int sequence_length) {
  constexpr unsigned int kHiddenSize = 2048U;
  constexpr unsigned int kIntermediateSize = 8192U;
  constexpr unsigned int kColumnsPerWorkgroup = 64U;
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int token = workgroup / (kHiddenSize / kColumnsPerWorkgroup);
  const unsigned int output_column =
      workgroup % (kHiddenSize / kColumnsPerWorkgroup) * kColumnsPerWorkgroup +
      __builtin_amdgcn_workitem_id_x();
  if (token >= sequence_length || output_column >= kHiddenSize) return;

  float accumulator = 0.0f;
  for (unsigned int intermediate = 0U; intermediate < kIntermediateSize; ++intermediate) {
    const float gate = (float)__builtin_bit_cast(
        _Float16, gate_input[(unsigned long long)token * kIntermediateSize + intermediate]);
    const float up = (float)__builtin_bit_cast(
        _Float16, up_input[(unsigned long long)token * kIntermediateSize + intermediate]);
    const float silu_gate = gate / (1.0f + __builtin_expf(-gate));
    const float down = (float)__builtin_bit_cast(
        _Float16, down_projection_weight[(unsigned long long)output_column * kIntermediateSize + intermediate]);
    accumulator += silu_gate * up * down;
  }
  const float res = (float)__builtin_bit_cast(
      _Float16, residual[(unsigned long long)token * kHiddenSize + output_column]);
  hidden[(unsigned long long)token * kHiddenSize + output_column] =
      __builtin_bit_cast(unsigned short, (_Float16)(accumulator + res));
}

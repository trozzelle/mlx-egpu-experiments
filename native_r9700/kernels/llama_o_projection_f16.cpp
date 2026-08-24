extern "C" __attribute__((global)) void llama_o_projection_f16(
    const unsigned short* context,
    const unsigned short* o_projection_weight,
    const unsigned short* residual,
    unsigned short* post_attention_hidden,
    unsigned int sequence_length) {
  constexpr unsigned int kHiddenSize = 2048U;
  constexpr unsigned int kColumnsPerWorkgroup = 64U;
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int token = workgroup / (kHiddenSize / kColumnsPerWorkgroup);
  const unsigned int output_column =
      workgroup % (kHiddenSize / kColumnsPerWorkgroup) * kColumnsPerWorkgroup +
      __builtin_amdgcn_workitem_id_x();
  if (token >= sequence_length || output_column >= kHiddenSize) return;
  float accumulator = 0.0f;
  for (unsigned int column = 0U; column < kHiddenSize; ++column) {
    const float input = (float)__builtin_bit_cast(
        _Float16, context[(unsigned long long)token * kHiddenSize + column]);
    const float weight = (float)__builtin_bit_cast(
        _Float16, o_projection_weight[(unsigned long long)output_column * kHiddenSize + column]);
    accumulator += input * weight;
  }
  const float skip = (float)__builtin_bit_cast(
      _Float16, residual[(unsigned long long)token * kHiddenSize + output_column]);
  post_attention_hidden[(unsigned long long)token * kHiddenSize + output_column] =
      __builtin_bit_cast(unsigned short, (_Float16)(accumulator + skip));
}

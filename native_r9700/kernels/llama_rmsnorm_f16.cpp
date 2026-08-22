extern "C" __attribute__((global)) void llama_rmsnorm_f16(
    const unsigned short* hidden_input,
    const unsigned short* scale,
    unsigned short* hidden_output,
    float epsilon) {
  const unsigned int row = __builtin_amdgcn_workgroup_id_x();
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  if (lane != 0U) return;

  const unsigned long long row_offset = (unsigned long long)row * 2048ULL;
  float sum_of_squares = 0.0f;
  for (unsigned int column = 0U; column < 2048U; ++column) {
    const _Float16 input = __builtin_bit_cast(_Float16, hidden_input[row_offset + column]);
    const float value = input;
    sum_of_squares += value * value;
  }

  const float inverse_rms = 1.0f / __builtin_sqrtf(
      sum_of_squares * (1.0f / 2048.0f) + epsilon);
  for (unsigned int column = 0U; column < 2048U; ++column) {
    const _Float16 input = __builtin_bit_cast(_Float16, hidden_input[row_offset + column]);
    const _Float16 weight = __builtin_bit_cast(_Float16, scale[column]);
    const _Float16 output = (_Float16)((float)input * (float)weight * inverse_rms);
    hidden_output[row_offset + column] = __builtin_bit_cast(unsigned short, output);
  }
}

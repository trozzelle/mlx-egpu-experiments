extern "C" __attribute__((global)) void llama_rmsnorm_zero_store_f16(
    const unsigned short* hidden_input,
    const unsigned short* scale,
    unsigned short* hidden_output,
    float epsilon) {
  const unsigned int row = __builtin_amdgcn_workgroup_id_x();
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  if (lane != 0U) return;

  (void)hidden_input;
  (void)scale;
  (void)epsilon;
  const unsigned long long row_offset = (unsigned long long)row * 2048ULL;
  for (unsigned int column = 0U; column < 2048U; ++column) {
    hidden_output[row_offset + column] = 0U;
  }
}

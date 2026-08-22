extern "C" __attribute__((global)) void llama_embed_row_f16(
    const unsigned short* embedding_rows,
    unsigned short* hidden_output,
    const unsigned long long* selected_row) {
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  const unsigned int index = workgroup * 256U + lane;
  if (index < 2048U) {
    hidden_output[index] = embedding_rows[(*selected_row * 2048ULL) + index];
  }
}

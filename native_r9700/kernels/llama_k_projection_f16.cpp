extern "C" __attribute__((global)) void llama_k_projection_f16(
    const unsigned short* normalized,
    const unsigned short* k_projection_weight,
    unsigned short* fresh_k,
    unsigned int sequence_length) {
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int head_dimension = __builtin_amdgcn_workitem_id_x();
  const unsigned int token = workgroup / 8U;
  const unsigned int kv_head = workgroup % 8U;
  if (token >= sequence_length || head_dimension >= 64U) return;

  const unsigned int projection_row = kv_head * 64U + head_dimension;
  const unsigned long long normalized_offset = (unsigned long long)token * 2048ULL;
  const unsigned long long weight_offset =
      (unsigned long long)projection_row * 2048ULL;
  float accumulator = 0.0f;
  for (unsigned int hidden_dimension = 0U; hidden_dimension < 2048U;
       ++hidden_dimension) {
    const _Float16 input =
        __builtin_bit_cast(_Float16, normalized[normalized_offset + hidden_dimension]);
    const _Float16 weight = __builtin_bit_cast(
        _Float16, k_projection_weight[weight_offset + hidden_dimension]);
    accumulator += (float)input * (float)weight;
  }

  const unsigned long long fresh_k_offset =
      ((unsigned long long)kv_head * sequence_length + token) * 64ULL +
      head_dimension;
  fresh_k[fresh_k_offset] = __builtin_bit_cast(unsigned short, (_Float16)accumulator);
}

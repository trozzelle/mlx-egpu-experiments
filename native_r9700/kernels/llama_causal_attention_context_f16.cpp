extern "C" __attribute__((global)) void llama_causal_attention_context_f16(
    const float* attention_probabilities,
    const unsigned short* v_cache,
    unsigned short* context,
    unsigned int sequence_length,
    unsigned int position,
    unsigned int cache_capacity_tokens) {
  constexpr unsigned int kQueryHeads = 32U;
  constexpr unsigned int kKvHeads = 8U;
  constexpr unsigned int kGqaGroupSize = kQueryHeads / kKvHeads;
  constexpr unsigned int kHeadDimension = 64U;
  constexpr unsigned int kMaximumPrefixTokens = 128U;
  if (sequence_length == 0U || sequence_length > kMaximumPrefixTokens ||
      cache_capacity_tokens == 0U || cache_capacity_tokens > kMaximumPrefixTokens ||
      (unsigned long long)position + sequence_length > cache_capacity_tokens) {
    return;
  }
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int query_head = workgroup / sequence_length;
  const unsigned int query_token = workgroup % sequence_length;
  const unsigned int dimension = __builtin_amdgcn_workitem_id_x();
  if (query_head >= kQueryHeads || query_token >= sequence_length ||
      dimension >= kHeadDimension) {
    return;
  }
  const unsigned int absolute_query = position + query_token;
  const unsigned int kv_head = query_head / kGqaGroupSize;
  const unsigned long long probability_offset =
      ((unsigned long long)query_head * sequence_length + query_token) *
      cache_capacity_tokens;
  float context_sum = 0.0f;
  for (unsigned int key_token = 0U; key_token <= absolute_query; ++key_token) {
    const float probability = attention_probabilities[probability_offset + key_token];
    const float value = (float)__builtin_bit_cast(
        _Float16, v_cache[((unsigned long long)kv_head * cache_capacity_tokens + key_token) *
                              kHeadDimension +
                          dimension]);
    context_sum += probability * value;
  }
  context[(unsigned long long)query_token * kQueryHeads * kHeadDimension +
          query_head * kHeadDimension + dimension] =
      __builtin_bit_cast(unsigned short, (_Float16)context_sum);
}

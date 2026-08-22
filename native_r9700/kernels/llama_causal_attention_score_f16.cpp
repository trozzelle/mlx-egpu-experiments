extern "C" __attribute__((global)) void llama_causal_attention_score_f16(
    const unsigned short* normalized,
    const unsigned short* q_projection_weight,
    const unsigned short* k_cache,
    float* attention_scores,
    unsigned int sequence_length,
    unsigned int position,
    unsigned int cache_capacity_tokens) {
  constexpr unsigned int kQueryHeads = 32U;
  constexpr unsigned int kKvHeads = 8U;
  constexpr unsigned int kGqaGroupSize = kQueryHeads / kKvHeads;
  constexpr unsigned int kHeadDimension = 64U;
  constexpr unsigned int kHiddenSize = 2048U;
  constexpr unsigned int kMaximumPrefixTokens = 128U;
  if (sequence_length == 0U || sequence_length > kMaximumPrefixTokens ||
      cache_capacity_tokens == 0U || cache_capacity_tokens > kMaximumPrefixTokens ||
      (unsigned long long)position + sequence_length > cache_capacity_tokens) {
    return;
  }
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int query_head = workgroup / sequence_length;
  const unsigned int query_token = workgroup % sequence_length;
  const unsigned int key_token = __builtin_amdgcn_workitem_id_x();
  if (query_head >= kQueryHeads || query_token >= sequence_length ||
      key_token >= cache_capacity_tokens) {
    return;
  }

  const unsigned long long score_offset =
      ((unsigned long long)query_head * sequence_length + query_token) *
          cache_capacity_tokens +
      key_token;
  const unsigned int absolute_query = position + query_token;
  if (key_token > absolute_query) {
    attention_scores[score_offset] = -3.402823466e+38F;
    return;
  }

  const unsigned int kv_head = query_head / kGqaGroupSize;
  float score = 0.0f;
  for (unsigned int dimension = 0U; dimension < kHeadDimension; ++dimension) {
    const unsigned int q_row = query_head * kHeadDimension + dimension;
    float q_value = 0.0f;
    for (unsigned int column = 0U; column < kHiddenSize; ++column) {
      const float activation =
          (float)__builtin_bit_cast(_Float16, normalized[(unsigned long long)query_token * kHiddenSize + column]);
      const float weight = (float)__builtin_bit_cast(
          _Float16, q_projection_weight[(unsigned long long)q_row * kHiddenSize + column]);
      q_value += activation * weight;
    }
    const float k_value = (float)__builtin_bit_cast(
        _Float16, k_cache[((unsigned long long)kv_head * cache_capacity_tokens + key_token) *
                              kHeadDimension +
                          dimension]);
    score += q_value * k_value;
  }
  attention_scores[score_offset] = score * 0.125f;
}

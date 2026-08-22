extern "C" __attribute__((global)) void llama_causal_attention_softmax_f32(
    const float* attention_scores,
    float* attention_probabilities,
    unsigned int sequence_length,
    unsigned int position,
    unsigned int cache_capacity_tokens) {
  constexpr unsigned int kMaximumPrefixTokens = 128U;
  constexpr unsigned int kQueryHeads = 32U;
  if (sequence_length == 0U || sequence_length > kMaximumPrefixTokens ||
      cache_capacity_tokens == 0U || cache_capacity_tokens > kMaximumPrefixTokens ||
      (unsigned long long)position + sequence_length > cache_capacity_tokens ||
      __builtin_amdgcn_workitem_id_x() != 0U) {
    return;
  }
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int query_head = workgroup / sequence_length;
  const unsigned int query_token = workgroup % sequence_length;
  if (query_head >= kQueryHeads || query_token >= sequence_length) return;

  const unsigned int absolute_query = position + query_token;
  const unsigned long long row_offset =
      ((unsigned long long)query_head * sequence_length + query_token) *
      cache_capacity_tokens;
  float row_max = -3.402823466e+38F;
  for (unsigned int key_token = 0U; key_token <= absolute_query; ++key_token) {
    const float value = attention_scores[row_offset + key_token];
    if (value > row_max) row_max = value;
  }
  float normalizer = 0.0f;
  for (unsigned int key_token = 0U; key_token <= absolute_query; ++key_token) {
    const float probability = __builtin_expf(attention_scores[row_offset + key_token] - row_max);
    attention_probabilities[row_offset + key_token] = probability;
    normalizer += probability;
  }
  for (unsigned int key_token = 0U; key_token < cache_capacity_tokens; ++key_token) {
    if (key_token > absolute_query) {
      attention_probabilities[row_offset + key_token] = 0.0f;
    } else {
      attention_probabilities[row_offset + key_token] /= normalizer;
    }
  }
}

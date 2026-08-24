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
  constexpr unsigned int kKeyTokensPerBlock = 64U;
  constexpr unsigned int kKeyBlocks = kMaximumPrefixTokens / kKeyTokensPerBlock;
  constexpr float kRopeTheta = 500000.0f;
  constexpr float kRopeFactor = 8.0f;
  constexpr float kOriginalContext = 8192.0f;
  constexpr float kLowFrequencyFactor = 1.0f;
  constexpr float kHighFrequencyFactor = 4.0f;
  if (sequence_length == 0U || sequence_length > kMaximumPrefixTokens ||
      cache_capacity_tokens == 0U || cache_capacity_tokens > kMaximumPrefixTokens ||
      (unsigned long long)position + sequence_length > cache_capacity_tokens) {
    return;
  }
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int query_head = workgroup / sequence_length;
  const unsigned int query_token = workgroup % sequence_length;
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  if (query_head >= kQueryHeads || query_token >= sequence_length) {
    return;
  }
  const unsigned int absolute_query = position + query_token;
  const unsigned int kv_head = query_head / kGqaGroupSize;
  for (unsigned int key_block = 0U; key_block < kKeyBlocks; ++key_block) {
    const unsigned int key_token = key_block * kKeyTokensPerBlock + lane;
    if (key_token >= cache_capacity_tokens) continue;
    const unsigned long long score_offset =
        ((unsigned long long)query_head * sequence_length + query_token) *
            cache_capacity_tokens +
        key_token;
    if (key_token > absolute_query) {
      attention_scores[score_offset] = -3.402823466e+38F;
      continue;
    }

    float score = 0.0f;
    // Split-half RoPE over the query head: the query is projected on the fly and
    // rotated with absolute_query, matching the K cache which is already rotated
    // with each key's absolute token.
    for (unsigned int pair = 0U; pair < kHeadDimension / 2U; ++pair) {
      const unsigned int q_row0 = query_head * kHeadDimension + pair;
      const unsigned int q_row1 = query_head * kHeadDimension + pair + kHeadDimension / 2U;
      float q0 = 0.0f;
      float q1 = 0.0f;
      for (unsigned int column = 0U; column < kHiddenSize; ++column) {
        const float activation = (float)__builtin_bit_cast(
            _Float16, normalized[(unsigned long long)query_token * kHiddenSize + column]);
        q0 += activation * (float)__builtin_bit_cast(
            _Float16, q_projection_weight[(unsigned long long)q_row0 * kHiddenSize + column]);
        q1 += activation * (float)__builtin_bit_cast(
            _Float16, q_projection_weight[(unsigned long long)q_row1 * kHiddenSize + column]);
      }
      float inv_frequency = __builtin_powf(
          kRopeTheta, -2.0f * (float)pair / (float)kHeadDimension);
      const float wavelength = 6.2831853071795864769f / inv_frequency;
      if (wavelength > kOriginalContext / kLowFrequencyFactor) {
        inv_frequency /= kRopeFactor;
      } else if (wavelength >= kOriginalContext / kHighFrequencyFactor) {
        const float smooth = (kOriginalContext / wavelength - kLowFrequencyFactor) /
                             (kHighFrequencyFactor - kLowFrequencyFactor);
        inv_frequency = (1.0f - smooth) * inv_frequency / kRopeFactor +
                        smooth * inv_frequency;
      }
      const float angle = (float)absolute_query * inv_frequency;
      const float cosine = __builtin_cosf(angle);
      const float sine = __builtin_sinf(angle);
      const float q0_rope = q0 * cosine - q1 * sine;
      const float q1_rope = q1 * cosine + q0 * sine;

      const float k0 = (float)__builtin_bit_cast(
          _Float16, k_cache[((unsigned long long)kv_head * cache_capacity_tokens + key_token) *
                                kHeadDimension +
                            pair]);
      const float k1 = (float)__builtin_bit_cast(
          _Float16, k_cache[((unsigned long long)kv_head * cache_capacity_tokens + key_token) *
                                kHeadDimension +
                            pair + kHeadDimension / 2U]);
      score += q0_rope * k0 + q1_rope * k1;
    }
    attention_scores[score_offset] = score * 0.125f;
  }
}

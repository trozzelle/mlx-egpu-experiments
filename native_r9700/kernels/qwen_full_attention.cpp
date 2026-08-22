static inline float qwen_bf16_load(const unsigned char* bytes, unsigned long long offset) {
  const unsigned short bits =
      (unsigned short)bytes[offset] | ((unsigned short)bytes[offset + 1U] << 8U);
  return __builtin_bit_cast(float, (unsigned int)bits << 16U);
}

static inline void qwen_bf16_store(unsigned char* bytes, unsigned long long offset, float value) {
  const unsigned short bits = (unsigned short)(__builtin_bit_cast(unsigned int, value) >> 16U);
  bytes[offset] = (unsigned char)bits;
  bytes[offset + 1U] = (unsigned char)(bits >> 8U);
}

extern "C" __attribute__((global)) void qwen_full_attention(
    const unsigned char* query_bytes,
    const unsigned char* k_cache_bytes,
    const unsigned char* v_cache_bytes,
    unsigned char* output_bytes,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dimension,
    unsigned int query_length,
    unsigned int key_length,
    unsigned int position,
    unsigned long long k_cache_capacity_bytes,
    unsigned long long v_cache_capacity_bytes) {
  const unsigned long long max_u64 = ~0ULL;
  if (query_heads == 0U || kv_heads == 0U || query_heads % kv_heads != 0U ||
      head_dimension == 0U || query_length == 0U || key_length == 0U ||
      position > key_length || query_length > key_length - position ||
      kv_heads > max_u64 / head_dimension) {
    return;
  }

  const unsigned long long cache_elements_per_token = (unsigned long long)kv_heads * head_dimension;
  if (key_length > max_u64 / cache_elements_per_token) return;
  const unsigned long long cache_elements = cache_elements_per_token * key_length;
  if (cache_elements > max_u64 / 2U) return;
  const unsigned long long k_cache_required_bytes = cache_elements * 2U;
  const unsigned long long v_cache_required_bytes = cache_elements * 2U;
  if (k_cache_required_bytes > k_cache_capacity_bytes ||
      v_cache_required_bytes > v_cache_capacity_bytes) {
    return;
  }

  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int query_head = workgroup / query_length;
  const unsigned int query_token = workgroup % query_length;
  const unsigned int dimension = __builtin_amdgcn_workitem_id_x();
  if (query_head >= query_heads || query_token >= query_length || dimension >= head_dimension) {
    return;
  }

  const unsigned int kv_head = query_head / (query_heads / kv_heads);
  const unsigned int absolute_query = position + query_token;
  const float scale = 1.0f / __builtin_sqrtf((float)head_dimension);
  float max_score = -3.402823466e38f;

  for (unsigned int key_token = 0U; key_token < key_length; ++key_token) {
    if (key_token > absolute_query) break;
    float score = 0.0f;
    for (unsigned int component = 0U; component < head_dimension; ++component) {
      const unsigned long long query_offset =
          (((unsigned long long)query_token * query_heads + query_head) * head_dimension + component) * 2U;
      const unsigned long long key_offset =
          (((unsigned long long)kv_head * key_length + key_token) * head_dimension + component) * 2U;
      score += qwen_bf16_load(query_bytes, query_offset) *
               qwen_bf16_load(k_cache_bytes, key_offset);
    }
    max_score = score > max_score ? score : max_score;
  }

  float normalizer = 0.0f;
  float context_sum = 0.0f;
  for (unsigned int key_token = 0U; key_token < key_length; ++key_token) {
    if (key_token > absolute_query) break;
    float score = 0.0f;
    for (unsigned int component = 0U; component < head_dimension; ++component) {
      const unsigned long long query_offset =
          (((unsigned long long)query_token * query_heads + query_head) * head_dimension + component) * 2U;
      const unsigned long long key_offset =
          (((unsigned long long)kv_head * key_length + key_token) * head_dimension + component) * 2U;
      score += qwen_bf16_load(query_bytes, query_offset) *
               qwen_bf16_load(k_cache_bytes, key_offset);
    }
    const float weight = __builtin_expf(score * scale - max_score * scale);
    const unsigned long long value_offset =
        (((unsigned long long)kv_head * key_length + key_token) * head_dimension + dimension) * 2U;
    context_sum += weight * qwen_bf16_load(v_cache_bytes, value_offset);
    normalizer += weight;
  }

  const unsigned long long output_offset =
      (((unsigned long long)query_token * query_heads + query_head) * head_dimension + dimension) * 2U;
  qwen_bf16_store(output_bytes, output_offset, context_sum / normalizer);
}

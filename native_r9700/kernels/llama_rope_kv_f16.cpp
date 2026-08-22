extern "C" __attribute__((global)) void llama_rope_kv_f16(
    const unsigned short* fresh_k,
    const unsigned short* fresh_v,
    unsigned short* k_cache,
    unsigned short* v_cache,
    unsigned int sequence_length,
    unsigned int position,
    unsigned int cache_capacity_tokens) {
  constexpr unsigned int kKvHeads = 8U;
  constexpr unsigned int kHeadDim = 64U;
  constexpr float kRopeTheta = 500000.0f;
  constexpr float kRopeFactor = 8.0f;
  constexpr float kOriginalContext = 8192.0f;
  constexpr float kLowFrequencyFactor = 1.0f;
  constexpr float kHighFrequencyFactor = 4.0f;

  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int token = workgroup / kKvHeads;
  const unsigned int kv_head = workgroup % kKvHeads;
  const unsigned int dimension = __builtin_amdgcn_workitem_id_x();
  if (token >= sequence_length || dimension >= kHeadDim) return;
  if ((unsigned long long)position + sequence_length > cache_capacity_tokens) return;

  const unsigned long long fresh_offset =
      ((unsigned long long)kv_head * sequence_length + token) * kHeadDim + dimension;
  const unsigned long long absolute_token = (unsigned long long)position + token;
  const unsigned long long cache_offset =
      ((unsigned long long)kv_head * cache_capacity_tokens + absolute_token) * kHeadDim +
      dimension;
  const unsigned int pair_dimension = dimension % (kHeadDim / 2U);
  float inv_frequency = __builtin_powf(
      kRopeTheta, -2.0f * (float)pair_dimension / (float)kHeadDim);
  const float wavelength = 6.2831853071795864769f / inv_frequency;
  if (wavelength > kOriginalContext / kLowFrequencyFactor) {
    inv_frequency /= kRopeFactor;
  } else if (wavelength >= kOriginalContext / kHighFrequencyFactor) {
    const float smooth = (kOriginalContext / wavelength - kLowFrequencyFactor) /
                         (kHighFrequencyFactor - kLowFrequencyFactor);
    inv_frequency = (1.0f - smooth) * inv_frequency / kRopeFactor +
                    smooth * inv_frequency;
  }
  const float angle = (float)absolute_token * inv_frequency;
  const unsigned int paired_dimension =
      dimension < kHeadDim / 2U ? dimension + kHeadDim / 2U : dimension - kHeadDim / 2U;
  const unsigned long long paired_offset =
      ((unsigned long long)kv_head * sequence_length + token) * kHeadDim + paired_dimension;
  const float current = (float)__builtin_bit_cast(_Float16, fresh_k[fresh_offset]);
  const float paired = (float)__builtin_bit_cast(_Float16, fresh_k[paired_offset]);
  const float cosine = __builtin_cosf(angle);
  const float sine = __builtin_sinf(angle);
  const float rotated = dimension < kHeadDim / 2U ? current * cosine - paired * sine
                                                   : current * cosine + paired * sine;
  k_cache[cache_offset] = __builtin_bit_cast(unsigned short, (_Float16)rotated);
  v_cache[cache_offset] = fresh_v[fresh_offset];
}

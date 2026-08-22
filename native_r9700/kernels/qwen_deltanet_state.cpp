extern "C" __attribute__((global)) void qwen_deltanet_state(
    const unsigned short* hidden_input,
    unsigned short* convolution_state,
    float* recurrent_state,
    unsigned short* hidden_output,
    unsigned int position,
    unsigned int convolution_width,
    unsigned int convolution_channels,
    unsigned int value_heads,
    unsigned int key_heads,
    unsigned int key_dimension,
    unsigned int value_dimension,
    unsigned long long hidden_capacity_elements,
    unsigned long long convolution_state_capacity_elements,
    unsigned long long recurrent_state_capacity_elements,
    unsigned long long output_capacity_elements) {
  if (convolution_width == 0U || convolution_channels == 0U ||
      value_heads == 0U || key_heads == 0U || value_heads % key_heads != 0U ||
      key_dimension == 0U || value_dimension == 0U) {
    return;
  }

  unsigned long long key_elements = 0ULL;
  unsigned long long value_elements = 0ULL;
  unsigned long long recurrent_required = 0ULL;
  unsigned long long convolution_required = 0ULL;
  if (__builtin_mul_overflow((unsigned long long)key_heads, key_dimension,
                             &key_elements) ||
      __builtin_mul_overflow((unsigned long long)value_heads, value_dimension,
                             &value_elements) ||
      __builtin_mul_overflow(value_elements, key_dimension, &recurrent_required) ||
      __builtin_mul_overflow((unsigned long long)convolution_width,
                             convolution_channels, &convolution_required) ||
      convolution_channels < value_elements) {
    return;
  }

  unsigned long long hidden_required = 0ULL;
  if (__builtin_mul_overflow(key_elements, 2ULL, &hidden_required) ||
      __builtin_add_overflow(hidden_required, value_elements, &hidden_required) ||
      __builtin_add_overflow(hidden_required, (unsigned long long)value_heads,
                             &hidden_required) ||
      __builtin_add_overflow(hidden_required, (unsigned long long)value_heads,
                             &hidden_required) ||
      hidden_capacity_elements < hidden_required ||
      convolution_state_capacity_elements < convolution_required ||
      recurrent_state_capacity_elements < recurrent_required ||
      output_capacity_elements < value_elements) {
    return;
  }

  const unsigned int value_head = __builtin_amdgcn_workgroup_id_x();
  const unsigned int value_channel = __builtin_amdgcn_workitem_id_x();
  if (value_head >= value_heads || value_channel >= value_dimension) {
    return;
  }

  const unsigned int key_head = value_head / (value_heads / key_heads);
  const unsigned long long value_index =
      (unsigned long long)value_head * value_dimension + value_channel;
  const unsigned long long query_offset = 0ULL;
  const unsigned long long key_offset = key_elements;
  const unsigned long long value_offset = key_elements * 2ULL;
  const unsigned long long decay_offset = value_offset + value_elements;
  const unsigned long long beta_offset = decay_offset + value_heads;
  const unsigned long long convolution_index =
      (unsigned long long)(position % convolution_width) * convolution_channels +
      value_index;
  const float convolution_prior = __builtin_bit_cast(
      float, (unsigned int)convolution_state[convolution_index] << 16U);
  convolution_state[convolution_index] = hidden_input[value_offset + value_index];

  const float decay = __builtin_bit_cast(
      float, (unsigned int)hidden_input[decay_offset + value_head] << 16U);
  const float beta = __builtin_bit_cast(
      float, (unsigned int)hidden_input[beta_offset + value_head] << 16U);
  const unsigned long long row =
      ((unsigned long long)value_head * value_dimension + value_channel) *
      key_dimension;
  float kv_memory = 0.0f;
  for (unsigned int key_channel = 0U; key_channel < key_dimension;
       ++key_channel) {
    const unsigned long long recurrent_index = row + key_channel;
    const float prior = recurrent_state[row + key_channel] * decay;
    recurrent_state[recurrent_index] = prior;
    const float key = __builtin_bit_cast(
        float,
        (unsigned int)hidden_input[key_offset +
                                    (unsigned long long)key_head * key_dimension +
                                    key_channel]
            << 16U);
    kv_memory += prior * key;
  }

  const float value = __builtin_bit_cast(
      float, (unsigned int)hidden_input[value_offset + value_index] << 16U);
  const float delta = (value + convolution_prior - kv_memory) * beta;
  float result = 0.0f;
  for (unsigned int key_channel = 0U; key_channel < key_dimension;
       ++key_channel) {
    const unsigned long long recurrent_index = row + key_channel;
    const float key = __builtin_bit_cast(
        float,
        (unsigned int)hidden_input[key_offset +
                                    (unsigned long long)key_head * key_dimension +
                                    key_channel]
            << 16U);
    const float query = __builtin_bit_cast(
        float,
        (unsigned int)hidden_input[query_offset +
                                    (unsigned long long)key_head * key_dimension +
                                    key_channel]
            << 16U);
    const float updated = recurrent_state[recurrent_index] + key * delta;
    recurrent_state[recurrent_index] = updated;
    result += updated * query;
  }
  hidden_output[value_index] =
      (unsigned short)(__builtin_bit_cast(unsigned int, result) >> 16U);
}

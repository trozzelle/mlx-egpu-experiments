extern "C" __attribute__((global)) void qwen_affine4_linear(
    const unsigned short* input,
    const unsigned char* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned short* output,
    unsigned long long input_features,
    unsigned long long output_features,
    unsigned long long input_capacity_elements,
    unsigned long long packed_weight_capacity_bytes,
    unsigned long long affine_group_capacity,
    unsigned long long output_capacity_elements) {
  if (input_features == 0ULL || input_features % 64ULL != 0ULL ||
      input_capacity_elements < input_features ||
      output_capacity_elements < output_features) {
    return;
  }

  const unsigned long long groups_per_output = input_features / 64ULL;
  unsigned long long weight_elements = 0ULL;
  unsigned long long group_count = 0ULL;
  if (__builtin_mul_overflow(output_features, input_features, &weight_elements) ||
      __builtin_mul_overflow(output_features, groups_per_output, &group_count)) {
    return;
  }
  const unsigned long long packed_bytes =
      weight_elements / 2ULL + (weight_elements & 1ULL);
  if (packed_weight_capacity_bytes < packed_bytes ||
      affine_group_capacity < group_count) {
    return;
  }

  const unsigned long long output_index =
      (unsigned long long)__builtin_amdgcn_workgroup_id_x();
  if (output_index >= output_features ||
      __builtin_amdgcn_workitem_id_x() != 0U) {
    return;
  }

  float accumulated = 0.0f;
  for (unsigned long long input_index = 0ULL;
       input_index < input_features;
       ++input_index) {
    const unsigned long long element_index = output_index * input_features + input_index;
    const unsigned int packed = packed_weight[element_index >> 1U];
    const unsigned int nibble_shift = (unsigned int)(element_index & 1ULL) * 4U;
    const unsigned int quantized = (packed >> nibble_shift) & 0x0fU;
    const unsigned long long group_index =
        output_index * groups_per_output + input_index / 64ULL;
    const float scale = (float)__builtin_bit_cast(_Float16, scales[group_index]);
    const float bias = (float)__builtin_bit_cast(_Float16, biases[group_index]);
    const float dequantized = (float)quantized * scale + bias;
    accumulated += (float)__builtin_bit_cast(_Float16, input[input_index]) * dequantized;
  }
  output[output_index] = __builtin_bit_cast(unsigned short, (_Float16)accumulated);
}

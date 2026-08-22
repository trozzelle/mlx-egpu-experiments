extern "C" __attribute__((global)) void llama_v_projection_f16(
    const unsigned short* normalized,
    const unsigned short* v_projection_weight,
    unsigned short* fresh_v,
    unsigned int sequence_length) {
  constexpr unsigned int kHiddenSize = 2048U;
  constexpr unsigned int kKvHeadCount = 8U;
  constexpr unsigned int kHeadDimension = 64U;

  const unsigned long long workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  const unsigned long long token_index = workgroup / kKvHeadCount;
  const unsigned int kv_head = (unsigned int)(workgroup % kKvHeadCount);
  if (token_index >= (unsigned long long)sequence_length) return;
  if (lane >= kHeadDimension) return;

  const unsigned int output_channel = kv_head * kHeadDimension + lane;
  const unsigned long long input_offset = token_index * kHiddenSize;
  const unsigned long long weight_offset =
      (unsigned long long)output_channel * kHiddenSize;
  float accumulator = 0.0f;
  for (unsigned int column = 0U; column < kHiddenSize; ++column) {
    const _Float16 input =
        __builtin_bit_cast(_Float16, normalized[input_offset + column]);
    const _Float16 weight =
        __builtin_bit_cast(_Float16, v_projection_weight[weight_offset + column]);
    accumulator += (float)input * (float)weight;
  }

  const unsigned long long output_offset =
      ((unsigned long long)kv_head * sequence_length + token_index) *
          kHeadDimension +
      lane;
  fresh_v[output_offset] = __builtin_bit_cast(unsigned short, (_Float16)accumulator);
}

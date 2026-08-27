typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x8 __attribute__((ext_vector_type(8)));

static inline __attribute__((device)) unsigned int pack_halves(unsigned short low, unsigned short high) {
  return (unsigned int)low | ((unsigned int)high << 16U);
}

extern "C" __attribute__((global)) void wmma_lane_map_gfx1201(
    const unsigned short* a,
    const unsigned short* b,
    const float* c,
    unsigned int* observations) {
  constexpr unsigned int kWaveSize = 32U;
  constexpr unsigned int kReadbackBytes = 2048U;
  const unsigned int workgroup = __builtin_amdgcn_workgroup_id_x();
  const unsigned int lane = __builtin_amdgcn_workitem_id_x();
  if (workgroup != 0U || lane >= kWaveSize) return;

  const unsigned int column_or_lane = lane & 15U;
  const unsigned int k_base = (lane >> 4U) * 4U;
  const unsigned int a_row_base = column_or_lane * 16U;

  const unsigned int A0 = pack_halves(
      a[a_row_base + k_base + 0U], a[a_row_base + k_base + 1U]);
  const unsigned int A1 = pack_halves(
      a[a_row_base + k_base + 2U], a[a_row_base + k_base + 3U]);
  const unsigned int A2 = pack_halves(
      a[a_row_base + k_base + 8U], a[a_row_base + k_base + 9U]);
  const unsigned int A3 = pack_halves(
      a[a_row_base + k_base + 10U], a[a_row_base + k_base + 11U]);
  const u32x4 a_words = {A0, A1, A2, A3};
  const f16x8 a_fragment = __builtin_bit_cast(f16x8, a_words);

  const unsigned int B0 = pack_halves(
      b[(k_base + 0U) * 16U + column_or_lane],
      b[(k_base + 1U) * 16U + column_or_lane]);
  const unsigned int B1 = pack_halves(
      b[(k_base + 2U) * 16U + column_or_lane],
      b[(k_base + 3U) * 16U + column_or_lane]);
  const unsigned int B2 = pack_halves(
      b[(k_base + 8U) * 16U + column_or_lane],
      b[(k_base + 9U) * 16U + column_or_lane]);
  const unsigned int B3 = pack_halves(
      b[(k_base + 10U) * 16U + column_or_lane],
      b[(k_base + 11U) * 16U + column_or_lane]);
  const u32x4 b_words = {B0, B1, B2, B3};
  const f16x8 b_fragment = __builtin_bit_cast(f16x8, b_words);

  const unsigned int c_row_base = (lane >> 4U) * 8U;
  const f32x8 c_fragment = {
      c[(c_row_base + 0U) * 16U + column_or_lane],
      c[(c_row_base + 1U) * 16U + column_or_lane],
      c[(c_row_base + 2U) * 16U + column_or_lane],
      c[(c_row_base + 3U) * 16U + column_or_lane],
      c[(c_row_base + 4U) * 16U + column_or_lane],
      c[(c_row_base + 5U) * 16U + column_or_lane],
      c[(c_row_base + 6U) * 16U + column_or_lane],
      c[(c_row_base + 7U) * 16U + column_or_lane]};

  f32x8 d_fragment;
  asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %3"
               : "=v"(d_fragment)
               : "v"(a_fragment), "v"(b_fragment), "v"(c_fragment));

  const u32x8 d_words = __builtin_bit_cast(u32x8, d_fragment);
  const unsigned int D0 = d_words[0];
  const unsigned int D1 = d_words[1];
  const unsigned int D2 = d_words[2];
  const unsigned int D3 = d_words[3];
  const unsigned int D4 = d_words[4];
  const unsigned int D5 = d_words[5];
  const unsigned int D6 = d_words[6];
  const unsigned int D7 = d_words[7];

  if (lane * 64U >= kReadbackBytes) return;
  observations[lane * 16U + 0U] = A0;
  observations[lane * 16U + 1U] = A1;
  observations[lane * 16U + 2U] = A2;
  observations[lane * 16U + 3U] = A3;
  observations[lane * 16U + 4U] = B0;
  observations[lane * 16U + 5U] = B1;
  observations[lane * 16U + 6U] = B2;
  observations[lane * 16U + 7U] = B3;
  observations[lane * 16U + 8U] = D0;
  observations[lane * 16U + 9U] = D1;
  observations[lane * 16U + 10U] = D2;
  observations[lane * 16U + 11U] = D3;
  observations[lane * 16U + 12U] = D4;
  observations[lane * 16U + 13U] = D5;
  observations[lane * 16U + 14U] = D6;
  observations[lane * 16U + 15U] = D7;
}

// F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1

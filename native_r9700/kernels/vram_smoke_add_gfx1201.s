.text
.globl vram_smoke_add
.p2align 8
.type vram_smoke_add,@function
vram_smoke_add:
  s_load_b64 s[2:3], s[0:1], 0x0
  s_load_b64 s[4:5], s[0:1], 0x8
  s_load_b64 s[6:7], s[0:1], 0x10
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32_e32 v0, 2, v0
  global_load_b32 v1, v0, s[2:3]
  global_load_b32 v2, v0, s[4:5]
  s_waitcnt vmcnt(0)
  v_add_nc_u32_e32 v1, v1, v2
  global_store_b32 v0, v1, s[6:7]
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel vram_smoke_add
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_kernarg_size 24
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_inst_pref_size 1
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_memory_ordered 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.kernels:
  - .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .offset: 16
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .name: vram_smoke_add
    .private_segment_fixed_size: 0
    .sgpr_count: 8
    .symbol: vram_smoke_add.kd
    .vgpr_count: 3
    .wavefront_size: 32
    .max_flat_workgroup_size: 64
...
.end_amdgpu_metadata

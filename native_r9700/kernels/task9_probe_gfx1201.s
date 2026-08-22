.text
.globl task9_probe_gfx1201
.p2align 8
.type task9_probe_gfx1201,@function
task9_probe_gfx1201:
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel task9_probe_gfx1201
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 8
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_wavefront_size32 1
  .amdhsa_kernarg_size 8
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .name: task9_probe_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 8
    .symbol: task9_probe_gfx1201.kd
    .vgpr_count: 8
    .wavefront_size: 32
    .max_flat_workgroup_size: 1
...
.end_amdgpu_metadata

# LN-2 RMSNorm asset second read

## Scope

Evidence-only inspection of the current RMSNorm HIP source, its checked-in
`gfx1201` image/manifest, the embedded HSA metadata and 64-byte kernel
descriptor, and the current bounded-trace result. No executor, PM4 builder,
later stage, cache, test, or hardware path was inspected or run.

## Image identity and ordinary code-object checks

- The manifest identifies `native_r9700/kernels/llama_rmsnorm_f16.cpp`, target
  `gfx1201`, a 15,857-byte image, and SHA-256
  `0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0`
  (`llama_rmsnorm_f16.json:84-119`). The current image has that exact byte
  count and digest.
- The manifest maps `.rodata` at image offset 1536 and `.text` at 5888
  (`llama_rmsnorm_f16.json:53-63`). The descriptor at 1536 has
  `kernarg_size = 0x20` and entry-relative offset `0x1100`; therefore its
  entry is exactly `1536 + 0x1100 = 5888`, agreeing with the manifest's
  `entry_offset` (`:2,21`).
- The embedded AMDGPU note says
  `amdgcn-amd-amdhsa--gfx1201`, HSA metadata version 1.2, wavefront size 32,
  18 SGPRs, 10 VGPRs, zero SGPR/VGPR spills, zero private/group segment bytes,
  and `uses_dynamic_stack = false`. It declares the four arguments at offsets
  0, 8, 16, and 24 with sizes 8, 8, 8, and 4. Its 28-byte logical kernarg span
  with 8-byte alignment occupies the descriptor's 32-byte allocation; this is
  normal tail padding, not a 28-versus-32 ABI defect.
- The manifest admits no relocations (`:3-19`). Together with the zero private
  segment, no dynamic stack, and no spill counts, the asset exposes no
  source-visible unresolved-call, scratch, LDS, or stack convention hazard.

## Source/instruction convention check

`llama_rmsnorm_f16.cpp:1-26` has no inline assembly or nonstandard register
convention. It uses the target's workgroup-X and workitem-X builtins, ordinary
global pointer loads/stores, fp16 bit reinterpretation, fp32 arithmetic, and
`__builtin_sqrtf`. The early return for lanes 1 through 63 is safe within this
source: lane 0 performs the complete sequential reduction and output loop,
and there is no barrier, LDS access, cross-lane operation, or dependence on a
lane that returned. The metadata's `uniform_work_group_size = 1` is consistent
with the source not relying on a compiler-assumed uniform workgroup.

Accordingly, this read found **no source-visible unsupported instruction,
barrier/divergence, LDS, scratch, or register-spill execution hazard** beyond
what is encoded in the image.

## Source-visible resource-register hazard

There is, however, a direct mismatch between the image's literal kernel
descriptor resource words and the resource words published by the manifest.
The 64 descriptor bytes beginning at image offset 1536 decode as:

```text
+0x00 group_segment_fixed_size  0x00000000
+0x04 private_segment_fixed_size 0x00000000
+0x08 kernarg_size               0x00000020
+0x10 kernel_code_entry_offset   0x0000000000001100
+0x2c compute_pgm_rsrc3          0x0000000a
+0x30 compute_pgm_rsrc1          0x08fc0001
+0x34 compute_pgm_rsrc2          0x00000084
+0x38 kernel_code_properties     0x00000408
```

The same image's manifest instead publishes:

```text
rsrc1 = 0xc00f0001  (3222208513)
rsrc2 = 0x00000084  (132)
rsrc3 = 0x000000a0  (160)
```

(`llama_rmsnorm_f16.json:113-116`). Thus `rsrc2` agrees, but the descriptor's
literal `rsrc1` and `rsrc3` do not: `0x08fc0001 != 0xc00f0001` and
`0x0000000a != 0x000000a0`.

This is an execution hazard, not a kernarg-ABI finding. A raw PM4 dispatch
must use the resource-register convention appropriate to the actual code
object; the inspected asset alone provides no evidence that the manifest's
different `rsrc1`/`rsrc3` values are a documented, target-correct translation
of the descriptor words. If they are programmed verbatim without such a
translation being required, register allocation/wave scheduling can disagree
with the compiled code even though the pointer/scalar ABI is correct. The
prior claim that the literal descriptor and manifest resource words agree is
not supported by the current checked-in image.

## Relation to current trace evidence

The bounded native trace proves byte-exact, finite input at `hidden`, then
fails closed at `normalized` with `failure_stage: trace_nonfinite` and
`trace output contains NaN or infinity`; the CPU oracle output is finite
(`ln-1c-first-stage.md:15-25`). This does not prove the resource mismatch is
causal, because the failed trace currently lacks the materialized PM4 tuple
and buffer/VA evidence. It does place the mismatch squarely in the remaining
asset/PM4 execution fault domain rather than later model stages.

## Verdict

**Hazard found:** manifest `rsrc1` and `rsrc3` are not the current image
kernel descriptor's literal compute-resource words. No independent
source-visible instruction or compiled register/scratch/LDS hazard was found.
The smallest next evidence is a failed-trace PM4 capture showing the actual
programmed `rsrc1`, `rsrc2`, and `rsrc3`, then an explicit comparison against
the descriptor above (or the documented gfx1201 translation, if one is used).

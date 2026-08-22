# Dynamic resident HSA dispatch review

## Scope
Reviewed `native_r9700/amdev_session.h` and `.cpp` only. No validation commands were run.

## Resolved preflight finding
The initial review found that image resources were not validated: zero `rsrc1`, `rsrc2`, or `rsrc3` would have been copied into PM4. The implementation now rejects any selected HSA image with a zero program resource register before PM4 submission, matching the established `kernel_catalog.cpp` convention.

## Confirmed review points
- Dynamic-PTB evidence is recorded immediately after resident image allocation and before cleanup: `dynamic_ptb_count` and `dynamic_ptb_physical_offset` are result fields.
- Kernarg binding range validation avoids unsigned underflow by first requiring `offset <= kernargs.size()` and then checking the remaining byte count.
- Once the compute queue is armed, failed submission/readback paths retire it before resident release. If retirement itself fails, resident mappings are deliberately retained rather than unmapped beneath an unproven-active queue; this is fail-closed.

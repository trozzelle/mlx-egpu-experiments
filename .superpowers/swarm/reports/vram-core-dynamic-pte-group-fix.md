# Dynamic PTE group fix

`DynamicPageTable` now treats `resident_gpu_va_limit` as an exclusive bound: a range ending exactly at that address is valid, while only ranges whose end exceeds it are rejected.

When a newly allocated dynamic PTB fails during zeroing, the released allocation is represented by a zero-sized pending PTB record while its leaf ownership remains reserved. Explicit unmap skips leaf and parent PTE writes for that unlinked record, clears any earlier mapped leaves in the existing safe order, then removes the pending ownership. This preserves collision protection until cleanup and never releases or writes the already released/unlinked PTB allocation.

The supervisor owns RED-to-GREEN execution; no commands were run for this change.

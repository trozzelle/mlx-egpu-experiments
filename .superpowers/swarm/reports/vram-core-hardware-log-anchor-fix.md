# VRAM smoke log anchor fix

## Root cause
The VRAM-smoke-only log path was created and later addressed through `logs/...` path strings. A `logs` symlink swap or directory-entry replacement could therefore disconnect a successful smoke result from the evidence file before the command returned.

## Change
`native_r9700/runtime.cpp` now holds a non-following current-working-directory descriptor, creates and opens `logs` through that descriptor, and records the opened `logs` directory identity with `fstat`. It creates the bounded-suffix log name with `openat(..., O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600)`, fully writes and syncs the file, verifies the open file and non-following leaf entry are the same regular inode, and syncs the anchored directory. Before command completion it re-reads `logs` from the held parent descriptor with non-following `fstatat` and requires that entry to retain the anchored directory device/inode identity.

Every open, write, sync, revalidation, or close failure returns the existing VRAM-smoke log failure path, which reports `failure_stage: log_write`, `log_path: not_written`, and a nonzero status. The ordinary runtime log helper is unchanged. This integrity check covers the command through completion; it does not claim that the returned lexical pathname remains protected from later filesystem mutation after descriptors close.

## Verification
Reviewed the modified source control flow against the required descriptor-anchoring sequence. No command, test, runner, session, or hardware invocation was performed, as required by this assignment.

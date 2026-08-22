# Task9 kernel-assets FIFO fix

## Root cause

The loader opened a manifest-selected direct child with `O_RDONLY | O_NOFOLLOW`. A no-writer FIFO satisfies that path constraint and does not traverse a symlink, but a blocking read-only FIFO open waits for a writer before the existing descriptor `fstat` can reject it as non-regular.

## Closure

The direct-child `openat` now adds `O_NONBLOCK` while retaining the root-fd anchor and `O_NOFOLLOW`. Opening a no-writer FIFO therefore returns a descriptor immediately; the existing same-descriptor `fstat` then rejects it with `kernel code path must name a regular non-symlink file`, before any read or output mutation.

The loader's regular-file path, 4096-byte limit, and output-immutability behavior are otherwise unchanged. Per task constraint, no commands or tests were run in this task.

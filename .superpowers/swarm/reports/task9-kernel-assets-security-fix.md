# Task 9 kernel asset security fix

## Root cause

The loader accepted any safe relative descendant, then canonicalized and inspected the path before reopening it with `std::ifstream`. That accepted nested paths and left a time-of-check/time-of-use gap between the inspected file and the bytes read. Its size check also trusted `file_size` before vector allocation, so a sparse attacker-controlled file could request an excessive allocation.

## Remediation

`load_verified_kernel_code` now:

- permits exactly one non-root, non-parent path component;
- rejects symlink and non-directory asset roots, canonicalizes the supplied root, then retains an `O_DIRECTORY | O_NOFOLLOW` root descriptor;
- opens the direct child through that descriptor with `openat(..., O_RDONLY | O_NOFOLLOW)` and verifies that same descriptor with `fstat`;
- requires a regular file and a nonnegative size at most 4096 bytes before allocating;
- reads exactly the fstat-reported byte count from the verified descriptor, retrying only interrupted reads;
- closes both descriptors by scope exit on every success and failure path.

The existing manifest schema, target, resource metadata, digest equality, descriptor validation, and output assignment ordering are unchanged. Future support for nested assets or larger code files requires an explicit reviewed contract.

## Verification status

No commands were run, per assignment constraint. The two observed RED cases are addressed by the direct-child rejection and pre-allocation 4096-byte `fstat` cap.

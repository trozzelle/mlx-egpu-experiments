# VRAM resident unmap-failure fix

`ResidentMemory` now stops cleanup for an allocation when its page-unmap callback fails and marks the allocation quarantined for the lifetime of the session. A quarantined record keeps both its allocator allocation and its GPU-VA reservation; later `release_all()` calls do not retry it.

Rollback failures commit the attempted GPU-VA reservation before returning the original map failure. During `release_all()`, fully unmapped allocations are released normally, while any unmap failure leaves its allocation resident. GPU-VA reuse resets only after every record has been released, preserving the existing deterministic reuse behavior for successful cleanup.

The supervisor owns the RED-to-GREEN test run; no commands were run for this change.

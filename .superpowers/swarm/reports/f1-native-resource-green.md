# F1 native resource worker GREEN implementation

**Scope:** native source and report only (`native_resource_worker.*`, `runner.cpp`, and the reusable Llama binder seam).

## Implemented

- Private JSONL framing now terminates at every physical newline. Oversized lines are discarded through that newline and malformed/invalid-UTF-8/duplicate-key lines produce one bounded null-correlation decode error before the next line is processed.
- Private `Prefill` parsing accepts an empty prefix through the 128-token resident limit and rejects 129 tokens before backend work. The public registry must pass only the `S-1` prefix to this boundary.
- `NativePrepareResult` carries the concrete `runner_binary_sha256`; the concrete backend hashes the bytes of the running executable (using the resolved executable path) and publishes that value. The existing client-side optional extension becomes a required comparison for the concrete result.
- `Prepare` builds the real Llama weight table and persistent dispatch once, selects the concrete HSA image descriptors, derives the ordered `sha256:` pack list from `image_sha256`, rejects declared/selected list mismatches and zero identities, prepares one `ResidentHsaSession`, and computes the producer fingerprint from those actual identities.
- `Prepare` checks concrete planned resident-buffer, kernel-image, scratch, and total byte counts against all three `ResourceSpec.resource_budget` limits before invoking `ResidentHsaSession::prepare`.
- `NativePersistentExecution` owns the opened `ModelWeightBinder`, validated spans, full embedding tensor, selected images, reusable dispatch/buffers, and resident session. Committed `Prefill` copies embedding rows from generation storage, never selects or reopens a shard, and only uploads request-specific hidden windows; model weights remain resident and are never re-uploaded.
- A zero-prefix `Prefill` emits canonical empty `(1,8,0,64)` K/V arrays and its hardware log without dispatch/upload/readback. Resident-session cleanup phase diagnostics are routed to stderr so private JSONL stdout remains uncontaminated.
- The Llama layer-table builder has an already-open-binder seam so the concrete owner retains the exact validated binder across warm requests.

## Verification status

No commands, builds, tests, formatters, package managers, hardware, or git operations were run for this implementation. Supervisor owns compile/test/hardware verification.

# ADR 0004 — macOS eGPU runtime selected as C1 substrate

**Status:** Accepted (2026-08-18)

## Decision

The local macOS eGPU runtime — TinyGPU.app / `APLRemotePCIDevice` / `PCIIface` native AMDev path — is selected as the initial production substrate for C1 (native R9700 producer). This follows the C0A25 kernel-proof PASS on that exact path (`--kernel-proof` green: kernel launch, CPU comparison, host↔device transfer). C1 proceeds on this macOS native substrate; Linux ROCm/HIP remains the reference and deferred fallback.

## Rejected alternative

- **Linux ROCm/HIP as the primary C1 substrate** — rejected because C0A25 proved the macOS native path end-to-end (load, store, launch, readback, parity), while the ROCm path lacks an equivalent C0 evidence gate here and targets a different silicon (Strix Halo/gfx1151). It stays reference, not primary.
- **Waiting for a generic/portable substrate before starting C1** — rejected because the proven macOS native path is sufficient to freeze the C1 contract and proceed; a wait defers integration value without reducing the kernel/runtime risk already retired by C0A25.

## Consequences

- The C1 contract is frozen on the macOS native path: 24-byte kernarg layout, KV interchange format, and the `P == R` producer/consumer parity gate.
- C1 parity work proceeds on macOS native; Linux ROCm/HIP is tracked as reference and can be promoted later if a parity or portability need outweighs the proven native path.
- Kernarg layout and kernel-store conventions locked by C0A21/C0A24/C0A25 (store/addressing) are treated as stable inputs to C1 task set 2.

**Links:** `CONTEXT.md` (Path C, Native R9700 producer), `docs/tasks/native-r9700-producer/README.md` (C0 → macOS substrate SELECTED for C1), `phase-c0a-macos-egpu-runtime-focus.md`, swarm report `c0a-compute-task-16-load-path-fix.md`.

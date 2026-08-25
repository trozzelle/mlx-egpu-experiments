# ADR 0007 — TinyGPU remains device owner behind a portable inference HAL

**Status:** Accepted (2026-08-25)

## Decision

**TinyGPU** remains the sole macOS DriverKit device owner for the R9700. It owns attachment and power lifecycle, protected BAR/device resources, buffer and virtual-address authority, queue creation, validated submission, fences, fault capture, and reset. Raw PCI/MMIO access remains diagnostic-only and is not the inference-client contract.

A deliberately small, vendor-neutral **Inference HAL** sits above TinyGPU. Its target concepts are Device, Buffer, Executable, CommandBuffer, Queue, Fence, timestamps, and fault queries. Only the AMD/TinyGPU backend is in current scope; portability is an interface constraint, not a requirement to implement another vendor.

`lemonade-sdk/mac-amdgpu`, tinygrad AMDev, and Linux amdgpu remain Port/Adapt or Normative reference sources. Their validated lifecycle, register, queue, and recovery sequences may be translated into TinyGPU after provenance, licensing, and differential-conformance review. They do not replace TinyGPU ownership.

## Considered alternatives

- **A new converged project DEXT** was rejected because it duplicates the proven TinyGPU ownership path and creates a high-risk migration boundary.
- **Adopting mac-amdgpu as device owner** was rejected because its cold-initialization evidence is valuable but its compute and public-ABI maturity trails the current native TinyGPU/AMDev path.
- **Using AMD-specific runtime objects as the public platform API** was rejected because engine and future device integrations should not depend on PM4, SDMA, MQD, or MMIO details.
- **Adopting IREE HAL or ROCr wholesale** was rejected because both exceed the required inference surface and assume runtime layers not present over macOS DriverKit.

## Consequences

- ADR 0004 remains valid historical evidence for the C1 substrate; this ADR extends the ownership decision beyond C1.
- TinyGPU hardening and cold-init work proceed without replatforming the accepted producer.
- The HAL may copy interface discipline from IREE and PJRT but owns its own minimal ABI and conformance suite.
- A future second vendor backend requires a separate capability decision; NVIDIA is not treated as an extension of the AMD runtime.

**Links:** `CONTEXT.md`, `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, `docs/REFERENCES.md`, ADR 0004.
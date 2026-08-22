# ADR 0003 — Path C uses a hybrid staged boundary

**Status:** Accepted (2026-08-16)

## Decision

Path C starts as a tinygrad-free **native R9700 producer** behind the existing KV interchange format, not as an immediate mlx-lm/oMLX backend rewrite or a DwarfStar fork. The first Path C phase runs a dual-track runtime spike — local macOS eGPU minimal-kernel launch and Linux ROCm/HIP reference build — then promotes whichever path can support the first native producer parity gate.

## Rejected alternative

- **Jump directly to a native consumer backend** — rejected for the first Path C gate because it couples driver/runtime risk, kernel correctness, mlx-lm scheduler changes, and oMLX integration before a tinygrad-free producer has proven KV parity.
- **Fork DwarfStar as the architecture** — rejected because DwarfStar is deliberately narrow and model-specific; it is source-level prior art for kernels, runtime structure, KV/session handling, and quality gates, not this project's product boundary.
- **Choose macOS eGPU or Linux ROCm upfront without evidence** — rejected because the current eGPU works through TinyGPU/tinygrad on macOS, while DwarfStar's ROCm path targets Linux Strix Halo/gfx1151. The first runtime phase must measure both before locking the substrate.

## Consequences

- The Phase 0 token-exact parity gate remains the producer-swap gate for Path C.
- `docs/ROADMAP.md` sequences Path C as runtime discovery → native producer parity → consumer integration → optional native backend.
- `docs/DESIGN.md` treats DwarfStar as a reference corpus, not a dependency or implementation plan.

**Links:** `CONTEXT.md` (Path C, Native R9700 producer, DwarfStar reference), `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, `docs/ROADMAP.md`.

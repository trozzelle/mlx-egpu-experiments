# Native R9700 Producer

## Goal

Build a tinygrad-free prefill producer on the AMD Radeon AI PRO R9700 that emits an mlx-lm-compatible prompt cache and passes token-exact `P == R` parity. The producer owns prefix KV truth; mlx-lm receives the final prompt token after importing the `S-1` cache.

## Canonical project documents

- `CONTEXT.md` — vocabulary and producer/consumer boundary.
- `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, and `docs/ROADMAP.md` — durable architecture, contracts, and phase gates.
- `docs/adr/` — accepted decisions. ADR 0005 prevents CPU-reference results from being labeled native acceptance.
- `docs/pinned-upstream-interfaces.md` — mlx-lm cache ABI and substrate facts.
- `docs/path-a-validation-results.md` — Phase 0 parity baseline.

## Current execution surface

- `2026-08-23-llama-numerical-debug-plan.md` — primary plan and acceptance gates.
- `phase-llama-numerical-trace.md` — bounded oracle/trace contract.
- `phase-llama-numerical-remediation.md` — one-stage-at-a-time repair and recurrence sequence.
- `phase-ln-2-rmsnorm-transcendental-diagnosis.md` — replanned RMSNorm root-cause (Class C compute,
  transcendental-isolated) after the PDF + mac-amdgpu reference landed.
- `phase-c1-c2-r9700-recovery-plan.md` — broader C1R/C2R objective and dependencies.
- `phase-c1r-native-llama-delivery.md` and `phase-native-producer-swarm-integration.md` — current native delivery integration.
- `phase-qwen3-8-native-text-delivery.md` — separate Qwen target-expansion contract; it remains blocked on meaningful Llama acceptance.
- `validation-commands.md` — exact validated commands and discovery rules.
- `.superpowers/swarm/progress.md` — compact current swarm status.

## Current status

- C0 substrate/runtime proof: complete. macOS TinyGPU.app / `APLRemotePCIDevice` / `PCIIface`, `1002:7551`, `gfx1201`, and the hardware kernel proof are the selected native substrate.
- C1R native producer: open. A two-token native prefill reaches all 16 layers and emits a schema-valid `r9700_native` NPZ, but prompt-0 parity fails and LN-1 localized the first numerical failure to layer-0 RMSNorm normalization.
- LN-2: blocked on TinyGPU/R9700 recovery plus a fresh C0 health-gate pass. No arithmetic repair is accepted until the bounded RMSNorm trace can run again.
- C2R and Qwen: blocked on finite, numerically valid native Llama K/V and C1R token-exact parity.

## Non-negotiable acceptance rules

- CPU/NumPy is an oracle only; it must never populate an accepted native artifact.
- Cache artifacts contain only the prefix; `generate_step` receives the final prompt token.
- After a cache is accepted, decode may not recompute or repair the prefix.
- Native acceptance requires fresh hardware identity, `exit_status: 0`, finite validated K/V, and token-exact `P == R`.

## Historical material

Completed C0, CPU-reference C1/C2/C3, prior product-worker, Superpowers, supervisor, and report material moved to `docs/archive/`. It is provenance only; begin planning from the documents above.

# ADR 0005 — CPU reference producer is not Native R9700 producer acceptance

**Status:** Accepted (2026-08-18)

## Decision

The existing `native_r9700.prefill` CPU/NumPy implementation and the C2 `native_r9700.serving` wrapper built around it are reclassified as **reference and ABI-oracle work**, not as completion evidence for the Native R9700 producer objective.

A Path C Native R9700 producer is accepted only when the Llama 3.2 1B prefill model-forward tensor work executes on the AMD Radeon AI PRO R9700/eGPU through the selected native runtime substrate, emits the `S-1` KV interchange prompt-cache artifact, and passes the token-exact `P == R` gate against mlx-lm.

A Path C C2 serving integration is accepted only when large prompts route through that actual R9700/eGPU producer, not through the CPU reference producer. The CPU reference path may remain available for tests, diagnostics, ABI validation, and fallback comparison, but it must be labeled as such in docs, logs, reports, and plans.

## Reclassified artifacts

- `native_r9700.prefill`: CPU/NumPy reference producer and prompt-cache ABI oracle.
- `native_r9700.kv_cache`: prompt-cache safetensors emitter; still reusable by the real R9700 producer if the R9700 path emits the same NPZ/intermediate shape, but not proof of R9700 compute by itself.
- `native_r9700.parity`: parity harness; valid gate harness when the producer under test is identified honestly.
- `native_r9700.serving`: mlx-lm imported-cache wrapper and fallback/security harness; C2 reference evidence until its producer route is backed by R9700/eGPU model-forward compute.
- `logs/c1-parity/*`, `logs/c2-serving/*`, and `docs/path-a-validation-results.md` Path C/Path C2 sections produced before this ADR: reference/ABI evidence, not Native R9700 acceptance evidence.

## Reason

The original project objective is to run prefill on the R9700/eGPU. The prior C1/C2 execution proved a valuable but weaker property: an independent tinygrad-free CPU implementation can emit mlx-lm-compatible prompt-cache artifacts and a serving wrapper can consume them.

That property de-risks the KV interchange ABI, RoPE/position semantics, `S-1` final-token injection, redacted logging, fallback policy, and mlx-lm cache import path. It does not prove that model-forward computation runs on the selected R9700 substrate.

Conflating "tinygrad-free" with "R9700/eGPU compute" hid the missing model-kernel work. This ADR restores the acceptance boundary.

## Consequences

- Phase C1 and Phase C2 are reopened for the original R9700/eGPU objective.
- Existing CPU/NumPy parity and C2 wrapper work stays useful as a reference oracle and integration harness.
- Future C1/C2 reports must include a producer implementation label such as `cpu_reference` or `r9700_native`, plus hardware evidence for `r9700_native` runs.
- C1 cannot pass unless R9700/eGPU model-forward prefill produces the accepted prompt cache.
- C2 cannot pass unless large prompts use the R9700/eGPU producer route and still satisfy fallback/security semantics.
- C3 direct consumer backend work remains downstream of real C2 evidence, not CPU-reference evidence.

## Resolution evidence

By 2026-08-25 the reopened gates were satisfied: native 16-layer Llama prefill produced hardware-backed `r9700_native` artifacts, C1R was token-exact through prompt-128, and C2R accepted the actual R9700 producer route with no fallback. The CPU/NumPy classification and hardware-evidence requirement remain in force.

**Links:** `docs/ARCHITECTURE.md`, `docs/DESIGN.md`, `docs/ROADMAP.md`, `docs/archive/tasks/native-r9700-producer/phase-c1-native-producer-parity.md`, `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md`.

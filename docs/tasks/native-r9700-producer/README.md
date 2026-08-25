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
- `docs/superpowers/plans/2026-08-24-native-prefill-launch-transport-optimization-v2.md` — launch/transport optimization plan; golden base `d902f06` (re-certified token-exact C1R/C2R 2026-08-24).
- `docs/superpowers/specs/2026-08-24-native-prefill-compute-side-optimization-design.md` and
  `docs/superpowers/plans/2026-08-24-native-prefill-compute-side-optimization.md` — approved
  measurement, GPU-profile, barrier, token-block, and targeted-kernel sequence.

## Current status

- The selected substrate remains macOS TinyGPU.app / `APLRemotePCIDevice` /
  `PCIIface`, `1002:7551`, `gfx1201`. A corrected GC TLB probe now uses the
  source-grounded GC `HDP → REQ → ACK` sequence; the MMHUB-only semaphore is
  no longer applied to GC.
- **Cold-native limitation:** after a true enclosure/server cold start, native
  GC invalidate REQ still receives no ACK until one explicit tinygrad AMD
  initialization runs. After that control initialization, native kernel proof
  and VRAM smoke pass. All performance/acceptance results below are therefore
  honest **warm-state native evidence**, not proof-complete tinygrad-free cold
  bring-up.
- Llama 3.2 1B C1R is token-exact at prompt 0/16/64/128. Native C2R prompt
  16/128 imports and decodes the accepted prompt cache without fallback.
- Launch/transport optimization reduced prompt-128 from 104.6 seconds to
  approximately 44.1 seconds with direct-ring ten-stage batching.
- Compute-side optimization evaluated token blocks `1,2,4,8,16,32`; every size
  passes the full C1R matrix and native C2R 16/128. Three-run prompt-128 medians:
  B1 44.137 s, B2 27.594 s, B4 20.319 s, B8 29.736 s, B16 33.165 s, B32
  34.084 s.
- Production defaults are now **block size 4**, **terminal timeline**, and
  **Full barriers**. B4 emits 5,120 kernels in 512 submissions across 32 blocks,
  reaches 6.30 prefix tokens/s, and is 53.96% faster than B1 / 80.58% faster
  than the original 104.6-second path. Ten consecutive prompt-128 B4 runs pass
  at 20.230–20.415 seconds without a TinyGPU restart.
- K/V projection overlap remains diagnostic-only: its 0.64% gain is below the
  3% promotion threshold. The selected B4 GPU profile ranks gate/up projection
  first (51.66%), attention score second (21.87%), MLP down third (12.01%), and
  RMSNorm fourth (10.47%). Query-RoPE tuning was correctly skipped because
  attention score did not rank first.
- Final changed-boundary verification is 258/258 passing and the full
  21-translation-unit runner builds without warnings. Whole-branch
  correctness/security and promotion reviews are clean. The pre-existing
  broader-suite test closures missing `hardware_lock.cpp` remain outside this
  optimization slice.
- Qwen3.8-27B remains a separate target-expansion slice; CPU/NumPy evidence is
  not native acceptance.

## Non-negotiable acceptance rules

- CPU/NumPy is an oracle only; it must never populate an accepted native artifact.
- Cache artifacts contain only the prefix; `generate_step` receives the final prompt token.
- After a cache is accepted, decode may not recompute or repair the prefix.
- Native acceptance requires fresh hardware identity, `exit_status: 0`, finite validated K/V, and token-exact `P == R`.

## Historical material

Completed C0, CPU-reference C1/C2/C3, prior product-worker, Superpowers, supervisor, and report material moved to `docs/archive/`. It is provenance only; begin planning from the documents above.

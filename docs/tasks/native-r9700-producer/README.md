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

- C0 substrate/runtime proof is established on macOS TinyGPU.app /
  `APLRemotePCIDevice` / `PCIIface`, `1002:7551`, `gfx1201`.
- Llama 3.2 1B C1R is token-exact at prompt 0/16/64/128. C2R prompt 16/128
  imports and decodes the accepted native prompt cache without fallback.
- The accepted launch/transport baseline uses persistent SDMA and a ten-stage
  direct-ring batch. Prompt-128 improved from 104.6 seconds to approximately
  43.7 seconds at `512a58a`, with 20,480 kernels and 2,048 compute submissions.
- Compute-side software on `opt/compute-side-token-blocks` now includes exclusive
  host timing, per-operation RPC accounting, optional T0–T10 GPU-clock capture
  in the proven compute-control page, diagnostic terminal/overlap barrier
  policies, and token blocks `1,2,4,8,16,32` with zero-padded exact-fill
  embedding uploads. Production defaults remain block size 1,
  `PerStageTimeline`, and `Full`; no unmeasured policy has been promoted.
- Hardware-free final changed-boundary verification is 232/232 passing and the
  full native runner builds without warnings. Whole-branch correctness/security
  review is clean for reachable software. The current broader native suite reports
  670 passed / 64 documented baseline failures, primarily test compile closures
  missing `hardware_lock.cpp`.
- Current hard blocker: three fresh kernel-proof attempts on 2026-08-25 reached
  TinyGPU but `CFG_READ` returned `Driver not available`; macOS
  `system_profiler SPPCIDataType` lists only the Thunderbolt Ethernet controller,
  not the R9700. No server restart or process kill was attempted. Until R9700
  access is restored, the GPU timestamp
  profile, terminal/overlap A/B, block-size ladder, C1R/C2R/stability recertification,
  and profile-gated query-RoPE optimization remain unverified.
- Qwen3.8-27B remains a separate target-expansion slice; CPU/NumPy evidence is
  not native acceptance.

## Non-negotiable acceptance rules

- CPU/NumPy is an oracle only; it must never populate an accepted native artifact.
- Cache artifacts contain only the prefix; `generate_step` receives the final prompt token.
- After a cache is accepted, decode may not recompute or repair the prefix.
- Native acceptance requires fresh hardware identity, `exit_status: 0`, finite validated K/V, and token-exact `P == R`.

## Historical material

Completed C0, CPU-reference C1/C2/C3, prior product-worker, Superpowers, supervisor, and report material moved to `docs/archive/`. It is provenance only; begin planning from the documents above.

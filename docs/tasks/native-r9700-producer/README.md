# Native R9700 Producer — Implementation Plan

## Source grounding

- `CONTEXT.md` — canonical terms: Path C, Native R9700 producer, Native consumer backend, DwarfStar reference, KV interchange format.
- `docs/ARCHITECTURE.md` — Path C hybrid staged boundary; native producer before native consumer backend; DwarfStar as reference only.
- `docs/DESIGN.md` — Native R9700 producer contract, Runtime-discovery gate, DwarfStar reference contract, validation/error gates.
- `docs/ROADMAP.md` — Phase C0 through C3 sequencing and promotion gates.
- `docs/adr/0001-kv-interchange-format-boundary.md` — KV interchange format remains the boundary for Path A and first Path C producer.
- `docs/adr/0002-producer-owns-kv-truth.md` — producer owns KV truth; consumer holds compatibility state.
- `docs/adr/0003-hybrid-staged-path-c.md` — Path C starts as native producer, not backend rewrite or DwarfStar fork.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md` — CPU reference producer and CPU-backed C2 wrapper are not native R9700 acceptance.
- `docs/pinned-upstream-interfaces.md` — mlx-lm KV ABI and `generate_step` S-1 prompt-cache contract; TinyGPU/tinygrad AMD facts for comparison.
- `docs/egpu-prefill-offload-reference.md` — Path C runtime-spike rationale and DwarfStar source facts.
- `docs/path-a-validation-results.md` — passing Phase 0 parity baseline.
- `docs/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md` — reopened C1R/C2R task plan for the original R9700/eGPU objective.
- `.superpowers/swarm/reports/c1r-task-2-runtime-proof.md` — C1R-2 reusable runtime hardware proof evidence.

## Goal

Build a tinygrad-free **Native R9700 producer** that runs prefill work on the AMD Radeon AI PRO R9700, emits the validated KV interchange format, and passes the same token-exact producer-swap gate against mlx-lm. Only after producer parity and serving integration are proven should the project decide whether to build a direct native mlx-lm/oMLX backend.

## Production milestone discipline

This project optimizes for the shortest working vertical slice that produces user-usable prefill offload. This is a race, not a marathon.

Do **not** get trapped in exhaustive primitive proofs, full proof ladders, proof-complete hardware implementation claims, or broad low-level comparison campaigns unless the current production blocker specifically requires that evidence. Use narrow diagnostics only to unblock the next product milestone.

The priority is a working prefill worker that can be posted on GitHub and used by others:
- benchmarkable end-to-end behavior before exhaustive proof completeness;
- honest logs and limitations, not theoretical proof artifacts;
- production API compatibility and repeatable setup over internal proof coverage;
- shortest path with engineering excellence: simple, maintainable, testable, and no fake success labels.

Qwen is a product goal. Do not silently defer it as “out of scope” when the user target includes Qwen. Keep the Llama vertical slice as the fastest first benchmark path when it is the shortest path, then choose the shortest honest Qwen target-expansion path and make blockers explicit.


## Non-goals

- Do not start with an mlx-lm/oMLX backend rewrite.
- Do not fork DwarfStar or adopt its model scope, server/API boundary, compressed KV/session format, or Strix Halo ROCm target as this project's architecture.
- Do not build a generic ROCm platform or general GGUF runner.
- Do not silently downgrade producer acceptance to semantic equivalence; `P == R` token-for-token is the gate.
- Do not expose network/TCP transport before a focused security/transport review.

## Current status

| Phase | State | Current reader guidance |
|---|---|---|
| C0 | **macOS substrate selected; GC compute recovery PASS** | Historical archive review verified the working C1/C0 BAR2 wire protocol. After a host/eGPU reset exposed GC-only MEC page faults, C0 now preserves the device-provided MEC firmware start pair and completes Tinygrad-derived GC AGP/invalidate-range setup. Fresh `--kernel-proof` passed with `doorbell_hit=1`, kernel launch, exact CPU comparison, and `exit_status: 0`. |
| C1 | **C1R-1/C1R-2 complete; real embedding hardware path PASS; model-forward work open** | `native_r9700::RuntimeSession::kernel_proof` wraps the native C0 proof. Fresh VRAM smoke passed, then the HSA Llama embedding-row path loaded a binder-selected 4096-byte safetensors row and produced exact fp16 output bytes on `1002:7551` / `gfx1201`. Actual C1 remains open until all Llama 3.2 1B prefill model-forward tensor work runs on the R9700/eGPU and passes `P == R`. |
| C2 | **Reopened after C1R** | The completed `native_r9700.serving` work proves mlx-lm imported-cache wrapper, fallback, and security behavior against the CPU reference producer. Actual C2 remains open until large prompts use the R9700/eGPU producer route. |
| C3 | Dependency-blocked | Wait for real C2 R9700 serving/performance evidence before any native backend decision or prototype. |

Required next action: execute C1R-3/C1R-4 in `phase-c1-c2-r9700-recovery-plan.md`. Do not treat CPU/NumPy parity, CPU-backed C2 serving, or C3 CPU-reference timing as completion evidence for the original R9700/eGPU producer objective.

## Phase documents

| Phase | Document | Outcome |
|---|---|---|
| C0 | `phase-c0-runtime-discovery.md`; mac-first continuation in `phase-c0a-macos-egpu-runtime-focus.md`; AMDev/SDMA unblock path in `phase-c0b-native-amdev-sdma-transfer.md`; completed VM prerequisite in `../native-r9700-gfx12-vm-pte-tlb/README.md` | C0A host-device transfer proof and minimal kernel launch proof are Done (C0A25 PASS); C0 substrate decision rerun Done: **macOS TinyGPU/AMDev native selected for C1**. |
| C1 CPU reference | `phase-c1-native-producer-parity.md`; ADR 0005 | CPU/NumPy reference producer, prompt-cache ABI oracle, and parity harness exist. Reclassified: not Native R9700 producer acceptance. |
| C1R/C2R recovery | `phase-c1-c2-r9700-recovery-plan.md` | Concrete plan to fulfill the original R9700/eGPU C1 and C2 objectives. |
| C2 CPU reference wrapper | `phase-c2-serving-integration.md`; ADR 0005 | mlx-lm imported-cache wrapper and security/fallback behavior exist against the CPU reference producer. Reclassified: not native C2 acceptance. |
| C3 | `phase-c3-native-backend-decision.md` | Historical decision was based on CPU-reference evidence; real C3 remains blocked pending real C2 R9700 evidence. |
| Validation | `validation-commands.md` | Shared exact commands and discovery rules. |

## Sequencing dependencies

```text
Phase 0 PASS (done)
    ↓
C0 runtime discovery
    ↓
C1R native R9700/eGPU producer parity
    ↓
C2R R9700/eGPU serving integration
    ↓
C3 native backend decision/prototype
```

Bridge Phase A1/A2 (tinygrad daemon/consumer wrapper) is optional and not a prerequisite for Path C. Use it only if a working service bridge is needed before the native producer lands.

## Shared contracts and artifacts

- **KV interchange format:** `.safetensors` prompt cache compatible with mlx-lm `load_prompt_cache`; per-layer `KVCache`, empty `meta_state`, global `offset`, fp16 K/V, Llama 3.2 1B `(1, 8, N, 64)` geometry for the first parity model.
- **mlx-lm injection contract:** imported cache covers `S-1`; final prompt token is supplied to `generate_step`.
- **RoPE contract:** Llama-3 scaling comes from the MLX `config.json` sidecar when GGUF lacks `rope_scaling`.
- **Parity gate:** native baseline `R` = mlx-lm prefill/decode; producer path `P` = native R9700 prefill/export/import + mlx-lm decode; success is `P == R` across the Phase 0 prompt suite.
- **Review artifacts:** every GPU/harness run writes a local log under `logs/`; logs and model files stay uncommitted.
- **Archived proof artifacts:** oversized C1 proof/source-as-data files were moved out of tracked
  source to the gitignored local archive
  `artifacts/native-r9700-c1-proof-archive/20260821T202312Z/` with a `MANIFEST.txt` recording
  original paths, byte counts, sha256 values, and pre-cleanup HEAD `d3ba1c4`. The archive is
  forensic-only: no product, runtime, build, test, help, or validation path may compile, link,
  parse, read, or display `c1_primitive_bridge.cpp`.
- **Retired primitive diagnostics:** the only compatibility seam is the explicitly named
  `--legacy-primitive-diagnostic <name>` route with an executable injected through
  `NATIVE_R9700_C1_PRIMITIVE_BRIDGE`. Without injection it fails nonzero with
  `failure_stage: legacy_proof_unavailable`; it is never native-prefill acceptance.
- **Reference corpus:** DwarfStar / `antirez/ds4` is prior art for narrow engine/kernels/testing only.

## Orchestration overview

- C0 completed source capture and proof-lane attempts, then continued through the serial macOS eGPU runtime path in `phase-c0a-macos-egpu-runtime-focus.md`; Linux ROCm/HIP remains a reference fallback rather than the initial work lane. The VM/PTE/TLB prerequisite, native SDMA host-device transfer proof, and minimal kernel launch/readback proof are complete.
- C1 CPU-reference work completed a useful prompt-cache ABI oracle but did not execute model-forward tensor math on the R9700/eGPU. Per ADR 0005, actual C1R serializes on building that R9700/eGPU model-forward producer.
- C2 CPU-reference work completed mlx-lm imported-cache wrapper/fallback/security behavior but did not route large prompts through an R9700/eGPU producer. Per ADR 0005, actual C2R serializes on C1R parity.
- C3 serializes on real C2R performance evidence; it must not start as implementation without a decision task and likely a new ADR if the KV interchange fast path is retired.

## Current blockers and future decisions

- C1R blocker: no Llama 3.2 1B model-forward tensor path currently runs on the R9700/eGPU outside tinygrad.
- C2R blocker: no accepted R9700/eGPU producer route exists for the serving wrapper.
- Qwen3.8-27B is a product goal, not a theoretical afterthought. It requires a separate target-expansion slice because the current local target has a different MLX-VLM/quantized/hybrid-cache ABI than the Llama prompt-cache path; pursue the shortest honest Qwen path after the first working R9700 prefill-worker benchmark slice, not an exhaustive proof ladder.
- C2 oMLX scope remains a future decision after C1 parity: mlx-lm only first, or include oMLX imported-cache seam.
- C3 backend seam remains a future decision after C2 evidence: mlx-lm first, oMLX first, shared backend layer, or no direct backend.
- Post-Llama model target remains a future decision after C1 parity.

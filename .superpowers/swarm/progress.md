# R9700 Products — Current Progress

## Authority

- Current implementation plan: `docs/IMPLEMENTATION_PLAN.md`.
- Current capability gates: `docs/ROADMAP.md`.
- Current supervisor/swarm packets: `docs/tasks/r9700-products/README.md`.
- Active command ledger: `docs/tasks/native-r9700-producer/validation-commands.md`.
- Completed producer packets, plans, handoffs, and diagnostic task sets: `docs/archive/README.md`.

## Current facts

- C0 kernel proof and resident-VRAM smoke pass on the R9700 (`1002:7551`, gfx1201).
- The native 16-layer prefill emits a schema-valid `r9700_native` NPZ with finite,
  numerically-correct K/V (all 16 layers, ULP-level vs the CPU reference).
- **C1R is token-exact at prompt-0, prompt-16, prompt-64, and prompt-128** (the
  full 128-token resident cache): `[12366,13,578,469]`, `[11,706,28995,12207]`,
  `[279,4216,62520,9478]`, `[13,578,30791,17604]`.
- **C2R imported-cache serving passes** (prompt-16 and prompt-128):
  `route=native_producer`, `accepted_cache=true`, `fallback_reason=none`,
  token-exact decode.
- Root causes fixed: single-dispatch compute ring (`5755f8d`), missing
  completion-timeline reset (`8f2f0ca`), launch geometry + o-proj/gated-MLP width
  (`c8f5770`), missing query RoPE (`36bf94a`), fused gated-MLP PCIe blowup
  (`6036802`), and the 64-key attention span (`c26f801`).
- B0 remains a scalar/native correctness baseline, not the target performance architecture. F1 persistent worker is promoted; F2 WMMA foundation, production Inference HAL/Kernel Pack integration, and native Qwen acceptance have not promoted.

## Current follow-up ledger

| Phase | Status | Evidence | Next gate |
|---|---|---|---|
| B0 native Llama producer/serving baseline | Complete | C0 hardware proof; C1R prompt-0/16/64/128 exact; C2R no-fallback | Preserve as regression control. |
| F1 persistent warm worker | Complete | Accepted producer/serving path; multi-PDB1 resident allocation; persistent private worker | Preserve the first authoritative cold/warm/GPU-compute baseline and no-reload service contract. |
| F2 gfx1201 WMMA foundation | Ready: task 3A | Tasks 1–2 Done; lane-map exact; offline admission contracts pass | Acquire pinned sources/select candidate image, then 3B admission→G0. |
| P1 TinyGPU Device Owner hardening | Ready: tasks 1A and 2A | Stable ABI/source foundations reviewed; nine host contracts and Xcode builds pass | Import ABI amendment and cold-firmware provenance run concurrently. |
| P2 Inference HAL | Ready: task 1 | Stable P1 ABI subset is Done; G0/P1 completion are promotion-only gates | Freeze portable/stable/deferred operation matrices, then run tasks 2/3A concurrently. |
| P3 first-class Kernel Packs | Blocked on G0 | Tasks 1–4 Done; 13 scalar packs and RMSNorm hardware evidence accepted | Consume exact G0 in task 5, then final promotion. |
| Q1 Qwen contract/oracle package | Ready: task 7 | Tasks 1–6 Done; deterministic 259-test oracle package | Close immutable base revision/license provenance. |
| F3 matrix projection graph | Blocked | F1 model-handle/prepacking accepted; F2 WMMA/G0 missing | Start after accepted F2/G0; promote gate/up → down → fused QKV → O. |

## Guardrails

- Stop at the first non-finite or out-of-tolerance stage.
- CPU/NumPy is oracle evidence only; it cannot produce an accepted native artifact.
- Preserve `S-1` cache semantics and final-token injection.
- Q1 implementation is complete; task set 7 provenance closure may proceed independently, while native Qwen work remains downstream of F2–F4/F6.

## Launch/Transport Optimization (2026-08-24)

Related archived plans: `docs/archive/superpowers/plans/2026-08-24-native-prefill-compute-side-optimization.md` and `docs/archive/superpowers/plans/2026-08-24-native-prefill-compute-batching-inpage-kernargs.md`.
Historical work boundary: branch `feature/native-r9700-producer` at `3d314bc`; this section records a completed/dropped experiment, not the current checkout authority.
Baseline: 128-token native prefill 104.6 s wall / 12.7 s CPU; kernel_count=20480; transfer_bytes=2072649728.

Waves: W1 = T1+T2 (parallel, disjoint files) → W2 = T3 (amdev_session.cpp) → W3 = T4+T5 (parallel, disjoint files) → W4 = T6 (supervisor verify).

| Task | Status | Owner | Deps | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| T1 Monotonic PM4 timeline value | Done | — | — | opt-t1-monotonic-timeline.md | 2/2 passed; field at struct end (positional-agg safe) | — |
| T2 Parameterize submit/poll | Done | — | — | opt-t2-transport-params.md | 24/24 passed; build exit 0 | — |
| T3 Batched resident dispatch | Dropped | — | T1, T2 | opt-t3-batched-dispatch.md + opt-t3b-perstage-kernargs.md | contract 38/38, but HW: CP never fetches (rptr=0) with kernargs ring (26-page control) | reverted — per-stage kernargs/26-page compute-control breaks CP fetch; needs HW debug |
| T4 Batched SDMA upload | Dropped | — | — | opt-t4-sdma-upload.md | contract pass, but HW: SDMA fence timeout | reverted — SDMA ring uses fixed offset 0 + reset-per-chunk; needs cumulative wptr + wrap |
| T5 Wire prefill loop | Dropped | — | T3 | opt-t5-wire-prefill.md | — | reverted with T3 |
| T6 Verify + measure | Dropped | — | T1–T5 | — | — | blocked by T3/T4 reverted; baseline 104.6s unchanged |

## R9700 Products Swarm Execution (2026-08-25)

### Shared work boundary

- Checkout: `<repo-root>`
- Branch: `feature/r9700-products-wave-a`
- Boundary kind: fallback linked worktree created from `main`; every executor/reviewer uses this checkout and branch.
- TinyGPU source, build, and task authority is the in-repository `tinygpu/` tree on this branch; upstream Tinygrad is read-only Port/Adapt provenance only, and no external TinyGPU checkout or branch is active.
- P1 executors may edit only `tinygpu/` after task set 1 review; all validation/build/install commands use that tree and emit binaries under `tinygpu/build/`.
- Agents never run git. The supervisor validates, reviews, updates ledgers/reports, and makes local checkpoint commits. Push and PR work remain user-owned unless separately requested.

### Orchestration map

- Wave A is closed as the foundation wave: F1 promoted; F2/P1/P3/Q1 completed the reachable source/contract work recorded below.
- Wave B0 starts five independent unblockers now: F2 task set 3A, P1 task sets 1A and 2A, Q1 task set 7, and P2 task set 1.
- Wave B1 consumes local freezes: F2 3B→4/5→6, P1 2B/3, and parallel P2 task sets 2/3A. Q1 provenance remains independent.
- Wave B2 starts F3 and P3 task set 5 as G0 consumers. Independently, P2 task sets 3B/4 consume accepted P1 extensions and do not wait for G0 to implement; P2 promotion still waits for P1 and G0.
- Wave C1 runs F4, P2 command/backend completion, and P3 promotion after their direct dependencies. Wave C2 is the serialized P4 convergence boundary.
- Wave D starts F5 after F4; F6 starts only after F4 plus Q1 task 7. P5 waits for P4 and a measured human-approved need.
- P1 task sets 1A/2A may research concurrently, but one P1↔P2 contract owner serializes edits to the P1/P2 packets and active validation ledger. One upstream-manifest owner serializes F2/P1/Q1 provenance changes after their disjoint reports are ready.
- Source/research lanes may run concurrently; every shared-file integration, DEXT install, and R9700 hardware command serializes through its named owner/lock.
- Verification and quality: executors record focused commands but run no project-wide/hardware/git work. Supervisor owns RED/GREEN, hardware evidence, broad suites, review gates, commits, and strict no-substitute promotion.

| Task | Status | Owner | Dependencies | Report | Evidence | Blocker |
|---|---|---|---|---|---|---|
| Shared baseline verification | Done | Supervisor | Task packets committed | `baseline-runtime-repair.md`, `baseline-raw-hip-repair.md` | Initial: 715 passed/26 failed. Focused repairs reviewed; final `tests/native_r9700 -v`: 744 passed, 2 dependency warnings in 662.69s (`artifact://276`). | — |
| Wave A / F1 persistent warm worker | Done | Supervisor integration | B0 | `f1-promotion.md`; F1 contract/integration/multi-PDB1 reports | One private child serves ten token-exact prompt-128 requests with accepted caches, zero warm reloads, and no fallback. Benchmark emits 10 identified raw rows plus 3 full native aggregate records with exact scoped counts and N-based throughput. Complete suite: 288 passed; final review: PASS. | — |
| Wave A / F2 gfx1201 WMMA foundation | Blocked | Supervisor / F2 admission | B0 | `f2-contract-freeze.md`, lane/admission reports | Task sets 1–2 complete. Fresh lane-map hardware/comparator evidence is exact on `1002:7551`/gfx1201; the offline layout/admission tool passes 34 tests. | Task set 3 real proof lacks pinned rocWMMA/AITER checkouts, a selected linear WMMA image, and bound ISA/resource/layout reports; G0/tasks 4–6 remain blocked. |
| Wave A / P1 TinyGPU Device Owner | Blocked | Supervisor / P1 source integration | B0, ADR 0007 | `p1-abi-freeze.md`; `p1-xcode-resume.md`; P1 RED/fix/review/integration reports; `p1-remaining-blockers.md` | Task set 1 Done. Reviewed task-set-2 safe source/package/common-client and task-set-3 bounded ownership plus real host-visible allocate/release/client-death source are checkpointed. Nine host contracts and both unsigned Xcode targets pass; task-set-3 controls: 62 passed; final focused reviews have zero Critical/Important findings. | Task set 2 lacks provenance-bound PSP/SOS/TMR transitions and signed install/hardware evidence. Task set 3's frozen 48-byte-plus-distinct-descriptor import is not representable by `IOConnectCall*`; device-local/private-VM PTE mapping waits for cold ownership. Tasks 4–6 remain downstream; task 6 also requires G0. |
| Wave A / P3 Kernel Packs | Blocked | P3 runtime/offline integration | B0; G0 for final migration | `p3-contract-freeze.md`; `p3-scalar-migration.md`; P3 RED/GREEN reports | Task sets 1–4 complete. All 13 scalar packs are allocation-free, evidence-sealed, selectable through existing admission, and pass the 27-test gate; both fresh R9700 RMSNorm traces exit 0. | Task set 5 and final P3 promotion require the blocked F2/G0 WMMA artifact. |
| Wave A / Q1 Qwen contract/oracle | Blocked | Q1Acceptance | B0 | Q1 identity/tensor/hybrid/oracle/shape/acceptance reports | Task sets 1–6 implemented. Pinned mlx-lm 0.32.0 / MLX 0.32.1 regeneration and model-bound parity pass; package gate passes 259 tests with oracle-only evidence. | Hard provenance blocker: `base_model_revision=unavailable_in_pinned_conversion_metadata` and applicable base license provenance. No native/performance claim. |
| Wave B0 / F2 task 3A source-image acquisition | Ready | Unassigned | F2 tasks 1–2 Done | `f2-source-image-selection.md` | Pinned revisions/ABI/lane map are frozen. | Acquire exact rocWMMA/AITER checkouts and select one candidate image; no hardware claim. |
| Wave B0 / P1 task 1A import ABI re-freeze | Ready | Unassigned | P1 stable task 1 Done | `p1-import-abi-amendment.md` | Infeasible current transport and security constraints are documented. | Select one public-API-representable import design and pass focused security/architecture review. |
| Wave B0 / P1 task 2A cold-firmware provenance | Ready | Unassigned | P1 stable task 1 Done | `p1-cold-firmware-provenance.md` | Cold stage/source blockers are isolated. | Bind exact firmware revisions, hashes, licenses, scope, and bundle/load policy. |
| Wave B0 / Q1 task 7 provenance closure | Ready | Unassigned | Q1 tasks 1–6 Done | `q1-provenance-closure.md` | Deterministic oracle package and 259-test gate pass. | Resolve immutable base revision and applicable base license; no native claim. |
| Wave B0 / P2 task 1 contract freeze | Ready | Unassigned | P1 stable ABI task 1 Done | `p2-contract-freeze.md` | Stable P1 operations are accepted; G0 blocks promotion only. | Freeze portable/stable/deferred operation matrices and exact commands. |
| Wave B1 / F2 task 3B admission | Blocked | Unassigned | F2 task 3A | `f2-admission-*` reports | Offline admission tool passes 34 tests. | Waiting for selected real source/image inputs. |
| Wave B1 / F2 tasks 4 and 5 | Blocked | Unassigned | F2 tasks 2 and 3B | WMMA source/numerical reports | Lane-map proof is accepted. | Task 4 starts after 3B; task 5 overlaps only after task-4 ABI freezes. |
| Wave B1 / F2 task 6 G0 | Blocked | Unassigned | F2 tasks 2–5 | `.superpowers/swarm/reports/g0-wmma-conformance.md` | — | Native benchmark/hardware publication serializes after all F2 gates. |
| Wave B1 / P1 task 2B cold ownership | Blocked | Unassigned | P1 task 2A | P1 cold reports | Reviewed cold source boundary exists. | Waiting for firmware provenance/bundle contract plus signing/hardware inputs. |
| Wave B1 / P1 task 3 mapping completion | Blocked | Unassigned | P1 task 1A and accepted 2B | P1 import/VM reports | Host-visible ownership source passes. | Waiting for import re-freeze and cold-owned device-local/private-VM mapping. |
| Wave B1 / P2 tasks 2 and 3A | Blocked | Unassigned | P2 task 1 | HAL contract reports | Task sets own disjoint portable versus AMD-stable-subset files. | Start concurrently immediately after task-set-1 freeze. |
| Wave B2 / F3 and P3 task 5 | Blocked | Unassigned | G0; F1 for F3 | F3/P3 reports | F1 and P3 foundations are ready. | Waiting for accepted G0; one F2→P3 catalog integration owner. |
| Wave B2 / P2 task 3B | Blocked | Unassigned | P2 task 3A; accepted P1 import/mapping extension | P2 AMD-backend reports | — | Does not wait for G0 to implement; promotion still waits for P1/G0. |
| Wave B2 / P2 task 4 | Blocked | Unassigned | P2 tasks 3A and 3B | P2 command/backend reports | — | Command/queue/fence work follows complete AMD memory/executable backend. |
| Wave C1 / F4, P2 task 5, P3 promotion | Blocked | Unassigned | F3; P2 tasks 2–4 plus P1/G0; P3 task 5 | phase promotion reports | — | Direct dependencies and serialized hardware gates remain. |
| Wave C2 / P4 service-platform convergence | Blocked | Unassigned | F1, accepted P2/P3, selected F2–F4 graph | P4 reports | — | Production cutover waits for accepted platform and graph contracts. |
| Wave D / F5, F6, P5 | Blocked | Unassigned | F4 for F5; F4 plus Q1 task 7 for F6; P4 plus decision for P5 | downstream phase reports | — | F5 starts after F4; F6 starts only after F4 and Q1 provenance closure. |

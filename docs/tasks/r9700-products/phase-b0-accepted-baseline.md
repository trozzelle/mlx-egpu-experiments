# Phase B0: Accepted native producer and cache-serving baseline

## Source grounding

- `docs/ROADMAP.md` §Shared baseline B0.
- `docs/IMPLEMENTATION_PLAN.md` §Authority and starting state.
- `docs/DESIGN.md` §Product correctness gates and §Numerical acceptance contract.
- `.superpowers/swarm/progress.md` §Current facts.
- `docs/adr/0004-macos-substrate-selection.md` and ADR 0005 resolution evidence.
- `docs/archive/tasks/native-r9700-producer/validation-commands-c0-c3.md` — historical exact commands.
- Archived C0–C2 task/evidence packets indexed by `docs/archive/README.md`.

## Goal

Preserve the accepted B0 native Llama producer and imported-cache serving behavior as an immutable regression baseline. This phase is complete; it authorizes later work to optimize or productize the path, not to rerun or reinterpret its acceptance.

## Dependencies

- None. B0 is the dependency root for F1, F2, P1, P3, and Q1.
- Later phases consume the same Llama model identity, `S-1` cache semantics, producer-kind labels, hardware evidence rules, and C1R/C2R corpus.

## Orchestration map

- Sequential blockers: none; all B0 task sets are completed evidence.
- Parallelizable task sets: none; no implementation work remains.
- Shared contracts/artifacts: C0 kernel/transfer logs, C1R prompt-0/16/64/128 tokens and per-layer deltas, C2R accepted-cache/no-fallback results, scalar/native controls, CPU oracle, prompt-cache schema.
- Coordination risks: future agents must not edit archived packets or replace B0 evidence with a new optimized-kernel tolerance policy.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. C0 native device/runtime proof | Done | Historical C0 supervisor | R9700 identity, kernel launch, transfer, and resident-VRAM evidence accepted. |
| 2. C1R native Llama producer parity | Done | Historical C1R supervisor | Token-exact at prompt lengths 0, 16, 64, and 128; all 16 layers finite and ULP-level versus CPU reference. |
| 3. C2R imported-cache serving | Done | Historical C2R supervisor | Actual hardware producer route, accepted cache, no fallback, token-exact decode. |
| 4. Baseline ownership and archive freeze | Done | Documentation supervisor | Current authority points to B0; completed packets and exact commands archived. |

Agents must not reopen or reset these rows. New regressions create a defect task in the affected current phase.

## Task set 1: Preserve C0 native device/runtime proof

### Source refs

- `docs/ROADMAP.md` §Shared baseline B0 accepted capabilities, bullets 2 and 6.
- `.superpowers/swarm/progress.md` §Current facts, C0 and root-cause evidence.
- ADR 0004 §Decision.

### Target

- Evidence only: current progress ledger, historical logs/reports, runtime proof contracts.
- Non-goals: rerun hardware, change TinyGPU/AMDev, or claim P1/P2 completion.

### Change

No implementation change. Preserve the accepted device identity (`1002:7551`, `gfx1201`), kernel launch, H2D/D2H comparison, resident-VRAM proof, and failure-field schema as downstream regression requirements.

### Acceptance

B0 references continue to bind native claims to R9700 hardware evidence and do not imply cold-init, HAL, or persistent-service completion.

### Validation

Historical commands are preserved under `docs/archive/tasks/native-r9700-producer/validation-commands-c0-c3.md` §Native lower-BAR VRAM mapping smoke and §2026-08-22 GC compute recovery proof. No command is rerun for this completed row.

## Task set 2: Preserve C1R native Llama producer parity

### Source refs

- `docs/ROADMAP.md` §Shared baseline B0 accepted capabilities, bullets 3–4.
- `.superpowers/swarm/progress.md` §Current facts, C1R evidence.
- ADR 0005 §Resolution evidence.

### Target

- `tests/native_r9700/test_parity.py`
- `tests/native_r9700/test_native_hsa_prefill_contract.py`
- `tests/native_r9700/test_native_worker_evidence.py`
- committed Llama fixtures and accepted token corpus.
- Non-goals: alter tolerances for F2/F3 or regenerate fixtures without a current phase task.

### Change

No implementation change. Preserve exact final tokens, finite K/V, model identity, `producer_kind=r9700_native`, request-bound hardware log, and `S-1` prefix length as downstream gates.

### Acceptance

Every later Llama graph/kernel phase cites and runs the C1R corpus before promotion; CPU/NumPy output remains oracle-only.

### Validation

Historical native prefill command:

```sh
APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock \
  build/native-r9700-runtime/native_r9700_runner --native-prefill-proof \
    --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct \
    --token-ids-json '[128000,128001]' \
    --out logs/full-native-prefill.npz \
    --log logs/full-native-prefill.log
```

This command is historical evidence, not a requested rerun.

## Task set 3: Preserve C2R imported-cache serving

### Source refs

- `docs/ROADMAP.md` §Shared baseline B0 accepted capabilities, bullet 5.
- `.superpowers/swarm/progress.md` §Current facts, C2R evidence.
- `docs/DESIGN.md` §mlx-lm prompt-cache adapter and §Prefill request lifecycle.

### Target

- `native_r9700/serving.py`
- `tests/native_r9700/test_serving.py`
- `native_r9700/kv_cache.py`
- `tests/native_r9700/test_kv_cache.py`
- Non-goals: introduce persistent-service, direct-transport, or HAL behavior in B0.

### Change

No implementation change. Preserve actual R9700 producer routing, cache validation before acceptance, fallback only before acceptance, final-token injection, and terminal post-acceptance failure.

### Acceptance

Later service/adapter phases cannot weaken cache acceptance or silently retry the full prefix.

### Validation

Historical focused contract:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_serving.py -v
```

Historical integration command is preserved in the archived validation ledger §C2 mlx-lm serving wrapper contract. No command is rerun for this completed row.

## Task set 4: Preserve baseline ownership and archive freeze

### Source refs

- `docs/tasks/r9700-products/README.md` §Authority and §Shared contracts and artifacts.
- `docs/archive/README.md` §Use policy.

### Target

- `docs/ROADMAP.md`
- `.superpowers/swarm/progress.md`
- `docs/archive/`
- this B0 document.
- Non-goals: edit historical claims or create compatibility aliases at old task paths.

### Change

Keep B0 marked complete; keep archived packets historical; route new execution through F/P/Q task docs. If a B0 regression appears, record it as a blocker in the consuming phase and preserve the original evidence.

### Acceptance

No current task doc instructs an agent to execute archived C0–C3 packets; B0 remains the named regression root.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-b0-accepted-baseline.md \
  docs/tasks/r9700-products/README.md .superpowers/swarm/progress.md
```

## Phase validation

B0 is complete when its evidence pointers resolve and every current phase treats it as a regression dependency rather than unfinished implementation. No hardware or test command is required to close this already-completed phase.

## Handoff notes

- F1 consumes accepted producer/serving behavior.
- F2 consumes scalar/native controls and code-image admission.
- P1 consumes accepted TinyGPU/AMDev behavior without claiming cold lifecycle.
- P3 consumes existing asset/catalog validation.
- Q1 consumes the native-evidence labeling rules but not Llama cache geometry.

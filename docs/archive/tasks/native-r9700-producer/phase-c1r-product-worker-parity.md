# Phase C1R-W3: Full prefill, C1R parity, and C2R handoff

## Source grounding
- `docs/archive/superpowers/plans/2026-08-21-native-r9700-product-worker-rearchitecture.md`, Wave 3 Tasks 9–11.
- `docs/archive/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md:32-52`: C1R/C2R acceptance definitions.
- `docs/DESIGN.md:47-64, 82-109, 148-168`: KV interchange, native producer, and consumer seams.
- `docs/adr/0002-producer-owns-kv-truth.md`: consumer never recomputes accepted prefix.
- `docs/adr/0005-cpu-reference-is-not-native-r9700-producer.md`: exact native evidence requirements.

## Goal
Extend real resident native execution through all 16 Llama layers, emit the unchanged NPZ/prompt-cache artifacts, pass the C1R token-exact gate, then hand the accepted producer to C2R serving without post-acceptance recomputation.

## Dependencies
- C1R-W2 layer-0 resident-dataflow evidence is Done and reviewed.
- Hardware access remains verified by the selected-substrate transfer and layer-0 logs.

## Orchestration map
- **Sequential blockers:** task set 1 → task set 2 → task set 3. No parallel implementation here: artifact schema, parity evidence, and serving routing are causally dependent.
- **Parallelizable review lanes:** after each task set freezes its candidate, dispatch code review and evidence review in parallel; after task set 3, dispatch code review and security/transport review in parallel.
- **Shared contracts/artifacts:** 16-layer fp16 NPZ schema, `producer_kind=r9700_native`, `S-1` prompt-cache rule, Phase-0 prompt fixtures, local hardware logs.
- **Coordination risks:** no task changes KV schema; no task labels CPU output as native; no task runs a hardware proof concurrently with another worker or modifies report/ledger before supervisor accepts evidence.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Full native NPZ worker | Not started | Native-worker owner | First possible `pass` source. |
| 2. C1R parity package | Not started | Parity owner | Opens only after task set 1. |
| 3. C2R serving handoff | Not started | Serving owner | Opens only after C1R review pass. |

## Task set 1: Full native prefill and atomic NPZ output

### Source refs
- Rearchitecture plan, Task 9.
- Recovery plan `:166-178`: all-layer prefill, explicit native route, hard geometry failures.
- `native_r9700/native_worker.py:142-199`: NPZ acceptance validator.

### Target
- Create `native_r9700/native_prefill_worker.h/.cpp`
- Modify `native_r9700/llama_layer_executor.cpp`, `native_r9700/runtime.cpp`
- Modify `native_r9700/native_worker.py`, `native_r9700/prefill.py`
- Modify `tests/native_r9700/test_native_worker_evidence.py`, `tests/native_r9700/test_prefill.py`

Non-goals: prompt-cache schema change, parity report change, serving routing, C3 backend work.

### Change
1. Add RED tests rejecting 15 layers, wrong scalar metadata, wrong K/V shape, non-fp16 K/V, CPU identity, missing hardware log, failed worker exit, and partial NPZ left behind after failure.
2. Implement `run_native_prefill` using the C1R-W2 layer executor through layers 0–15. GPU-produced hidden state flows directly layer to layer. If weight residency is bounded, stream through `DeviceMemory` with recorded chunk byte counts; CPU never supplies the next layer’s tensor math.
3. Validate all K/V arrays and metadata in memory, then write the NPZ to a sibling temporary path and atomically replace the requested path only after complete success.
4. Keep Python’s current validation authoritative for the handoff: rejection deletes any output; `cpu_reference` behavior remains unchanged.
5. Run prompt-0 first, convert with existing `native_r9700.kv_cache`, and retain hardware evidence. The worker may report `pass` only if all full-result requirements hold.

### Acceptance
A prompt-0 `r9700_native` run produces an atomic valid 16-layer fp16 K/V NPZ with selected-hardware evidence and converts through the existing cache emitter.

### Validation
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_native_worker_evidence.py tests/native_r9700/test_prefill.py -q
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.kv_cache --prefill-npz <observed-native-prefill.npz> --out <observed-native-prompt-cache.safetensors> --log <observed-kv-cache.log>
```

The second command is run only with observed paths emitted by the supervisor-owned native prompt-0 run.

## Task set 2: C1R token-exact parity and review package

### Source refs
- Rearchitecture plan, Task 10.
- Recovery plan `:180-191`: C1R exact parity and review requirements.
- `docs/DESIGN.md:172-180`: `P == R` is the load-bearing gate.

### Target
- Modify `native_r9700/parity.py`, `tests/native_r9700/test_parity.py`
- Modify only after observed success: `docs/tasks/native-r9700-producer/validation-commands.md`, `docs/path-a-validation-results.md`, `.superpowers/swarm/progress.md`

Non-goals: tolerate token mismatch, run C2R early, alter accepted prompt suite.

### Change
1. Add RED report tests rejecting missing hardware log/cache path, CPU identity for a native run, and any `P != R` token mismatch. Cover `S-1` cached prefix plus final prompt-token injection.
2. Run prompt-0 native parity first. Diagnose only observed RoPE/position, K/V layout, layer order, geometry, or precision deltas; preserve token exactness.
3. Run prompt-1 and prompt-2 independently after prompt-0 passes. Keep run logs isolated by prompt.
4. Aggregate evidence only after all three runs pass. Add exact observed commands and results to the validation ledger, Path C report, and swarm ledger. Do not prewrite a passing status.
5. Submit one frozen review package for geometry/RoPE/KV/atomic-output/hardware-evidence inspection. Critical or Important findings block C2R.

### Acceptance
All Phase-0 prompts have `producer_kind=r9700_native`, `P == R`, valid cache and hardware log paths, and a reviewer-approved C1R report.

### Validation
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_parity.py -q
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

The supervisor runs the exact native parity commands discovered by this task only after focused tests and review pass.

## Task set 3: C2R serving route and final review handoff

### Source refs
- Rearchitecture plan, Task 11.
- Recovery plan `:195-232`: native serving identity, fallback, security, and review requirements.
- ADR 0002: no recomputation after cache acceptance.

### Target
- Modify `native_r9700/serving.py`, `tests/native_r9700/test_serving.py`
- Modify only if a pass-through argument is necessary: `tinygrad_kv_worker/harness.py`, `tests/test_harness_c2_serving.py`
- Modify only after observed success: validation ledger, Path C report, swarm ledger

Non-goals: TCP/network transport, duplicate cache-emitter logic in harness, C3 implementation.

### Change
1. Add RED serving tests for unavailable/malformed/no-evidence/CPU-labelled native worker results. Before cache acceptance they must fail or native-mlx fallback with a redacted reason; after acceptance, decode failure must surface without prefix recomputation.
2. Route accepted large requests through only `prefill.py --producer-kind r9700_native`, existing `kv_cache.py`, cache validation/import, and mlx-lm final-token decode.
3. Run direct serving for prompt-0 and one larger Phase-0 prompt. Run below-threshold, unavailable-worker, and malformed-output fallback cases.
4. Have the harness delegate to the serving route only; no duplicate validation or conversion.
5. After observed evidence, submit code review and security/transport review in parallel. Update durable evidence only after both approve.

### Acceptance
Large prompts use the accepted native producer and remain token-exact. Pre-acceptance fallback works; accepted-cache decode never recomputes prefill; logs preserve local-only/redacted behavior.

### Validation
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py tests/test_harness_c2_serving.py -q
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
git diff --check
```

## Phase validation
The supervisor verifies C1R first: all prompt parity reports, hardware logs, and reviewer decision. Only then verifies direct C2R serving, fallback scenarios, harness delegation, focused security review, broad native suite, and whitespace check.

## Handoff notes
A passing C2R package is evidence for a later C3 decision only. It does not authorize direct native-consumer implementation or a Qwen expansion; those require their own decision phase.
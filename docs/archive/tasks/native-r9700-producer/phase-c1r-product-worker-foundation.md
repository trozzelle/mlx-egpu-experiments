# Phase C1R-W1: Parallel runtime and oracle foundation

## Source grounding
- `docs/archive/superpowers/plans/2026-08-21-native-r9700-product-worker-rearchitecture.md`, Wave 1 Tasks 3–6.
- `docs/archive/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md:121-150`: reusable memory/transfer manager and native primitive requirements.
- `docs/adr/0004-macos-substrate-selection.md:7-18`: selected macOS TinyGPU/AMDev substrate and frozen 24-byte kernarg facts.
- `.superpowers/swarm/progress.md:434-463`: broad proof coverage exists but remains partial/proof-only.

## Goal
Create reusable, source-grounded AMDev and oracle modules with small independent test surfaces. This phase prepares the native worker; it does not execute or accept model-forward prefill.

## Dependencies
- C1R-W0 must be Done.
- All packets consume the fixed substrate (`1002:7551`, `gfx1201`) and current CPU fixture data.

## Orchestration map
- **Sequential blockers:** C1R-W0.
- **Parallelizable task sets:** 1–4 are four independent implementation lanes. Dispatch them in a single wave.
- **Shared contracts/artifacts:** Frozen 24-byte kernarg layout; C0 packet encodings; committed fixture archives; selected hardware identity.
- **Coordination risks:** task set 1 alone may modify `runtime.cpp`; task set 2 alone owns new AMDev/device-memory files and `c1_transfer_bridge.cpp`; task set 3 alone owns fixture files; task set 4 alone owns runtime-test file splitting. No worker changes another packet’s files.
- **Review:** dispatch four read-only reviews in parallel after each implementation lane completes. The supervisor resolves nothing by cherry-picking overlapping edits; overlap is a task failure.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Pure AMDev packets | Not started | Packet worker | `runtime.cpp` extraction owner. |
| 2. Session and device memory | Not started | Memory worker | Owns AMDev mechanics extraction. |
| 3. Fixture catalog | Not started | Oracle worker | Owns fixture metadata/generation. |
| 4. Runtime test seams | Not started | Test worker | Moves tests; removes archive parsing. |

## Task set 1: Pure AMDev packet extraction

### Source refs
- Rearchitecture plan, Task 3.
- `native_r9700/runtime.h:6249-6259`: current public SDMA/PM4 word builders.

### Target
- Create `native_r9700/amdev_packets.h`, `native_r9700/amdev_packets.cpp`
- Modify `native_r9700/runtime.cpp`
- Create `tests/native_r9700/test_amdev_packets.py`

Non-goals: TinyGPU socket/BAR access, buffer allocation, kernel catalog, model execution.

### Change
1. Port existing no-hardware SDMA opcode/length/fence and PM4 59-dword assertions into dedicated packet tests.
2. Move the exact `build_sdma_copy_words` and `build_pm4_dispatch_words` implementations into `amdev_packets.cpp` without changing emitted words.
3. Include `amdev_packets.h` from callers; retain no duplicate encoder body.

### Acceptance
Pure packet tests prove byte-identical legacy packet values. `runtime.cpp` no longer implements the pure encoders.

### Validation
```sh
${PY} -m pytest tests/native_r9700/test_amdev_packets.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra -c native_r9700/amdev_packets.cpp -I native_r9700 -o build/native-r9700-runtime/amdev_packets.o
```

## Task set 2: AMDev session and bounded device memory

### Source refs
- Rearchitecture plan, Task 4.
- `native_r9700/c1_transfer_bridge.cpp:109-263`: compact transfer-adapter precedent.
- C0B transfer command and required markers in `validation-commands.md:134-142`.

### Target
- Create `native_r9700/amdev_session.h/.cpp`, `native_r9700/device_memory.h/.cpp`
- Create `tests/native_r9700/test_device_memory_contract.py`
- Modify `native_r9700/c1_transfer_bridge.cpp`

Non-goals: kernel execution semantics, Llama model loading, native prefill CLI.

### Change
1. Define `DeviceBuffer { gpu_va, size_bytes, name }` and `DeviceMemory.allocate/upload/download/release_all` as in the source plan.
2. Extract one source-grounded TinyGPU connection/BAR/VM implementation into `amdev_session`; do not copy C0 setup logic into more than one active module.
3. Validate buffer names, sizes, upload/download ranges, transfer accounting, and release state before touching hardware.
4. Make the transfer bridge use these modules or delete it only after equivalent runtime transfer behavior passes.

### Acceptance
No-hardware validation rejects invalid ownership/range transitions; C0B transfer proof still reports exact round-trip and selected hardware identity.

### Validation
```sh
${PY} -m pytest tests/native_r9700/test_device_memory_contract.py -q
${PY} -m pytest tests/test_native_amdev_transfer_contract.py -v
```

The supervisor, not the task worker, runs the existing C0B hardware command from `validation-commands.md` after review.

## Task set 3: Data-driven fixture catalog

### Source refs
- Rearchitecture plan, Task 5.
- `native_r9700/ref_fixtures.py:69-166`: frozen geometry and repeated fixture dimensions.
- Recovery plan `:109-119`: fixture metadata must record names, shapes, dtypes, tolerances, and digests.

### Target
- Create `native_r9700/fixture_catalog.py`, `tests/native_r9700/test_fixture_catalog.py`
- Modify `native_r9700/ref_fixtures.py`, `tests/native_r9700/test_ref_fixtures.py`

Non-goals: no GPU execution; do not regenerate committed fixture bytes unless an oracle bug is independently demonstrated.

### Change
1. Define frozen `FixtureSpec(name, archive_name, arrays, shape, dtype, tolerance, sha256)` entries plus `fixture_specs()` and `fixture_spec(name)`.
2. Replace head/band-specific metadata wrapper functions with a parameterized fixture generator driven by catalog entries.
3. Test every declared fixture’s archive, arrays, dtype, geometry, and digest.

### Acceptance
The catalog is the one source of fixture metadata. Tests no longer need the retired bridge to validate operand bytes.

### Validation
```sh
${PY} -m pytest tests/native_r9700/test_fixture_catalog.py tests/native_r9700/test_ref_fixtures.py -q
```

## Task set 4: Focus runtime tests on behavior

### Source refs
- Rearchitecture plan, Task 6.
- `tests/native_r9700/test_runtime_contract.py:1-20`: current mixed lifecycle/protocol/worker test purpose.

### Target
- Create `tests/native_r9700/test_runtime_lifecycle.py`, `tests/native_r9700/test_runtime_protocol.py`, `tests/native_r9700/test_native_worker_evidence.py`
- Modify `tests/native_r9700/test_runtime_contract.py`

Non-goals: do not loosen evidence requirements, change runtime behavior, or preserve archive-array assertions under another name.

### Change
1. Move lifecycle/kernarg/packet assertions to `test_runtime_lifecycle.py`.
2. Move fake executable, marker, and command protocol assertions to `test_runtime_protocol.py`.
3. Move Python worker NPZ/evidence acceptance to `test_native_worker_evidence.py`.
4. Remove `_archived_bridge_source_text_or_skip` and every direct archive parse.

### Acceptance
All runtime tests execute without opening archived C++ and still reject missing/incorrect evidence.

### Validation
```sh
${PY} -m pytest tests/native_r9700/test_runtime_lifecycle.py tests/native_r9700/test_runtime_protocol.py tests/native_r9700/test_native_worker_evidence.py -q
```

## Phase validation
The supervisor runs all four focused Python commands, compiles `amdev_packets.cpp`, then performs one reviewed C0B hardware transfer proof. Hardware evidence is written to the configured ignored run root; no C1R acceptance status changes.

## Handoff notes
Only after the supervisor confirms the four lanes share no duplicate AMDev mechanism and all reviews pass may C1R-W2 begin.
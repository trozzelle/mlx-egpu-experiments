# Phase C1R-W2: Native worker integration and layer-0 evidence

## Source grounding
- `docs/superpowers/plans/2026-08-21-native-r9700-product-worker-rearchitecture.md`, Wave 2 Tasks 7–8.
- `docs/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md:152-164`: layer-0 must compose hardware primitives in model order and compare K/V/hidden state to the CPU oracle.
- `docs/DESIGN.md:82-105`: native producer inputs, output, and loud failure requirements.

## Goal
Integrate the extracted AMDev/oracle modules into one thin runner-to-worker path and prove one resident layer-0 real-model vertical slice. The result remains `native_prefill_acceptance: open`.

## Dependencies
- C1R-W0 and all C1R-W1 task sets are Done and reviewed.
- The supervisor has a passing selected-substrate C0B transfer proof after the extraction.

## Orchestration map
- **Sequential blocker:** C1R-W1 integration review.
- **Parallelizable task sets:** none. Task set 2 depends on Task set 1’s C++ request/result seam and source list.
- **Shared contracts/artifacts:** `NativePrefillRequest`, `NativePrefillResult`, `DeviceMemory`, `KernelDescriptor`, fixture catalog, Llama model directory.
- **Coordination risks:** one integration owner edits `runtime.h`, `runtime.cpp`, and `runner.cpp`. Parallel implementation here would fracture the C++ contract. A reviewer and a hardware-evidence reviewer may assess the result in parallel after the owner stops editing.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Thin runner/worker seam | Not started | Integration owner | Reduces public runtime surface. |
| 2. Kernel catalog and resident layer-0 | Not started | Same integration owner | Opens only after task set 1 review. |

## Task set 1: Integrate the thin runner/worker seam

### Source refs
- Rearchitecture plan, Task 7.
- `native_r9700/native_worker.py:87-117`: Python-side native evidence consumer.
- `native_r9700/runtime.h:6182-6355`: current public runtime contract and overgrown proof APIs.

### Target
- Modify `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, `native_r9700/runner.cpp`
- Create `native_r9700/runtime_contract.cpp`
- Modify `tests/native_r9700/test_runtime_lifecycle.py`, `tests/native_r9700/test_runtime_protocol.py`

Non-goals: full layer loop, KV cache serialization, consumer routing, native acceptance.

### Change
1. Add a RED runner test for `--native-prefill-proof --model missing --token-ids-json '[1]' --out <path> --log <path>` requiring nonzero exit, `producer_kind: r9700_native`, `native_prefill_acceptance: open`, explicit failure stage, and no NPZ.
2. Define `NativePrefillRequest`, `NativePrefillResult`, and `run_native_prefill` exactly as specified in the rearchitecture plan.
3. Retain only stable lifecycle/log/Kernargs declarations in `runtime.h`; move packet declarations to `amdev_packets.h`; remove primitive-chain metadata and proof-only public APIs.
4. Make `runner.cpp` parse only model, token JSON, NPZ output, and log paths for the native prefill command. Reject empty, negative, malformed, or non-integer tokens before device connection.
5. Emit JSON and `key: value` logs on all error/success paths. Do not create an NPZ before full native worker validation.

### Acceptance
The active runner has one small native-prefill command contract. Missing model/input fails loudly, produces no artifact, and cannot be confused with C1R acceptance.

### Validation
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_lifecycle.py tests/native_r9700/test_runtime_protocol.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

## Task set 2: Add compact kernel catalog and resident layer-0 executor

### Source refs
- Rearchitecture plan, Task 8.
- Recovery plan `:133-150`: required Llama primitive classes.
- Recovery plan `:154-164`: full layer-0 output requirements.

### Target
- Create `native_r9700/kernel_catalog.h/.cpp`, `native_r9700/llama_layer_executor.h/.cpp`
- Create `tests/native_r9700/test_kernel_catalog.py`, `tests/native_r9700/test_layer0_executor_contract.py`
- Modify `native_r9700/runtime.cpp`

Non-goals: embedded fixture bytes, expected output arrays in C++, prompt-cache serialization, 16-layer acceptance.

### Change
1. Add RED tests rejecting duplicate/unknown kernel names, zero workgroups, malformed digest, missing model weights, and an executor evidence record that claims fixture-sourced intermediate inputs.
2. Define compact `KernelDescriptor` data and `find_kernel(name)` exactly as the rearchitecture plan specifies. Descriptor content is name, digest, launch geometry, and kernarg bytes only.
3. Source executable kernel material from named reviewed assets or compiler output; record digest. Do not embed model weights, fixture arrays, expected outputs, or generated byte arrays in C++.
4. Implement prompt-0 layer-0 execution with real tokens and model weights. Allocate device buffers once, preserve GPU intermediates through attention and MLP stages, then read back K/V and post-layer hidden for CPU-oracle comparison.
5. Record kernel count, transfer bytes, K/V shape, hidden shape, model/fixture identity, hardware identity, and a specific failure stage. Set `native_prefill_acceptance: open` unconditionally for this layer-only mode.
6. The implementer discovers the exact hardware invocation from the final runner mode, runs it only after reviewer approval, and appends it to `validation-commands.md` only after a successful observed log.

### Acceptance
A layer-0 log demonstrates real token/model input, GPU-resident intermediate dataflow, selected R9700 identity, K/V and hidden comparison, and `exit_status: 0`; it does not claim full native prefill acceptance.

### Validation
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kernel_catalog.py tests/native_r9700/test_layer0_executor_contract.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/kernel_catalog.cpp native_r9700/llama_layer_executor.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

## Phase validation
1. Run both focused test commands.
2. Compile the explicit full source list.
3. Have separate code and hardware-evidence reviewers inspect the same immutable candidate.
4. Run exactly one supervisor-owned hardware layer-0 proof after review. Store its log under the configured ignored run root.

## Handoff notes
C1R-W3 is blocked until layer-0 evidence proves no fixture intermediate is fed into accepted computation. The phase status remains `open`, not native accepted.
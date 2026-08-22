# Phase 4: Kernel Image, Direct PM4 Dispatch, Timeline, and Readback

## Source grounding
- Source plan read: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 1-23, 47-61, 604-779.
- Existing blocker report read: `.superpowers/swarm/reports/c0a-task-4-kernel-proof.md` lines 31-39 and 55-68.
- Exact hardware command source read: `docs/tasks/native-r9700-producer/validation-commands.md` lines 144-152.

## Goal
Convert a passing compute ring into an actual tinygrad-free gfx1201 kernel proof: load the reviewed fixed kernel text, write kernargs, submit separated SDMA H2D/D2H copies, emit direct PM4 dispatch packets, poll a compute timeline, read back output, and compare exact CPU bytes.

## Dependencies
- Phase 3 complete: compute ring setup and HQD activation pass, including the chosen write-pointer unit for doorbell submission.
- Existing kernel metadata constants in `native_amdev_transfer_probe.cpp` remain source of truth for mode `minimal-u32-add-one`, arch `gfx1201`, expected output `[2..9]`, RSRC fields, kernarg size, and reference hashes.
- This phase does not unblock C0A-6 unless both hardware runs pass and final review in Phase 5 accepts the result.

## Orchestration map
- Sequential blockers: Task set 1 must preserve SDMA transfer behavior; Task set 2 must produce reviewed kernel text/kernargs and VM mappings; Task set 3 submits PM4/timeline/readback; Task set 4 records repeated-run evidence and review context.
- Parallelizable task sets: after Phase 3, Task set 1 and Task set 2 may be developed in parallel if agents coordinate on `am_compute` VM mappings and do not both edit the same helper bodies. Task set 3 depends on both.
- Shared contracts/artifacts: `submit_sdma_copy`, `load_kernel_blob`, `write_kernel_kernargs`, `build_compute_dispatch_words`, `submit_compute_dispatch`, `poll_compute_timeline`, `kernel_blob_load_status`, `kernarg_write_status`, `sdma_h2d_status`, `sdma_d2h_status`, `kernel_launch_status`, `cpu_comparison_status`, `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`.
- Coordination risks: kernel byte provenance is a source gap in the plan; resolve it before embedding bytes. Do not claim `kernel_blob_load_status: pass` from metadata only; bytes must be written and read back from BAR0.

## Progress ledger
| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Single-copy SDMA primitive | Not started | TBD | Factors SDMA copy while preserving `--transfer-proof` behavior byte-for-byte. |
| 2. Kernel text provenance, kernargs, and VM mappings | Not started | TBD | Must resolve durable 512-byte text source before embedding/loading. |
| 3. Direct PM4 dispatch and readback compare | Not started | TBD | Consumes Phase 3 write-pointer unit and Task set 2 loaded code/kernargs. |
| 4. Repeated-run report and review packet | Not started | TBD | Records two hardware runs or precise blocker with logs. |

Agents update only their row and append evidence/notes as work completes.

## Task set 1: Single-copy SDMA primitive

### Source refs
- Plan Task 5 interfaces and Step 1: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 604-630.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Read existing SDMA submission helpers and `--transfer-proof` path.
- Non-goals: no kernel text embedding, no kernargs, no PM4 dispatch, no transfer-proof behavior drift.

### Change
1. Factor current `submit_sdma_transfer` into `submit_sdma_copy(const RemoteClient&, DiscoveryLog*, SysmemMapping*, uint64_t src_va, uint64_t dst_va, uint32_t byte_count, uint32_t fence_value, uint64_t submit_byte_offset, std::string*)`.
2. Use the primitive for H2D input: staging sysmem to `am_compute::kInputVramVa`.
3. Use the primitive for D2H output: `am_compute::kOutputVramVa` to readback sysmem.
4. Preserve existing `--transfer-proof` byte-for-byte by wrapping copy calls around the same fixed transfer as today.

### Acceptance
- Existing SDMA transfer proof semantics remain unchanged.
- Kernel proof can record distinct `sdma_h2d_status` and `sdma_d2h_status` without conflating them with the existing transfer proof.

### Validation
Executor records these exact supervisor commands; executor does not run them in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

Supervisor should also run the exact C0B SDMA transfer proof if this task changes transfer behavior:

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0b-native-amdev-sdma-transfer.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --transfer-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Task set 2: Kernel text provenance, kernargs, and VM mappings

### Source refs
- Plan Task 5 Steps 2-4: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 631-678.
- Source gap: the plan says "continue exact bytes from local c0a notes" for the 512-byte `.text` but does not name a durable checked-in file containing those bytes. This task must resolve that provenance before code embedding/loading.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Modify: `tests/test_native_amdev_transfer_contract.py` if self-test expectations need exact byte-count/hash/prologue/epilogue coverage.
- Create or update report section in `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`.
- Non-goals: no PM4 dispatch submission, no final pass claim, no runtime tinygrad dependency, no generated artifact required at runtime.

### Change
1. Locate a durable source for the reviewed 512-byte kernel `.text` bytes or regenerate and record provenance in the report. Acceptable provenance must include file path or generation command, byte count 512, SHA-256 `kKernelReferenceTextSha256`, and source comments. Do not proceed with anonymous bytes.
2. Add compile-time `kKernelText` or an equivalent tinygrad-free embedded artifact with source comments.
3. Validate byte count equals 512 and either SHA-256 equals `kKernelReferenceTextSha256` or a deterministic self-test checks first 64 bytes, last 16 bytes, full byte count, and logged text SHA.
4. Implement `load_kernel_blob(const RemoteClient&, DiscoveryLog*, std::string*)`; write bytes to code VRAM and read them back through BAR0 before `kernel_blob_load_status: pass`.
5. Implement `write_kernel_kernargs(SysmemMapping*, uint64_t output_va, uint64_t input_va, uint64_t scalar_va, std::string*)` with three 64-bit pointers in order: output, input, scalar/addend pointer. Write scalar value `1` at `kKernargsVa + 24` or mapped compute-control storage behind that VA.
6. Extend `write_fixed_page_tables` to map output/code/kernargs/ring/rptr/wptr/timeline/eop VAs exactly as specified in plan lines 664-674.

### Acceptance
- Report cites durable kernel text provenance and exact validation evidence.
- `kernel_blob_load_status: pass` is reachable only after write/readback of code bytes.
- `kernarg_write_status: pass` is reachable only after exact pointer/scalar layout is written.
- Page-table self-test lists every new mapped VA and PTB index.

### Validation
Executor records these exact supervisor commands; executor does not run them in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

Expected after this task set if PM4 dispatch is not implemented yet: code/kernargs/H2D pass, compute ring setup pass, and blocker moves to `kernel_dispatch_submit`; earlier failures classify as `kernel_blob_load`, `kernarg_write`, `sdma_h2d_submit`, or `vm_mapping`.

## Task set 3: Direct PM4 dispatch and readback compare

### Source refs
- Plan Task 6 Steps 1-4: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 682-759.
- Direct PM4 source facts: same plan lines 55-57 and 60.

### Target
- Modify: `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`.
- Modify: `tests/test_native_amdev_transfer_contract.py` if packet-order self-test needs updates.
- Non-goals: no AQL, no multi-XCC workaround, no production queue scheduler, no broad runtime API.

### Change
1. Implement `build_compute_dispatch_words(uint64_t code_va, uint64_t kernargs_va, uint64_t timeline_va)` with packet order from plan lines 701-713: acquire_mem, set SH PGM, RSRC1/2/3, TMPRING, restart, user data, resource limits, start/thread dims, `PACKET3_DISPATCH_DIRECT`, CS partial flush event, release_mem timeline write.
2. Cite every PM4 packet header constant from local `pm4_soc15.py` or `pm4_nv.py` source.
3. Implement `submit_compute_dispatch(const RemoteClient&, DiscoveryLog*, SysmemMapping*, const std::vector<uint32_t>&, std::string*)`: write words into compute ring, write wptr in the unit chosen by Phase 3, issue seq_cst fence, write BAR2 doorbell qword at `am_compute::kMecDoorbellBar2ByteOffset`.
4. Implement `poll_compute_timeline(const SysmemMapping&, std::string*)` with bounded 3-second wait. On timeout set `failure_stage: kernel_timeline_timeout` and include observed value.
5. After compute timeline pass, submit SDMA D2H copy, poll SDMA fence, compare readback bytes to `kKernelExpectedOutputBytesHex`, and on mismatch set `failure_stage: readback_mismatch` with expected/observed hex.
6. On pass print `kernel_launch_status: pass`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`, `failure_stage: none`, `failure_text: none`, and `exit_status: 0`.

### Acceptance
- PM4 sequence self-test reports expected packet order, sizes, dispatch initiator, and timeline value.
- Hardware pass is based on compute timeline plus D2H CPU byte comparison, not on dispatch submission alone.
- Failure stages are precise: `kernel_dispatch_submit`, `kernel_timeline_timeout`, or `readback_mismatch`.

### Validation
Executor records these exact supervisor commands; executor does not run them in OMP task mode:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_native_amdev_transfer_contract.py -v
```

```sh
/bin/bash -o pipefail -c 'mkdir -p build/native-r9700-runtime logs; log=logs/c0-macos-egpu-minimal-runtime.log; { printf "%s\n" "command: xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp -o build/native-r9700-runtime/native_amdev_transfer_probe && build/native-r9700-runtime/native_amdev_transfer_probe --kernel-proof; status=$?; printf "wrapper_exit_status: %d\n" "$status"; exit "$status"; } 2>&1 | tee "$log"'
```

## Task set 4: Repeated-run report and review packet

### Source refs
- Plan Task 6 Steps 5-6: `docs/superpowers/plans/2026-08-17-gfx1201-compute-dispatch.md` lines 760-779.

### Target
- Update report: `.superpowers/swarm/reports/c0a-compute-task-5-dispatch.md`.
- Prepare review packet for final reviewer in Phase 5.
- Non-goals: no fixing review findings in this task set unless separately dispatched; no commit or push.

### Change
1. Record focused pytest command/output supplied by supervisor.
2. Record first hardware `--kernel-proof` log path and pass/blocker tokens.
3. If first hardware run passes, supervisor reruns the same command once more; record second log path and tokens.
4. If first run passes and second fails, set blocker `compute_repeated_run_reset` and record both logs.
5. Include exact observed output bytes/digest for any pass.
6. Request reviewer with source and hardware log context.

### Acceptance
- Report contains commands, log paths, pass/blocker tokens, observed output bytes/digest, and repeated-run result.
- Review packet is sufficient for a reviewer to evaluate correctness, maintainability, architectural fit, and simplicity/no over-engineering.

### Validation
Pass requires two supervisor hardware runs to contain:

```text
kernel_launch_status: pass
cpu_comparison_status: pass
host_device_transfer_status: pass
failure_stage: none
exit_status: 0
wrapper_exit_status: 0
```

If not passing, report must carry the precise blocker and no phase may mark C0A-5 `Done`.

## Phase validation
Supervisor runs focused pytest, hardware `--kernel-proof`, and a second hardware `--kernel-proof` if the first passes. Phase complete requires either two passing runs with exact tokens above or a precise blocker report for Phase 5 decision.

## Handoff notes
- Phase 5 consumes the final report and reviewer findings.
- A single hardware pass is not enough; repeated-run safety is part of this phase's acceptance.

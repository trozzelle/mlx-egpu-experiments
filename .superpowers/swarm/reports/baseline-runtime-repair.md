# Baseline runtime repair

## Baseline and scope

The supervisor baseline was **715 passed, 26 failed** in `artifact://209`. Twenty-five failures were runtime/AMDev contract failures; the remaining failure was the raw HIP asset generator and is outside this repair. No test, build, formatter, linter, package-manager, hardware, or project-wide validation command was run by this worker, as required.

The runtime failures had three root causes:

1. `HardwareLock` is a required transitive implementation of `amdev_session.cpp` after commit `ef5b7a4`, but several local compile closures and the dynamic C1 bridge build still omitted `hardware_lock.cpp`. The linker therefore reported the complete `HardwareLock::*` symbol set as undefined.
2. Three source contracts retained pre-optimization expectations: the PM4 GCR payload still expected `0x000003f0`, the GC-flush call assertion omitted the optional timing argument, and the post-doorbell helper count expected one call site that the batched resident path intentionally bypasses.
3. The committed implementations are source-grounded: `amdev_packets.cpp` now emits the code-install barrier value `0x0000c3f1`; `setup_fixed_vm_mapping` passes `hdp_flush_usec` to the GC helper; and `ResidentHsaSession::dispatch_batch` writes one combined stream directly rather than invoking the single-dispatch diagnostic wrapper.

## Edits

### HardwareLock link closures

- `native_r9700/runtime.cpp`: added `native_r9700/hardware_lock.cpp` to the generated default C1 transfer-bridge `build_cmd` immediately after `kernel_catalog.cpp`.
- `tests/native_r9700/test_device_memory_contract.py`: added a named `HARDWARE_LOCK_SOURCE`, asserted that it exists, and added it to the device-memory probe link command.
- `tests/native_r9700/test_resident_kernel_dispatch_contract.py`: added a named `HARDWARE_LOCK_SOURCE`, asserted that it exists, and added it to the resident-dispatch probe link command.
- `tests/native_r9700/test_runtime_lifecycle.py`: added a named `HARDWARE_LOCK_SOURCE`, added it to `RUNNER_SOURCES`, and asserted the runner closure retains that source.
- `tests/native_r9700/test_runtime_llama_embed_contract.py`: added a named `HARDWARE_LOCK_SOURCE`, added it to `RUNNER_SOURCES`, and asserted the runner closure retains that source.
- `tests/native_r9700/test_runtime_vram_contract.py`: added a named `HARDWARE_LOCK_SOURCE`, added it to `RUNNER_SOURCES`, and asserted the runner closure retains that source; all runner-based trace/publication harnesses inherit that closure.
- `tests/native_r9700/test_runtime_protocol.py`: named and asserted the canonical runner source entry, added `hardware_lock.cpp` to the bridge `required_sources` assertion, and added it to the direct runtime API probe command. The bridge source parser now observes the production `build_cmd` entry, so a future omission fails the explicit required-source assertion as well as the link.

### Source-grounded stale contracts

- `tests/native_r9700/test_amdev_packets.py`: corrected the full frozen PM4 prefix, retaining the three zero payload dwords at indices 4-6 and setting the ACQUIRE_MEM GCR payload at dword 7 from `0x000003f0` to `0x0000c3f1`. This matches the current encoder's GLI/GLM/GLK/GLV/GL1/GL2 invalidation bits and the already-aligned `test_gpu_timestamp_pm4_contract.py` fixture; no production packet behavior was changed.
- `tests/native_r9700/test_runtime_vram_contract.py`: changed the GC setup assertion to the committed call shape `flush_gc_tlb_vmid0_native(client, log, &error, hdp_flush_usec)`. The body remains REQ then ACK polling and still excludes ENG17 semaphore I/O.
- `tests/native_r9700/test_resident_kernel_dispatch_contract.py`: changed the post-doorbell wrapper occurrence count from five to four (definition plus the three single-dispatch call sites). The comment records that the resident batch path submits one combined PM4 stream directly and intentionally bypasses this helper.

No runtime execution semantics, helper framework, raw HIP generator, phase packet, validation ledger, or progress ledger was changed.

## Original runtime failures accounted for

- **PM4 (1):** `test_amdev_packets.py::test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream` — stale GCR fixture; updated to the source-grounded `0xc3f1` value.
- **Device memory (1):** `test_device_memory_contract.py::test_device_memory_rejects_invalid_transitions_without_mutating_accounting` — probe link omitted `hardware_lock.cpp`; fixed closure.
- **Resident dispatch (7):**
  - `test_resident_kernel_dispatch_contract.py::test_resident_dispatch_rejects_a_kernel_without_reviewable_code`
  - `test_resident_kernel_dispatch_contract.py::test_resident_dispatch_rejects_kernarg_layout_drift`
  - `test_resident_kernel_dispatch_contract.py::test_resident_dispatch_rejects_code_that_exceeds_the_c0_page`
  - `test_resident_kernel_dispatch_contract.py::test_physical_dispatch_rejects_unreviewed_asset_before_tinygpu_connection`
  - `test_resident_kernel_dispatch_contract.py::test_physical_dispatch_rejects_a_digest_mismatch_before_bar0_operation`
  - `test_resident_kernel_dispatch_contract.py::test_resident_dispatch_preflight_accepts_a_complete_bounded_launch`
  - `test_resident_kernel_dispatch_contract.py::test_native_queue_diagnostics_identify_post_doorbell_receipt_and_fetch_state`
  The first six had the incomplete probe link closure; the last had the stale helper-count assertion.
- **Lifecycle (7):** `test_runtime_lifecycle.py::test_dry_run_exit_status_zero`, `test_runtime_lifecycle.py::test_dry_run_requires_log_fields`, `test_runtime_lifecycle.py::test_dry_run_reports_kernarg_layout`, `test_runtime_lifecycle.py::test_dry_run_reports_packet_encodings`, `test_runtime_lifecycle.py::test_dry_run_rejects_reinit_and_skip`, `test_runtime_lifecycle.py::test_dry_run_hardware_free`, and `test_runtime_lifecycle.py::test_help_lists_lifecycle_c0_transfer_and_legacy_diagnostic_modes` — canonical runner list omitted `hardware_lock.cpp`.
- **Llama embedding (1):** `test_runtime_llama_embed_contract.py::test_help_lists_llama_embed_smoke_without_opening_tinygpu` — canonical runner list omitted `hardware_lock.cpp`.
- **Runtime protocol (2):** `test_runtime_protocol.py::test_default_transfer_bridge_build_links_all_amdev_modules` — dynamic bridge build omitted `hardware_lock.cpp`; `test_runtime_protocol.py::test_transfer_round_trip_bytes_returns_caller_owned_output` — direct runtime API probe omitted it.
- **Resident VRAM/trace (6):** `test_runtime_vram_contract.py::test_help_lists_vram_smoke_without_opening_tinygpu`, `test_runtime_vram_contract.py::test_llama_stage_trace_help_and_invalid_arguments_never_open_tinygpu`, `test_runtime_vram_contract.py::test_llama_stage_trace_unit_scale_rejects_non_normalized_stage_before_device`, `test_runtime_vram_contract.py::test_llama_stage_trace_zero_input_requires_unit_scale_before_device`, and `test_runtime_vram_contract.py::test_llama_trace_publication_failure_seam_scalar_values_and_nonfinite_diagnostic` inherited the incomplete runner closure; `test_runtime_vram_contract.py::test_fixed_vm_gc_flush_uses_req_ack_without_semaphore` had the stale optional-argument assertion.

This accounts for 25 of the 26 baseline failures. The remaining baseline failure, `test_raw_hip_asset_generator.py::test_fresh_embed_row_source_generates_admitted_raw_code_and_manifest`, remains an explicit non-goal and was not edited.

## Residual status

The supervisor's first focused rerun (`artifact://236`) reported 24 passed and the PM4 test failed because the fixture was one dword short: it omitted one of the three zero payload dwords before dword 7. This worker restored that missing zero and did not rerun validation. The raw HIP generator failure above remains an explicit non-goal; the corrected runtime status still requires the focused supervisor command below.

## Focused supervisor command

Run this exact no-hardware focused selection from the assigned worktree:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest -q \
  tests/native_r9700/test_amdev_packets.py::test_pm4_dispatch_words_preserve_the_frozen_59_dword_c0a25_stream \
  tests/native_r9700/test_device_memory_contract.py::test_device_memory_rejects_invalid_transitions_without_mutating_accounting \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_resident_dispatch_rejects_a_kernel_without_reviewable_code \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_resident_dispatch_rejects_kernarg_layout_drift \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_resident_dispatch_rejects_code_that_exceeds_the_c0_page \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_physical_dispatch_rejects_unreviewed_asset_before_tinygpu_connection \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_physical_dispatch_rejects_a_digest_mismatch_before_bar0_operation \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_resident_dispatch_preflight_accepts_a_complete_bounded_launch \
  tests/native_r9700/test_resident_kernel_dispatch_contract.py::test_native_queue_diagnostics_identify_post_doorbell_receipt_and_fetch_state \
  tests/native_r9700/test_runtime_lifecycle.py::test_dry_run_exit_status_zero \
  tests/native_r9700/test_runtime_lifecycle.py::test_dry_run_requires_log_fields \
  tests/native_r9700/test_runtime_lifecycle.py::test_dry_run_reports_kernarg_layout \
  tests/native_r9700/test_runtime_lifecycle.py::test_dry_run_reports_packet_encodings \
  tests/native_r9700/test_runtime_lifecycle.py::test_dry_run_rejects_reinit_and_skip \
  tests/native_r9700/test_runtime_lifecycle.py::test_dry_run_hardware_free \
  tests/native_r9700/test_runtime_lifecycle.py::test_help_lists_lifecycle_c0_transfer_and_legacy_diagnostic_modes \
  tests/native_r9700/test_runtime_llama_embed_contract.py::test_help_lists_llama_embed_smoke_without_opening_tinygpu \
  tests/native_r9700/test_runtime_protocol.py::test_default_transfer_bridge_build_links_all_amdev_modules \
  tests/native_r9700/test_runtime_protocol.py::test_transfer_round_trip_bytes_returns_caller_owned_output \
  tests/native_r9700/test_runtime_vram_contract.py::test_help_lists_vram_smoke_without_opening_tinygpu \
  tests/native_r9700/test_runtime_vram_contract.py::test_fixed_vm_gc_flush_uses_req_ack_without_semaphore \
  tests/native_r9700/test_runtime_vram_contract.py::test_llama_stage_trace_help_and_invalid_arguments_never_open_tinygpu \
  tests/native_r9700/test_runtime_vram_contract.py::test_llama_stage_trace_unit_scale_rejects_non_normalized_stage_before_device \
  tests/native_r9700/test_runtime_vram_contract.py::test_llama_stage_trace_zero_input_requires_unit_scale_before_device \
  tests/native_r9700/test_runtime_vram_contract.py::test_llama_trace_publication_failure_seam_scalar_values_and_nonfinite_diagnostic
```

To record the known non-runtime residual separately, the supervisor may run:

```sh
cd ${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a && ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest -q tests/native_r9700/test_raw_hip_asset_generator.py::test_fresh_embed_row_source_generates_admitted_raw_code_and_manifest
```

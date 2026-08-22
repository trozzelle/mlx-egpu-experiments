# C1R Task 9 — native R9700 prefill producer blocker

## Result

The Task 8 seam remains **non-accepting**. No complete native prefill producer was added because the repository has no production GPU-resident Llama layer dispatcher. `r9700_native` remains fail-closed and cannot emit or retain an NPZ.

## Reachable fail-closed path

`runner --native-prefill-proof` now validates its request and removes any pre-existing regular output before reporting:

- `native_prefill_acceptance: open`
- `native_prefill_full_layer_loop_status: blocked`
- `failure_stage: native_layer0_kernel_sequence_unavailable`
- `native_prefill_blocker_source: native_r9700/llama_layer_executor.cpp:execute_llama_layer0`

The runner serializes the layer-loop status and blocker source in both its key/value hardware log and JSON. The Python worker preserves those fields; its acceptance validator and CLI require `native_prefill_full_layer_loop_status=pass` in addition to the existing producer identity, R9700 substrate, hardware-log, exit-zero, nonzero-kernel/transfer, and complete fp16 NPZ schema checks.

## Source-grounded blocker

1. `native_r9700/llama_layer_executor.cpp`, `execute_llama_layer0`, binds all ten real layer-0 fp16 safetensors spans through `ModelWeightBinder` before any device work, then explicitly stops at `kernel_sequence` because there is no reviewed Llama stage-kernel sequence.
2. `native_r9700/amdev_session.h/.cpp` now exposes `AMDevSession::dispatch_resident_kernel`: a bounded physical C0 seam that preflights a caller-provided reviewed descriptor, BAR0-loads and readbacks its code, binds exact kernargs, uploads input with SDMA, submits descriptor-sourced 59-dword PM4, polls the compute timeline, and SDMA-readbacks output. Its connection/VM/queue sequence is ported from `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp:6339-6609`; its code-load and kernarg mechanisms are ported from lines `3203-3281`.
3. `native_r9700/kernel_catalog.cpp` deliberately contains no executable descriptor. The C0 add-one blob remains probe provenance only, never a product or Llama asset.
4. `ResidentKernelDispatch` fails before TinyGPU connection unless supplied code, SHA-256 identity, nonzero resource registers and geometry, exact kernarg layout, and nonempty one-page-bounded input/output spans are present. This is a reusable hardware seam, not a host-vector or bridge fallback.

## Required unblock

Provide a reviewed descriptor and real executable gfx1201 code for every Llama stage in order: embedding/input preparation, RMS norms, Q/K/V/O projections, RoPE, attention, residuals, MLP, and K/V extraction. Each descriptor must specify its exact code bytes and SHA-256 identity, PM4 resources, geometry, and kernarg layout; the executor must bind real safetensors weight spans and sequence those descriptors for layers `0..15`. Only after that path has written and read back all 32 fp16 K/V tensors atomically, and produced the explicit R9700 log with exit status zero, may the worker set `native_prefill_full_layer_loop_status: pass` and `native_prefill_acceptance: pass`.

## Inherited partial-work disposition

The stopped predecessor's generic `KernelDescriptor` fields, parameterized `Pm4DispatchConfig`, empty catalog, and preflight-contract test were retained because they match the C0 bounded-launch contract. It had no physical dispatch method. The stale claim that the catalog contained `fp32_add_scalar` and that no arbitrary dispatch seam existed was discarded; the catalog remains empty and the new physical seam is not wired to Llama without reviewed stage assets.

## Tests changed

- `tests/native_r9700/test_resident_kernel_dispatch_contract.py` covers fail-closed rejection of missing code, mismatched kernargs, and code that exceeds the fixed C0 page; it also proves the physical API returns at preflight rather than connecting to TinyGPU for an unreviewed asset.
- `tests/native_r9700/test_amdev_packets.py` proves the generic PM4 form binds descriptor code address, resources, geometry, kernarg address, and timeline rather than silently using frozen C0 values.
- `tests/native_r9700/test_layer0_executor_contract.py` compiles the executor with `ModelWeightBinder`, so an empty/non-bindable model directory fails before device work.
- `tests/native_r9700/test_runtime_protocol.py` and `test_native_worker_evidence.py` retain the non-accepting full-layer-loop gate.

No commands were run: the task constraint prohibits test, hardware, and other command execution.

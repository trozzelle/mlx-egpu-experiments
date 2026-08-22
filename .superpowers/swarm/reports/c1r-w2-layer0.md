# C1R-W2-2 Layer-0 Foundation

## Scope

Task8 creates the compact kernel-catalog API and a fail-closed Llama layer-0 executor foundation. It does not perform device work, serialize a prompt cache, execute a full layer loop, embed kernel operands or model/fixture data, or claim native-prefill acceptance.

## Files and symbols

- `native_r9700/kernel_catalog.h/.cpp`
  - `KernelDescriptor`
  - `validate_kernel_descriptors(...)`
  - `find_kernel(...)`
- `native_r9700/llama_layer_executor.h/.cpp`
  - `LayerExecutionEvidence`
  - `validate_layer_execution_evidence(...)`
  - `execute_llama_layer0(...)`
- `tests/native_r9700/test_kernel_catalog.py`
  - `test_kernel_descriptor_validation_rejects_malformed_catalog_entries`
  - `test_unknown_kernel_lookup_returns_null`
- `tests/native_r9700/test_layer0_executor_contract.py`
  - `test_layer0_rejects_existing_model_directory_without_real_weights`
  - `test_layer0_rejects_fixture_sourced_intermediate_evidence_and_stays_open`
  - `test_layer0_rejects_fixture_sourced_model_input_evidence_and_stays_open`

## RED contracts

- `C1R-W2-2-RED-KERNEL-DESCRIPTOR`: `test_kernel_descriptor_validation_rejects_malformed_catalog_entries` rejects duplicate names, zero launch dimensions, and noncanonical digests; `test_unknown_kernel_lookup_returns_null` rejects an unknown lookup by returning `nullptr`.
- `C1R-W2-2-RED-MISSING-WEIGHTS`: `test_layer0_rejects_existing_model_directory_without_real_weights` requires a `model weights` error before `DeviceMemory` is required.
- `C1R-W2-2-RED-FIXTURE-INPUT`: `test_layer0_rejects_fixture_sourced_intermediate_evidence_and_stays_open` and `test_layer0_rejects_fixture_sourced_model_input_evidence_and_stays_open` reject `fixture:` intermediate or model sources while retaining `native_prefill_acceptance: open`.

## Fail-closed boundary

`execute_llama_layer0` resets the evidence to its open-only default, validates model-weight presence before device-memory validation, and then rejects because no reviewed resident layer-0 kernel sequence is available. The executor neither allocates nor transfers device buffers in this foundation. `validate_layer_execution_evidence` requires layer index, kernel/transfer counts, K/V/hidden shapes, model/intermediate source identities, and hardware identity for a prospective successful layer record; it rejects any non-open acceptance value.

## Supervisor verification commands

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_kernel_catalog.py tests/native_r9700/test_layer0_executor_contract.py -q
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra native_r9700/runtime_contract.cpp native_r9700/amdev_packets.cpp native_r9700/amdev_session.cpp native_r9700/device_memory.cpp native_r9700/kernel_catalog.cpp native_r9700/llama_layer_executor.cpp native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 -o build/native-r9700-runtime/native_r9700_runner
```

These commands are supervisor-owned and intentionally not run for this foundation handoff.

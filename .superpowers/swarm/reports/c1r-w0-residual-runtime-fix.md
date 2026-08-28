# C1R-W0 residual runtime fix

## Changed branch
`feature/native-r9700-producer`

## Changed fields
`RuntimeSession::native_prefill_proof` now compares `out_path` and `requested_log_path` as absolute, lexically normalized `std::filesystem::path` values before its existing conflict return, output removal, or log write. Relative and absolute aliases therefore collide without symlink resolution.

The native-prefill result field is now exactly `token_ids_json: <redacted>` in both runner stdout and the hardware log. Request parsing and token computation are unchanged.

The existing `failure_stage: output_path_conflict` response, direct and `./` alias handling, cleanup-failure path, fail-closed acceptance, and C0 wrappers remain unchanged.

## Supervisor commands

Focused RED-contract selectors:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'native_prefill_proof_redacts_token_ids_from_stdout_and_hardware_log or native_prefill_proof_rejects_relative_output_and_absolute_log_aliases'
```

Full runtime-contract validation:

```sh
${PY} -m pytest tests/native_r9700/test_runtime_contract.py -q
```

C++ compile validation:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```

Per assignment, this worker did not run commands.

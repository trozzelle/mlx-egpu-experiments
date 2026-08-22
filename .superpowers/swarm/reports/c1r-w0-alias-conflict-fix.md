# C1R-W0 alias-conflict fix

## Changed branch
`feature/native-r9700-producer`

## Minimal change
`RuntimeSession::native_prefill_proof` now compares `out_path` and `requested_log_path` as `std::filesystem::path(...).lexically_normal()` values before its existing output-path conflict response. This treats `<tmp>/native-prefill.npz` and `<tmp>/./native-prefill.npz` as the same lexical target without filesystem probing or symlink resolution.

The existing `failure_stage: output_path_conflict` response remains before `std::remove` and before any log write. Distinct paths and the existing no-injection/cleanup behavior are unchanged.

## Supervisor commands

Focused GREEN selector:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k native_prefill_proof_rejects_lexically_distinct_output_and_log_path_aliases
```

Full runtime-contract validation:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q
```

C++ compile validation:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```

Per assignment, this worker did not run commands.

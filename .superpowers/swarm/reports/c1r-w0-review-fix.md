# C1R-W0-1 review fixes

## Changed symbol

- `RuntimeSession::legacy_primitive_diagnostic` now treats any injected `native_prefill_acceptance: pass` marker as a legacy-diagnostic protocol violation. The injected payload is not copied into the wrapper CLI/log result when it contains that marker, so neither surface can publish the prohibited native-prefill pass claim. The result remains nonzero with `legacy_diagnostic_status: fail` and `failure_stage: legacy_diagnostic_protocol`.
- `RuntimeSession::native_prefill_proof` now rejects identical output/log paths before removal or log writing with `failure_stage: output_path_conflict`.
- `RuntimeSession::native_prefill_proof` now checks `std::remove` for errors other than `ENOENT`. A pre-existing output that cannot be removed returns `failure_stage: output_path_cleanup`, preserves the output, and does not fall through to `legacy_proof_unavailable`. A nonexistent output remains valid and follows the existing fail-closed `legacy_proof_unavailable` result.

No test source adjustment was needed: the RED selector coverage (and its `json` import) is already present in `tests/native_r9700/test_runtime_contract.py`. C0 lifecycle/kernel/transfer wrappers and the no-injection legacy/native-prefill behavior are unchanged; this work adds no archive, primitive-chain product route, or native acceptance path.

## RED-to-GREEN mapping

- `test_legacy_primitive_diagnostic_rejects_injected_native_prefill_acceptance`: the prohibited injected marker forces legacy protocol failure and is excluded from emitted diagnostic output.
- `test_native_prefill_proof_rejects_equal_output_and_log_paths`: equal paths return `output_path_conflict` before either `remove` or `write_all_text_file`.
- `test_native_prefill_proof_reports_output_cleanup_failure`: a pre-existing directory makes `std::remove` fail with a non-`ENOENT` error and returns `output_path_cleanup` before the generic legacy-prefill blocker.

## Supervisor commands

Focused GREEN selector (including the retained no-injection guard):

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'legacy_primitive_diagnostic_reports_legacy_proof_unavailable_without_bridge or legacy_primitive_diagnostic_rejects_injected_native_prefill_acceptance or native_prefill_proof_rejects_equal_output_and_log_paths or native_prefill_proof_reports_output_cleanup_failure'
```

Full runtime-contract suite:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q
```

C++ compile:

```sh
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
```

Re-review diff hygiene:

```sh
git diff --check -- native_r9700/runtime.cpp tests/native_r9700/test_runtime_contract.py .superpowers/swarm/reports/c1r-w0-review-fix.md
```

Per executor policy, no validation command was run by this worker.

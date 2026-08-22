# C1R-W0 review RED regressions

## Contract tests

- `test_legacy_primitive_diagnostic_reports_legacy_proof_unavailable_without_bridge` exercises the direct `--legacy-primitive-diagnostic fp32_add_scalar` path with no injected bridge. It requires a nonzero exit, `failure_stage: legacy_proof_unavailable`, and no native-prefill pass claim.
- `test_legacy_primitive_diagnostic_rejects_injected_native_prefill_acceptance` supplies every valid `fp32_add_scalar` primitive marker plus `native_prefill_acceptance: pass`. The legacy diagnostic must reject it, return nonzero, and never publish that native pass claim.
- `test_native_prefill_proof_rejects_equal_output_and_log_paths` requires an equal `--out`/`--log` invocation to report `failure_stage: output_path_conflict` and leave no file at the requested NPZ path.
- `test_native_prefill_proof_reports_output_cleanup_failure` uses a pre-existing nonempty directory at `--out`, containing a small child file. The prior empty-directory setup was not portable because `remove` can delete empty directories; this corrected fixture requires a nonzero result with `failure_stage: output_path_cleanup`, no native-prefill pass claim, and retention of the directory and child file rather than replacement with an NPZ.

## Expected pre-fix status

- The direct no-injection legacy-diagnostic test is already GREEN after the C1R-W0 cutover; it protects that completed failure mode rather than serving as a RED test.
- The injected native-pass test is RED: the current diagnostic accepts the otherwise-valid injected primitive output and copies `native_prefill_acceptance: pass` to stdout.
- The equal-path test is RED: the current implementation writes its log at the requested NPZ path instead of rejecting the conflicting paths.
- The cleanup-path test is RED: the current implementation ignores failure to remove the pre-existing output directory and reports the later legacy-proof failure instead of an output cleanup failure.

No validation command was run by this executor; the supervisor owns RED/GREEN observation.

## Supervisor selectors

RED before the review fixes:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'legacy_primitive_diagnostic_rejects_injected_native_prefill_acceptance or native_prefill_proof_rejects_equal_output_and_log_paths or native_prefill_proof_reports_output_cleanup_failure'
```

GREEN after the review fixes (including the already-green direct no-injection guard):

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q -k 'legacy_primitive_diagnostic_reports_legacy_proof_unavailable_without_bridge or legacy_primitive_diagnostic_rejects_injected_native_prefill_acceptance or native_prefill_proof_rejects_equal_output_and_log_paths or native_prefill_proof_reports_output_cleanup_failure'
```

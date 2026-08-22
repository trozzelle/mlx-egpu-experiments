# Native-prefill output/log alias regression

- Test: `test_prefill_cli_rejects_output_log_alias_conflict_without_generic_side_effects`
- RED: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py::test_prefill_cli_rejects_output_log_alias_conflict_without_generic_side_effects -q`
  - Expected before the production fix: failure because `prefill.main()` processes the blocked `output_path_conflict` worker result through generic rejected-output cleanup and generic log writing; with lexical `--out x.npz` and `--log ./x.npz`, both side effects target the same resolved path.
- GREEN: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py::test_prefill_cli_rejects_output_log_alias_conflict_without_generic_side_effects -q`
  - Expected after the production fix: pass, proving the native worker result is handled with exit status 1 without generic cleanup or log-write calls at the conflicted target.

The commands were intentionally not run because this task explicitly prohibits command execution.

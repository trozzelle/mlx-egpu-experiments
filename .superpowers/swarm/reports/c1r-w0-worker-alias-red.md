# Native-worker output/log alias regression

- Test: `test_native_worker_blocks_lexically_distinct_output_and_log_aliases_before_runner`
- RED: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_native_worker_blocks_lexically_distinct_output_and_log_aliases_before_runner -q`
  - Expected before the production fix: failure because `run_native_prefill` invokes the patched runner and subsequently writes its rejection log through the `./` alias, recreating the rejected NPZ target; it also does not return `native_prefill_acceptance: blocked` with `failure_stage: output_path_conflict`.
- GREEN: `${PY} -m pytest tests/native_r9700/test_runtime_contract.py::test_native_worker_blocks_lexically_distinct_output_and_log_aliases_before_runner -q`
  - Expected after the production fix: pass, proving normalized alias rejection happens before subprocess execution and before any log write.

The commands were intentionally not run because this task explicitly prohibits command execution.

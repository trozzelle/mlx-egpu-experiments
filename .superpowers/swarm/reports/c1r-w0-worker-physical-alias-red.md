# Worker physical-path alias regression

- Test: `test_native_worker_blocks_dangling_log_symlink_to_output_before_side_effects`
- RED expected before the production repair: failure because `run_native_prefill()` compares only lexical absolute paths. A dangling `native-prefill.log` symlink to `native-prefill.npz` bypasses that check, launches the fake runner, attempts rejected-output cleanup, and writes the worker result through the symlink. The fake runner's `native_prefill_acceptance: open` was incidental output, not the intended guard schema.
- GREEN expected after the repair: pass, proving the worker returns the established pre-launch `output_path_conflict` result schema with `native_prefill_acceptance: blocked` before runner launch, cleanup, or log writing; the symlink target remains absent.

No commands were run, per assignment.

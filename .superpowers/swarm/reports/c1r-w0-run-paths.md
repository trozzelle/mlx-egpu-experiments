# C1R-W0 run-paths implementation

## Delivered

- Added `native_r9700.run_paths.run_root()`: it returns
  `Path(NATIVE_R9700_RUN_ROOT)` when configured, otherwise
  `Path("logs/native-r9700-runs")`, without creating either root.
- Added `native_r9700.run_paths.new_run_dir(label)`: it rejects `/` and `\\`
  before filesystem mutation, then creates and returns one directory named
  `<label>-YYYYMMDDTHHMMSSZ` under the selected root.
- Documented `NATIVE_R9700_RUN_ROOT` and the default generated-run root in the
  validation ledger.

## Ignore policy

No `.gitignore` change was needed: the default root is already covered by the
existing `logs/` rule. A configured root is user-selected and therefore cannot
be safely ignored by a repository-wide fixed path rule.

## Validation

The assignment prohibits commands, so the focused run-path tests were not run.
The implementation matches the RED public contract in
`tests/native_r9700/test_run_paths.py`.

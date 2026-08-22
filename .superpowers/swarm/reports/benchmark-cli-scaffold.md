# Benchmark CLI Scaffold

## Changed files
- `native_r9700/benchmark.py`
- `tests/native_r9700/test_benchmark.py`
- `.superpowers/swarm/reports/benchmark-cli-scaffold.md`

## Exact behavior
- Added `native_r9700.benchmark` with row validation, serving-result-to-benchmark-row conversion, JSON/report/log writers, and CLI parsing for `--model`, `--fixtures-dir`, `--producer-kind`, `--artifacts-dir`, `--json`, `--report`, `--log`, and repeated `--serving-result` inputs.
- The CLI does not run benchmark/model/hardware commands; it consumes C2 serving result JSON and fails closed if no accepted native C2 result is supplied.
- Native benchmark rows require `gate_result=pass`, `producer_kind=r9700_native`, `row_role=native_benchmark`, `route=native_producer`, `accepted_cache=true`, no fallback reason, a non-empty `hardware_log_path`, nonzero native timing/transfer evidence, and token-exact decoded-vs-reference evidence.
- `cpu_reference` rows validate only as `row_role=baseline` with an explicit `baseline_name`; Path A rows validate only as `row_role=control` using `producer_kind=path_a_tinygrad` and an explicit label.
- Added focused tests for the public API, native row schema/writers, fake-native rejection, and baseline/control labeling.

## Remaining blocker
- Real benchmark execution remains blocked until Task 5 emits accepted native C2 serving JSON with hardware log path, timing counters, and token-exact evidence.

## Supervisor commands
- `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_benchmark.py -q`

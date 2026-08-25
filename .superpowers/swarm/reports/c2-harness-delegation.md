# C2 harness delegation to imported-cache serving wrapper

## Scope
Added the smallest external worker/harness integration path for C2: `tinygrad_kv_worker.harness --c2-serving` delegates to the existing `native_r9700.serving` CLI. It does not duplicate prompt-cache validation, fallback logic, report rendering, redaction, or native producer policy.

## RED evidence
Command:
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_harness_c2_serving.py -q
```
Result before implementation: `2 failed` with argparse `unrecognized arguments: --c2-serving ...` (`artifact://4534`).

## Implementation
Changed `tinygrad_kv_worker/harness.py`:
- Added `_build_c2_serving_argv(args, forward_max_new_tokens)` to build `[sys.executable, "-m", "native_r9700.serving", ...]`.
- Added `_run_c2_serving(args, forward_max_new_tokens)` to return the child process return code.
- Added `--c2-serving` plus C2 wrapper arguments to the existing harness parser.
- Branches before Phase 0 Path A logging and `--gguf/--mlx` validation when `--c2-serving` is selected.
- Leaves `--threshold-tokens` and `--producer-timeout-s` as child-owned CLI strings so the wrapper, not the harness, owns type coercion.

Added `tests/test_harness_c2_serving.py`:
- Delegation without `--gguf/--mlx` builds the wrapper command and returns the child status.
- `--producer-kind r9700_native` is passed through to `native_r9700.serving` so fail-closed native policy stays in the wrapper.
- Omitted `--max-new-tokens` is not forwarded, preserving the wrapper's own default instead of the legacy Path A default.

## Verification
Focused tests:
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_harness_c2_serving.py -q
```
Result: `3 passed, 2 warnings in 1.64s`.

Focused C2/harness regression:
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/test_harness_c2_serving.py tests/test_harness_injected_path.py tests/test_harness_logging.py tests/test_harness_report.py tests/native_r9700/test_serving.py -q
```
Result: `27 passed, 2 warnings in 1.64s`.

Real harness smoke:
```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m tinygrad_kv_worker.harness --c2-serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --producer-model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-0 --threshold-tokens 2 --max-new-tokens 4 --artifacts-dir artifacts/c2-harness-smoke --json .superpowers/swarm/reports/c2-harness-smoke-result.json --log logs/c2-harness-smoke.log --report .superpowers/swarm/reports/c2-harness-smoke-report.md
```
Result: exit `0`, `gate_result=pass`, `route=native_producer`, `accepted_cache=true`, `producer_kind=cpu_reference`, decoded tokens `[12366, 13, 578, 469]` exactly matched baseline.

Task-doc alignment: `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md` and `docs/archive/tasks/native-r9700-producer/phase-c1-c2-r9700-recovery-plan.md` now include the 2026-08-21 pivot note so future agents see the accepted product/reference path separately from the still-open native R9700 objective.

## Acceptance status
- C2 reference-wrapper integration: accepted for Llama CPU-reference imported-cache serving.
- Native R9700 C2: open; `r9700_native` still intentionally fails closed until the producer exists.
- Qwen3.8-27B: out of this C2 scope; Qwen remains a separate target-expansion phase because its local MLX-VLM/quantized/hybrid-cache ABI does not match the Llama prompt-cache ABI.

# C2 serving integration plan after C1R pivot

## Starting point
- C1R product-smoke acceptance is the mlx-lm imported-cache wrapper path, not exhaustive per-primitive hardware parity.
- Passing smoke artifact: `.superpowers/swarm/reports/c1r-prefill-smoke-result.json`.
- Passing smoke report: `.superpowers/swarm/reports/c1r-prefill-smoke-report.md`.
- Accepted interface: `native_r9700.prefill` NPZ -> `native_r9700.kv_cache` safetensors -> `native_r9700.serving` imported prompt cache.

## Verified C1R smoke
Command:
```sh
${PY} -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --producer-model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-0 --threshold-tokens 2 --max-new-tokens 4 --artifacts-dir artifacts/c1r-prefill-smoke --json .superpowers/swarm/reports/c1r-prefill-smoke-result.json --log logs/c1r-prefill-smoke.log --report .superpowers/swarm/reports/c1r-prefill-smoke-report.md
```
Result: exit `0`, `gate_result=pass`, `route=native_producer`, `accepted_cache=true`, decoded tokens `[12366, 13, 578, 469]` match baseline exactly.

## Honest limitations
- `producer_kind=cpu_reference`; this is a usable imported-cache serving path, not a native R9700 prefill-speed win.
- `producer_kind=r9700_native` is still rejected before spawning producer subprocesses.
- Qwen3.8-27B remains outside C1R because its loader/config/KV ABI differs and was explicitly deferred by `.superpowers/swarm/reports/c1-qwen-target-decision.md`.

## C2 target
C2 should harden and expose the existing imported-cache serving wrapper as the integration seam:
1. Keep the C2 public contract centered on `native_r9700.serving` result JSON/log/report fields.
2. Preserve fail-closed behavior for unsupported `r9700_native` requests until the producer exists.
3. Add or verify one external worker/harness invocation path that calls the wrapper with model, fixtures/prompt or token ids, threshold, artifacts dir, JSON, and log paths.
4. Keep C2 acceptance labels explicit: `REFERENCE WRAPPER PASS; NATIVE R9700 C2 OPEN` for CPU-reference producer, native C2 pass only after `producer_kind=r9700_native` emits and validates a cache.
5. Defer Qwen to a separate target-expansion phase unless a Qwen MLX safetensors loader/KV ABI task is opened deliberately.

## Next execution wave
- C2 task 1: inspect `tinygrad_kv_worker.harness` and any existing worker CLI to identify the narrow integration point for invoking `native_r9700.serving` without changing model math.
- C2 task 2: add focused tests for that worker/harness invocation path if absent; preserve existing serving unit contracts.
- C2 task 3: run a real wrapper smoke after integration and update the Path C report section.

## Execution result
- Implemented `tinygrad_kv_worker.harness --c2-serving` as a child-process delegate to `native_r9700.serving`; the harness does not duplicate cache validation or fallback policy.
- Added focused tests in `tests/test_harness_c2_serving.py`.
- Verification: `python -m pytest tests/test_harness_c2_serving.py -q` -> `3 passed, 2 warnings in 1.64s`.
- Verification: `python -m pytest tests/test_harness_c2_serving.py tests/test_harness_injected_path.py tests/test_harness_logging.py tests/test_harness_report.py tests/native_r9700/test_serving.py -q` -> `27 passed, 2 warnings in 1.64s`.
- Real harness smoke: `python -m tinygrad_kv_worker.harness --c2-serving ... prompt-0 ...` -> exit `0`, `gate_result=pass`, `accepted_cache=true`, `route=native_producer`, `producer_kind=cpu_reference`, exact decoded tokens `[12366, 13, 578, 469]`.
- Report: `.superpowers/swarm/reports/c2-harness-delegation.md`.

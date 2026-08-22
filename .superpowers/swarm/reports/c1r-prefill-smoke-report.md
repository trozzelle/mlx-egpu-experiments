## Path C2 — mlx-lm imported-cache serving wrapper

Status: **REFERENCE WRAPPER PASS; NATIVE R9700 C2 OPEN**

gate_result: pass
status: pass
requested_producer_kind: cpu_reference
producer_kind: cpu_reference
model: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
producer_model_dir: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
fixtures_dir: tests/native_r9700/fixtures
prompt_count: 1
threshold_tokens: 2
producer_timeout_s: 300
json_path: .superpowers/swarm/reports/c1r-prefill-smoke-result.json
log_path: logs/c1r-prefill-smoke.log
artifacts_dir: artifacts/c1r-prefill-smoke
exit_status: 0

| Prompt | S | N prefix | Route | Fallback | Accepted cache | Decoded tokens | R tokens | Exact | Mismatches | Cache |
|---|---:|---:|---|---|---|---|---|---|---|---|
| prompt-0 | 6 | 5 | native_producer |  | True | `[12366, 13, 578, 469]` | `[12366, 13, 578, 469]` | True | `[]` | `artifacts/c1r-prefill-smoke/prompt-0.prompt-cache.safetensors` |

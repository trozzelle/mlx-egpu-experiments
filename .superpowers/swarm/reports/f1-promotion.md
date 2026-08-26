# F1 persistent warm worker promotion

Status: **PASS**

## Accepted implementation

- One public Python worker starts one private `native_r9700_runner --model-service-worker` child.
- One warm model generation serves ten `prompt-128` requests without resident-weight reload.
- Every request has `S=129`, `N=128`, `producer_kind=r9700_native`, `route=native_producer`, `accepted_cache=true`, token-exact comparison, and no fallback.
- Explicit process smoke completed ten Prefills on the first generation followed by unload, reload, and unload.
- Multi-PDB1 resident mappings support the model allocation across the full large-BAR VRAM window without overwriting the fixed C0 staging mapping.

## Hardware promotion evidence

Evidence root: `logs/f1-persistent-worker/`

- Process smoke: `logs/f1-persistent-worker/process-smoke/result.json`
- Warm serving samples: `logs/f1-persistent-worker/warm/serving.json`
- Scoped benchmark: `logs/f1-persistent-worker/warm/benchmark.json`
- Benchmark report: `logs/f1-persistent-worker/warm/benchmark.md`
- Device path: `TinyGPU.app/APLRemotePCIDevice/PCIIface`
- Warm operations: one `LoadModel`, ten `Prefill`, one `UnloadModel`
- `load_preparation_count=1`
- `prefill_count=10`
- `warm_prefill_weight_reload_count=0`
- Benchmark status: `pass`
- Full native benchmark records: 13
- `raw_warm_sample_count=10`
- `scope_aggregate_count=3`
- `records_by_scope={\"cold_process\":1,\"warm_prefill\":11,\"gpu_compute\":1}`
- `total_record_count=13`
- Every aggregate record retains the complete `native_r9700_benchmark_v1` timing, transfer, correctness, route/cache, hardware-log, and `row_role=native_benchmark` fields.

## Scope separation

| Scope | Samples | Median seconds | Median absolute deviation seconds | Minimum seconds | Maximum seconds |
|---|---:|---:|---:|---:|---:|
| `cold_process` | 1 | 16.133592666999903 | 0.0 | 16.133592666999903 | 16.133592666999903 |
| `warm_prefill` | 10 | 14.448696 | 0.024237999999999538 | 14.381851 | 14.645698 |
| `gpu_compute` | 10 | 13.137389 | 0.022178500000000767 | 13.07518 | 13.192957 |

`cold_process` measures worker entry through the first completed resident `LoadModel`. `warm_prefill` uses each request's native prefill elapsed time, reports throughput from the private `N=128` prefix rather than public `S=129`, and excludes the one-time load. `gpu_compute` uses the measured persistent dispatch-loop duration. No warm-up sample was discarded; all ten measured warm requests are included.

## Verification

- Complete F1 protocol/resource/service/serving/benchmark acceptance suite: 288 passed.
- F1 task-set-5 promotion contract: 190 passed, with two dependency deprecation warnings.
- Multi-PDB1/process acceptance gate: 125 passed.
- Real process smoke: exit 0.
- Real ten-sample warm command: exit 0.
- Real benchmark command: `benchmark status=pass rows=13`.
- Final acceptance re-review: PASS; zero remaining Critical/Important findings.

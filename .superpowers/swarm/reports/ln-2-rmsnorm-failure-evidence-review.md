# LN-2 RMSNorm nonfinite failure-evidence review

## Verdict: FAIL

### Findings

- **HIGH — `native_r9700/runtime_contract.cpp:889-901`**: diagnostic-publication failure is deliberately ignored. If `publish_trace_failure_diagnostic` fails, the code logs to stderr and still returns a `trace_nonfinite` failure. In particular, `publish_trace_failure_diagnostic` can leave its staging file behind when cleanup itself fails (`runtime_contract.cpp:424-447`). This fails the stated failed-trace contract that the only emitted artifact is an atomically published `<stage>.failure.json`; a caller cannot distinguish “nonfinite with durable evidence” from “nonfinite with missing or stale staging evidence.” The result/error path must surface diagnostic-publication failure and preserve the no-artifact invariant as far as cleanup permits.

- **MEDIUM — `tests/native_r9700/test_runtime_vram_contract.py:582-620`**: the focused test covers only successful direct invocation of `capture_trace_failure_diagnostic` and `publish_trace_failure_diagnostic`. It does not exercise either diagnostic helper failure path (write, file fsync, rename, parent fsync, or cleanup), nor does it exercise the `run_llama_stage_trace` nonfinite branch. The generic successful-artifact fault seam at lines 397-487 is not a contract test for the metadata-only file path. Consequently, the test does not defend the required error-path/atomicity claim.

## Verified properties

- **PASS — ordering (`native_r9700/runtime_contract.cpp:823-900`)**: `ResidentHsaSession::prepare` completes before `materialize_trace_kernargs` (844); failure metadata is captured (855-860) before dispatch, readback, and finite validation (862-889). RMSNorm explicitly requires a 32-byte materialized block (849-854).
- **PASS — materialized metadata source (`runtime_contract.cpp:173-190`, `native_r9700/amdev_session.cpp:2261-2289`)**: trace materialization copies stage kernargs and applies the same resident-buffer VA bindings used by `ResidentHsaSession::dispatch`. The report's PM4 entry VA, fixed kernargs VA, resource registers, and geometry are taken from the same stage/image inputs used to build the dispatch packet.
- **PASS — successful-artifact nonleakage on the normal nonfinite branch (`native_r9700/runtime_contract.cpp:885-902, 930-952`)**: finite validation precedes construction/publication of the raw success artifact. The output preflight rejects any pre-existing success directory, staging directory, legacy raw/JSON file, or failure file (`796-803`). The nonfinite branch writes no readback bytes to disk and invokes only the failure-diagnostic publisher.
- **PASS — success-path metadata shape (`runtime_contract.cpp:305-331`; `tests/native_r9700/test_runtime_vram_contract.py:598-620`)**: the diagnostic has the materialized kernarg hex; resident buffers 0, 1, and 11 with name/allocation size/live VA/physical offset; PM4 image/entry/kernargs/resource/geometry fields; and excludes success fields (`raw_path`, `sha256`, `finite_count`, successful exit status).

## Validation

Not run, per assignment: review/report only; no tests, git, or hardware.

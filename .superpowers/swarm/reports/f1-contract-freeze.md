# F1 contract freeze

**Task set:** F1 task set 1, Contract and validation-command freeze  
**Owner:** `F1Contract`  
**Status:** Needs review  
**Report:** `.superpowers/swarm/reports/f1-contract-freeze.md`

This report freezes the local service boundary for task sets 2–5. It does not add a production service, launch a process, load a model, execute hardware, or modify the shared validation ledger.

## Grounding

The authoritative contract is `docs/DESIGN.md`: operations/model ownership are in §Persistent model service contract (lines 124–149), canonical KV fields and `S-1` injection are in §Canonical KV description and §mlx-lm prompt-cache adapter (lines 151–182), benchmark fields/scopes are in §Benchmark contract (lines 254–268), and lifecycle/error vocabulary is in §Lifecycle and state transitions and §Validation and errors (lines 270–318). `docs/ROADMAP.md` §Phase F1 (lines 50–80) requires one resident model, repeated warm requests, separate cold/warm/GPU records, and an actual load/repeated-prefill/unload/reload smoke. `docs/IMPLEMENTATION_PLAN.md` F1 (lines 121–154) assigns protocol, registry, native-resource lifetime, persistent routing, and benchmark work packages.

The local source is the binding evidence:

- `native_r9700/native_worker.py`: `R9700_NATIVE_PRODUCER_KIND`, `_REQUIRED_FIELDS`, `_OPTIONAL_EVIDENCE_FIELDS`, `_INTEGER_ABI_RANGES`, `_SELECTED_COMPLETION_POLICY`, `_SELECTED_BARRIER_POLICY`, `run_native_prefill`, `validate_native_prefill_npz`, `_normalize_result`, `_acceptance_problems`, `_open_result`, and `_write_result_log`.
- `native_r9700/serving.py`: `NativePrefillConfig`, `_SAFE_REQUEST_ID_RE`, `_safe_request_id`, `_producer_artifact_paths`, `_base_result`, `_run_producer`, `_validate_prompt_cache`, `_with_fallback`, and `generate_with_native_prefill`.
- `native_r9700/kv_cache.py`: pinned mlx-lm metadata writer `_metadata`, including the 16 empty per-layer `meta_state` values.
- `native_r9700/benchmark.py`: `BENCHMARK_SCHEMA_VERSION`, `_REQUIRED_ROW_FIELDS`, `_TIMING_FIELDS`, `benchmark_row_from_serving_result`, `validate_benchmark_row`, `validate_benchmark_rows`, `build_benchmark_result`, and `_parser`.
- `native_r9700/resident_memory.h/.cpp`: `ResidentBuffer`, `ResidentMemory::allocate`, and `ResidentMemory::release_all`.
- `native_r9700/model_weight_binder.h/.cpp`: `ModelWeightBinder::open`, `bind_tensor`, `bind_llama_layer0`, and `bind_llama_stage_layer`.
- `native_r9700/runner.cpp`: existing C++ runner entrypoint and build source-list anchor; task set 3 adds the private worker mode without creating a second runner executable.
- `native_r9700/runtime.h/.cpp`, `native_r9700/native_resource_worker.h/.cpp`: task-set-3 native resource-owner implementation files; the worker header/source are private child-boundary files, not public service or registry files.


`docs/REFERENCES.md` classifies these local worker/serving/resident files as the first reference for proven behavior and explicitly says they do not yet define persistent lifetime or a public provider protocol. The wire details below are therefore explicit F1 decisions constrained by those sources.

## Frozen protocol v1

### Envelope and operations

- `protocol_version` is exactly `r9700_prefill_service_v1`.
- The initial transport is one JSON object per line over local stdio (`stdio_jsonl`). There is no TCP/network operation in F1. A later Unix-socket transport may reuse this envelope only.
- Every request has exactly `{protocol_version, request_id, operation, body}`.
- Every decoded-request response has exactly `{protocol_version, request_id, operation, status, result, error, evidence}`. `result` is an object; `error` and `evidence` are objects or `null`. Raw-frame failures use the separate exact transport-error envelope below, with unavailable correlation fields set to `null`.
- Each raw JSONL frame is at most `65,536` bytes including its trailing newline. `MAX_FRAME_BYTES=65536` is enforced on raw bytes before UTF-8 or JSON decoding; an oversized frame is rejected as `blocked`/`invalid_request` at `frame_size` using the pre-decode envelope below. Responses, including their evidence and error envelopes, obey the same cap.
- v1 rejects unknown top-level fields, duplicate JSON keys, non-object bodies, malformed types, and any version other than `r9700_prefill_service_v1`. Prompt/token values are never included in errors or logs.
- The operation list is exactly, in this order: `GetCapabilities`, `Health`, `LoadModel`, `UnloadModel`, `Prefill`, `GetMetrics`, `CaptureTrace`. `Decode` is deliberately absent because `docs/DESIGN.md` permits it only after a later roadmap promotion.
### Pre-decode transport errors

A frame-size or frame-decode failure occurs before a validated request object exists. It emits exactly this seven-key envelope, with no extra keys:

```json
{
  "protocol_version": "r9700_prefill_service_v1",
  "request_id": null,
  "operation": null,
  "status": "blocked",
  "result": {},
  "error": {
    "domain": "invalid_request",
    "message": "raw frame rejected before decode",
    "failure_stage": "frame_size"
  },
  "evidence": null
}
```

For `frame_decode`, only `failure_stage` changes to `"frame_decode"`; the message is the same bounded, non-sensitive UTF-8 string. `message` and `failure_stage` are each bounded by the 16 KiB string limit and never contain raw frame, token, or prompt bytes. The reader emits exactly one transport envelope for each rejected frame, discards through the next newline, then continues with the next frame. If EOF occurs while discarding a rejected frame, it emits no second envelope and exits normally; a failed envelope write exits rather than retrying. Transport errors never reserve a request ID or enter operation dispatch.

### Parsed request/schema errors

After successful UTF-8 and JSON decoding, schema validation precedes correlation recovery. A normal response echoes `request_id` only when the parsed value is a string passing the exact `_safe_request_id` grammar and 1–128 ASCII-character bound, and echoes `operation` only when the parsed value is one of `OPERATIONS`; otherwise that field is `null`. No unvalidated or malformed value is copied into a response. The response `protocol_version` remains the frozen constant. Every recovered valid request ID is checked against the process-lifetime namespace before returning a parsed validation error; a new valid ID is reserved exactly once, while a reused ID is reported with its validated value and is never re-reserved.

The minimal exact request bodies are the signatures in `docs/DESIGN.md` plus the explicit empty bodies shown here:

| Operation | Exact body | Result fields |
|---|---|---|
| `GetCapabilities` | `{}` | `service_name`, `protocol_version`, `operations`, `transport`, `model_formats`, `quantizations`, `cache_formats`, `model_family`, `geometry` |
| `Health` | `{}` | `service_available`, `service_unavailable_reason`, `device_state`, `model_state`, `runtime_substrate`, `loaded_model_count`, `active_request_count`, `last_failure_stage` |
| `LoadModel` | `{model_uri, model_digest, format, quantization}` | `model_handle`, `model_state`, `model_fingerprint`, `kernel_pack_digests` |
| `UnloadModel` | `{model_handle}` | `model_handle`, `model_state` |
| `Prefill` | `{model_handle, token_ids, cache_spec, request_options}` | `model_handle`, `request_state`, `prompt_token_count`, `prefix_token_count`, `cache` |
| `GetMetrics` | `{}` | `model_handle`, `model_state`, `metrics` |
| `CaptureTrace` | `{}` | `trace_format`, `trace_path`, `snapshot` |

`LoadModel` and `Prefill` field names are the operation signatures specified by `DESIGN.md`; the remaining result names are the smallest projections needed to expose existing lifecycle, artifact, and benchmark evidence. `GetMetrics` exposes only service-owned load/native-prefill/resource counters and timings; cache import, decode, and consumer acceptance remain serving/benchmark-owned. `CaptureTrace` has no target selector: its `request_id` identifies the trace call itself, never an earlier `Prefill`.
`request_options` is exactly `{timeout_ms}`, where `timeout_ms` is an integer in `[1, 300000]`; `300000` is the existing `serving.py::_DEFAULT_PRODUCER_TIMEOUT_S` default expressed in milliseconds. `service_available` is a control-plane boolean: it is true while the process and registry can accept control operations and false only for `process_faulted` or `shutting_down`; it is not a model-readiness state. `service_unavailable_reason` is `null` when available and exactly one of those two values otherwise. `device_state` is a read-only observation of the DESIGN device lifecycle (`disconnected`, `initializing`, `ready`, `degraded`, `faulted`, `resetting`, or `unavailable`); TinyGPU is the sole device-lifecycle authority, and the service never advances or repairs it. `model_state` remains the separate model lifecycle below. All count fields are non-negative integers. `metrics` is the service-owned timing/transfer projection plus exactly `load_preparation_count`, `warm_prefill_weight_reload_count`, `prefill_count`, `active_request_count`, `resident_bytes`, `resident_bytes_baseline`, and `resource_drift_bytes`; it contains no consumer acceptance or decode counters.
`GetMetrics` passes while the service is available even when no model has ever loaded or the prior model has been unloaded. In that `model_state:"unloaded"` result, `model_handle` is exactly `null`, all process-lifetime counters remain observable, and the current-model resource fields `resident_bytes`, `resident_bytes_baseline`, and `resource_drift_bytes` are exactly zero. In a loaded result (`resident-ready` or `draining`), `model_handle` is the live `mh_[0-9a-f]{32}` handle; `model_handle` is nullable only when `model_state` is `unloaded`. The registry serializes `GetMetrics` with an in-flight `LoadModel` preparation so a snapshot observes either the pre-load `unloaded` state or the committed loaded state, never an uncommitted transient.
The service-owned timing/transfer projection fields are exactly `prefill_elapsed_sec`, `kernel_elapsed_usec`, `transfer_elapsed_sec`, `cache_emit_elapsed_sec`, `total_elapsed_sec`, `tokens_per_sec_prefill`, `transfer_h2d_bytes`, and `transfer_d2h_bytes`, copied from `benchmark.py` where applicable. Cache-import, decode, end-to-end, and accepted/rejected-consumer fields remain in `serving.py`/benchmark records and are not returned by `GetMetrics`.
`load_preparation_count` increments once for each successful explicit `LoadModel` preparation (the initial load and an explicit reload each count); `warm_prefill_weight_reload_count` increments only when a warm `Prefill` reloads weights and does not count an explicit unload/reload. These names and scopes are fixed for the smoke evidence.


### Private persistent native-resource child boundary

The Python service has one private cross-language child boundary in addition to the public `r9700_prefill_service_v1` protocol. Task set 2 creates `native_r9700/native_resource_client.py`. At public-service startup it launches the existing C++ `native_r9700_runner` exactly once with `--model-service-worker` using `subprocess.Popen` and dedicated pipes. Those pipes carry private local JSONL only; the public service's stdin/stdout are never passed to, read by, or written by the child. The child stays alive from service startup through service shutdown and is the sole owner of one native resource generation. There is no socket, network transport, generic RPC framework, shared library, or per-Prefill child launch. The warm path must not call the one-shot `runner.cpp --native-prefill-proof` branch or `subprocess.run`.
Runner selection is explicit rather than discovered. Public `model_service` and benchmark commands require `--native-runner build/native-r9700-runtime/native_r9700_runner`, and task set 4 propagates that exact value to `NativeResourceClient(runner_path=...)`. The constructor rejects a missing option, PATH lookup, fallback default, or `NATIVE_R9700_PREFILL_RUNNER` on the persistent-service path. It `lstat`s the supplied path and rejects symlinks, resolves it to an absolute canonical path, requires a regular file and owner-executable permission, opens that exact file read-only, and hashes its exact bytes before `Popen`. The client records the opened-file identity (`st_dev`, `st_ino`, mode, size, and nanosecond timestamps), rechecks the canonical path and open descriptor before launch, and rejects any changed identity. The child-reported `runner_binary_sha256` must exactly equal that pre-launch hash before `Prepare`/resource acceptance; a mismatch faults/rejects the child boundary. The persistent path performs no PATH search, fallback selection, or environment-variable override.

The private protocol is exactly `r9700_native_resource_v1` and is not an extension of the public protocol. Every request has exactly `{protocol_version, request_id, operation, body}`. Every response has exactly `{protocol_version, request_id, operation, status, result, error}`; `result` is an object (empty on failure), `error` is `null` on `pass` and otherwise exactly `{domain, message, failure_stage}`. The private operation list, in order, is exactly `Prepare`, `Commit`, `Rollback`, `Release`, `Prefill`, `Health`, `Shutdown`. The private reader uses the public 65,536-byte framing and discard/EOF rules, but its pre-decode error envelope is the exact six-key private envelope below and never contains the public `evidence` key. It rejects UTF-8/JSON/duplicate-key/schema failures with that private envelope, recovers only validated correlation fields after decode, and uses bounded non-sensitive error strings. It permits only one in-flight request. The client fails closed on a mismatched response `request_id` or `operation`, a duplicate response/request ID, an unexpected result/error shape, child EOF, or child exit; it never retries a native operation or repairs an accepted prefix after such a fault.
For private frame-size and frame-decode failures, the child emits exactly one six-key response and no other keys:
```json
{
  "protocol_version": "r9700_native_resource_v1",
  "request_id": null,
  "operation": null,
  "status": "error",
  "result": {},
  "error": {
    "domain": "invalid_request",
    "message": "raw frame rejected before decode",
    "failure_stage": "frame_size"
  }
}
```
`failure_stage` is `"frame_size"` for an oversized raw frame and `"frame_decode"` for invalid UTF-8, malformed JSON, or duplicate keys; the fixed message and stage are bounded non-sensitive UTF-8 strings (at most 16 KiB) and never contain frame, prompt, or token bytes. The reader emits one response, discards through the next newline, then continues with the next frame; if EOF occurs while discarding, it emits no second response and exits normally. A failed error-envelope write exits rather than retrying. This private envelope is never augmented with `evidence`.

The private schemas are exact and contain no additional fields. `ResourceSpec` is the immutable object already assembled by task set 2: `{model_uri, model_digest, model_fingerprint, cache_capacity, kernel_pack, resource_budget}`, where `cache_capacity` is `{batch:1,prefix_positions:128}`, `kernel_pack` is `{name,version,digests}` with ordered lowercase `sha256:<64 hex>` digests, and `resource_budget` is `{resident_bytes_max,scratch_bytes_max,total_bytes_max}` of unsigned limits. Native `PreparedResources` and committed `ResidentResources` are opaque child-held values and are never serialized. The operation bodies and successful result objects are:

| Operation | Exact body | Exact `pass` result |
|---|---|---|
| `Prepare` | `{"resource_spec":<ResourceSpec>}` | `{"resource_generation":<uint64>,"state":"prepared","producer_fingerprint":"sha256:<64 lowercase hex>","runner_binary_sha256":"sha256:<64 lowercase hex>"}` |
| `Commit` | `{"resource_generation":<uint64>}` | `{"resource_generation":<uint64>,"state":"resident-ready","producer_fingerprint":"sha256:<64 lowercase hex>"}` |
| `Rollback` | `{"resource_generation":<uint64>}` | `{"resource_generation":<uint64>,"state":"released","already_released":<bool>}` |
| `Release` | `{"resource_generation":<uint64>}` | `{"resource_generation":<uint64>,"state":"released","already_released":<bool>}` |
| `Prefill` | `{"resource_generation":<uint64>,"request_id":<safe request ID>,"token_ids":[<uint32>...],"prefill_npz_path":<string>,"hardware_log_path":<string>}` | `{"resource_generation":<uint64>,"producer_fingerprint":"sha256:<64 lowercase hex>",<accepted native evidence fields>}` |
| `Health` | `{}` | `{"child_state":"ready","resource_generation":<uint64>|null,"resource_state":"none"|"prepared"|"resident-ready"|"release-failed","producer_fingerprint":"sha256:<64 lowercase hex>"|null,"error_summary":{"domain":<string>,"message":<bounded string>,"failure_stage":<bounded string>}|null}` |
| `Shutdown` | `{}` | `{"state":"shutdown"}` |

The `<accepted native evidence fields>` in a successful `Prefill` result are exactly `native_prefill_acceptance`, `native_prefill_full_layer_loop_status`, `runtime_substrate`, `hardware_log_path`, `compute_completion_policy`, `compute_barrier_policy`, `prefill_npz_path`, `kernel_count`, `transfer_bytes`, `block_tokens`, `block_count`, `failure_stage`, `exit_status`, and `failure_text`; no public-only field or second evidence schema is introduced. The child requires `resource_generation` to be the committed generation, uses the caller-supplied exclusive service artifact paths, and never receives a model path on Prefill. `Prepare`/`Commit`/`Prefill` identity fields are immutable for the generation. `Shutdown` is sent only after the public registry has completed teardown; after its response is flushed the child exits normally.

Private `Prefill.token_ids` contains only the producer-owned prefix and has length `N=0..128`; the public registry removes the final prompt token before the private call. The consumer-owned final token never crosses this boundary or appears in the native prompt-cache artifact.

`hardware_log_path` in the private body/result is the same service-owned artifact path exposed as `prefill_log_path` in the public cache projection; `request_id` in the body is the public request ID, while the private envelope ID remains the private correlation ID. The client passes only canonical service-created exclusive paths and the child never derives an artifact path from an unvalidated value.

### Identifier rules

- `request_id` preserves the existing `serving.py` character/path validator: `_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")` and `_safe_request_id` rejects empty values, NUL, `/`, `\\`, `.` and `..`. Protocol v1 adds a hardening cap of 1–128 ASCII characters; this 128-character cap is a new task-set-2 RED contract and required `_safe_request_id` change, not an existing source limit. The service must not widen the character set or add a second request-ID grammar. Every client `request_id` is process-lifetime unique across every operation (including failed requests and non-`Prefill` operations): the service reserves it before dispatch, rejects reuse as `blocked`/`invalid_request` at `request_id_reuse`, and echoes a valid ID unchanged. It uses the ID for the existing `_producer_artifact_paths` names, but creates every artifact exclusively (no overwrite); an unexpected path collision is a terminal artifact-creation error.
- `model_handle` is service-generated and matches `mh_[0-9a-f]{32}`. The suffix is 16 bytes (128 bits) obtained from a cryptographically secure random source; it is not derived from `model_uri`, `request_id`, a digest, or a filesystem path. The registry checks the live-handle map before publication; an accidental collision causes regeneration, never replacement. Handles are not exposed as paths, GPU addresses, or parseable selectors, and are never reused while a process is alive. A malformed/stale handle is rejected with `invalid_request`.
- `model_digest` and `producer_fingerprint` are `sha256:` followed by 64 lowercase hexadecimal characters. `model_digest` covers the verified model/config identity; `producer_fingerprint` identifies the native producer evidence used for the cache.

### Common status/error domains

`status` is exactly one of `pass`, `blocked`, or `error`, preserving the current worker/serving vocabulary (`native_prefill_acceptance`, serving `status`, and benchmark `gate_result` remain separate fields):

- `pass`: the requested operation completed; `LoadModel` has reached `resident-ready` and a cache is only called accepted after the consumer gate below;
- `blocked`: validation, lifecycle policy, timeout, or fail-closed evidence refusal;
- `error`: execution/device failure or a post-acceptance terminal failure.

`error` is exactly `{domain, message, failure_stage}`. `domain` is exactly one of the `docs/DESIGN.md` error distinctions: `invalid_request`, `unsupported_capability`, `executable_rejection`, `resource_exhaustion`, `timeout`, `device_lost_or_faulted`, `numerical_rejection`, `cache_rejection`, or `consumer_decode_failure`. A stale/unknown model handle is `invalid_request`; no second lifecycle-specific error domain is introduced. `message` and `failure_stage` are bounded UTF-8 strings by `native_worker.py`'s `_MAX_STRING_EVIDENCE_BYTES` (16 KiB) rule and must not contain token/prompt values. Successful responses have `error=null`.

### Frozen input/error mapping

| Raw frame over `MAX_FRAME_BYTES` | `blocked` / `invalid_request` / `frame_size` |
| Invalid UTF-8 or malformed JSON, including duplicate JSON keys | `blocked` / `invalid_request` / `frame_decode` |
| Unknown top-level field, wrong envelope type, or non-object body | `blocked` / `invalid_request` / `envelope_validation` |
| Unsupported protocol version | `blocked` / `invalid_request` / `protocol_version` |
| Unknown operation | `blocked` / `invalid_request` / `operation_validation` |
| Malformed operation body or invalid field type | `blocked` / `invalid_request` / `operation_validation` |
| Unsupported request ID syntax or length | `blocked` / `invalid_request` / `request_id_validation` |
| Reused process-lifetime request ID | `blocked` / `invalid_request` / `request_id_reuse` |
| Exclusive artifact collision | `blocked` / `invalid_request` / `artifact_creation` |
| `token_ids` element is not an unsigned 32-bit integer | `blocked` / `invalid_request` / `token_validation` |
| `S` is outside `1..129` | `blocked` / `invalid_request` / `token_bounds` |
| Caller digest does not match the verified canonical bytes | `blocked` / `invalid_request` / `model_digest_verification` |
| Unsupported model format or quantization | `blocked` / `unsupported_capability` / `load_capability` |
| Capability unavailable because the observed device is not ready | `blocked` / `device_lost_or_faulted` / `device_state` |
| Any other unavailable capability | `blocked` / `unsupported_capability` / `capability_check` |
| The sole model slot is occupied, including `validating`, `preparing`, `resident-ready`, or `draining` | `blocked` / `resource_exhaustion` / `model_capacity` |
| Stale/malformed handle | `blocked` / `invalid_request` / `handle_lookup` |
| Operation forbidden by the handle's model state | `blocked` / `invalid_request` / `model_state` |
| Drain deadline expires before teardown completion | `blocked` / `timeout` / `drain_timeout` |
| Prefill request deadline expires before production | `blocked` / `timeout` / `prefill_timeout` |
| Device/executable/numerical failure after dispatch | `error` / `device_lost_or_faulted`, `executable_rejection`, or `numerical_rejection` / the exact native failure stage |
| Cache metadata/evidence fails before acceptance | `blocked` / `cache_rejection` / `cache_validation` |
| Consumer decode fails after cache acceptance | `error` / `consumer_decode_failure` / `consumer_decode` |

## Model identity and cache specification

### Load/fingerprint

`model_uri` is resolved by task set 2 to the canonical local model directory containing `model.safetensors` or `model.safetensors.index.json` whose entries name only sibling `.safetensors` shards. Task set 2 verifies this path, the file inventory, and the caller digest before native allocation, and owns the canonical path lifetime for the resource preparation/committed model lifetime. Task set 3 receives that canonical path through `resource_spec` and opens it read-only; it never resolves, replaces, deletes, or otherwise owns the caller's model path. `format` is exactly `safetensors`; `quantization` is exactly `fp16` for the first F1 target. The binder's `bind_tensor`/`bind_llama_layer0`/`bind_llama_stage_layer` checks establish the 16-layer Llama geometry and F16 file-backed ranges without decoding tensor payloads.

`model_fingerprint` is exactly:

```json
{
  "model_digest": "sha256:<64 lowercase hex>",
  "format": "safetensors",
  "quantization": "fp16",
  "model_family": "llama",
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "geometry": {
    "num_layers": 16,
    "num_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 131072
  },
  "rms_norm_eps": 0.00001,
  "rope_theta": 500000.0,
  "rope_scaling": {
    "rope_type": "llama3",
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192
  }
}
```

Before any native allocation, the service recomputes the caller-supplied `model_digest` from the canonical JSON bytes of the input object below and compares it byte-for-byte with the request value. The canonical input object is:

```json
{
  "config": {
    "architectures": ["LlamaForCausalLM"],
    "geometry": {
      "num_layers": 16,
      "num_heads": 32,
      "n_kv_heads": 8,
      "head_dim": 64,
      "hidden_size": 2048,
      "intermediate_size": 8192,
      "vocab_size": 128256,
      "max_position_embeddings": 131072
    },
    "model_family": "llama",
    "model_type": "llama",
    "rms_norm_eps": 0.00001,
    "rope_scaling": {
      "rope_type": "llama3",
      "factor": 32.0,
      "high_freq_factor": 4.0,
      "low_freq_factor": 1.0,
      "original_max_position_embeddings": 8192
    },
    "rope_theta": 500000.0
  },
  "files": [
    {"path": "config.json", "size": "<uint64>", "sha256": "<64 lowercase hex>"},
    {"path": "<verified relative model/index/shard path>", "size": "<uint64>", "sha256": "<64 lowercase hex>"}
  ],
  "format": "safetensors",
  "model_family": "llama",
  "quantization": "fp16",
  "shard_index": {
    "index_path": "<relative index path or null>",
    "members": [
      {"shard": "<relative shard path>", "tensor_name": "<verified tensor name>"}
    ]
  }
}
```

The angle-bracket values above are typed placeholders, not literal bytes. `files` contains exactly `config.json`, the unindexed `model.safetensors` or the index plus every referenced `.safetensors` shard, with POSIX relative paths only, raw byte sizes, and raw lowercase SHA-256 values; paths are sorted bytewise. `shard_index.index_path` is the exact relative string `model.safetensors.index.json` when that index is present, otherwise JSON `null`. `shard_index.members` contains every verified tensor-to-shard membership from the index, or every verified safetensors-header tensor mapped to `model.safetensors` when no index exists; members are sorted by `(tensor_name, shard)`.
RFC 8785 JSON Canonicalization Scheme (JCS) is the sole serialization rule for this object: object keys use RFC 8785 UTF-16 code-unit ordering, arrays retain the prescribed orders, JSON numbers use the RFC 8785/ECMAScript canonical lexical form (including `-0` as `0`), strings use the RFC 8785 JSON escaping rules, and the result is UTF-8 bytes with no BOM, insignificant whitespace, or trailing newline. Non-finite numbers (`NaN`, `Infinity`, and `-Infinity`) are rejected before canonicalization. The service sets `model_digest = "sha256:" + SHA256(canonical_bytes)`. No absolute path or mtime is included. A mismatch is `blocked`/`invalid_request` at `model_digest_verification`.

Task set 2 RED tests must pin this checked JCS fixture, including its bytes and digest (the fixture is independent of the model inventory but exercises canonical numbers, key ordering, and UTF-8 strings):

```json
{"z":-0.0,"a":1e-5,"n":500000.0,"u":"é"}
```

Its exact canonical UTF-8 bytes, with no trailing newline, are:

```text
{"a":0.00001,"n":500000,"u":"é","z":0}
```

The corresponding UTF-8 hex bytes are `7b2261223a302e30303030312c226e223a3530303030302c2275223a22c3a9222c227a223a307d`, and the expected SHA-256 is `sha256:a5f32101f172484252004bacdcb9b2f194e82948b19be1634ffd6a39d60a65fd`.

The fingerprint is immutable after `resident-ready`; `kernel_pack_digests` is a separate ordered array of immutable selected-pack `sha256:<64>` identities because the model identity and executable-pack identity have separate owners. `model_digest` is the SHA-256 of the canonical model/config inventory defined above, not an implementation choice left to task set 2. This preserves the design requirement for verified model/config identity without adding a second model abstraction.

### Prefill/cache

`token_ids` is an array of exact unsigned 32-bit integers from `serving.py::_coerce_token_ids`; if `S=len(token_ids)`, v1 accepts exactly `1 ≤ S ≤ 129` for the resident prefix capacity `N=128`. The producer offloads the first `N=S-1` positions and the consumer receives only the final prompt token after import; `S=1` therefore has `N=0`, while `S=129` has `N=128`. The array length is never capped at 128.

`cache_spec` is exactly `{schema_version, cache_class, transport}` with values:

```json
{
  "schema_version": "mlx_lm_prompt_cache_v1",
  "cache_class": "KVCache",
  "transport": "file"
}
```

The response `cache` is exactly the existing serving artifact/evidence projection plus canonical metadata:

```json
{
  "prompt_cache_path": "<service-artifact-root>/<request-id>.prompt-cache.safetensors",
  "metadata": {
    "schema_version": "mlx_lm_prompt_cache_v1",
    "producer_fingerprint": "sha256:<64 lowercase hex>",
    "producer_kind": "r9700_native",
    "model_digest": "sha256:<64 lowercase hex>",
    "num_layers": 16,
    "n_kv_heads": 8,
    "offset": N,
    "head_dim": 64,
    "batch": 1,
    "sequence_length": N,
    "dtype": "float16",
    "physical_layout": "B,H,S,D",
    "absolute_start_position": 0,
    "absolute_end_position": N,
    "rope_theta": 500000.0,
    "rope_scaling": {
      "rope_type": "llama3",
      "factor": 32.0,
      "high_freq_factor": 4.0,
      "low_freq_factor": 1.0,
      "original_max_position_embeddings": 8192
    },
    "cache_class": "KVCache",
    "cache_variant": "llama3.2_1b_fp16",
    "request_id": "<request-id>",
    "meta_state": [
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      ""
    ]
  },
  "prefill_npz_path": "<service-artifact-root>/<request-id>.prefill.npz",
  "prefill_log_path": "<service-artifact-root>/<request-id>.prefill.log",
  "kv_cache_log_path": "<service-artifact-root>/<request-id>.kv-cache.log",
  "payload_digest": "sha256:<64 lowercase hex>",
  "payload_length_bytes": "<positive uint64>"
}
```

`model_fingerprint` is always an object in protocol results; cache metadata does not overload that name with a scalar. Its scalar digest is named `model_digest` and must equal `model_fingerprint.model_digest`; the cache's `rope_theta` and `rope_scaling` must equal the loaded fingerprint exactly. `meta_state` has exactly 16 entries, one empty string per layer in layer order. In the pinned mlx-lm safetensors metadata this is serialized as exactly the keys `0.0` through `0.15`, each with the empty value `""`, with no missing, non-empty, or extra per-layer value. Task set 4 owns the canonical writer in `kv_cache.py` and the corresponding validator in `serving.py`.

`prompt_cache_path`, `prefill_npz_path`, `prefill_log_path`, and `kv_cache_log_path` are the names produced by `serving.py::_producer_artifact_paths`; the service, not the caller, roots them under its artifact directory and creates them without overwrite. The typed `metadata` descriptor and its flattened safetensors header must pass `serving.py::_validate_prompt_cache`: 16 ordered `KVCache` layers, K/V shapes `(1,8,N,64)`, finite values, layer offset/size `N`, integer geometry, exact absolute position/RoPE/model identity, and exactly the 16 empty `meta_state` values above.
The response-level `metadata` is a typed descriptor; the safetensors header is a flat `dict[str,str]` and never contains nested objects, arrays, booleans, or JSON `null` values. Task set 4 must emit the exact F1 identity keys `schema_version`, `producer_kind`, `producer_fingerprint`, `model_digest`, `request_id`, `num_layers`, `batch`, `n_kv_heads`, `sequence_length`, `head_dim`, `absolute_start_position`, `absolute_end_position`, `rope_theta`, and `rope_scaling`, plus the pinned per-layer cache keys. String identities (`schema_version`, `producer_kind`, `producer_fingerprint`, `model_digest`, and `request_id`) use their exact scalar strings. `num_layers`, `batch`, `n_kv_heads`, `sequence_length`, `head_dim`, `absolute_start_position`, `absolute_end_position`, and `offset` use canonical unsigned decimal strings (zero is exactly `"0"`; no sign, decimal point, exponent, leading zero, or surrounding whitespace). `rope_theta` is the RFC 8785 JCS number token encoded as a string (`"500000"` for this model), and `rope_scaling` is the RFC 8785 canonical JSON object encoded as one UTF-8 JSON string (`"{\"factor\":32,\"high_freq_factor\":4,\"low_freq_factor\":1,\"original_max_position_embeddings\":8192,\"rope_type\":\"llama3\"}"` for this model). `dtype`, `physical_layout`, `cache_class`, and `cache_variant` are exact scalar strings. The logical `meta_state.0` through `meta_state.15` entries are all the empty string; the pinned mlx-lm flat keys are exactly `0.0` through `0.15`, each `""`, with no missing, non-empty, or extra per-layer entry. If a future optional boolean is present, its only accepted strings are lowercase `"true"` and `"false"`; F1 emits no boolean metadata. Pinned structural `KVCache` class/offset keys remain flat scalar strings as required by the adapter, and no nested identity object is permitted.


## Evidence and lifecycle

### Exact producer evidence

Protocol `evidence` is the existing producer evidence object with one mandatory F1 identity extension, `producer_fingerprint`; it is not a second evidence schema. Its required fields are exactly the existing `native_worker.py::_REQUIRED_FIELDS` plus `producer_fingerprint`:

- `producer_kind`;
- `producer_fingerprint`;
- `native_prefill_acceptance`;
- `native_prefill_full_layer_loop_status`;
- `runtime_substrate`;
- `hardware_log_path`;
- `compute_completion_policy`;
- `compute_barrier_policy`;
- `prefill_npz_path`;
- `kernel_count`;
- `transfer_bytes`;
- `block_tokens`;
- `block_count`;
- `failure_stage`;
- `exit_status`.

`failure_text` is also carried because `native_worker.py::_PARSED_FIELDS` and `_write_result_log` emit it. The only permitted diagnostic additions beyond this fixed identity extension are the exact names in `_OPTIONAL_EVIDENCE_FIELDS` (including layer-0/resident-subgraph/KV-projection source/status fields); no other evidence key is invented by F1. Numeric types/ranges are exactly `_INTEGER_ABI_RANGES`; all other evidence values follow `_STRING_EVIDENCE_FIELDS` and the 16 KiB string bound. `producer_fingerprint` is finite/known by construction, must match the committed handle and private request result, and is never omitted from an accepted native Prefill evidence object.

For acceptance, `native_prefill_acceptance=pass`, `native_prefill_full_layer_loop_status=pass`, `runtime_substrate=TinyGPU.app/APLRemotePCIDevice/PCIIface`, matching existing hardware/NPZ paths, `compute_completion_policy=terminal`, `compute_barrier_policy=full`, `exit_status=0`, nonzero `kernel_count` and `transfer_bytes`, exact `block_tokens`/`block_count`, and strict finite F16 NPZ schema are required by `native_worker.py::_acceptance_problems` and `validate_native_prefill_npz`. `serving.py::_native_prefill_evidence_problems` independently requires `producer_kind=r9700_native`, `native_prefill_acceptance=pass`, matching readable hardware log and NPZ paths, and nonzero kernel/transfer evidence. This preserves the distinction between `cpu_reference` evidence and request-bound `r9700_native` evidence.

The benchmark projection remains the existing `native_r9700_benchmark_v1` row: `_REQUIRED_ROW_FIELDS` (`prompt_name`, `prompt_tokens`, `producer_kind`, `gate_result`, all timing/transfer fields, `baseline_name`, `speedup_vs_baseline`, and `row_role`) plus the source's optional route/cache/fallback/hardware/token-exact fields. `validate_benchmark_row` requires native rows to have `gate_result=pass`, `accepted_cache=true`, `route=native_producer`, no fallback, a hardware log, token-exact evidence, positive prefill/kernel/total timings, and nonzero transfers. Task set 5 adds explicit `cold_process`, `warm_prefill`, and `gpu_compute` scope labels, ten raw warm sample records, three separate aggregate scope records, per-scope/total counts, and no-warm-reload accounting while retaining those existing fields.

### Model lifecycle

The only model states are exactly `unloaded → validating → preparing → resident-ready → draining → unloaded` from `docs/DESIGN.md`:

- `LoadModel` enters `validating`; no handle is visible until `resident-ready`.
- F1 has exactly one model slot. It is occupied from the first `validating` transition through `preparing`, `resident-ready`, and `draining`; every second `LoadModel`, regardless of digest, is blocked with `resource_exhaustion/model_capacity` until the sole handle reaches `unloaded`. `loaded_model_count` is therefore always `0` or `1`.
- Validation failure or any preparation/upload/executable/allocation failure unwinds to `unloaded`; partial resources are not reachable.
- A successful preparation commits one handle owning verified identity, resident/prepacked weights, selected pack identities, scratch/reusable request buffers, KV policy/capacity (`N=128` prefix positions), graph variants, and quantization metadata.
- `Prefill` is legal only in `resident-ready` and never reloads weights.
- `UnloadModel` enters `draining`, refuses new requests, waits for active requests using the fixed service-side `UNLOAD_DRAIN_TIMEOUT_MS=30000` (30 seconds), and releases dependent resources exactly once. A timeout leaves the handle `draining` and returns `blocked/timeout`; it does not cancel secretly or create a replacement handle. A repeated `UnloadModel` for that same draining handle joins the same teardown, waits on the same active-request set for the same bounded interval, and observes the same `release_once` result; a later repeat may complete after the earlier timeout. New loads remain blocked until teardown reaches `unloaded`.
- Every stale/malformed handle request is refused with `blocked/invalid_request`; there is one registry/lifecycle owner, not a second service abstraction.

`ResidentMemory::allocate`/`release_all` own page-rounded resident mappings and reverse-order release/quarantine. Task set 3 exposes only the native `prepare`, `commit`, `rollback`, and idempotent `release_once` resource primitives plus resource counters; task set 2's `ModelRegistry` alone owns model states, active-request exclusion, draining, the fixed timeout, and repeat-unload policy.

### Native child ownership, cleanup, and fault semantics

The child owns exactly one generation at a time. `Prepare` allocates and returns an opaque prepared generation; every partial allocation, mapping, upload, scratch buffer, or binder state is self-cleaned before a Prepare error, so no cleanup token is returned on that error. `Commit` consumes that prepared value and publishes `resident-ready`; a failed commit self-cleans its consumed value. `Rollback` is for a successful Prepare that the caller abandons before Commit. `Release` is for a committed resident generation. Both cleanup operations are idempotent: the first successful pass returns `already_released:false`, and a repeat of the same operation for the same generation returns `already_released:true` with the exact same `resource_generation` and `state:"released"`. Generation numbers are monotonic and never reused.

On a cleanup error, the response is `status:"error"`, `result:{}`, and `error:{domain,message,failure_stage}`; the child retains ownership in `resource_state:"release-failed"`. While in `release-failed`, the only allowed operations are read-only `Health` and a retry of the matching cleanup operation (`Rollback` or `Release`) for the same `resource_generation`; every other operation, including `Shutdown`, is rejected until cleanup passes. `Health` remains a successful read-only observation and reports the retained `resource_state`, `resource_generation`, and bounded `error_summary` (`domain`, `message`, `failure_stage`). The Python handle remains `draining`, new `LoadModel` is blocked, and the registry publishes `unloaded` only after a cleanup pass. A child crash or device loss faults the public service, leaves no accepted-prefix repair or fallback path, and requires a process restart; the service does not respawn the child inside the same process.

### Producer fingerprint binding

`producer_fingerprint` is `"sha256:"` plus lowercase SHA-256 over the UTF-8 bytes of the RFC 8785 JCS serialization of this exact object (the displayed member order is descriptive; JCS UTF-16 key ordering determines the bytes):

```json
{
  "domain": "r9700-producer-fingerprint-v1",
  "protocol_version": "r9700_native_resource_v1",
  "runner_binary_sha256": "sha256:<64 lowercase hex>",
  "ordered_kernel_pack_sha256": ["sha256:<64 lowercase hex>", "..."],
  "target": "gfx1201",
  "runtime_substrate": "TinyGPU.app/APLRemotePCIDevice/PCIIface",
  "completion_policy": "terminal",
  "barrier_policy": "full",
  "device_identity": {
    "vendor_id": "1002",
    "device_id": "7551"
  }
}
```

The runner computes `runner_binary_sha256` over the exact executable image running the child and computes/publishes the fingerprint in the successful `Prepare` result. `ordered_kernel_pack_sha256` preserves the selected pack order; it is not sorted, replaced, or inferred by the client. The preimage contains no model, request, filesystem path, timestamp, or timing value. Unknown, missing, non-finite, malformed, or extra fields reject before hashing. Python stores the returned fingerprint in the pending/native handle and requires the same value from `Commit`; every native `Prefill` result/evidence and cache metadata repeats it. The consumer accepts a cache only when that value is byte-for-byte equal to the committed handle fingerprint and the request-bound native evidence fingerprint; absent/unknown/mismatched identity is `blocked/cache_rejection/cache_validation` and cannot fall back after acceptance.


### CaptureTrace

`CaptureTrace` with body `{}` writes a bounded JSON trace artifact and returns `trace_format=json`, `trace_path`, and a `snapshot` captured at call time. The snapshot is an aggregate service/registry view containing only bounded service-owned fields (`service_available`, `service_unavailable_reason`, observed `device_state`, current `model_state`, `loaded_model_count`, `active_request_count`, the service metrics counters/timings, and `last_failure_stage`) and is itself limited by `MAX_FRAME_BYTES`. It contains no token IDs, prompts, previous-request/native-evidence claim, or request selector. The common response `request_id` identifies this `CaptureTrace` call; request-specific hardware logs and native evidence remain exclusively in each `Prefill` response.

### Request lifecycle and fallback

The only request states are exactly `received → validated → queued → running → produced → adapter-validating → accepted|rejected`, from `docs/DESIGN.md` §Prefill request lifecycle. Task set 2/model service owns `received` through `produced`, including active-request accounting and the unload drain policy; task set 4/`serving.py` owns adapter validation and the consumer terminal state, including cache `meta_state` validation. Malformed input, stale handle, timeout, resource/device/numerical failure, or cache rejection is terminal `rejected` with the exact error mapping above. Fallback is legal only before `accepted`, as implemented by `serving.py::_with_fallback`; after `accepted`, consumer decode failure is terminal and must not recompute the prefix or invoke fallback.

## Ownership matrix

| Task set | Owns | Must not own |
|---|---|---|
| 2 — local protocol, registry, and private client | `native_r9700/service_protocol.py`, `native_r9700/model_service.py`, and new `native_r9700/native_resource_client.py`; the public `r9700_prefill_service_v1` envelope, `ModelRegistry`, one-slot lifecycle, active requests, drain/retry policy, ResourceSpec verification/assembly, private-child `Popen` lifetime, and Python-side fingerprint binding. Extend `tests/native_r9700/test_service_protocol.py` and `test_model_service.py`. | Native allocations, C++ runner/worker implementation, public-stdio sharing, sockets/network, or a second service lifecycle. |
| 3 — native model-resource lifetime and private worker | `native_r9700/runner.cpp`, `native_r9700/runtime.h/.cpp`, existing `resident_memory.*` and `model_weight_binder.*`, and new narrowly named `native_r9700/native_resource_worker.h/.cpp`; implement `--model-service-worker`, the exact `r9700_native_resource_v1` JSONL loop, native resource ownership, cleanup, and producer-fingerprint computation. Update the existing runner build `RUNNER_SOURCES`/validation source list to include `native_resource_worker.cpp` while retaining one runner binary; extend the native lifecycle tests. | Public protocol, Python registry/model states, active-request waiting, drain timeout/retry policy, generic RPC/network, WMMA pack schema, HAL migration, or a second executable/service. |
| 4 — worker/consumer integration | `native_r9700/native_worker.py` and `native_r9700/serving.py` wire public Load/Unload/Prefill/Health/Shutdown through task set 2's client; retain `kv_cache.py`'s canonical writer and `serving.py`'s validator, including exact fingerprint equality and 16 empty `meta_state` values. Extend the four named worker/serving/cache/accounting tests and assert private child-process evidence. | One-shot `subprocess.run`/`--native-prefill-proof` on the warm path, direct C++ pipes on public stdio, native resource ownership, alternate cache schema, network transport, or post-acceptance fallback. |
| 5 — warm smoke/benchmark promotion | `native_r9700/benchmark.py`: `benchmark_row_from_serving_result`, `validate_benchmark_row`, `validate_benchmark_rows`, `build_benchmark_result`, `render_benchmark_report`, `write_benchmark_log`, and CLI `main` add scope labels, explicit load-preparation accounting, raw sample identity, scope aggregates, median/dispersion, and no-warm-reload evidence. Extend only `tests/native_r9700/test_benchmark.py`; write `logs/f1-persistent-worker/` evidence and `.superpowers/swarm/reports/f1-promotion.md`. | No model lifecycle, kernel optimization, block-size change, direct transport, or mixed-scope warm metric. |
### Runner compile/link closure inventory

Task set 3 owns the following independent runner-linked source closures and must add `native_r9700/native_resource_worker.cpp` to each one; no shared build abstraction or centralized list is introduced in this contract phase:

- `tests/native_r9700/test_block_prefill_runtime_contract.py::RUNNER_SOURCES`.
- `tests/native_r9700/test_compute_barrier_policy.py::RUNNER_SOURCES`.
- `tests/native_r9700/test_native_hsa_prefill_contract.py::RUNNER_SOURCES`.
- `tests/native_r9700/test_runtime_lifecycle.py::RUNNER_SOURCES`.
- `tests/native_r9700/test_runtime_llama_embed_contract.py::RUNNER_SOURCES`.
- `tests/native_r9700/test_runtime_protocol.py::RUNNER_SOURCES` and its `compile_runner` path; its separate `c1_transfer_bridge.cpp`/AMDev probe closure does not link `runner.cpp` and remains unchanged.
- `tests/native_r9700/test_runtime_vram_contract.py::RUNNER_SOURCES`.
- `tests/native_r9700/test_gpu_stage_profile_contract.py::FORMAT_PROBE_SOURCES`, because its generated `FORMAT_PROBE_SOURCE` includes `runner.cpp` before linking the listed sources.

The current active ledger has three additional independent `clang++` runner closures, all owned by task set 3: `Current native runner build and no-model smokes` (the `build/native-r9700-runtime/native_r9700_runner` command), `P3 schema`, and `P3 scalar migration`. Each source list must add `native_resource_worker.cpp` while retaining `runner.cpp` as the sole `main`/entrypoint and one output binary. The F1 process-smoke and warm-benchmark blocks invoke that output but do not compile it; both pass the explicit `--native-runner` path below. No current production dynamic/default build list under `native_r9700/` compiles `runner.cpp`; the dynamic bridge command in `runtime.cpp` is a separate AMDev bridge closure. If a production list is introduced that compiles `runner.cpp` or the worker-referencing AMDev closure, task set 3 must add the worker source and a RED source-set contract before use.


### Cross-set native resource interface

Task set 2 assembles this minimal immutable `ResourceSpec` after verification: `{"model_uri":<canonical model-directory string>,"model_digest":"sha256:<64 lowercase hex>","model_fingerprint":<exact model fingerprint object>,"cache_capacity":{"batch":1,"prefix_positions":128},"kernel_pack":{"name":<string>,"version":<string>,"digests":[<ordered sha256 identity>...]},"resource_budget":{"resident_bytes_max":<uint64>,"scratch_bytes_max":<uint64>,"total_bytes_max":<uint64>}}`. It owns URI resolution, digest/fingerprint/geometry/RoPE verification, budget selection, artifact paths, and the model path lifetime. Task set 3 receives this object once over `Prepare`, opens the canonical path read-only, and owns opaque native prepared/committed values; it never resolves, replaces, deletes, or otherwise owns the caller's path.

`native_r9700/native_resource_client.py` is the only cross-language caller. It maps the public lifecycle to the exact private `r9700_native_resource_v1` operations and never serializes `PreparedResources` or `ResidentResources`; the child retains those values under `resource_generation:uint64`. `Prepare` returns `state:"prepared"` plus the computed producer fingerprint, `Commit` returns `state:"resident-ready"` plus the same fingerprint, and `Prefill` returns the current accepted evidence fields plus generation/fingerprint. `Rollback` and `Release` return exactly `{resource_generation:uint64,state:"released",already_released:bool}` on pass or `{domain,message,failure_stage}` in `error`; a cleanup error retains `release-failed` ownership and Python `draining` until a same-operation/generation retry passes. This is the sole native resource boundary; the public registry owns lifecycle policy and no C++ code owns public protocol/model states.

F2 owns WMMA-specific source/image contracts and P3 owns generic Kernel Pack records/tooling. Neither F1 task set owns `kernel_assets.cpp`, `kernel_catalog.cpp`, or generated catalogs; F2/P3 must nominate one supervisor-selected integration owner for that shared boundary.

## Active validation ledger blocks

The following two sections mirror the already-reconciled F1 command blocks in `docs/tasks/native-r9700-producer/validation-commands.md`. The commands are frozen and are not run by this agent; active-ledger synchronization is complete. Task set 3 must update every runner-linked source closure enumerated in `§Runner compile/link closure inventory` with `native_resource_worker.cpp` while retaining one `native_r9700_runner` binary; task set 4 must expose the explicit `--native-runner` option. Each command's expected evidence must include the private child process, protocol, generation, and fingerprint observations below.

### F1 persistent process smoke

```sh
mkdir -p logs/f1-persistent-worker/process-smoke
${PY} -m native_r9700.native_worker \
  --smoke-load-unload-reload \
  --model mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-128 \
  --samples 10 \
  --producer-kind r9700_native \
  --native-runner build/native-r9700-runtime/native_r9700_runner \
  --artifacts-dir logs/f1-persistent-worker/process-smoke/artifacts \
  --json logs/f1-persistent-worker/process-smoke/result.json \
  --log logs/f1-persistent-worker/process-smoke/run.log \
  --trace logs/f1-persistent-worker/process-smoke/trace.json
```

Expected evidence: one public Python service process launches exactly one `native_r9700_runner --model-service-worker` child at service startup and keeps that child PID alive through final shutdown; the public stdin/stdout and private child pipes are distinct. The child performs `Prepare → Commit` for each explicit load generation and ten `Prefill` operations in the resident generation, then `Release → released` before the public `draining → unloaded` transition; there is no per-request launch, `subprocess.run`, socket, or one-shot `--native-prefill-proof` call. The child log/result records `r9700_native_resource_v1`, one-in-flight correlation, the generation, `runner_binary_sha256`, ordered pack digests, and the computed `producer_fingerprint`. The smoke also performs an initial `LoadModel → validating → preparing → resident-ready`, ten independent `prompt-128` requests with `S=129` and `N=128`, `UnloadModel → draining → unloaded`, a second explicit `LoadModel → validating → preparing → resident-ready`, and final unload. The result must show `load_preparation_count=2` (the initial load plus the explicit reload), `raw_warm_sample_count=10`, `warm_prefill_weight_reload_count=0` across those ten warm Prefills, no resource drift, no fallback after acceptance, and no stale request/model association. Every accepted request must have `producer_kind=r9700_native`, `native_prefill_acceptance=pass`, `native_prefill_full_layer_loop_status=pass`, the exact runtime substrate, an existing request-bound hardware log, nonzero kernel/transfer evidence, strict S-1 cache metadata including the 16 empty `meta_state` values, an exact handle/evidence/cache `producer_fingerprint` match, empty `failure_stage`, and `exit_status=0`.

### F1 warm benchmark promotion

```sh
mkdir -p logs/f1-persistent-worker/warm
${PY} -m native_r9700.native_worker \
  --warm-prefill-samples \
  --model mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --prompt-name prompt-128 \
  --samples 10 \
  --producer-kind r9700_native \
  --native-runner build/native-r9700-runtime/native_r9700_runner \
  --artifacts-dir logs/f1-persistent-worker/warm/artifacts \
  --json logs/f1-persistent-worker/warm/serving.json \
  --log logs/f1-persistent-worker/warm/worker.log \
  --trace logs/f1-persistent-worker/warm/trace.json
${PY} -m native_r9700.benchmark \
  --model mlx_models/meta-Llama-3.2-1B-Instruct \
  --fixtures-dir tests/native_r9700/fixtures \
  --artifacts-dir logs/f1-persistent-worker/warm/artifacts \
  --json logs/f1-persistent-worker/warm/benchmark.json \
  --report logs/f1-persistent-worker/warm/benchmark.md \
  --log logs/f1-persistent-worker/warm/benchmark.log \
  --producer-kind r9700_native \
  --serving-result logs/f1-persistent-worker/warm/serving.json
```

Expected evidence: one public Python service process and one persistent private `native_r9700_runner --model-service-worker` child provide all ten warm requests; the child is not relaunched between requests, its private pipes are not the public service pipes, and the warm path never invokes one-shot `--native-prefill-proof`, `subprocess.run`, TCP, or a fallback after acceptance. The worker JSON has exactly ten raw warm samples for the concrete `prompt-128` fixture, each with `S=129` and `N=128`, all accepted with `route=native_producer`, `accepted_cache=true`, empty `fallback_reason`, token-exact evidence, request-bound hardware logs, `warm_prefill_weight_reload_count=0`, and exact equality among committed-handle, private-Prefill evidence, and cache-metadata `producer_fingerprint`. The child process evidence records `r9700_native_resource_v1`, one generation, ten `Prefill` operations, the immutable runner binary SHA, ordered kernel-pack SHA values, and the JCS-derived fingerprint. The benchmark JSON/report/log has ten raw warm sample records plus exactly three aggregate scope records (`cold_process`, `warm_prefill`, and `gpu_compute`), for `raw_warm_sample_count=10`, `scope_aggregate_count=3`, and `total_record_count=13`; with the raw samples labeled `warm_prefill`, `records_by_scope` is `{"cold_process":1,"warm_prefill":11,"gpu_compute":1}`. No one-time load time is included in a warm-prefill timing. Every record retains the existing `native_r9700_benchmark_v1` required timing/transfer/correctness fields and `row_role=native_benchmark`; aggregate records carry their explicit scope and aggregate identity. The 2026-08-25 B4 observation (18.012 seconds / 7.11 prefix tok/s for prompt-128) is recorded as the first comparator, not an automatic threshold; promotion requires no warm reload, no resource drift, cache integrity, fail-closed behavior, exact-token evidence, and persistent-child/fingerprint evidence.

## Review corrections

This section explicitly addresses every Important finding in `agent://F1FreezeReview`; anchors are the exact report headings/fields where each correction is frozen.

1. **Bound JSONL frames and token arrays.** Anchor: `§Frozen protocol v1 > Envelope and operations` (`MAX_FRAME_BYTES`) and `§Frozen input/error mapping` (`frame_size`, `token_bounds`). Correction: raw frames are capped at 65,536 bytes before decode, and `token_ids` accepts only `S=1..129` with `N=S-1≤128`; oversize and out-of-range input map to bounded `blocked`/`invalid_request` errors.
2. **Enforce the single-resident-model limit.** Anchor: `§Evidence and lifecycle > Model lifecycle` (one model slot, `loaded_model_count`). Correction: capacity is exactly one across `validating`, `preparing`, `resident-ready`, and `draining`; every second load blocks until `unloaded`.
3. **Define the canonical model-digest bytes.** Anchor: `§Model identity and cache specification > Load/fingerprint` (canonical input object, JCS fixture, and `model_digest`). Correction: the caller digest is verified against RFC 8785 JCS canonical UTF-8 bytes over the prescribed relative file inventory, geometry/config/RoPE identity, and shard/index membership; JCS fixes UTF-16 key ordering, number/string forms, no whitespace/newline, and non-finite rejection, while absolute paths and mtimes are excluded.
4. **Bind RoPE into one model fingerprint.** Anchor: `§Load/fingerprint` (`model_fingerprint`) and `§Prefill/cache` (`model_digest`, `rope_theta`, `rope_scaling`). Correction: `model_fingerprint` is one object, cache metadata uses the scalar `model_digest` name, and exact Llama-3 RoPE identity is required in both identities.
5. **Validate the pinned cache meta-state.** Anchor: `§Prefill/cache` (`meta_state`) and ownership row `4 — worker/consumer integration`. Correction: exactly 16 empty per-layer values, serialized as `0.0..0.15` empty mlx-lm metadata, are written and validated by task set 4's `kv_cache.py`/`serving.py` changes.
6. **Make request artifact namespaces unique.** Anchor: `§Identifier rules` (`request_id`) and `§Prefill/cache` (artifact paths). Correction: IDs are process-lifetime unique across all operations, echoed unchanged, reserved before dispatch, and every artifact is created exclusively without overwrite.
7. **Keep consumer metrics out of GetMetrics.** Anchor: `§Envelope and operations` (`GetMetrics`, `metrics`). Correction: GetMetrics contains only service-owned load/native-prefill/resource counters and timings; import, decode, and consumer acceptance remain serving/benchmark-owned.
8. **Define unload timeout and retry semantics.** Anchor: `§Model lifecycle` (`UNLOAD_DRAIN_TIMEOUT_MS`, repeated `UnloadModel`). Correction: the service uses a fixed 30-second drain wait; timeout leaves `draining`, and a repeat joins the same teardown and `release_once` result while new loads stay blocked.
9. **Keep draining policy in ModelRegistry.** Anchor: `§Ownership matrix` rows `2 — local protocol and registry` and `3 — native model-resource lifetime`. Correction: task set 2 alone owns active-request/draining/timeout/repeat policy; task set 3 exposes only `prepare/commit/rollback/release_once` and resource counters.
10. **Use the existing prompt-128 fixture.** Anchor: `§Active validation ledger blocks > F1 persistent process smoke` and `> F1 warm benchmark promotion`. Correction: both commands use `--prompt-name prompt-128` and assert `S=129`, `N=128`.
11. **Make benchmark record counts consistent.** Anchor: `§F1 warm benchmark promotion` expected evidence. Correction: ten raw warm samples are joined by three separate aggregate scope records; `raw_warm_sample_count=10`, `scope_aggregate_count=3`, `records_by_scope={"cold_process":1,"warm_prefill":11,"gpu_compute":1}`, and `total_record_count=13`.
12. **Distinguish explicit reload from warm-path reload.** Anchor: `§F1 persistent process smoke` expected evidence and `metrics`. Correction: the smoke expects `load_preparation_count=2` for two explicit preparations and `warm_prefill_weight_reload_count=0` during all ten warm Prefills.
13. **Separate service health from device lifecycle.** Anchor: `Health` result fields and the `service_available`/`device_state` definitions under `§Envelope and operations`. Correction: no `service_state` lifecycle exists; service availability/reason, observed device state, and model state are separate, and the service never advances device state.
14. **Give CaptureTrace a defined target.** Anchor: `CaptureTrace` result fields and `§Evidence and lifecycle > CaptureTrace`. Correction: CaptureTrace is a bounded aggregate service/registry snapshot at call time with no previous-request/native-evidence claim; request-specific hardware logs remain Prefill evidence.
15. **Use a separate pre-decode transport envelope.** Anchor: `§Frozen protocol v1 > Pre-decode transport errors` and `§Parsed request/schema errors`. Correction: public `frame_size` and `frame_decode` emit the exact seven-key envelope with fixed protocol, `request_id:null`, `operation:null`, `status:"blocked"`, `result:{}`, `error.domain:"invalid_request"`, bounded non-sensitive message/stage, and `evidence:null`; one envelope is emitted per rejected frame, then the reader discards to newline and continues or exits once at EOF. Parsed errors recover only validated correlation fields; private predecode uses the separate exact six-key envelope defined under `§Private persistent native-resource child boundary`.
16. **Use RFC 8785 JCS for model digests.** Anchor: `§Model identity and cache specification > Load/fingerprint`. Correction: RFC 8785 fixes UTF-16 key ordering, canonical number and string forms, UTF-8 bytes, no whitespace/newline, and non-finite rejection; task-set-2 RED tests pin the exact fixture bytes and expected `sha256:a5f32101f172484252004bacdcb9b2f194e82948b19be1634ffd6a39d60a65fd`.
17. **Complete the cache meta-state cardinality.** Anchor: `§Prefill/cache` (`meta_state`). Correction: the exact response descriptor has 16 ordered empty strings, and the pinned flat writer/validator requires the complete `0.0` through `0.15` set.
18. **Make preparation and cleanup failure semantics exact.** Anchor: `§Cross-set native resource interface` and `§Native child ownership, cleanup, and fault semantics`. Correction: `Prepare` self-rolls back every partial allocation before returning `{domain,message,failure_stage}` in `error`; `Rollback` and `Release` pass only with `{resource_generation,state:"released",already_released}` and retain `release-failed` ownership on cleanup error until a same-operation/generation retry passes, while read-only `Health` remains allowed to expose state/generation/error summary.
19. **Keep the reconciled F1 commands aligned with the active validation ledger.** Anchor: `§Active validation ledger blocks` and `§Final-freeze correction block`. Correction: the two headings and command blocks below mirror the already-present active-ledger sections, and each expected-evidence paragraph requires one private child PID, private protocol/generation trace, runner SHA, and exact fingerprint equality; the worker does not edit the shared ledger, and only final correction/re-review remains.
20. **Freeze flat safetensors metadata.** Anchor: `§Prefill/cache` (`metadata`) and `§Active validation ledger blocks`. Correction: task set 4 writes a flat string map with the exact identity/geometry/position keys, JCS number/JSON-string encodings, complete empty per-layer keys, lowercase booleans if ever present, and no nested values.
21. **Define unloaded GetMetrics.** Anchor: `§Frozen protocol v1 > Envelope and operations` (`GetMetrics`). Correction: an available unloaded registry passes with `model_handle:null`, `model_state:"unloaded"`, process-lifetime counters, and zero current-model resource fields; loaded snapshots carry the live handle, so nullability exists only for `unloaded`.
22. **Specify the native ResourceSpec boundary.** Anchor: `§Cross-set native resource interface`. Correction: task set 2 assembles the canonical URI, verified digest/fingerprint, fixed geometry/RoPE through that fingerprint, N=128 cache capacity, selected pack name/version/digests, and resource budget; task set 2 owns verification/path lifetime, while task set 3 opens read-only and owns prepared resources. The pack schema is not duplicated.

### Final-freeze correction block

`agent://F1FinalFreezeReview` found three residual gaps. The active validation ledger already contains the two F1 command blocks; this section records the corresponding report/packet corrections and alignment requirements. The phase/progress row remains **Needs review** until the final correction and re-review, not because ledger state is missing.

1. **Callable persistent bridge:** task set 2 owns `native_r9700/native_resource_client.py`, which starts the existing runner once with `--model-service-worker` via `subprocess.Popen` and private pipes. The child alone owns the native generation from service startup through shutdown. The private protocol is exactly `r9700_native_resource_v1` with the seven frozen operations and exact request/response envelopes; one in-flight request and mismatched/duplicate IDs fail closed. Task set 3 owns `runner.cpp`, `runtime.h/.cpp`, existing resource files, and `native_resource_worker.h/.cpp`, and updates the existing runner build/validation source lists to compile the worker into the one runner. Task set 4 routes warm Prefill through the client and removes one-shot runner use from production warm dispatch.
2. **Cleanup result/state:** Rollback and Release pass only with `{resource_generation:uint64,state:"released",already_released:bool}`; the first pass is `already_released:false` and an idempotent same-operation repeat is `true`. Any cleanup error returns `{domain,message,failure_stage}` in `error`, retains child ownership as `release-failed`, leaves Python `draining`, blocks new LoadModel, permits read-only `Health` and only the same operation/generation cleanup retry, rejects every other operation including `Shutdown`, and reaches `unloaded` only after a pass. Child crash/device loss faults the service and disallows accepted-prefix repair/fallback until process restart.
3. **Verifiable producer identity:** the runner computes `producer_fingerprint` as `sha256:` of SHA-256 over UTF-8 RFC 8785 JCS of the exact `r9700-producer-fingerprint-v1` preimage in `§Producer fingerprint binding`; the runner binary SHA is included and published during Prepare. Python binds it to the model handle, and every native Prefill evidence/cache metadata repeats it. The consumer requires exact equality to both the handle and request-bound evidence and rejects unknown/missing/non-finite/mismatched identity.

For both F1 command blocks, private-child evidence is mandatory: one child PID spanning startup-to-shutdown, the private protocol version and operation/generation trace, the runner executable SHA, and exact fingerprint equality across Prepare/Commit/Prefill evidence/cache. These observations are evidence requirements, not a claim that this agent ran the commands.
### Final native-boundary review correction map

`agent://F1NativeBoundaryReview` identified four final technical findings; the corrected anchors and owners are:

1. **Every runner link closure:** task set 3 owns `native_resource_worker.cpp` in all eight runner-linked test closures (`test_block_prefill_runtime_contract.py`, `test_compute_barrier_policy.py`, `test_native_hsa_prefill_contract.py`, `test_runtime_lifecycle.py`, `test_runtime_llama_embed_contract.py`, `test_runtime_protocol.py`, `test_runtime_vram_contract.py`, and the generated runner format probe in `test_gpu_stage_profile_contract.py`), plus the three active-ledger `clang++` runner blocks. There is no current production dynamic/default runner compile list; any new one receives the same RED source-set contract.
2. **Explicit runner identity:** task set 2's constructor receives the task-set-4-propagated `--native-runner build/native-r9700-runtime/native_r9700_runner` path, canonicalizes and hashes the owner-executable regular file before launch, rejects symlinks/non-files/permission or identity changes, and verifies the child-reported `runner_binary_sha256`; no PATH/default/environment fallback is permitted on persistent service paths. Both F1 command blocks pass the explicit option.
3. **Private pre-decode:** task sets 2 and 3 share the exact six-key `r9700_native_resource_v1` error envelope with null correlation, `status:"error"`, `result:{}`, `invalid_request` error, bounded message/stage, no `evidence`, and public-mirroring one-response/newline-discard/continue-or-EOF behavior.
4. **Cleanup observability:** in `release-failed`, task set 3 permits only read-only `Health` and matching-generation same-operation cleanup retry; `Health` reports state, generation, and bounded error summary, while all other operations including `Shutdown` reject until cleanup passes. Task set 2 preserves Python `draining` and the same retry gate.

Task sets 2 and 3 are ready to begin after final review. The report and packet remain **Needs review** until that correction map is re-reviewed; this agent ran no commands.

## Supervisor validation command from the packet

The packet records this exact command for supervisor validation; the agent does not run it:

```sh
git diff --check docs/tasks/r9700-products/phase-f1-persistent-warm-worker.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/f1-contract-freeze.md
```

Expected observation: exit status 0 with no whitespace errors in the F1 phase ledger, active-ledger F1 headings/commands, or this report. The supervisor then runs the focused task-set-2–5 commands and the actual process/hardware commands above; this report records them but does not claim their execution. The active-ledger F1 blocks retain the private-child process/fingerprint evidence requirements from `§Final-freeze correction block`.

## Unresolved external blockers

This report and packet now contain the corrected persistent child boundary, exact cleanup semantics, and verifiable producer fingerprint. The report remains **Needs review** until final correction/re-review; the active ledger already contains the reconciled F1 command blocks. Supervisor-owned execution prerequisites remain the concrete model directory `mlx_models/meta-Llama-3.2-1B-Instruct`, `tests/native_r9700/fixtures/prompts.json`, a fresh R9700/TinyGPU run, and task-set-4/5 implementation of the frozen worker switches. No production source or shared validation ledger was changed by this task.

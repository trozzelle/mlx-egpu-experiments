# Phase F1: Persistent warm worker

## Source grounding

- `docs/ROADMAP.md` §Phase F1: Persistent warm worker.
- `docs/IMPLEMENTATION_PLAN.md` §F1 — Persistent warm worker and §Repository and file responsibility map.
- `docs/DESIGN.md` §Persistent model service contract, §Model lifecycle, §Prefill request lifecycle, §Benchmark contract, and §Security and review gates.
- ADR 0006 — independent product tracks.
- `.superpowers/swarm/progress.md` F1 row: keep **Needs review** until the final native-boundary correction and re-review; the active ledger already contains both exact F1 command blocks, and this phase must not become Done at this stage.
- `docs/REFERENCES.md` phase/source matrix F1; mlx-lm cache (Normative), oMLX (Pattern/adapter source), vLLM connector (Pattern).
- Manifest IDs: `mlx-lm-cache`, `omlx`, `vllm-kv-connector`.

## Goal

Deliver a local long-lived R9700 Prefill Service that loads a verified model once, owns resident/prepacked resources through an opaque model handle, serves repeated warm native-prefill requests without weight reload, emits accepted prompt caches, and unloads without leaks or hidden fallback.

## Dependencies

- B0 is Done.
- P2/P3 are not required for F1; F1 remains on the accepted direct AMDev path until Gate G1.
- F3 depends on F1's frozen model-handle and prepacking contracts.
- P4 depends on the completed persistent service.

## Reference resources

- **Normative:** mlx-lm cache/save/load/final-token behavior (`mlx-lm-cache`).
- **Pattern:** oMLX external worker lifecycle and vLLM connector roles; copy process/lifecycle shape only.
- **Local authority:** `native_r9700/native_worker.py`, `native_resource_client.py`, `serving.py`, `kv_cache.py`, `resident_memory.*`, `model_weight_binder.*`, `runtime.*`, `runner.cpp`, `native_resource_worker.*`, and `benchmark.py`.
- **Do not port:** oMLX cluster/distributed scope or vLLM runtime; F1 has no socket/network or generic RPC boundary.

## Orchestration map

- Sequential blockers: task set 1 freezes the public protocol, private child protocol, lifecycle, ownership, fingerprint, and validation-command names before implementation. Task set 4 waits for task sets 2 and 3. Task set 5 waits for task set 4 and review.
- Parallelizable task sets: task sets 2 (Python public protocol/model registry/private client) and 3 (native runner/resource lifetime/private worker) are independently implementable after task set 1 acceptance through the frozen `ResourceSpec` and `r9700_native_resource_v1` operation/result contract.
- Shared contracts/artifacts: public `r9700_prefill_service_v1`, private `r9700_native_resource_v1`, request/model handle IDs, process-lifetime request uniqueness, 64KiB predecode/error rules, RFC 8785 model digest, exact producer fingerprint JCS preimage, one-slot model state, private child generation/lifetime, cleanup retry states including read-only `Health` during `release-failed`, canonical runner-path/identity/hash binding, native runner evidence, prompt-cache artifact paths/flat metadata/meta-state, and benchmark-scope fields.
- Coordination risks: task set 2 is the sole public registry/active-request/draining/timeout-policy and private-client/Popen/runner-path/fingerprint-binding owner; task set 3 is the sole C++ runner/private-worker/native-resource/fingerprint-computation owner and updates every independent runner source closure; task set 4 is the sole public worker/consumer and `kv_cache.py`/`serving.py` cache writer/validator owner; benchmark.py changes serialize in task set 5.

## Progress ledger

| Task set | Status | Owner | Notes |
|---|---|---|---|
| 1. Contract and validation-command freeze | Done | F1Contract | Frozen in `.superpowers/swarm/reports/f1-contract-freeze.md`; final review found zero Critical/Important issues and authorized task sets 2–3. Active ledger commands are reconciled.
| 2. Local protocol, model registry, and private resource client | Done | F1Protocol | Final current-source review zero findings; combined task-set-2/3 focused command 107 passed.
| 3. Native runner/resource lifetime and private worker | Done | F1Native | Persistent all-layer/embedding generation, injected budgets, N=0 cache, actual runner/pack identities, clean JSONL diagnostics, and teardown contracts accepted.
| 4. Worker/consumer integration | Done | F1Integration | One production registry/session lifecycle, strict request-bound NPZ-to-mlx-lm cache conversion, `1.*` identity metadata, reconstructed cache validation, S=1/N=0, model/evidence binding, exact worker modes, and no native one-shot route. Task-set gate: 313 passed. |
| 5. Repeated warm smoke and benchmark promotion | Done | Supervisor | Real R9700 process smoke and ten-sample warm execution pass. The benchmark emits 13 full native records with exact scoped counts, request identities, N-based throughput, and no lifecycle contradiction. Complete F1 acceptance suite: 288 passed; final review: PASS. Report: `.superpowers/swarm/reports/f1-promotion.md`. |

Agents update only their row and append evidence/notes as work completes.

### Task set 1 evidence / notes
- Report: `.superpowers/swarm/reports/f1-contract-freeze.md`; final review passed and the active ledger contains both exact F1 command blocks.
- Frozen protocol: `r9700_prefill_service_v1`; raw JSONL frames are capped at 65,536 bytes before decode, pre-decode failures use the exact null-correlation envelope with one response then newline-discard/continue-or-EOF-exit behavior, and `token_ids` accepts `S=1..129` with `N=S-1≤128`.
- RFC 8785 JCS fixes model-digest number/string/key encoding and non-finite rejection; the report pins canonical UTF-8 fixture bytes and the expected SHA-256. The service has one model slot across `validating`, `preparing`, `resident-ready`, and `draining`; health separates service availability, observed read-only device state (TinyGPU remains the sole lifecycle authority), and model state.
- Request IDs are process-lifetime unique across all operations and artifacts are exclusive/no-overwrite. Unload uses the fixed 30-second drain timeout; repeated unload joins the same teardown and `release_once` result. `GetMetrics` passes unloaded with `model_handle:null`, process-lifetime counters, and zero current-model resource fields; loaded snapshots use the live handle.
- Task set 2 solely owns registry/draining/active-request policy, ResourceSpec verification/assembly/path lifetime, and the Popen-backed private client including runner-path identity/hash validation; task set 3 owns the one runner child/resource generation, every runner-linked source closure, and exact private protocol; task set 4 routes all warm Prefill through that client, propagates the explicit runner path, and binds evidence/cache fingerprints. The report mirrors the two reconciled active-ledger command blocks; each requires one child PID spanning service startup-to-shutdown plus protocol/generation/fingerprint evidence. The worker edited no shared ledger and ran no commands.
### Final native-boundary review correction map

`agent://F1NativeBoundaryReview` identified four final technical findings; the report and this packet now close them as follows:

1. **Runner compile/link closures — task set 3:** add `native_resource_worker.cpp` to the independent `RUNNER_SOURCES`/runner-linked closures in `test_block_prefill_runtime_contract.py`, `test_compute_barrier_policy.py`, `test_native_hsa_prefill_contract.py`, `test_runtime_lifecycle.py`, `test_runtime_llama_embed_contract.py`, `test_runtime_protocol.py`, `test_runtime_vram_contract.py`, and the generated runner format probe in `test_gpu_stage_profile_contract.py`, plus the active-ledger `Current native runner build and no-model smokes`, `P3 schema`, and `P3 scalar migration` clang blocks. No current production dynamic/default build list compiles `runner.cpp`; any future one receives the same RED source-set contract.
2. **Runner selection/hash — task sets 2 and 4:** each public service/native-worker command that starts the persistent child requires `--native-runner build/native-r9700-runtime/native_r9700_runner`; the separate benchmark command consumes `--serving-result` and never launches a runner. The client receives the service option, rejects symlinks/non-files/non-executable or changed identities, hashes the exact canonical file before launch, and verifies child `runner_binary_sha256`; no PATH/default/environment fallback exists on the persistent path.
3. **Private predecode — task sets 2 and 3:** private `frame_size`/`frame_decode` failures use exactly six keys with null correlation, `status:"error"`, `result:{}`, `invalid_request` error, bounded non-sensitive message/stage, no `evidence`, and public-mirroring one-response/discard/continue-or-EOF behavior.
4. **Release-failed observability — task sets 2 and 3:** only read-only `Health` and matching-generation same-operation cleanup retry are allowed; `Health` reports state/generation/error summary, while all other operations including `Shutdown` reject until cleanup passes.

Task sets 2 and 3 are ready to begin after final review; this report and packet remain **Needs review** until that correction map is re-reviewed. No commands are run by this task.

## Task set 1: Freeze service contract and validation commands

### Source refs

- `docs/IMPLEMENTATION_PLAN.md` F1 work package 1.
- `docs/DESIGN.md` §Persistent model service contract and lifecycle states.
- `docs/ROADMAP.md` F1 promotion gate.

### Target

- Freeze the exact task-set-1 sections already represented in `docs/tasks/native-r9700-producer/validation-commands.md`; the task-set-1 worker must not edit the shared validation ledger.
- Inspect `native_r9700/native_worker.py`, `native_resource_client.py`, `serving.py`, `benchmark.py`, `resident_memory.*`, `model_weight_binder.*`, `runtime.*`, `runner.cpp`, `native_resource_worker.*`, and `RUNNER_SOURCES` only to ground signatures/fields and source-list updates.
- Write `.superpowers/swarm/reports/f1-contract-freeze.md`.
- Non-goals: production implementation, process launch, model load, hardware command, TCP transport.

### Change

1. Freeze protocol version and operations: public `GetCapabilities`, `Health`, `LoadModel`, `UnloadModel`, `Prefill`, `GetMetrics`, `CaptureTrace`; private `Prepare`, `Commit`, `Rollback`, `Release`, `Prefill`, `Health`, `Shutdown` under `r9700_native_resource_v1`.
2. Freeze opaque model-handle and request-ID formats, exact `ResourceSpec`/opaque-generation boundary, model fingerprint, producer fingerprint JCS bytes, cache specification, evidence fields, and status/error domains.
3. Freeze states: validating, preparing, resident-ready, draining, unloaded; private prepared/resident/release-failed ownership and exact cleanup retry transitions; request states from received through accepted/rejected.
4. Assign exact Python/C++/worker/client files, runner build/validation source-list updates, and no-public-stdio/no-socket process topology for task sets 2–5.
5. Record the exact F1 process-smoke and warm-benchmark commands mirrored in the active validation ledger. Commands must name the concrete model, prompt corpus, output/log paths, sample count, explicit `--native-runner build/native-r9700-runtime/native_r9700_runner` on the two service/native-worker invocations, and expected public/private child/fingerprint evidence; the benchmark-only invocation consumes the serving result and has no runner option.

### Acceptance

- Report contains exact protocol fields/states, ownership matrix, frozen ResourceSpec/native error boundary, and no unresolved interface names.
- Report contains clean verbatim sections headed `F1 persistent process smoke` and `F1 warm benchmark promotion`, aligned with the already-present active-ledger blocks; final native-boundary correction/re-review is required before task set 1 is accepted. Until then, the phase row remains **Needs review**.
- No production source or shared validation ledger changed in this task set.

### Validation

```sh
git diff --check docs/tasks/r9700-products/phase-f1-persistent-warm-worker.md \
  docs/tasks/native-r9700-producer/validation-commands.md \
  .superpowers/swarm/reports/f1-contract-freeze.md
```

## Task set 2: Implement local protocol, model registry, and private resource client

### Source refs

- Task set 1 accepted contract/report.
- `docs/IMPLEMENTATION_PLAN.md` F1 work packages 2 and 4.
- `docs/DESIGN.md` §Persistent model service contract.
- oMLX and vLLM Pattern references in `docs/REFERENCES.md`.

### Target

- Create `native_r9700/service_protocol.py`, `native_r9700/model_service.py`, and `native_r9700/native_resource_client.py`.
- Create/extend `tests/native_r9700/test_service_protocol.py`, `test_model_service.py`, and `test_native_resource_client.py`.
- `native_resource_client.py` is the only Python-to-native resource bridge: it starts the existing runner once with `--model-service-worker` via `subprocess.Popen`, using the required `--native-runner build/native-r9700-runtime/native_r9700_runner` CLI value passed into its constructor, owns dedicated private stdin/stdout pipes, serializes one private request at a time, and closes/shuts down that child with the public service. The public `r9700_prefill_service_v1` service and `ModelRegistry` remain Python-owned.
- Non-goals: native allocation/C++ runner implementation, public-stdio sharing, one-shot warm Prefill, network/TCP/socket transport, shared memory, generic RPC, or a second service lifecycle.

### Change

1. Write RED tests for the exact public and private pre-decode envelopes, newline-discard/continue-or-EOF behavior, 64KiB bounds, RFC 8785 model-digest and producer-fingerprint canonical bytes/non-finite and unknown-field rejection, version/struct validation, opaque IDs, one-in-flight correlation, mismatched/duplicate IDs, process-lifetime request uniqueness, exclusive artifacts, sensitive-input redaction, invalid transitions, one-slot duplicate loads, draining behavior, cleanup retry/fault semantics, unloaded/loaded `GetMetrics`, and partial-load cleanup.
2. Implement the frozen public request/response schema and exact input/error mapping for `r9700_prefill_service_v1`, with Python owning external stdin/stdout and `ModelRegistry`.
3. Implement `NativeResourceClient` with exact `r9700_native_resource_v1` request/result schemas, dedicated `Popen` pipes, one in-flight request, response correlation/error validation, bounded frames, no auto-respawn/retry, explicit `Shutdown`, and child crash/device-loss faulting.
- The constructor must `lstat` the supplied path (rejecting symlinks), resolve an absolute canonical path, require a regular owner-executable file, open/hash that exact file before launch, compare pre-open/open/post-hash and pre-launch identity, and reject any changed identity. It then requires the child-reported `runner_binary_sha256` to equal the pre-launch hash before accepting `Prepare`; no PATH search, fallback default, or `NATIVE_R9700_PREFILL_RUNNER` is allowed on the persistent service path.
4. Implement the one-slot model registry state machine with dependency injection for the client/resource owner; the registry alone owns active-request exclusion, draining, the fixed 30-second timeout, repeated-unload policy, and Python `draining`/`unloaded` transitions.
5. Assemble and verify the minimal immutable `ResourceSpec`, pass only that object to `Prepare`, bind the returned producer fingerprint to the pending/committed model handle, invoke only `Prepare/Commit/Rollback/Release`, and keep native `PreparedResources`/`ResidentResources` opaque. Call `Rollback` only for caller-aborted successful Prepare-before-Commit; rely on native Prepare self-cleanup for Prepare errors and permit same-operation/generation cleanup retries while keeping read-only `Health` available during `release-failed`.
6. On child EOF/exit or device loss, fault the public service, reject accepted-prefix repair/fallback, and require process restart; never silently create a replacement child or resource generation.
### Acceptance

- Public and private schemas reject oversized/malformed frames with their exact pre-decode envelopes: public uses the seven-key envelope with `evidence:null`, while private `r9700_native_resource_v1` uses exactly six keys with `status:"error"`, null correlation, `result:{}`, and no `evidence`; both discard through newline and continue or exit normally at EOF with bounded non-sensitive strings.
- The client proves one persistent child PID from service startup through shutdown, private pipes distinct from public stdin/stdout, no socket/network, one in-flight request, exact operation/result correlation, no per-Prefill launch, explicit canonical runner-path/hash binding, child-SHA equality, and fail-closed child-crash/device-loss handling.
- Registry exposes exactly one occupied model slot, valid public/private state transitions, fixed drain/repeat semantics, exact `{resource_generation,state:"released",already_released}` cleanup results, `release-failed` retry behavior with read-only `Health` allowed for state/generation/error-summary observation, rejection of every other operation including `Shutdown` until cleanup passes, unloaded/loaded metrics nullability, and no reachable partial state after a Prepare/Commit failure.
- No model weights are loaded into Python numerical arrays by this layer; cache import/decode/consumer acceptance metrics remain outside `GetMetrics`. A committed handle's producer fingerprint is exact-equal to Prepare/Commit, every native Prefill evidence result, and cache metadata.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_service_protocol.py \
  tests/native_r9700/test_model_service.py \
  tests/native_r9700/test_native_resource_client.py -v
```

## Task set 3: Bind native resource lifetime and private runner worker

### Source refs

- Task set 1 ownership/lifecycle contract.
- `docs/IMPLEMENTATION_PLAN.md` F1 work package 3.
- `docs/DESIGN.md` model-handle ownership list.

### Target

- Modify `native_r9700/runner.cpp`, `native_r9700/runtime.h/.cpp`, `native_r9700/resident_memory.h/.cpp`, and `native_r9700/model_weight_binder.h/.cpp`.
- Create `native_r9700/native_resource_worker.h` and `native_r9700/native_resource_worker.cpp`; implement the narrowly named `--model-service-worker` mode inside the existing runner binary.
- Extend `tests/native_r9700/test_resident_memory_contract.py`, `test_model_weight_binder_contract.py`, `test_runtime_lifecycle.py`, and add `test_native_resource_worker_contract.py` as needed for the private boundary.
- Update every independent runner-linked source closure (do not centralize lists): `RUNNER_SOURCES` in `tests/native_r9700/test_block_prefill_runtime_contract.py`, `test_compute_barrier_policy.py`, `test_native_hsa_prefill_contract.py`, `test_runtime_lifecycle.py`, `test_runtime_llama_embed_contract.py`, `test_runtime_protocol.py`, and `test_runtime_vram_contract.py`; add `native_resource_worker.cpp` to the generated runner format-probe closure `tests/native_r9700/test_gpu_stage_profile_contract.py::FORMAT_PROBE_SOURCES` because its generated source includes `runner.cpp`; and update the active-ledger `Current native runner build and no-model smokes`, `P3 schema`, and `P3 scalar migration` `clang++` blocks. Keep `runner.cpp` as the sole entrypoint and one output binary. Standalone AMDev-only probes that do not link the worker-referencing runner remain unchanged. No current production dynamic/default build list compiles `runner.cpp`; any future such list requires the same RED source-set contract.
- Non-goals: public protocol schema, Python registry/model states, active-request waiting, drain timeout/repeat policy, generic retry/RPC framework, socket/network, WMMA prepacking format, HAL migration, queue ABI changes, or a second executable/service.

### Change

1. Add RED contracts for immutable `ResourceSpec` identity/budget input, opaque prepared/committed generation ownership, resident allocations, selected executable identities, scratch/reusable request buffers, exact private operation/result schemas, one-in-flight JSONL correlation, cleanup idempotence/retry, child fault behavior, and producer-fingerprint derivation.
2. Implement the private `r9700_native_resource_v1` JSONL reader/writer in `native_resource_worker.*` behind `--model-service-worker`; use the same raw 65,536-byte framing/discard/EOF behavior, but freeze private predecode failures as exactly `{protocol_version:"r9700_native_resource_v1",request_id:null,operation:null,status:"error",result:{},error:{domain:"invalid_request",message,failure_stage}}` with no `evidence`; use `failure_stage:"frame_size"` or `"frame_decode"`, bounded non-sensitive strings, and the exact seven-operation order. Keep private child pipes separate from public service stdio and do not add a socket.
3. Implement native `Prepare`/`Commit`/`Rollback`/`Release` around `ResidentMemory::allocate`, `ResidentMemory::release_all`, `ResidentBuffer` ownership/rollback, `ModelWeightBinder`, and stable request buffers. `Prepare` self-cleans every partial failure; `Rollback`/`Release` return exact cleanup objects, retain `release-failed` ownership on error, allow only same-operation/generation retry plus read-only `Health`, report `error_summary` with state/generation, and reject all other operations including `Shutdown` until cleanup passes.
4. Implement private `Prefill` using the committed generation and current accepted native evidence fields; it must never resolve a model path or reload resident weights. Implement `Health` and `Shutdown` with explicit generation/state reporting and normal post-response child exit.
5. Compute `producer_fingerprint` during `Prepare` from the exact JCS preimage (`runner_binary_sha256`, ordered kernel-pack digests, target/device/substrate, completion/barrier policies), publish it in Prepare/Commit/Prefill, and reject missing/unknown/non-finite/extra identity inputs. The fingerprint contains no model/request/path/timing values.
6. Update every independent runner build/validation source list and no-model smoke dependency closure to include `native_resource_worker.cpp` without replacing the one runner executable; do not add model registry or public protocol behavior to C++.

### Acceptance

- Native tests and source-list evidence prove one `native_r9700_runner --model-service-worker` child can process Prepare/Commit, repeated Prefill, cleanup, Health, and Shutdown over private JSONL with one in-flight correlation and no public-protocol/network path; private predecode failures use the exact six-key envelope and public-mirroring discard behavior.
- Resource counters and marker logs prove one native generation, repeated reuse without weight reload, one teardown for a committed lifetime, exact first/repeat cleanup results, retained `release-failed` ownership on cleanup error, read-only Health state/generation/error-summary observability, same-generation retry, rejection of every other operation including Shutdown until pass, and no reachable native state after Prepare failure.
- The runner publishes a deterministic `producer_fingerprint` whose JCS preimage and binary/kernel/device identities are exact; fingerprint identity is immutable after resident-ready and repeated in every native Prefill result/evidence.
- Every runner-linked test closure and the three active-ledger `clang++` runner blocks include `native_resource_worker.cpp`; `runner.cpp` remains the sole entrypoint/one executable, and C++ exposes no public registry/protocol or second lifecycle.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_resident_memory_contract.py \
  tests/native_r9700/test_model_weight_binder_contract.py \
  tests/native_r9700/test_runtime_lifecycle.py \
  tests/native_r9700/test_native_resource_worker_contract.py -v
```

The supervisor's existing native-runner build/no-model smoke and the active-ledger P3 schema/scalar runner builds must be rerun after adding `native_r9700/native_resource_worker.cpp` to every closure enumerated above; they must still produce exactly one `build/native-r9700-runtime/native_r9700_runner` with `runner.cpp` as sole entrypoint and exercise `--model-service-worker` over private JSONL without running a one-shot warm Prefill.

## Task set 4: Integrate persistent worker and consumer boundary

### Source refs

- Accepted task sets 2–3.
- `docs/IMPLEMENTATION_PLAN.md` F1 work packages 4–5.
- `docs/DESIGN.md` §mlx-lm prompt-cache adapter and request lifecycle.
- B0 C2R task set 3.

### Target

- Modify `native_r9700/native_worker.py` as the Python public service entrypoint and `native_r9700/serving.py` as the public consumer projection; consume (do not reimplement) task set 2's `native_resource_client.py`, and propagate the required `--native-runner build/native-r9700-runtime/native_r9700_runner` CLI value from each public service/native-worker invocation into the client. `benchmark.py` remains task-set-5-owned and consumes only the persisted serving result.
- Modify `native_r9700/kv_cache.py` and `serving.py` for the pinned canonical metadata writer/validator: the response descriptor has the exact 16 empty per-layer `meta_state` values, while the safetensors header is a flat string map with exact identity, geometry, absolute-position, JCS RoPE, and lowercase-boolean encodings.
- Extend `tests/native_r9700/test_native_worker_evidence.py`, `test_serving.py`, `test_kv_cache.py`, and `test_prefill_phase_accounting.py` for public/private process evidence, fingerprint equality, and persistent warm routing.
- Non-goals: implementing the client/C++ worker, alternate cache schema, direct-memory adapter, public/child stdio sharing, HAL, network transport, one-shot warm runner, or post-acceptance fallback.

### Change

1. Write RED tests for public service startup/shutdown, exactly one private child PID, private pipe isolation, handle lookup, Prepare/Commit/Release sequencing, repeated Prefill through one generation, request isolation, child crash/device-loss faulting, cleanup retry/draining behavior, exact flat safetensors metadata encoding, 16 empty per-layer `meta_state` acceptance/rejection, exact JCS producer/model/RoPE identity, and no accepted-prefix fallback.
2. Start task set 2's `NativeResourceClient` once at public-service startup with the propagated canonical runner path; keep it alive through public shutdown, call `Shutdown` only after registry teardown, and expose no child pipes on public stdin/stdout. The persistent path never consults PATH, a default runner, or `NATIVE_R9700_PREFILL_RUNNER`.
3. Route public LoadModel through `ResourceSpec → Prepare → Commit`, bind the returned producer fingerprint to the opaque model handle, and route every public Prefill through private `Prefill` using the committed generation and exclusive artifact paths. Do not call `subprocess.run`, `--native-prefill-proof`, or any per-request runner; require the child `runner_binary_sha256` to equal the client-hashed runner before `Prepare` acceptance.
4. Preserve request-bound native evidence and atomic prompt-cache output; have `kv_cache.py` write and `serving.py` validate the typed descriptor plus exact flat metadata encodings, canonical RoPE/model identity, exact handle/evidence/cache producer-fingerprint equality, and exactly 16 empty per-layer `meta_state` values.
5. Map private cleanup errors to Python `draining`/`release-failed` behavior without unloading early; allow read-only `Health` and only same-operation/generation cleanup retry while `release-failed`, reject every other private operation including `Shutdown` until a pass, and map child crash/device loss to service fault with no accepted-prefix repair/fallback until process restart.
6. Keep fallback legal only before cache acceptance; post-acceptance decode remains terminal. Remove the one-shot production route after all callers migrate; an explicit diagnostic branch, if retained, cannot be reachable from public warm Prefill.

### Acceptance

- Multiple public requests use one loaded model and one persistent private child/generation, with distinct public/private pipes, no per-request child launch, no TCP/network, no one-shot warm runner, explicit canonical runner-path/hash binding, and independently validated exclusive artifacts.
- Process evidence proves one child PID from service startup through shutdown, exact `r9700_native_resource_v1` operation/generation correlation, `Prepare/Commit` once per explicit load, repeated `Prefill` without weight reload, `Health` observability during any `release-failed` retry, and `Release`/`Rollback` retry semantics before `unloaded`.
- Every accepted request's native evidence and cache metadata has a finite, known producer fingerprint exactly equal to the committed handle and request-bound private evidence; the consumer rejects any mismatch/missing/unknown identity and never repairs an accepted prefix.
- Evidence proves no warm-Prefill weight reload and no stale request/model association; cache validation proves exact meta-state/RoPE/model identity. B0 serving/fallback tests remain unchanged in outcome.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py \
  tests/native_r9700/test_kv_cache.py \
  tests/native_r9700/test_prefill_phase_accounting.py -v
```

## Task set 5: Prove repeated warm service and benchmark scopes

### Source refs

- Task set 1 exact smoke/benchmark commands.
- Accepted task set 4.
- `docs/ROADMAP.md` F1 promotion gate.
- `docs/DESIGN.md` §Benchmark contract.

### Target

- Modify `native_r9700/benchmark.py` and `tests/native_r9700/test_benchmark.py` only for ten raw warm samples plus separate `cold_process`, `warm_prefill`, and `gpu_compute` aggregate records, with explicit per-scope and total counts.
- Produce `logs/f1-persistent-worker/` evidence and `.superpowers/swarm/reports/f1-promotion.md`.
- Update this ledger and `.superpowers/swarm/progress.md` only after supervisor validation/review.
- Non-goals: optimize kernels, change block size, direct transport, or compare cold startup as warm throughput.

### Change

1. Add RED benchmark-row contracts for scope labels, explicit load-preparation accounting, ten raw warm sample identities, three scope aggregates, median/dispersion, and no-warm-reload evidence.
2. Run the exact `prompt-128` fixture with `S=129/N=128`, exactly ten warm prefills, and the explicit load → unload → reload smoke through the actual process.
3. Record ten raw warm samples plus separate `cold_process`, `warm_prefill`, and `gpu_compute` aggregate records; report per-scope and total counts without mixing one-time load time into warm-prefill timing.
4. Dispatch final review; fix and re-review every Critical/Important finding.
5. Promote F1 only after C1R/C2R and process/resource checks pass.

### Acceptance

- Exactly ten warm requests complete with no warm-Prefill weight reload, cache corruption, resource drift, or fallback after acceptance.
- Report names the first authoritative warm prompt-128 baseline and separates all benchmark scopes with consistent raw/aggregate/total counts.
- Process unload/reload succeeds and all request artifacts remain model/request bound.

### Validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_benchmark.py \
  tests/native_r9700/test_native_worker_evidence.py \
  tests/native_r9700/test_serving.py -v
```

Then the supervisor runs the exact commands recorded by task set 1 under `F1 persistent process smoke` and `F1 warm benchmark promotion`.

## Phase validation

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
```

Supervisor also requires the task-set-5 process smoke, fresh hardware logs, final review with zero Critical/Important findings, and `git diff --check` before marking F1 Done.

## Handoff notes

- F3 consumes the immutable model-handle, prepacking identity, and warm benchmark baseline.
- P4 consumes the persistent service API and evidence schema; F1 must not depend on HAL objects.
- F5 consumes the canonical KV validator and process lifetime but owns any direct-local transport decision.

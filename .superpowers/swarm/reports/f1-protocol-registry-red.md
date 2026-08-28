# F1 task-set-2 protocol/registry/client RED

**Owner:** `F1ProtocolRed`
**Task set:** F1 persistent warm worker, task set 2 — local protocol, model registry, and private resource client
**Status:** RED contracts authored; supervisor verification pending

## Scope

Created only the three assigned focused test files:

- `tests/native_r9700/test_service_protocol.py`
- `tests/native_r9700/test_model_service.py`
- `tests/native_r9700/test_native_resource_client.py`

No production source, phase packet, active ledger, or other test file was changed. The native-resource-client tests use executable temporary fake child scripts that speak the frozen private JSONL protocol; they do not invoke the native runner, allocate resources, or use hardware.

## Contracts covered

### `test_service_protocol.py`

- Freezes `r9700_prefill_service_v1` and `r9700_native_resource_v1`, exact operation order, and `MAX_FRAME_BYTES=65536`.
- Requires exact public seven-key pre-decode envelope (`evidence:null`) for oversized, malformed UTF-8/JSON, and duplicate-key frames; verifies bounded non-sensitive messages, valid-correlation recovery only, and no prompt/token redaction leaks.
- Requires exact public request/response key sets, unknown-field/type/version/operation rejection, and public/private envelope separation.
- Pins the RFC 8785 fixture bytes, UTF-8 hex, no-newline rule, and model digest:
  `sha256:a5f32101f172484252004bacdcb9b2f194e82948b19be1634ffd6a39d60a65fd`.
- Pins producer-fingerprint JCS preimage ordering and digest:
  `sha256:a1c2948871b161bccad64ce551cc32277bd5872c664fb8424029fe6e3f708c7b`.
- Rejects non-finite values, unknown/missing/malformed producer identity, and path/timing fields before hashing.

### `test_model_service.py`

- Freezes immutable `ResourceSpec` fields and nested cache/kernel-pack/resource-budget shapes.
- Uses `ModelRegistry(resource_client=..., artifact_dir=...)` and `dispatch(request_mapping)` as the public seam.
- Covers service-process request-ID uniqueness/reuse, one occupied model slot, cryptographically shaped opaque handle, `validating → preparing → resident-ready → draining → unloaded`, failed Prepare/Commit unwinding, active-prefill drain exclusion, and service-owned exclusive artifact paths.
- Covers unloaded/loaded/unloaded `GetMetrics` nullability and zero current-resource counters.
- Covers exact cleanup result shape, release-failed observability through Health, same-generation Release retry, refusal of new work while draining, and close ordering (`Release` before one `Shutdown`).

### `test_native_resource_client.py`

- Uses fake executable children and a `Popen` recorder to require one canonical runner path, regular owner-executable non-symlink validation, explicit `--model-service-worker`, one persistent PID, and private stdin/stdout pipes distinct from public stdio.
- Covers exact immutable `ResourceSpec` serialization to Prepare, pre-launch SHA/child-reported `runner_binary_sha256` binding, no model-path resolution on private Prefill, and one process-lifetime private request-ID namespace.
- Covers one-in-flight serialization, mismatched/duplicate response correlation, bounded oversized private response handling, exact six-key private errors without public `evidence`, redacted token/path values, child EOF/crash/device-loss faulting, no auto-respawn/retry, explicit Shutdown, and release-failed Health plus same-generation cleanup retry.

## Production changes required for GREEN

1. Add `native_r9700/service_protocol.py` exporting the frozen protocol constants, `ServiceProtocolError`, `decode_request_frame(bytes)`, `encode_response(mapping)`, `canonical_jcs(mapping)`, `compute_model_digest(identity)`, and `compute_producer_fingerprint(identity)` with the exact validation/error/redaction behavior asserted above.
2. Add `native_r9700/model_service.py` exporting frozen immutable `ResourceSpec` and one-slot `ModelRegistry`; inject only a `resource_client` and artifact root, route through `dispatch(request_mapping)`, reserve request IDs for the process lifetime, and keep the 30-second drain policy internal/fixed.
3. Add `native_r9700/native_resource_client.py` exporting `NativeResourceClient`; use one `subprocess.Popen` child with dedicated pipes, exact private frames/results, canonical executable identity/hash checks, one-in-flight correlation, fail-closed child faults, and explicit cleanup/shutdown semantics.
4. Reconcile the private Prepare result's child `runner_binary_sha256` binding with the frozen native result schema while keeping the six-key private envelope and exact operation result validation. The fake child test exercises the required child-SHA check.

## Expected RED cause

The tests deliberately import production modules through test-local `_require_*` helpers so missing files/API exports fail during test execution with a clear RED assertion rather than collection errors. Before the three production modules exist, each focused lane reports its specific `RED: ... is required` failure; after scaffolding, remaining failures are contract-local: exact envelope/frame validation, RFC8785/JCS and fingerprint identity, ResourceSpec/registry state transitions, Popen/path/hash/correlation behavior, and private-child fault/cleanup handling. No test intentionally depends on a build, package install, hardware device, real native runner, or unrelated existing test behavior.

## Supervisor verification command

```sh
${PY} -m pytest \
  tests/native_r9700/test_service_protocol.py \
  tests/native_r9700/test_model_service.py \
  tests/native_r9700/test_native_resource_client.py -v
```

The executor did not run pytest, builds, formatters, package managers, hardware commands, or git commands; RED/GREEN execution is supervisor-owned.

## Post-review RED additions for findings 1, 9, 10, and 12

The client lane now adds four focused contracts from `agent://F1ProtocolReview`:

- `test_prepare_requires_child_runner_sha256_even_when_other_identity_fields_are_valid`
  uses a valid three-field `Prepare` result with no `runner_binary_sha256` and
  requires rejection rather than accepting an unbound native child.
- `test_unterminated_oversized_private_response_faults_without_waiting_for_eof`
  uses a FIFO event barrier and a fake child that emits 65,537 bytes without a
  newline, then remains open. The client must raise a bounded `frame_size`
  fault before EOF; the test releases the child only after that assertion.
- `test_recoverable_private_request_error_does_not_poison_client` makes the
  first `Prepare` return `resource_exhaustion`, then requires `Health` and a
  second `Prepare` to succeed on the same child.
- `test_launch_executes_retained_verified_file_when_path_replaced_before_popen`
  replaces the verified pathname inside the `Popen` seam. The launched child
  must still be the bytes that were hashed before the replacement, not the
  replacement script (which writes an observable marker).

The fake child emits the expected runner hash for ordinary successful
`Prepare` responses; `missing-runner-sha` is reserved for the mandatory-field
RED contract. No commands, tests, builds, formatters, package managers, git
operations, or hardware were run in this RED addition.

## Finding 11 RED extension

Added `test_canonical_jcs_accepts_binary64_exact_safe_integer_boundaries` to
pin both signs of the maximum exact-safe integer
(`±9007199254740991`), and
`test_canonical_jcs_rejects_integer_outside_binary64_exact_safe_range` to
reject `9007199254740993` and `-9007199254740993` before canonical hashing.

## Registry RED extensions for findings 3-8, 13-15

Added focused contracts in `test_model_service.py` for the registry review:

- `test_release_failed_gate_rejects_everything_except_health_and_retry` requires
  exact `release_failed` gating for `GetMetrics`, `CaptureTrace`, `LoadModel`,
  and `Prefill` while retaining read-only `Health` access.
- `test_load_publishes_only_nonzero_kernel_pack_identities` rejects the
  all-zero pack digest sentinel and binds the selected nonzero identities into
  both `Prepare` and the public load result.
- `test_metrics_never_publishes_a_transitional_load_snapshot` holds native
  `Commit` behind an event barrier and rejects a `model_handle:null` snapshot
  paired with `validating`/`preparing`.
- `test_prefill_artifact_reservation_rejects_an_exclusive_open_race` injects an
  `EEXIST` race at the exclusive artifact open and requires
  `artifact_creation` without invoking native `Prefill`.
- `test_indexed_model_shards_are_streamed_without_read_bytes` patches
  `Path.read_bytes` for indexed safetensors shards and requires a successful
  load through bounded/header plus streamed hashing.
- `test_load_binds_native_prepare_to_the_verified_model_inventory` replaces a
  verified shard during `Prepare` and requires no resident handle to publish.
- `test_metrics_expose_only_the_declared_transfer_counters` rejects the
  undeclared `transfer_bytes` metric in both `GetMetrics` and `CaptureTrace`.
- `test_close_marks_shutdown_before_a_concurrent_load_can_start` blocks child
  shutdown and requires concurrent `LoadModel` rejection at `shutting_down`.
- `test_prefill_rejects_missing_or_empty_native_artifact` covers both absent
  and zero-length service-owned prefill payloads, requiring
  `cache_rejection/cache_validation`.

The existing release-failure Prefill assertion was aligned to the same exact
`release_failed` gate. No commit-failure Rollback contract was added: failed
Commit consumes and self-cleans the prepared native value per the corrected
review, while Rollback remains a successful-Prepare-before-Commit operation.
No commands, tests, builds, formatters, package managers, git operations, or
hardware were run; supervisor RED/GREEN verification remains pending.

## GREEN fixture alignment

The registry test seam now injects the exact nonzero test pack
`r9700-llama-fp16/v1/(sha256: + "3" * 64)` required by the verified-pack
constructor.
The fake native client writes a nonempty Prefill payload by default; the
missing-artifact case disables publication and the empty-artifact case writes
zero bytes explicitly. These are fixture-only updates; no production source
or verification command was changed.

## Final-review residual: private blocked request errors

Added `test_blocked_private_request_error_is_request_scoped_and_health_recovers`.
Its fake child returns an exact six-key private response with
`status:"blocked"` and a `resource_exhaustion` error for the first `Prepare`.
The test asserts the complete response shape/content, then requires `Health`
to succeed on the same child. This remains RED until the private decoder and
client treat a request-scoped blocked response as recoverable instead of
poisoning the client. No commands or verification runs were performed.

## Final-review residual: issued model handles

Added `test_model_handles_skip_issued_values_after_unload`. It forces
`secrets.token_hex` to repeat the first unloaded handle before producing a new
value, then requires the registry to skip the issued value and retain both
handles in its process-lifetime `_issued_handles` set. No commands or
verification runs were performed.

## Final-review residual: public Prefill evidence

Added `test_successful_prefill_response_is_public_protocol_encodable`. It
round-trips a successful registry Prefill through `service_protocol.encode_response`
and requires public evidence to carry `producer_kind:"r9700_native"` while
excluding private `resource_generation`.

## Public Prefill token validation RED extension

Added boundary contracts for empty and 130-token `Prefill` bodies, which must
return the exact `token_bounds` failure stage, plus non-integer, boolean,
negative, and above-`uint32` token values, which must return
`token_validation` rather than the generic `operation_validation` stage.

## Final-review residual: injected budgets and rollback retry

`_registry` now injects the exact non-default
`resident_bytes_max/scratch_bytes_max/total_bytes_max` mapping, and
`test_registry_passes_injected_resource_budget_exactly_to_prepare` verifies
that no hardcoded 1 GiB budget reaches `Prepare`.

`test_failed_prepare_inventory_rollback_is_health_visible_and_close_retried`
forces post-Prepare inventory invalidation, a failed first `Rollback`, and a
successful retry. It requires no public handle, `draining` plus
`release-failed` Health visibility, and the retry to precede child `Shutdown`.

## JCS decimal formatting RED extension

Added exact-byte checks that `canonical_jcs({"x": 0.1})` emits
`b'{"x":0.1}'` and `canonical_jcs({"x": 0.001})` emits
`b'{"x":0.001}'`.

## Final-review residual: Prefill token boundaries

Added direct `ModelRegistry.dispatch` contracts for empty and 130-token
`Prefill` bodies, requiring `token_bounds`, and for string, boolean, negative,
and above-`uint32` token values, requiring `token_validation` rather than the
generic `operation_validation` stage.

## Checkpoint residual: persistent child stderr backpressure

Added `test_persistent_child_stderr_backpressure_cannot_block_private_response`.
Its fake child writes 1 MiB to stderr before sending the `Health` response;
the test requires completion despite exceeding a normal pipe buffer. Cleanup
terminates only the intentionally blocked RED child, while successful
implementations receive normal `Shutdown`. No commands or verification runs
were performed.

# Q1 task-set 3: hybrid-cache RED contracts

**Status:** RED contracts added; supervisor verification pending
**Owner:** `Q1HybridRed`
**Scope:** Qwen3.8-27B text-only hybrid cache ownership, opaque spill metadata, executable MLX restore, and capture/restore CLI surface.
**Production files intentionally unchanged:** `native_r9700/qwen_hybrid_cache.py`, `native_r9700/qwen_spill.py`.

## Contracts added

### `tests/native_r9700/test_qwen_hybrid_state_spill.py`

- `test_capture_serializes_all_64_hybrid_layers_in_runtime_order_without_cpu_tensors` keeps the 64-entry runtime order, 48 `ArraysCache`/16 `KVCache` split, exact component shapes/dtypes, leaf digests, and opaque `tobytes()`-only capture. The fixture uses the frozen Q1 fingerprint and `committed_position = 4 = S-1` for `[760, 6511, 314, 9338, 369]`.
- `test_serialized_state_freezes_layer_order_and_every_component_metadata` requires version-1 wire metadata for the exact runtime class list and every component ID, owner, update, position, trim support, shape, dtype, byte count, and digest. It also requires the frozen 48/16 class counts.
- `test_capture_requires_the_frozen_qwen_model_fingerprint` requires fail-closed model identity rather than accepting an arbitrary nonempty label.
- `test_hybrid_cache_exposes_task_set_3_restore_api` makes the MLX conversion boundary an explicit public API even when the optional MLX dependency is unavailable.
- `test_deserialize_rejects_every_runtime_order_and_component_metadata_mutation` covers reordered, missing, extra, wrong-class, wrong-shape, wrong-dtype, wrong-offset, and wrong-owner records while preserving a valid outer checksum, so failures identify semantic validation rather than checksum corruption.
- The existing round-trip test now also captures/serializes the same opaque cache twice and requires byte-for-byte deterministic output.
- `test_executable_mlx_restore_decodes_little_endian_arrays_separately_from_opaque_bridge` is optional-MLX and hardware-free. It uses real `mlx_lm` `ArraysCache`/`KVCache` instances, checks canonical little-endian bfloat16/fp32 words, exact shapes/dtypes, actual MLX array objects, persisted KV offset, and separation from the opaque bridge.
- `test_executable_mlx_restore_rejects_nonfinite_state_before_partial_assignment` is optional-MLX and requires NaN/Inf rejection before any cache layer is partially assigned.
- `test_hybrid_state_cli_help_exposes_capture_and_restore_modes` and `test_hybrid_state_cli_parses_and_dispatches_without_running_a_model` pin `--capture-hybrid-state`, `--restore-hybrid-state`, `--token-ids-json`, and fail-closed dispatch on a missing model/spill path without loading a model.

The existing upload and opaque-bridge tests remain in place and continue to distinguish bounded raw-byte upload from executable MLX reconstruction.

### `tests/native_r9700/test_qwen_layer_executor.py`

- The synthetic executor state now carries the frozen model fingerprint, exact recurrent/full-attention shapes and byte extents, and `S-1 = 4`.
- `test_qwen_stage_plan_covers_all_64_runtime_layers_with_48_arrays_and_16_kv_entries` walks every layer and requires the exact class/asset schedule: `qwen_affine4_linear` plus `qwen_deltanet_state` for 48 recurrent layers and `qwen_affine4_linear` plus `qwen_full_attention` for the 16 full-attention layers.

### `tests/native_r9700/test_qwen_layer_executor_contract.py`

The existing no-hardware C++ stage-plan probe remains the executor boundary check for valid Arrays/KV plans and multimodal, runtime-order, missing-resident-state, and binding rejection paths. No C++ production or probe changes were made in this RED lane.

## Expected RED causes and required production changes

The current implementation has no task-set-3 `restore_qwen_hybrid_cache_into_mlx` API or capture/restore CLI, and its spill header does not emit or validate the frozen runtime-layer/component metadata, exact Q1 fingerprint, exact Qwen shapes/dtypes, owner/update/position/trim semantics, or non-finite payload rejection. The expected GREEN work is therefore limited to the task-set-3 production ownership:

1. Extend `qwen_spill.py`'s deterministic version-1 schema/validation while keeping capture, serialization, and native upload opaque and allocation-free.
2. Add `restore_qwen_hybrid_cache_into_mlx(model, state)` in `qwen_hybrid_cache.py`; validate all metadata/lengths/digests/finite values before assignment, decode canonical little-endian bytes into exact real MLX arrays, and assign them to the pinned `ArraysCache.state`/`KVCache.state` with `KVCache.offset = N`.
3. Add the `qwen_hybrid_cache` module CLI with separate capture and restore dispatch, full-token `S-1` handling, and fail-closed model/spill validation without a fallback or model recomputation path.

No fixture generation, `qwen_parity.py` edits, native hardware, or Llama cache behavior belongs to this lane.

## Supervisor RED command

From the worktree root, run exactly:

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py -v
```

This worker did not run tests, builds, linters, formatters, package managers, model loads, hardware, or git commands.

## Review-derived RED corrections

The review findings in `agent://Q1HybridReview` are converted into seven
focused contracts in `tests/native_r9700/test_qwen_hybrid_state_spill.py`:

1. `test_capture_accepts_runtime_arrays_cache_list_state` is parameterized over
   `bfloat16` and `float32` and uses real `mlx_lm.models.cache.ArraysCache`
   objects when installed (with a source-pinned pure fallback). The current
   capture boundary rejects the actual mutable list returned by
   `ArraysCache.state` before any spill is produced.
2. `test_capture_normalizes_runtime_mlx_dtype_objects` is parameterized over
   both Qwen scalar dtypes and supplies real MLX `Dtype` objects when available.
   The current boundary compares those objects directly with schema strings and
   rejects the leaves.
3. `test_restore_cli_passes_language_model_make_cache_to_bridge` removes the
   attached `.cache`, supplies only `language_model.make_cache()`, and records
   the explicit cache passed to the bridge. The current restore CLI never calls
   `make_cache()` and invokes the bridge without the explicit target.
4. `test_mlx_restore_keeps_arrays_cache_state_mutable_for_resume` is
   parameterized over both recurrent scalar leaves, uses real optional MLX cache
   classes, and performs a resumed `ArraysCache.__setitem__`. The current
   restore assigns a tuple, so the mutable resumed update cannot succeed.
5. `test_deserialize_rejects_noncanonical_json_scalar_types` parameterizes a
   float shape dimension, float component position, and integer trim flag while
   rebuilding a valid outer checksum. The current equality-only metadata checks
   accept these noncanonical JSON scalar types.
6. `test_hybrid_state_cli_requires_matching_source_pin_before_model_load` is
   parameterized over capture and restore, passes the canonical schema-v1
   `--source-pin-report` interface, and uses a mismatched fingerprint while
   asserting `_load_model` is never called. The current parser has no
   source-pin argument or pre-load identity gate.
7. `test_hybrid_cli_reports_frozen_identity_digests_and_assignment_evidence` is
   parameterized over capture and restore with deterministic synthetic state
   helpers. It requires the frozen identity/count/token/layer fields,
   `state_digest`, `record_digest`, and restore-only `assigned_layers`,
   `arrays_assigned`, `kv_assigned`, and `mlx_restore` evidence. The
   `state_digest` preimage is the JCS-canonical JSON object with domain
   `qwen-hybrid-state-digest-v1`, model fingerprint, committed position, runtime
   layer order, and ordered component metadata/payload digests; `record_digest`
   hashes the exact serialized spill bytes. The current reports retain
   `model_identity` and omit these acceptance fields.

The final residual review adds four more focused contracts:

8. `test_atomic_write_failure_preserves_existing_destination` injects an
   `os.replace` failure after an existing spill is present and requires the
   previous destination bytes to survive. The current cleanup unlinks the
   destination on every exception.
9. `test_capture_rejects_same_resolved_output_and_report_before_model_load`
   passes distinct path spellings that resolve to the same capture artifact and
   asserts the loader is not reached. The current dispatch has no collision
   check.
10. `test_capture_accepts_buffer_protocol_leaf_without_tobytes` is parameterized
    over `bfloat16` and `float32` and supplies a raw buffer leaf without a
    `tobytes` method. The current spill boundary requires that method instead of
    accepting the buffer protocol.
11. `test_capture_prefills_prefix_without_sampling_and_commits_n` records the
    prompt length, `max_tokens=0`, sampled-token count, and committed position.
    The current capture requests one generated token, allowing an extra cache
    update and sampled-token append instead of consuming only the S-1 prefix.

12. `test_restore_rejects_same_resolved_spill_and_output_before_source_pin`
    uses distinct `--spill`/`--out` spellings with one resolved destination,
    injects source-pin/model-load sentinels, and verifies the original spill
    remains byte-for-byte intact. The current restore dispatch verifies source
    identity and then reads/replaces the colliding spill instead of rejecting
    the path pair first.

## Supervisor RED command and expected failures

From `.worktrees/r9700-products-wave-a`, the exact supervisor command is:

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_hybrid_state_spill.py \
  tests/native_r9700/test_qwen_layer_executor.py \
  tests/native_r9700/test_qwen_layer_executor_contract.py -v
```
The existing 26 contracts remain the baseline. Before the corresponding
production fixes, the command is expected to fail the twelve new contracts
(with their scalar/mode parameter cases): list-state capture, MLX dtype
normalization, explicit restore-cache handoff, mutable ArraysCache resume,
noncanonical JSON scalar rejection, source-pin CLI admission, frozen
capture/restore report evidence, atomic destination preservation, same-path
capture rejection, buffer-protocol capture, zero-token prefix capture, and
same-path restore rejection. No command, test, model load, build, formatter,
package-manager, hardware, or git execution was performed for this RED
correction pass.

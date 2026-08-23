# LN-1B native bounded trace review

## Verdict: FAIL

The command gate and resident-prefix execution are appropriately bounded, and the trace is separated from accepted prefill/NPZ publication. However, the native artifact cannot satisfy the shared oracle/native comparison contract: eight of the ten stage schemas have incompatible shapes (and two attention artifacts have incompatible byte counts). The reported scalar metadata is also not faithful to the dispatched kernargs. The promised atomic two-file publication is not actually atomic.

## Findings

### 1. Blocker — native and oracle stage schemas are incompatible

`native_r9700/runtime_contract.cpp` declares physical resident-buffer shapes:

| Stage group | Native shape | Oracle shape |
|---|---:|---:|
| `hidden`, `normalized`, `post_attention_hidden` | `[1,2048]` | `[1,1,2048]` |
| `fresh_k`, `fresh_v` | `[1,8,64]` | `[1,8,1,64]` |
| `attention_scores`, `attention_probabilities` | `[1,32,128]`, 16,384 bytes | `[1,32,1,1]`, 128 bytes |
| `context` | `[1,32,64]` | `[1,1,2048]` |

Only `k_cache` and `v_cache` agree on `[1,8,1,64]` / 1,024 bytes. The native attention kernels intentionally materialize the entire 128-token resident capacity for a token-0 query (`llama_causal_attention_score_f16.cpp:29-35` and `llama_causal_attention_softmax_f32.cpp:35-40`), whereas the CPU oracle emits only its logical one-token tensor. Therefore the LN-1C comparator must reject these artifacts at its required shape/dtype gate, before it can compare any values. A single canonical artifact shape and byte extent must be selected for every boundary; the oracle and native producer must then emit that same representation.

### 2. Blocker — `scalars` does not record the dispatched scalar fields faithfully

`trace_scalars_json` reports `{"output_columns":1}` for stage indices 1, 2, and 7. Those kernargs are instead `sequence_length` in the K-projection, V-projection, and O-projection kernels (for example `llama_v_projection_f16.cpp:1-5`; O-projection takes `sequence_length` at `llama_o_projection_f16.cpp:1-6`). For indices 3–6, it reports `head_count: 1`, although head count is not a dispatched kernarg; the actual mutable scalar fields are sequence length, position, and cache capacity. This makes the required forensic scalar metadata misleading even though `kernarg_hex` is present. Serialize the actual named kernarg fields and values, without invented fields.

### 3. High — publication is not atomic as the interface promises

The worker writes two temporary files, then renames the raw file before the JSON file (`runtime_contract.cpp:632-645`). A process crash between those renames leaves a published `.bin` without matching JSON. On a JSON-rename failure, cleanup of the already-published raw file is attempted but its result is ignored, so the documented guarantee that *any failure leaves no trace artifact* is not enforceable. Publish a complete staged directory (or equivalent single atomic commit unit) only after both files are durable, and treat cleanup failure as a surfaced failure condition.

### 4. Medium — focused tests cannot catch the contract break

`test_llama_stage_trace_contract_is_bounded_and_non_accepting` only searches C++ source text for field literals and the absence of `write_native_prefill_npz`. It does not compare every native trace stage's `shape`, `dtype`, `byte_count`, or scalar keys against the oracle's shared schema. The CLI test correctly proves help/nonzero layer/unknown stage avoid creating the trace directory, but it does not cover the two-file publication failure path or an artifact-pair atomicity invariant. Add contract tests for the canonical stage table and fault-injected publication failures once the schema is reconciled.

## Passed review points

- **Command parsing / early rejection:** `runner.cpp` requires the exact option order and argument count; strict uint32 parsing and layer/position `== 0` checks occur before `run_llama_stage_trace`. Unknown stages are rejected in `run_llama_stage_trace` before model binding or `ResidentHsaSession::prepare`.
- **Bounded stage selection:** the fixed ten-entry stage allowlist maps each accepted stage to one buffer and a final stage index. The dispatch loop executes only indices `0..stage_index`; `hidden` dispatches none. The buffer-name assertion prevents reading a different resident buffer.
- **Readback extent and token-0 cache position:** the selected output's `readback_byte_count` is set before `prepare`; resident readback uses that recorded count. K/V cache readback is 1,024 bytes from buffer offset zero, which is the logical position-0 slot. The full cache allocation is not read back.
- **Non-finite failure:** both fp16 and fp32 values are bit-tested for exponent-all-ones. Any NaN/infinity, malformed extent, dispatch failure, or readback-size mismatch returns before output-directory creation/publication.
- **Path containment within this API:** the accepted stage is a whitelist, filenames are fixed, and outputs are constructed beneath canonicalized `trace_dir`; no caller-controlled filename can escape that directory. Unlike the oracle API, this C++ API has no distinct `run_root`/`run_dir` pair, so it cannot enforce a parent-root policy beyond treating `trace_dir` itself as the requested root.
- **Full-prefill isolation:** `LlamaStageTraceRequest` has no NPZ/cache path, the trace worker uses its own resident session and closes it before publication, and it never calls `run_native_prefill` or `write_native_prefill_npz`. It performs only the selected prefix and no full prefill loop.

## Validation

Not run, as directed. This review is source/test inspection only; no hardware, git, or test commands were used.

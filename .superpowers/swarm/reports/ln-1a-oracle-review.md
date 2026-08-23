# LN-1A Llama stage oracle review

## Verdict: **FAIL**

The implementation can emit only the `hidden` artifact. Every computational boundary beginning at `normalized` fails before it can produce an oracle tensor, and the emitted metadata/layout is not compatible with the current native trace contract for most stages. The focused tests do not exercise either failure.

## Findings

### P0 — All requested computed stages fail due to an invalid hidden tensor rank

- **Location:** `native_r9700/llama_stage_oracle.py:75-91, 152-155`
- `_load_embedding_row` reads a `[1, hidden_size]` embedding row and adds another leading axis before returning it, producing `[1, 1, hidden_size]` (line 91).
- `_stage_tensor` passes that rank-3 value directly to `primitives.rms_norm` for every stage except `hidden` (line 155). `primitives.rms_norm` explicitly accepts only rank-1 or rank-2 inputs and rejects rank-3 input (`native_r9700/primitives.py:159-165`).
- Therefore `normalized`, `fresh_k`, `fresh_v`, `k_cache`, `v_cache`, `attention_scores`, `attention_probabilities`, `context`, and `post_attention_hidden` cannot be emitted. Simply removing the added axis is not an isolated repair: `_project_heads` then expects a rank-3 matmul result at `llama_stage_oracle.py:111-116`, while the reference primitive returns a rank-2 result for a rank-2 input (`native_r9700/primitives.py:134-143`). The context/post-attention path also retains incompatible rank assumptions.
- This violates the required ten named boundaries and prevents the oracle from identifying the earliest native numerical failure.

### P1 — The oracle metadata and raw tensor layout do not match the native trace contract

- **Location:** `native_r9700/llama_stage_oracle.py:91, 116, 134, 207-210`; native counterpart `native_r9700/runtime_contract.cpp:89-108`
- The oracle reports the NumPy tensor's literal shapes and uses the bare stage name as `buffer` (lines 207-210). Those values differ from the resident trace schema:
  - `hidden` / `normalized` / `post_attention_hidden`: oracle `[1,1,2048]` versus native `[1,2048]`.
  - `fresh_k` / `fresh_v`: oracle `[1,8,1,64]` versus native `[1,8,64]`.
  - `context`: oracle `[1,1,2048]` versus native `[1,32,64]`.
  - `attention_scores` / `attention_probabilities`: oracle `[1,32,1,1]` and 128 bytes versus native `[1,32,128]` and 16,384 bytes.
  - The oracle's `buffer` is e.g. `"hidden"`; the native trace uses e.g. `"layer0.embedding_row"`, `"layer0.normalized"`, and so on.
- A comparator is required to check shape and dtype before numerical comparison. These incompatible artifacts therefore cannot be compared, even where contiguous byte order could otherwise agree at token 0.

### P1 — Attention artifacts describe a different extent from the selected native output

- **Location:** `native_r9700/llama_stage_oracle.py:119-135`; native counterpart `native_r9700/runtime_contract.cpp:102-105`, `native_r9700/kernels/llama_causal_attention_score_f16.cpp:20-35`
- The oracle calculates only the logical `[query=0, key=0]` element. The native trace declares and reads the full 128-key resident score/probability buffer. Native score dispatch writes causal-mask values for keys it executes (lines 23-35 of the kernel) and the trace byte count includes the full allocation. No padding/masking/extent policy is shared in the oracle metadata.
- Thus the two artifacts differ in shape, byte count, digest scope, finite count, and potentially values outside key 0. This makes the asserted shared digest/finite-count contract false for the attention stages.

### P1 — Tests validate only `hidden`; all successful-computation coverage is externally skipped

- **Location:** `tests/native_r9700/test_llama_stage_oracle.py:61-105`
- The sole success-path test invokes `stage="hidden"` twice (lines 66-83), the only stage that avoids the rank failure. It never invokes any of the nine computed stages, checks their metadata/shape against the shared native schema, or validates their numerical values.
- That test is skipped outright when an external model directory is absent (lines 61-63). The remaining local tests only prove rejection paths, so a normal focused test pass can contain no successful oracle invocation.
- Consequently the test suite fails to defend the main LN-1A contract: generating deterministic, usable layer-0/token-0 artifacts for every named boundary.

### P2 — The module depends on a private prefill implementation detail for its required shard lookup

- **Location:** `native_r9700/llama_stage_oracle.py:22, 143`; private implementation `native_r9700/prefill.py:102-127`
- The oracle imports and calls `prefill._tensor_shards`, a private helper owned by the broader prefill path. This is specifically contrary to a narrow oracle built on the existing strict loader/reference-primitives boundary, and it creates an undocumented coupling to prefill's index parsing and exception types.
- The executor report acknowledges the private dependency. No test establishes that the oracle remains correct if prefill's private helper changes, nor is shard resolution exposed as a public strict-loader API.

## Confirmed non-findings

- Request validation occurs before `load_model_metadata` and before `destination.mkdir`, and rejects nonzero layer/position, unknown stages, and a resolved `run_dir` outside the resolved `run_root` (`native_r9700/llama_stage_oracle.py:57-72, 198-215`).
- This code path has no NPZ/cache writer and writes only its named raw/JSON diagnostic files after model/tensor computation. It does not itself create or alter an accepted native NPZ/cache.

## Review scope

Source inspection only; no tests or hardware commands were run, as requested.

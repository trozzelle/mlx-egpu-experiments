# LN-1 remediation re-review

## Verdict: **FAIL — one prior High finding remains incomplete**

Source-only re-review; no tests, git, or hardware commands were run.

## Previous findings

| Prior finding | Status | Evidence |
|---|---|---|
| LN-1A P0 — computed stages fail from rank-3 hidden input | **ADDRESSED** | `native_r9700/llama_stage_oracle.py:103-119` now returns the embedding row as rank-2 `[1, 2048]`. RMSNorm receives that rank at line 214; projections produce their internal `[1, heads, 1, 64]` representation (lines 139-145); `_canonical_stage_tensor` converts only output boundaries (lines 178-194). The subsequent K/V, attention, context, and post-attention paths have compatible ranks. |
| LN-1A P1 / LN-1B blocker — producer schemas disagree | **ADDRESSED** | Oracle `STAGE_SPECS` (`llama_stage_oracle.py:38-53`) and native `kLlamaStageTraceStages` (`runtime_contract.cpp:90-110`) agree for all ten buffer names, shapes, dtypes, and byte counts. `test_llama_stage_trace_native_table_matches_canonical_oracle_schema` compares each native row against the Python table. |
| LN-1A P1 — attention describes only key 0 | **ADDRESSED** | `_attention_parts` materializes `[1, 32, 128]` score/probability buffers, fills masked scores with finite `float32` minimum, writes the token-0 score, and zeroes all non-token-0 probabilities (`llama_stage_oracle.py:162-174`). This matches the native resident extent and token-0 causal behavior. |
| LN-1A P1 — successful oracle coverage is external-model-only / hidden-only | **ADDRESSED** | The local synthetic test invokes `emit_stage_oracle` for every `STAGE_SPECS` entry and checks emitted metadata, finite count, and raw byte count (`tests/native_r9700/test_llama_stage_oracle.py:67-127`). It needs no external model. |
| LN-1A P2 — private `prefill._tensor_shards` dependency | **ADDRESSED** | `resolve_tensor_shards` is a public strict-loader function (`native_r9700/loader.py:54-95`), and the oracle imports/calls it directly (`llama_stage_oracle.py:22, 201-202`). The narrow test verifies indexed resolution without a prefill import. |
| LN-1B blocker — scalar metadata invents fields | **ADDRESSED** | `trace_scalars_json` selects actual scalar names and reads their actual offsets from materialized kernargs (`runtime_contract.cpp:198-268`). The offsets match the supplied HSA metadata: RMSNorm 24; K/V 24; RoPE/score 32/36/40; softmax 16/20/24; context 24/28/32; O projection 32. No invented `output_columns` or `head_count` labels remain. |
| LN-1B High — raw/JSON artifact pair publication is atomic and cleanup failure is surfaced | **NOT ADDRESSED** | The staged-directory rename fixes the former **process-crash-between-two-renames** exposure and cleanup errors are surfaced (`runtime_contract.cpp:675-727`). However, the promised publication occurs without making either staged file durable: `std::ofstream::flush()` at lines 689 and 709 is not `fsync`, and neither the files nor `trace_staging` are synced before the directory rename at line 719. A system crash/power loss after the rename can therefore expose a final artifact directory with a missing, empty, or stale raw/JSON member. The original required contract explicitly required both files to be durable before the single commit. This remains a High correctness defect. |
| LN-1B Medium — focused tests cannot catch the contract break | **NOT ADDRESSED** | The added tests improve table coverage, but remain source-text assertions: the schema test parses C++ literals (`tests/native_r9700/test_runtime_vram_contract.py:261-275`) and the publication/scalar test merely searches for identifiers and call counts (lines 278-294). There is no fault-injected publication test and no execution-level assertion of scalar values/offsets or all artifact-pair failure paths. Thus the requested regression defense is still absent. |

## New Critical/Important findings

None beyond the unresolved prior High publication-durability defect above.

## Assessment

- **Correctness:** Oracle rank/shape and canonical representation repairs are coherent. Native scalar reporting now derives values from dispatched kernarg bytes. The native output pair is atomically visible to ordinary concurrent readers, but is not crash-durable.
- **Maintainability:** `StageSpec`/`STAGE_SPECS` and the native fixed table make the comparison contract clear and bounded. The loader seam cleanly removes the prefill-private coupling. The native scalar switch is direct and appropriately narrow for the fixed ten-stage trace.
- **Architecture:** The CPU oracle remains on the strict-loader/reference-primitives boundary; native publication remains outside accepted prefill/NPZ output. No CPU/native boundary violation was introduced.
- **Simplicity:** The canonicalization-at-emission approach and one-directory commit are simpler than propagating resident layout throughout computation. No needless abstraction was added.

## Required disposition

Do not accept LN-1 as complete until staged raw and JSON files, and the containing staging directory, are durably synced before the atomic publish rename; add failure-injected behavioral coverage for write/rename/cleanup paths.

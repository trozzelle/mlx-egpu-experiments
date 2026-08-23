# LN-1 review remediation

## Scope

Resolved the Critical/Important findings from `ln-1a-oracle-review.md` and `ln-1b-native-trace-review.md` in the bounded layer-0/token-0 oracle and native trace lanes only. Validation was intentionally not run by this worker.

## Finding-to-change mapping

| Review finding | Source change | Focused coverage |
|---|---|---|
| LN-1A P0: computed oracle stages fail from rank-3 embedding input and inconsistent projection/context ranks | `llama_stage_oracle.py` now keeps the embedding and RMSNorm compute path rank-2 (`[1,2048]`), keeps head computation in its required internal rank, and applies canonical stage reshaping only in `_canonical_stage_tensor` at emission boundaries. All ten stages now pass through that boundary. | `test_emit_stage_oracle_materializes_every_canonical_stage_without_external_model` drives all ten stages with a local synthetic strict-loader model and real oracle control flow. |
| LN-1A P1 / LN-1B blocker: producer schemas disagree | `STAGE_SPECS` establishes the oracle canonical buffer names, shapes, dtypes, and byte counts. Oracle metadata comes from that table. Native table remains the matching resident representation: hidden/normalized/post `[1,2048]`; fresh K/V `[1,8,64]`; cache `[1,8,1,64]`; scores/probabilities `[1,32,128]`; context `[1,32,64]`. | `test_llama_stage_trace_native_table_matches_canonical_oracle_schema` parses the native table and compares every row with `STAGE_SPECS`. The oracle synthetic success test asserts each metadata row and raw byte count. |
| LN-1A P1: oracle attention has only logical key 0 | `_attention_parts` materializes the native 128-key extent: token 0 holds the computed score/probability, later scores hold finite causal-mask `float32` minimum, and later probabilities are zero. | The all-stage synthetic oracle test asserts the full attention metadata and 16,384-byte raw files. |
| LN-1A P1: success coverage depends on external model and covers only hidden | Replaced the model-dependent sole success path with a synthetic, local, fully computed all-stage test; it does not require external MLX weights. | `test_emit_stage_oracle_materializes_every_canonical_stage_without_external_model`. |
| LN-1A P2: private `prefill._tensor_shards` coupling | Added public strict-loader `resolve_tensor_shards(ModelData, tensor_names)` and changed the oracle to use it. It validates indexed/single-file shards and required names without importing prefill. | `test_strict_loader_resolves_oracle_tensor_shards_without_prefill_coupling`. |
| LN-1B blocker: invented native scalar names | `trace_scalars_json` reads the materialized kernarg block by actual argument offsets and serializes only `epsilon`, `sequence_length`, `position`, and `cache_capacity_tokens` for the applicable dispatched stage. | `test_llama_trace_publication_failure_seam_and_scalar_values` drives real scalar serialization with known kernarg bytes and asserts exact JSON values. |
| LN-1B High: raw/JSON pair publication is not atomic/durable and cleanup failure is ignored | Native trace writes and closes both files in one hidden staging directory, `fsync`s each file and the staging directory, atomically renames the directory, then `fsync`s its parent. Any failed write/sync/rename cleans staging or final output as appropriate; failed cleanup is surfaced. | `test_llama_trace_publication_failure_seam_and_scalar_values` uses the narrow injected publication operations to execute write, file-sync, staging-sync, rename, parent-sync, and cleanup failures, plus normal publication ordering and visibility. |

## Supervisor validation commands

Not run by this worker. Run exactly:

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
$PY -m pytest tests/native_r9700/test_llama_stage_oracle.py tests/native_r9700/test_runtime_vram_contract.py -q
```

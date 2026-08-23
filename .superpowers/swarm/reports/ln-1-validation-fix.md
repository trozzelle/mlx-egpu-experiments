# LN-1 focused validation fixes

## Scope

Remediated only the four supervisor-observed focused-test failures in the bounded Llama oracle/native-trace lane. Validation was intentionally not run by this worker.

## Finding-to-change mapping

| Observed failure | Change | Focused regression coverage |
|---|---|---|
| A one-file `model.safetensors` was parsed as a JSON index because `ModelData.weight_index_path` also records single-file provenance. | `resolve_tensor_shards` now recognizes only `.index.json` paths as indexes; all other validated provenance paths use the one-shard resolution branch. `_stage_tensor` translates strict-loader `ConfigError`s into `LlamaStageOracleError`, preserving the oracle's public failure type. | Added one-file shard resolution coverage and an oracle public-error normalization test. The existing wrong-embedding-shape test now reaches the safetensors shape check and asserts `LlamaStageOracleError`. |
| The bounded-trace contract test required JSON literals to stay in `runtime_contract.cpp`. | It now checks the declared public `LlamaStageTraceResult` schema in `runtime.h`, while retaining the bounded/no-prefill assertions on the trace implementation. | All externally published metadata values remain represented by stable public result fields; the test no longer depends on serialization-helper placement. |
| The native table parser assumed one row-closing format and could omit a canonical entry. | The parser bounds itself to the table declaration, accepts whitespace and either byte-count closing form, and asserts the complete ten-row count before exact schema comparison. | The exact native/oracle buffer, shape, dtype, and byte-count comparison remains intact for all ten stages. |
| The publication seam compiled its harness alongside `runner.cpp`, introducing a second `main`. | The harness compile closure now excludes both its inlined `runtime_contract.cpp` and `runner.cpp`; it retains the non-entrypoint production dependencies needed by the publication/scalar functions. | The publication harness still executes all write, sync, rename, cleanup, and scalar assertions without linking a duplicate entrypoint. |

## Supervisor validation

Not run by this worker. Run the focused LN-1 oracle/runtime trace tests specified by the supervisor.

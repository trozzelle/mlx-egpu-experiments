# Task9 live-binding implementation

- `LayerExecutionEvidence` now records the ten file-backed layer-0 safetensor span names and ten named Llama stage assets.
- Evidence validation requires the complete exact sets, a `tokens:embedding_gather` model-input origin, and a `device:embedding_gather` intermediate origin. Fixture and CPU-derived model/intermediate sources are rejected.
- `execute_llama_layer0` binds real safetensors spans, requires token IDs, records their names and the required stage names, and then fails closed before any dispatch when reviewed stage assets are unavailable. It never changes `native_prefill_acceptance` from `open` and performs no tensor math.
- No commands were run, per Task9 Wave B constraints. Native prefill acceptance remains open.

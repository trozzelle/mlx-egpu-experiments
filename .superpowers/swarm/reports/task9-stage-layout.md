# Task9 stage-layout implementation

Created the schema-only `LlamaStageLayout`, `StageBufferBinding`, and `StageLaunch` boundary with `build_layer0_stage_launches`.

The builder validates all required live layer-0 bindings against layout-derived dtype, shape, byte-span, 8-byte GPU-VA alignment, and `live_layer0_device_buffer` provenance. It also requires exactly one of each ten named stage assets, the `task9-kernarg-v1` schema, nonempty kernarg storage, and nonzero descriptor grid dimensions.

Launches are built in a temporary vector and committed only after every validation succeeds; failures leave the supplied output vector unchanged. Successful launches preserve each supplied asset's name and grid and allocate only zero-initialized opaque kernarg bytes at the asset-declared size. The implementation performs no asset loading, device work, dispatch, or model math.

Per assignment, no commands or tests were run.

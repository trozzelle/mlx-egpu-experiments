# Resident layer0 implementation

## Changed files
- `native_r9700/c1_primitive_bridge.cpp`
- `native_r9700/runtime.cpp`
- `tests/native_r9700/test_runtime_contract.py`

## Exact behavior
- Replaced the `--native-layer0` pure scaffold path with a bridge-owned model/prompt input materialization step.
- The bridge now parses `--token-ids-json`, reads the MLX model `config.json`, and inspects MLX safetensors metadata for the layer0 tensors needed by the first K/V path (`embed_tokens`, layer0 input norm, K proj, V proj).
- The bridge reports model/prompt input markers (`model_prompt_input_status`, config/weight/prompt statuses, `stage0_input_source: model_prompt`, `resident_subgraph_scope: layer0_model_prompt_input_materialization`) without fixture boundary markers.
- `native_prefill_acceptance` remains `open`; no fake native prefill pass is introduced.
- The remaining failure is now the later precise blocker `layer0_resident_kv_projection_dispatch_not_parameterized` after model/prompt inputs are identified, instead of `layer0_resident_dataflow_not_implemented`.
- Runtime fallback failure text/stage now matches that more precise blocker when bridge output is incomplete.
- Focused contract tests now assert the model/prompt resident subgraph markers and the later failure stage.

## Remaining blocker
- The first resident K/V projection dispatch still needs parameterized embedding/weight upload and hardware kernel launch using those model/prompt-derived buffers. Current counters remain `kernel_count: 0` and `transfer_bytes: 0`, so the path remains blocked/open.

## Supervisor commands
- Per assignment constraint, I did not run validation commands. Minimal suggested supervisor check: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py -q`

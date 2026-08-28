# C1R-3 fixture trace report

Status: Done

Scope:
- Added compact CPU prefill trace generation to `native_r9700/ref_fixtures.py`.
- Added private trace capture in `native_r9700/prefill._run_layer` so fixtures come from the same CPU reference layer implementation used by prefill parity.
- Generated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` and updated `fixtures_schema.json` metadata/digest.
- Updated `tests/native_r9700/test_ref_fixtures.py` to validate the new trace schema, shapes, dtypes, nontrivial K/V anchors, and attention probability normalization.

Fixture contract:
- Prompt: `prompt-0`, S=6, n_prefix=5.
- Layers: 0 and 15.
- Slice: first 2 prefix tokens, first 2 heads, first 16 hidden/head dims; attention probabilities/scores retain all 5 source positions for the sliced target tokens.
- Arrays cover hidden input, RMSNorm output, Q/K/V projections, RoPE Q/K, attention scores/probabilities/context, O projection, attention residual, post-attention norm, MLP gate/up/SiLU/gated/down, MLP residual output, final K/V.
- Size: `layer_trace_fixtures.npz` is 13,065 bytes, below the 512 KiB fixture cap.

Verification:
- RED before implementation: `python3 -m pytest tests/native_r9700/test_ref_fixtures.py -q` failed with missing `layer_trace_fixtures.npz` and missing schema entry.
- Generation: `${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures` wrote 6 fixture files.
- Focused fixture suite: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q` -> `10 passed in 0.06s`.
- Native focused suite after review fixes/refactor: `${PY} -m pytest tests/native_r9700 -q` -> `128 passed, 2 warnings in 11.16s`.

Decisions:
- Kept one compact NPZ rather than two per-layer files because the compressed output is 13 KiB and the schema records exact per-layer keys; this avoids another file-management convention.
- Did not add Qwen fixtures in C1R-3: ADR 0005 and the recovery plan keep Qwen as a separate post-Llama target phase unless Llama C1R/C2R lands first.

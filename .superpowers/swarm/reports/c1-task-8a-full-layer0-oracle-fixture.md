# C1R-8a full layer0 post-layer oracle fixture

## Scope

Add a small full-width CPU oracle fixture for layer 0's final hidden state so future full-layer hardware acceptance has a committed comparison target.

## Work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch.

## Implemented

- Added `native_r9700.ref_fixtures._layer0_post_layer_hidden_full_width`.
- `generate_layer_trace` now records `layer0_post_layer_hidden_fp16` from `mlp_residual_out_fp16` for layer 0.
- `generate_all` splits that array into `tests/native_r9700/fixtures/layer_trace_layer0_post_layer_hidden_fixtures.npz`.
- The fixture shape is `[8, 2048]` fp16: rows0:5 contain prompt-0 prefix hidden states, rows5:8 are zero padding matching current tile contracts.
- `fixtures_schema.json` now records the new file and digest.
- `tests/native_r9700/test_ref_fixtures.py` now validates schema, file digest, compact-trace agreement on rows0:2/cols0:16, non-zero full-width content outside compact dims, and zero padded rows.
- Updated stale schema file-list guard to include already-existing split fixture files plus the new post-layer hidden file.

## Decision

Commit a final post-layer hidden oracle, not bulky full intermediate projections. The final hidden tensor is enough to anchor future full-layer parity while staying well below the 512 KiB committed-fixture bound. It remains CPU oracle evidence only; hardware acceptance is still blocked until resident full-width/all-head layer execution exists.

## Generator issue handled

- Initial `python -m native_r9700.ref_fixtures --generate` failed because this worktree has no `mlx_models/` directory, so the loader treated `mlx_models/meta-Llama-3.2-1B-Instruct` as a Hugging Face repo ID and returned a 404.
- Root cause confirmed by directory reads: `mlx_models` absent in the worktree; sibling local model exists at `../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`.
- Regenerated with explicit local model path.

## Verification

- RED before implementation:
  - Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_post_layer_hidden_full_width_fixture_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_post_layer_hidden_full_width_fixture_matches_compact_trace -q`
  - Result: failed with missing schema key and missing fixture file.
- Fixture regeneration:
  - Command: `${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`
  - Result: wrote 15 fixture files including `layer_trace_layer0_post_layer_hidden_fixtures.npz`.
- New fixture digest/size:
  - SHA256: `feb3f5f10bca2182d677f0edb5f386270b2e1f91c21275d7ed95c419d14bc7a7`
  - Size: `19290` bytes.
- Focused green:
  - Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_post_layer_hidden_full_width_fixture_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_post_layer_hidden_full_width_fixture_matches_compact_trace tests/native_r9700/test_ref_fixtures.py::test_layer_trace_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests -q`
  - Result: `5 passed in 0.16s`.
- Full fixture suite:
  - Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q`
  - Result: `44 passed in 0.09s`.

## Remaining blocker

Full C1 layer acceptance still needs hardware-side full hidden production. This fixture only supplies the oracle target.

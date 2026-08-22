# C1R-8d MLP down full-inner to cols0:64 oracle fixtures

## Scope

Add full-inner layer0 MLP down-projection CPU oracle data for output cols0:64. This bridges from the new full 8192-column activation oracle to the first 64 hidden output columns and prepares future hardware accumulation across all MLP intermediate columns.

## Work boundary

- Path: `${HOME}/Development/ml/tools/egpu/.worktrees/native-r9700-producer`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch.

## Implemented

- Added `native_r9700.ref_fixtures._layer0_mlp_down_projection_full_inner_to_cols0_64`.
- `generate_layer_trace` now records four 2048-column partial down-projection chunks and a final fp32/fp16 sum for output cols0:64.
- `generate_all` writes:
  - `layer_trace_mlp_down_projection_full_inner_to_cols0_64_fixtures.npz`
  - `layer_trace_mlp_down_projection_full_inner_to_cols0_64_chunk0_fixtures.npz`
  - `layer_trace_mlp_down_projection_full_inner_to_cols0_64_chunk1_fixtures.npz`
  - `layer_trace_mlp_down_projection_full_inner_to_cols0_64_chunk2_fixtures.npz`
  - `layer_trace_mlp_down_projection_full_inner_to_cols0_64_chunk3_fixtures.npz`
- Each chunk fixture contains activation `[8,2048]`, weight `[2048,64]`, partial expected fp32 `[8,64]`, and partial expected fp16 `[8,64]`.
- Final fixture contains the accumulated full-inner expected output fp32/fp16 `[8,64]`.
- Tests validate chunk schemas, exact fp32 chunk matmul, exact fp16 casts, exact final sum of chunks, compact trace agreement on rows0:2/cols0:16, digest guards, and committed-size bounds.

## Decision

Split down weights by four 2048-column inner windows. A single 8192×64 weight fixture would exceed the committed-file taste/size bar; chunk files are each about 225 KiB and reusable as hardware dispatch windows. The final sum fixture is tiny and provides the cols0:64 full-inner target.

## Verification

- RED before implementation:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_to_cols0_64_chunk_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_to_cols0_64_fixtures_match_fp32_oracle -q`
  - Result: failed with missing chunk schema key and missing final fixture file.
- Regeneration:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`
  - Result: wrote 21 fixture files including the full-inner down final and four chunk fixtures.
- New fixture digests/sizes:
  - Final fixture SHA256: `e3aab29d893f849fc4627e4781ca36fef1574ccf4d5dda562fcdacf3438bb338`
  - Final expected fp32 SHA256: `84f9ddf66e1e71849b928caa061b6abcca81d00bea59081635592ca7d58f4d7e`
  - Sizes: final `2515` bytes; chunk0 `224919`; chunk1 `224877`; chunk2 `224829`; chunk3 `224796`.
- Focused green:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_to_cols0_64_chunk_fixtures_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_full_inner_to_cols0_64_fixtures_match_fp32_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_down_proj_inner_cols0_64_to_cols0_64_fixture_matches_partial_fp32_oracle tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests -q`
  - Result: `5 passed in 0.16s`.
- Full fixture suite:
  - Command: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_ref_fixtures.py -q`
  - Result: `48 passed in 0.10s`.

## Remaining blocker

Hardware still needs a full-inner down-projection chain that dispatches these four 2048-column windows and accumulates them into output cols0:64; output cols64:2048 and full attention width remain open.

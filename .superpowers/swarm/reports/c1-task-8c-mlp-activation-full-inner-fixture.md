# C1R-8c MLP activation full-inner oracle fixture

## Scope

Add a full-width layer0 MLP activation CPU oracle across all 8192 intermediate columns. This is prerequisite data for full MLP down-projection hardware work; the prior committed activation fixture covered only cols0:64.

## Work boundary

- Path: `<former-native-r9700-worktree>`
- Branch: `feature/native-r9700-producer`
- Boundary type: current feature branch.

## Implemented

- Added `INTERMEDIATE_SIZE = 8192` to `native_r9700.ref_fixtures` frozen geometry constants and schema geometry.
- Added `native_r9700.ref_fixtures._layer0_mlp_activation_full_inner`.
- `generate_layer_trace` now records:
  - `layer0_mlp_activation_full_inner_gate_fp16` shape `[8, 8192]`
  - `layer0_mlp_activation_full_inner_up_fp16` shape `[8, 8192]`
  - `layer0_mlp_activation_full_inner_expected_fp16` shape `[8, 8192]`
- `generate_all` splits those arrays into `tests/native_r9700/fixtures/layer_trace_mlp_activation_full_inner_fixtures.npz` and schema metadata.
- `tests/native_r9700/test_ref_fixtures.py` now validates schema, digest, SiLU(gate)*up recomputation, consistency with the existing cols0:64 fixture, consistency with compact `layer0_gated_mlp_fp16`, non-zero columns beyond cols0:64, zero padded rows, schema digest guard, and size guard.

## Decision

Commit the full MLP activation oracle as one NPZ. Raw payload is 393 KiB and compressed fixture size is 230,375 bytes, below the 512 KiB committed-fixture limit. This avoids prematurely adding split-file complexity while providing the exact activation operand needed for future full down-projection windows.

## Scout grounding

- `FullMlpWidthScout` confirmed the C1 blocker is widening/windowing existing 8x16x8 GEMM and fused activation chains, not adding a new math opcode.
- Current hardware path proves only MLP gate/up cols0:64, activation cols0:64, and partial down inner cols0:64 -> output cols0:64.

## Verification

- RED before implementation:
  - Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_activation_full_inner_fixture_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_activation_full_inner_fixture_matches_silu_multiply_oracle -q`
  - Result: failed with missing schema key and missing fixture file.
- Regeneration:
  - Command: `${PY} -m native_r9700.ref_fixtures --generate --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`
  - Result: wrote 16 fixture files including `layer_trace_mlp_activation_full_inner_fixtures.npz`.
- New fixture digest/size:
  - SHA256: `c4ac8b5c351d57097cc0fb6f68539f1aa2996591c13e27064f0a146b5e2d6ad9`
  - Size: `230375` bytes.
- Focused green:
  - Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_activation_full_inner_fixture_schema_shape_dtype tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_activation_full_inner_fixture_matches_silu_multiply_oracle tests/native_r9700/test_ref_fixtures.py::test_layer0_mlp_activation_cols0_64_fixture_matches_silu_multiply_oracle tests/native_r9700/test_ref_fixtures.py::test_all_fixture_files_small_enough tests/native_r9700/test_ref_fixtures.py::test_schema_json_matches_disk_digests -q`
  - Result: `5 passed in 0.16s`.
- Full fixture suite:
  - Command: `${PY} -m pytest tests/native_r9700/test_ref_fixtures.py -q`
  - Result: `46 passed in 0.10s`.
- Renamed full-hidden seam re-verification after `FullLayerAcceptanceScout`:
  - Compile exited `0` with no output.
  - Focused runtime tests exited `0` with `2 passed in 10.30s`.
  - Direct `--layer0-full-hidden-proof` emitted expected blocked markers and exited `1`; log `logs/c1-runner-layer0-full-hidden-proof-2026-08-20T13:54:38Z.log`.

## Remaining blocker

Full MLP down-projection still needs weight/output window contracts and hardware accumulation across all 8192 activation columns and 2048 output columns.

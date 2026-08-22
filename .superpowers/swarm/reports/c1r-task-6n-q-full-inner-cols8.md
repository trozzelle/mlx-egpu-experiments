# C1R-6n full-inner Q-projection cols8 accumulator

Status: proof-only hardware slice complete; full C1R-6 layer-0 acceptance remains open.

Decision
- Add `layer0_q_proj_full_inner_cols8_accum_chain` as the Q-projection analogue of the proven K/V full-inner cols8 accumulators.
- Reuse the existing full-inner input-norm activation chunks and the same 128-stage split-A/B resident accumulator topology.
- Pack Q model weights into the proven dot2 pair layout per 16-inner chunk.
- Keep `native_prefill_acceptance: open`; this is one real layer-0 projection cols8 tile, not a full native layer.

Fixture evidence
- Regenerated `tests/native_r9700/fixtures/layer_trace_fixtures.npz` with Q arrays.
- Current `layer_trace_fixtures.npz` SHA: `e138c82eab58403bb018d0c96089941ac3b382144cc81bf36b198a3c08c2a5e1`.
- Q source arrays:
  - `layer0_q_proj_full_inner_cols8_a_fp16`: shape `(8, 2048)`, SHA `26261261843662c4475a0aca530b2f160f7875e7ac507955b456af614b116920`.
  - `layer0_q_proj_full_inner_cols8_b_fp16`: shape `(2048, 8)`, SHA `575476239a216b096d715cbdc4c5fdc6e7a6d948a7e3f98db06f1380e3b17d50`.
  - `layer0_q_proj_full_inner_cols8_expected_fp32`: shape `(8, 8)`, SHA `af0ac8143d62beee4eac7d493ab903de185f48d1c8ee726c51f7ca7b9f70a3a3`.
  - `layer0_q_proj_full_inner_cols8_expected_fp16`: shape `(8, 8)`, SHA `87d71b8c21ca808ed8ca62d6f7d4bccdeb9205440cf4152a7b70bd622c7faf1c`.

RED before implementation
- Focused Q fixture/runtime tests failed before implementation because Q fixture arrays were absent and `native-r9700-runner --primitive-chain-proof layer0_q_proj_full_inner_cols8_accum_chain` reported `unsupported primitive chain`.

Implementation
- `native_r9700/ref_fixtures.py` now emits Q full-inner cols8 arrays from `input_norm_fp16 @ q_proj[:8].T`.
- `tests/native_r9700/test_ref_fixtures.py` covers Q schema, shape/dtype, fp32 matmul oracle, fp16 oracle, and shared K/V/Q activation rows.
- `native_r9700/c1_primitive_bridge.cpp` embeds packed Q model weights, Q expected fp32 bytes, a Q `FullInnerCols8ChainSpec`, and Q bridge entrypoint routing.
- `native_r9700/runtime.h`, `native_r9700/runtime.cpp`, and `native_r9700/runner.cpp` expose and validate the Q wrapper chain.
- `tests/native_r9700/test_runtime_contract.py` covers Q help text, required log markers, stage names, source arrays, expected SHA, measured tolerance, and missing-source-array rejection.

Hardware proof
- Wrapper command: `build/native-r9700-runtime/native_r9700_runner --primitive-chain-proof layer0_q_proj_full_inner_cols8_accum_chain`.
- Exit status: `0`.
- Log: `logs/c1-runner-primitive-chain-proof-layer0_q_proj_full_inner_cols8_accum_chain-2026-08-19T18:19:17Z.log`.
- Required markers include `producer_kind: hardware_primitive_chain`, `primitive_backend: hardware`, `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `chain_stage_count: 128`, `chain_readback_between_stages: no`, `resident_data_page_count: 17`, `data_region_residency: seventeen_distinct_vram_pages`, all activation/model/output PTE statuses `pass`, and wrapper status `pass`.
- Numeric result: `expected_fp32_sha256: af0ac8143d62beee4eac7d493ab903de185f48d1c8ee726c51f7ca7b9f70a3a3`, `tolerance: fp32_abs<=2e-6_or_ulp<=64`, `max_abs_diff: 1.1444091796875e-05`, `max_ulp_diff: 54`, `mismatch_count: 0`, `byte_mismatch_count: 42`, `cpu_comparison_status: pass`, `host_device_transfer_status: pass`.

Remaining C1R-6 blockers
- Full-shape tiled/multi-workgroup projection coverage beyond one cols8 Q/K/V tile.
- Full-shape RoPE over Q/K.
- Attention score/softmax/context kernels.
- Residual/RMSNorm/SiLU/gated MLP composition at full shape.
- Post-layer hidden-state oracle or regenerated fixtures large enough to validate complete layer-0 output.

Regression proof after C1R-6n
- Review gate `agent://C1R6nReview` approved blocking correctness and reported two P3 accuracy findings: runner help omitted Q and the Q wrapper test docstring said V.
- Fixes landed in `native_r9700/runner.cpp` and `tests/native_r9700/test_runtime_contract.py`; added direct help assertion for Q.
- Focused review-fix tests: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_runtime_contract.py::test_help_lists_dry_run_kernel_proof_and_transfer_proof_modes tests/native_r9700/test_runtime_contract.py::test_primitive_chain_proof_wraps_supplied_bridge_and_logs_layer0_q_proj_full_inner_cols8_accum_chain -q` exited `0` with `2 passed in 2.71s`.
- Native focused suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -q` exited `0` with `162 passed, 2 warnings in 51.08s` after review fixes.
- Full repository suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -q` exited `0` with `202 passed, 2 warnings in 84.36s` after review fixes.
- Re-review gate `agent://C1R6nReReview` approved with no findings.


C1R-6o fixture extension note
- Later C1R-6o regenerated the shared `layer_trace_fixtures.npz` to add Q RoPE pair arrays. Current fixture SHA is `e138c82eab58403bb018d0c96089941ac3b382144cc81bf36b198a3c08c2a5e1`; existing Q-proj array byte hashes and hardware log evidence above remain unchanged for the Q-proj slice.

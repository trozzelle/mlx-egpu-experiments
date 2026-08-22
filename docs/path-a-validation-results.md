# Path A — Phase 0 numeric-parity validation results

Status: **RUN COMPLETED — GATE PASSED**

Gate: injected path `P` must equal native baseline `R` token-for-token across the prompt set; per-layer numeric deltas are reported and flagged above the `1e-3` fp16 probe tolerance for diagnosis.

## Result summary

- Gate result: **PASS** (3/3 prompts token-exact).
- Run log: `${HOME}/Development/ml/tools/egpu/.worktrees/tinygrad-kv-worker-phase0/logs/runs/20260816-191810-659350000_meta-f16-final.log`.
- Producer weights: `mlx_models/meta-Llama-3.2-1B-Instruct.F16.gguf`.
- Consumer weights: `mlx_models/meta-Llama-3.2-1B-Instruct`.
- Source provenance: official fp16 `meta-llama/Llama-3.2-1B-Instruct` weights on both sides (F16 GGUF producer + mlx safetensors consumer).
- MLX prompt-cache contract: export the `S-1` prefix cache and pass the final prompt token to `generate_step`; passing full `S` plus the full prompt duplicates the prompt.
- Llama-3 RoPE scaling loaded from the MLX `config.json` sidecar and applied to tinygrad's RoPE precompute; the generated GGUF metadata records `rope.freq_base` but not `rope_scaling`.

## Prompt suite

| # | Prompt | S (tokens) | P == R |
|---|---|---|---|
| 0 | `prompt-0` | 6 | True |
| 1 | `prompt-1` | 222 | True |
| 2 | `prompt-2` | 661 | True |

## Per-layer numeric deltas (max|Δ| / mean|Δ| vs native KV)

| Layer | K max|Δ| | K mean|Δ| | V max|Δ| | V mean|Δ| | > 1e-3? |
|---|---|---|---|---|---|
| 0 | 0.0076389312744140625 | 0.00034630033769644797 | 0.00037539005279541016 | 2.0771845811395906e-05 | True |
| 1 | 0.012783050537109375 | 0.0007303535821847618 | 0.0011725425720214844 | 9.534387208987027e-05 | True |
| 2 | 0.020800083875656128 | 0.0009972760453820229 | 0.003493070602416992 | 0.00019622135732788593 | True |
| 3 | 0.03212451934814453 | 0.0011345782550051808 | 0.0035077929496765137 | 0.00024412901257164776 | True |
| 4 | 0.019255638122558594 | 0.0009987273951992393 | 0.003443121910095215 | 0.00026612283545546234 | True |
| 5 | 0.023622244596481323 | 0.0011769216507673264 | 0.004047870635986328 | 0.00027579572633840144 | True |
| 6 | 0.01784515380859375 | 0.0012270527658984065 | 0.003530248999595642 | 0.0003416658437345177 | True |
| 7 | 0.019090652465820312 | 0.0011853931937366724 | 0.00464707612991333 | 0.00039114351966418326 | True |
| 8 | 0.015564918518066406 | 0.0012953929835930467 | 0.0034224987030029297 | 0.0003630796854849905 | True |
| 9 | 0.015559196472167969 | 0.0010691970819607377 | 0.005664646625518799 | 0.0003188189584761858 | True |
| 10 | 0.014158546924591064 | 0.0010957547929137945 | 0.0030842944979667664 | 0.0002942346327472478 | True |
| 11 | 0.01657867431640625 | 0.0010965528199449182 | 0.002843424677848816 | 0.00026680887094698846 | True |
| 12 | 0.01378631591796875 | 0.001003173179924488 | 0.003891170024871826 | 0.00027916315593756735 | True |
| 13 | 0.012420654296875 | 0.001003806246444583 | 0.0032302141189575195 | 0.00034421152668073773 | True |
| 14 | 0.011698722839355469 | 0.001017598551698029 | 0.006541907787322998 | 0.0004919123603031039 | True |
| 15 | 0.01645660400390625 | 0.0010221578413620591 | 0.0058591365814208984 | 0.0005629300139844418 | True |

## Notes

- Flagged layers > 1e-3 fp16 probe tolerance: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]. These are diagnostic tinygrad-vs-MLX implementation deltas; the token gate passed.

## Path C – C1 CPU reference / prompt-cache ABI results (reclassified)

Status: **REFERENCE PASS; NATIVE R9700 C1 OPEN**

The `gate_result` / `status` values below describe CPU/NumPy reference parity only. Per ADR 0005,
they do not satisfy Native R9700 producer acceptance because model-forward tensor work did not run on
the R9700/eGPU.

r_source: both
model: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
fixtures: tests/native_r9700/fixtures
log_path: logs/c1-parity/run.log
json_path: logs/c1-parity/result.json
config_path: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct/config.json
weight_provenance: official fp16 meta-llama/Llama-3.2-1B-Instruct MLX safetensors
rope_config_note: Llama-3 rope_scaling loaded from the MLX config.json sidecar
artifacts: logs/c1-parity
runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface
pci_id: 1002:7551
arch: gfx1201

Native C1 acceptance blocker: no completed run proves Llama 3.2 1B prefill model-forward tensor work
executed on the R9700/eGPU and emitted the accepted prompt-cache artifact.

### C1R hardware primitive evidence (does not close C1)

- `fp32_add_scalar`, `fp16_to_fp32_cast`, `fp32_to_fp16_cast`, and `fp16_matmul_8x16x8` hardware primitive proofs ran through `TinyGPU.app/APLRemotePCIDevice/PCIIface` on `pci_id: 1002:7551`, `arch: gfx1201`, and exited `0` with `host_device_transfer_status: pass`.
- C1R-6b `fp16_matmul_8x16x8` proved a source-grounded RDNA4 `v_dot2_f32_f16` 8x16x8 kernel with fp32 output bytes: `kernel_blob_sha256: 56e4faa6c8fa01ca6d9ea97ac5857ee9fc074d1cd51a883313c97c2fbb6cb28f`, `kernel_text_byte_count: 2508`, `input_layout: a_row_major_then_b_kpair_col_packed`, `output_byte_count: 256`, `tolerance: exact_bytes`, `mismatch_count: 0`.
- C1R-6d `fp32_to_fp16_cast` proved the cast-back primitive needed for fp16 tensor materialization: `kernel_blob_sha256: dc5dd58390142a22d249986d015be589ea62732d36303b68b8528e09a010735d`, `kernel_text_byte_count: 64`, `input_byte_count: 32`, `output_byte_count: 16`, `tolerance: exact_bytes`, `mismatch_count: 0`.
- C1R-6c `fp16_matmul_8x16x8_layer0_k_tile` proved the same kernel on a real Llama layer-0 K-projection partial tile from `tests/native_r9700/fixtures/layer_trace_fixtures.npz` (`fixture_sha256: b13e1c8b5651b638787a0c5061a7cb8f7a0483482aafc1c1041ae7770e2159b3`): `acceptance_scope: hardware_primitive_tile_only`, `model_forward_scope: layer0_k_proj_partial_tile`, `native_prefill_acceptance: open`, `rows_valid: 5`, `tile_rows: 8`, `tile_inner: 16`, `tile_cols: 8`, `tolerance: fp32_ulp<=1`, `max_ulp_diff: 1`, `mismatch_count: 0`, `byte_mismatch_count: 1`.
- C1R-6e `layer0_k_tile_matmul_to_fp16_chain` proved resident on-device chaining for the same real Llama K-projection partial tile: stage 0 wrote fp32 `(8,8)` into a resident intermediate, stages 1-8 cast each row to fp16 without CPU readback between stages, and only the final fp16 tile was downloaded. The final fp16 output VRAM region is explicitly cleared before the compute chain so stale bytes cannot satisfy the proof. Hardware/wrapper evidence: `chain_stage_count: 9`, `chain_readback_between_stages: no`, `kernarg_rewrite_count: 9`, `compute_dispatch_count: 9`, `output_byte_count: 128`, `final_fp16_sha256: 7d8818f895f3e51bce24da8580fb10d76bffa457cba2c061ef2c7c1c0f5ee027`, `tolerance: exact_fp16_bytes`, `mismatch_count: 0`, `byte_mismatch_count: 0`, `final_output_clear_status: pass`, `host_device_transfer_status: pass`, `primitive_chain_proof_wrapper_status: pass`, `failure_stage: none`, `native_prefill_acceptance: open`.
- C1R-6f `fp16_residual_add_layer0_attention_slice8` proved a real Llama layer-0 attention residual-add slice from the same fixture: input packs `layer0_hidden_in_fp16[0,0:8]` then `layer0_o_proj_output_fp16[0,0:8]`, output matches `layer0_attention_residual_fp16[0,0:8]`, `kernel_blob_sha256: 57309c2e2441d96284b716ad71e5612e4b689055fc4e6d8a9be8aebb76764122`, `kernel_text_byte_count: 128`, `acceptance_scope: hardware_primitive_slice_only`, `model_forward_scope: layer0_attention_residual_partial_slice`, `native_prefill_acceptance: open`, `source_arrays: layer0_hidden_in_fp16,layer0_o_proj_output_fp16,layer0_attention_residual_fp16`, `fixture_slice: token=0,hidden_dim=0:8`, `full_fixture_shape: 2x16`, `covered_element_count: 8`, `full_element_count: 32`, `tolerance: exact_fp16_bytes`, `mismatch_count: 0`, `byte_mismatch_count: 0`, and wrapper/hardware exit 0.
- This evidence proves hardware kernel execution, a real layer-0 GEMM tile, stale-output-protected final-only readback, a resident primitive chain, and a real layer-0 residual-add slice; it is still not an accepted `r9700_native` prefill/prompt-cache producer route.

| Prompt | S | N prefix | P tokens | R tokens | Exact | Mismatches | Cache |
|---|---:|---:|---|---|---|---|---|
| prompt-0 | 6 | 5 | `[12366, 13, 578, 469]` | `[12366, 13, 578, 469]` | True | `[]` | `logs/c1-parity/prompt-0-prompt-cache.safetensors` |
| prompt-1 | 222 | 221 | `[128009, 128006, 78191, 271]` | `[128009, 128006, 78191, 271]` | True | `[]` | `logs/c1-parity/prompt-1-prompt-cache.safetensors` |
| prompt-2 | 661 | 660 | `[128009, 128006, 128006, 128006]` | `[128009, 128006, 128006, 128006]` | True | `[]` | `logs/c1-parity/prompt-2-prompt-cache.safetensors` |

flagged_layers_over_1e-3: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`

| Layer | max K | mean K | max V | mean V | >1e-3 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.0078125 | 0.00017515558 | 0.00048828125 | 2.0148847e-05 | True |
| 1 | 0.015625 | 0.00067116303 | 0.0012207031 | 0.00010435124 | True |
| 2 | 0.025390625 | 0.0010470577 | 0.0035400391 | 0.00022782081 | True |
| 3 | 0.015625 | 0.00099507486 | 0.0046386719 | 0.0002902229 | True |
| 4 | 0.015625 | 0.0010966347 | 0.0036621094 | 0.00031915866 | True |
| 5 | 0.0390625 | 0.0013487824 | 0.0031738281 | 0.00032265516 | True |
| 6 | 0.0234375 | 0.0013403689 | 0.00390625 | 0.00039757355 | True |
| 7 | 0.015625 | 0.0013509322 | 0.005859375 | 0.00044677936 | True |
| 8 | 0.016601562 | 0.0014306575 | 0.0048828125 | 0.00039588293 | True |
| 9 | 0.016601562 | 0.0011877895 | 0.0049743652 | 0.00037250351 | True |
| 10 | 0.017822266 | 0.0012244057 | 0.0043945312 | 0.00035624945 | True |
| 11 | 0.015625 | 0.0012250248 | 0.0037841797 | 0.00032808041 | True |
| 12 | 0.018310547 | 0.0011188368 | 0.0043945312 | 0.00034460862 | True |
| 13 | 0.015625 | 0.0011248104 | 0.005859375 | 0.00042447503 | True |
| 14 | 0.014648438 | 0.0011601 | 0.0087890625 | 0.00061878149 | True |
| 15 | 0.015625 | 0.0011586983 | 0.0083007812 | 0.00070930121 | True |

## Path C2 – CPU reference serving integration results (reclassified)

Status: **REFERENCE WRAPPER PASS; NATIVE R9700 C2 OPEN**

The wrapper/fallback/security evidence below was produced with the CPU reference producer. Per ADR
0005, it does not satisfy C2 native acceptance because large-prompt prefill did not route through an
accepted R9700/eGPU producer.

model: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
producer_model_dir: ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct
fixtures_dir: tests/native_r9700/fixtures
prompt_count: 3
threshold_tokens: 128
producer_timeout_s: 300.0
json_path: logs/c2-serving/result.json
log_path: logs/c2-serving/run.log
artifacts_dir: logs/c2-serving
exit_status: 0

Native C2 acceptance blocker: C1R R9700/eGPU producer parity is still open, so no accepted large
prompt serving route through R9700/eGPU prefill exists yet.

| Prompt | S | N prefix | Route | Fallback | Accepted cache | Decoded tokens | R tokens | Exact | Mismatches | Cache |
|---|---:|---:|---|---|---|---|---|---|---|---|
| prompt-0 | 6 | 5 | native_mlx_fallback | below_threshold | False | `[12366, 13, 578, 469]` | `[12366, 13, 578, 469]` | True | `[]` | `` |
| prompt-1 | 222 | 221 | native_producer |  | True | `[128009, 128006, 78191, 271]` | `[128009, 128006, 78191, 271]` | True | `[]` | `logs/c2-serving/prompt-1.prompt-cache.safetensors` |
| prompt-2 | 661 | 660 | native_producer |  | True | `[128009, 128006, 128006, 128006]` | `[128009, 128006, 128006, 128006]` | True | `[]` | `logs/c2-serving/prompt-2.prompt-cache.safetensors` |

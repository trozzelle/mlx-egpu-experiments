# P3 task set 4 — scalar-control Kernel Pack migration

## Status

- **Owner:** `P3ScalarMigration`
- **Worktree:** `feature/r9700-products-wave-a`
- **Result:** all 13 scalar records and their offline/native evidence are generated, sealed, and selectable through the existing asset/catalog boundary. Fresh task-4 native traces close `llama_rmsnorm_f16` and `llama_rmsnorm_epsilon_arithmetic_f16`.
- **Supervisor verification:** the focused pack/asset/catalog/runtime gate passes 27 tests; both hardware-wrapper commands pass on the R9700 with finite 2,048-element outputs and exit status 0.

## Inventory and preservation

The existing `kLlamaKernelManifest` order is retained exactly:

1. `llama_k_projection_f16`
2. `llama_v_projection_f16`
3. `llama_rmsnorm_f16`
4. `llama_rmsnorm_zero_store_f16`
5. `llama_rmsnorm_epsilon_arithmetic_f16`
6. `llama_rope_kv_f16`
7. `llama_causal_attention_score_f16`
8. `llama_causal_attention_softmax_f32`
9. `llama_causal_attention_context_f16`
10. `llama_o_projection_f16`
11. `llama_gated_mlp_f16`
12. `llama_gate_up_projection_f16`
13. `llama_mlp_down_f16`

The legacy `KernelDescriptor` values and all existing `.image` bytes remain unchanged. The asset loader still performs the same code-free manifest, target/provenance, safe-path, no-symlink, size, digest, and descriptor validation. The reviewed task-set-2 K/V attestation resource view (`8` SGPR/`8` VGPR) remains unchanged; the other pack attestations carry the decoded source AMDGPU metadata values without changing the legacy descriptor.

## Generated files

- Generator: `tools/generate_scalar_kernel_packs.py`
- Generated runtime include: `native_r9700/kernel_packs_generated.inc`
- One canonical owning record beside every existing HSA image sidecar:
  - `native_r9700/kernels/llama-k-projection-hsa-assets/llama_k_projection_f16.pack.json`
  - `native_r9700/kernels/llama-v-projection-hsa-assets/llama_v_projection_f16.pack.json`
  - `native_r9700/kernels/llama-rmsnorm-hsa-assets/llama_rmsnorm_f16.pack.json`
  - `native_r9700/kernels/llama-rmsnorm-zero-store-hsa-assets/llama_rmsnorm_zero_store_f16.pack.json`
  - `native_r9700/kernels/llama-rmsnorm-epsilon-arithmetic-hsa-assets/llama_rmsnorm_epsilon_arithmetic_f16.pack.json`
  - `native_r9700/kernels/llama-rope-kv-hsa-assets/llama_rope_kv_f16.pack.json`
  - `native_r9700/kernels/llama-attention-score-hsa-assets/llama_causal_attention_score_f16.pack.json`
  - `native_r9700/kernels/llama-attention-softmax-hsa-assets/llama_causal_attention_softmax_f32.pack.json`
  - `native_r9700/kernels/llama-attention-context-hsa-assets/llama_causal_attention_context_f16.pack.json`
  - `native_r9700/kernels/llama-o-projection-hsa-assets/llama_o_projection_f16.pack.json`
  - `native_r9700/kernels/llama-gated-mlp-hsa-assets/llama_gated_mlp_f16.pack.json`
  - `native_r9700/kernels/llama-gate-up-projection-hsa-assets/llama_gate_up_projection_f16.pack.json`
  - `native_r9700/kernels/llama-mlp-down-hsa-assets/llama_mlp_down_f16.pack.json`
- Each record has six adjacent evidence JSON files: `evidence/numpy-oracle.json`, `source-review.json`, `resource-review.json`, `isa-review.json`, `conformance.json`, and `native-run.json` (78 evidence files total).

The generator loads each `.pack.json`, calls `load_manifest`, `validate_manifest`, `compute_pack_sha256`, and `generate_cpp_initializers`, and emits records in the preserved manifest order. Pack digests are computed from the canonical nonrecursive preimage; no pack digest is hand-maintained in tests or C++.

## Runtime integration

- `kernel_catalog.cpp` includes the generated static view and exposes `llama_kernel_pack_records()` over all 13 records. The legacy `kCatalog` and `find_kernel()` behavior remain unchanged.
- `kernel_pack.h/.cpp` expose exact `find_llama_kernel_pack(name, version, error)` and `admit_llama_kernel_pack(...)` wrappers. Lookup uses the generated, evidence-admitted subset and never ranks, aliases, or falls back.
- `admit_llama_kernel_pack` first resolves the exact selected compatibility key, then delegates to `admit_kernel_pack`. The latter still calls `find_llama_kernel_asset` and `load_verified_kernel_code`; it compares image filename/digest/size/code-object, offsets, kernarg fields, resources, and selected geometry against the asset-owned attestation before publishing the descriptor.
- Canonical pack image paths are repository-relative while legacy asset locations remain direct children of their existing HSA asset roots. Admission derives the existing image root from the declared canonical path; legacy direct-child loading is unchanged.
- `kernel_assets.cpp` now owns exact ABI/resource attestations for all 13 Llama records. No second catalog, compatibility alias, runtime JSON/YAML parser, or F2/G0 path was added.

## Evidence state and fail-closed boundary

All source, license, image, ABI, resource, ISA, numerical, geometry, conformance, and native-run fields are concrete. The generated audit and production-selection spans both contain all 13 records.

- `llama_rmsnorm_f16`: request digest `bed6763cb18850f4aeb076a786b64e473f08ace4acb6418a42599aa6e391507a`, image `0878234b9282e8e83970542e3defed11e081dcae4dc7412c319ac77d179b63d0`, output `ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7`, pack `31f930ef3b456f7960f79f5b92f680c5eeed4cd5c6410112969e63b5fa18c9d3`. The trace-declared expectation remains `none`; the independently validated output contract is `all_zero_fp16_2048`.
- `llama_rmsnorm_epsilon_arithmetic_f16`: request digest `d6a31485901c135d6b2c4bd2e99d1c3d80ffe64b3b888a95436981077626e84a`, image `e440884d246d20580826888b6d279ce61eb24018b2b0196e1a1285071d41e037`, output `6c77b9aae94e81bededb4fc7be64e3ccbb7f6555bbc4741c3e9fa590b08d30a5`, pack `90a0138aed003196ff1091a81af9f1ec9c4c987899ca4a83334cb12faf5a7f39`. Trace expectation is `f16_0x5cf1_316.25`; validated output contract is `uniform_f16_0x5cf1_2048`.
- Both records bind runner SHA-256 `a854988bc8c9b47484c5f5532b013fc1ce35aa2e7818cc705203b566fee04d6c`, `TinyGPU.app/APLRemotePCIDevice/PCIIface`, `1002:7551`, `gfx1201`, `finite_count=2048`, `failure_stage=none`, `exit_status=0`, and `wrapper_exit_status=0`.

The canonical generator resealed each affected evidence payload/reference and pack preimage, regenerated the allocation-free C++ view, and reported `13 selectable`. No pending record or copied pass template remains.

## Supervisor commands

The supervisor executed both frozen hardware-wrapper boundaries:

```sh
tools/native-r9700-hardware-run \
  build/native-r9700-runtime/native_r9700_runner \
  --llama-stage-trace --model <mlx-model-dir> --token-id 0 --layer 0 --position 0 \
  --stage normalized --trace-dir logs/p3-scalar/llama_rmsnorm_f16 \
  --rmsnorm-unit-scale --rmsnorm-zero-input --rmsnorm-output-sentinel

tools/native-r9700-hardware-run \
  build/native-r9700-runtime/native_r9700_runner \
  --llama-stage-trace --model <mlx-model-dir> --token-id 0 --layer 0 --position 0 \
  --stage normalized --trace-dir logs/p3-scalar/llama_rmsnorm_epsilon_arithmetic_f16 \
  --rmsnorm-unit-scale --rmsnorm-zero-input --rmsnorm-output-sentinel \
  --rmsnorm-epsilon-arithmetic
```

Both boundaries satisfy `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, target `1002:7551`/`gfx1201`, matching image digests, finite output, `failure_stage: none`, `exit_status: 0`, `wrapper_exit_status: 0`, request-bound input/output digests, and the trace-specific output contract.

## Focused supervisor gate

```sh
${PY} -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_runtime_contract.py -v
```

The focused supervisor gate passes 27 tests. The real RMSNorm and epsilon-arithmetic dispatch traces pass and are sealed into the generated selection span.

# F2 lane-map RED contract — task set 2

**Status:** RED contract ready for supervisor verification
**Owner:** `F2LaneRed`
**Phase row:** `docs/tasks/r9700-products/phase-f2-gfx1201-wmma-foundation.md`, task set 2
**Changed files:** `tests/native_r9700/test_wmma_lane_map_asset.py`, `tests/native_r9700/test_wmma_lane_map_runner.py`, and this report only

## Scope and guardrails

This lane adds no production source, generated image, shared catalog entry, validation-ledger section, hardware command, compiler invocation, test run, formatter run, or package-manager invocation. The test file is hardware-free. The supervisor owns the post-GREEN R9700 proof and must compare its fresh readback to the calculator record; a CPU/tinygrad result cannot satisfy this gate.

The binding source is the reviewed F2 freeze in `.superpowers/swarm/reports/f2-contract-freeze.md` §2, §3.4, and the `F2 lane-map proof` command. The test freezes the corrected diagnostic ABI separately from the production linear-family ABI. Request-bound target-conformance evidence is not embedded in the immutable HSA asset manifest; the later hardware record binds to the immutable image digest and is linked by the G0 report.

## Frozen diagnostic source/asset contract

- Source: `native_r9700/kernels/wmma_lane_map_gfx1201.cpp`.
- Device symbol: `wmma_lane_map_gfx1201`, one C-linkage global kernel. It must execute exactly `v_wmma_f32_16x16x16_f16` on `gfx1201` wave32 device code; host, CPU, HIP-launch, fixture, and tinygrad paths are not admitted.
- Asset root: `native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets`; direct children are `wmma_lane_map_gfx1201.image` and `wmma_lane_map_gfx1201.json`.
- Kernarg: exactly four 8-byte pointer fields in order `a` (offset 0), `b` (offset 8), `c` (offset 16), and `observations` (offset 24), schema name `wmma-lane-map-gfx1201-v1`, segment size 32, alignment 8, no preload and no tail padding. `a`/`b` are FP16 operand matrices, `c` is the FP32 accumulator matrix, and `observations` receives raw words.
- Launch: one wave, `workgroup=(32,1,1)` and `global=(32,1,1)`.
- Readback: every lane writes exactly sixteen raw `uint32` words in order `A0..A3, B0..B3, D0..D7`; lane stride is 64 bytes and total readback is exactly 2048 bytes per case. The proof has three matrix-loaded cases: `a_map`, `b_map`, and `d_map`.
- Matrix cases: `a_map` loads the FP16 A element tag `(row*16+column+1)/256` with inactive matrices zero; `b_map` loads the same exact FP16 B tags with inactive matrices zero; `d_map` loads the FP32 C element tag `row*16+column+1` with inactive matrices zero so D readback exposes the D map. Exact FP16 halves are packed into raw A/B words and exact FP32 bits into raw D words.
- Result mapping: the expected comparator record maps every matrix element to calculator lane/register/half ownership for A and B and lane/register ownership for D. The manifest retains the frozen calculator A/B/D register-count, lane, GPR, bit-range, and point records. The source must make raw register ownership observable through the instruction/readback path; no hidden transpose or compensating permutation is acceptable.
- Descriptor/admission: manifest binds target `gfx1201`, wave32 code properties, positive descriptor/entry/resource values, exact 32-byte kernarg, zero private/LDS/preload fields, 256-byte entry alignment, and one admitted ELF symbol. Source/image paths and SHA-256 digests bind the immutable pair.
- Numerical identity: the source/manifest carries `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`; this finite FP16/FP32 diagnostic policy does not replace exact lane-map equality.
- Comparator/evidence: `native_r9700.wmma_lane_map.validate_lane_map_conformance(expected_records, observed_records, asset_identity)` consumes normalized calculator `{schema_version, calculator_revision, layout_digest, instruction, wave_size, a, b, d}` records, request-bound observed identity plus `{schema_version, request_id, runtime_substrate, pci_id, arch, wave_size, instruction, cases:{a_map:{raw_words},b_map:{raw_words},d_map:{raw_words}}}`, and immutable target/source/image/manifest/pack identity. Each case is exactly 32×16 `uint32` words. The hardware-free tests require one exact nonzero pass and reject lane/order/bit/value mutations. The pure function returns fixed `record_id: f2-wmma-lane-map-conformance-v1`, exact equality/status, and core EvidenceRef fields; the writer/CLI adds record path and canonical record digest while emitting a separate record with exactly `record_kind: target_conformance`, `evidence_slot: conformance`, `producer_kind: r9700_native`, empty `tool_digest`, and nonempty request-bound target/image/pack/input/output digests. Offline-oracle, offline-review, CPU, filename-only, shape-only, or request-unbound evidence is rejected. The immutable manifest carries only source/image identity; it must not be mutated with a later hardware digest.
- Selection: the probe remains diagnostic-only (`diagnostic_only=true`, `model_selectable=false`) and is absent from `kernel_catalog.cpp` and `llama_layer_executor.cpp`. It may be loaded through its dedicated proof path, but it is not a model kernel.

## Tests and expected RED reasons

| Test | Contract exercised | Expected RED reason before production implementation | Production change needed |
|---|---|---|---|
| `test_lane_map_source_is_one_freestanding_wave32_wmma_probe` | C-linkage symbol, four-pointer ABI, exact WMMA instruction, wave/lane IDs, raw A/B/D register readback, 2048-byte per-case layout, numerical-policy token, no host fallback | `native_r9700/kernels/wmma_lane_map_gfx1201.cpp` is absent (or lacks the required device instruction/readback ABI) | Add the freestanding diagnostic source with A/B/C/observations pointers, one wave32 dispatch, the three tagged matrix cases, one WMMA instruction, and raw A0-A3/B0-B3/D0-D7 stores. |
| `test_lane_map_manifest_binds_descriptor_kernarg_wave_and_launch_geometry` | Dedicated image/manifest existence, gfx1201/wave32/instruction identity, descriptor/resource/kernarg fields, one-wave geometry, exact 2048-byte readback and three cases, source/image digests | `wmma-lane-map-gfx1201-hsa-assets/` and its image/manifest are absent; once present, missing generated metadata or digest drift is an admission failure | Register only this checked-in source with the existing HSA generator/loader path, generate the direct-child image/manifest, and bind descriptor, launch, raw-readback, source, image, and case metadata without copying scalar gate/up resources. |
| `test_lane_map_manifest_retains_calculator_result_register_mapping` | Expected A/B/D register counts, lane/GPR/bit equations, point records, and raw output order | The manifest and its expected-layout/result-register mapping are absent | Persist the calculator-derived expected record as explicit diagnostic metadata and preserve it for supervisor comparison; do not call it accepted hardware evidence or compensate in a production kernel. |
| `test_lane_map_conformance_record_contract_is_request_bound` | `native_r9700.wmma_lane_map.validate_lane_map_conformance(...)` over normalized calculator element records and all three 32×16 raw-word cases; exact equality and core `target_conformance/conformance` fields, with nonempty target/image/pack/input/output digests and empty `tool_digest` | `native_r9700/wmma_lane_map.py` and its comparator seam are absent; malformed runner normalization would fail the matrix | Add the calculator/readback comparator and its module CLI, consume normalized expected/observed/asset mappings, return exact equality/status and core EvidenceRef fields, and let the CLI writer add record path/ID/canonical record digest outside the immutable HSA manifest. |
| `test_lane_map_conformance_rejects_lane_order_bit_and_value_mutations` | Four fail-closed mutations (lane swap, register-order swap, bit flip, and word-value change) across the nonzero matrix cases | The comparator seam is absent; once present, any mutation that returns `status: pass` is a conformance-admission defect | Reject every changed raw word with `status: fail`, `lane_map_status: fail`, and `exact_equality: false`; do not reduce the proof to shape-only validation. |
| `test_lane_map_probe_cannot_be_selected_as_a_model_kernel` | Diagnostic-only metadata and absence from product catalog/model graph | This guard is expected to pass against the current empty catalog; it fails if implementation accidentally registers the probe as a selectable model kernel | Keep the dedicated proof loader separate from `kCatalog` and `llama_layer_executor` model-stage selection; do not add a model graph alias or catalog entry. |

## Focused supervisor commands

After this RED wave, the exact focused contract command is:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_wmma_lane_map_asset.py \
  tests/native_r9700/test_hsa_code_image_generator.py \
  tests/native_r9700/test_hsa_code_image_loader.py -v
```

Only after the focused tests are GREEN may the supervisor run the hardware proof frozen by task set 1:

```sh
tools/native-r9700-hardware-run \
  build/f2-wmma/wmma_lane_map_gfx1201 \
  --asset-root native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets \
  --log logs/f2/wmma-lane-map-proof.json
```

The hardware log must identify `1002:7551`/`gfx1201`, TinyGPU.app/APLRemotePCIDevice/PCIIface, wave32, the exact instruction, source/image/manifest digests, all three observed A/B/D raw-word cases, exact calculator equality, and the passing separate `target_conformance/conformance` EvidenceRef. A mismatch is a failed proof and requires a reviewed F2 contract update; it must not be hidden by a transpose.

No verification command was run by this lane, per the swarm execution contract.

## F2LaneReview findings converted to focused RED contracts

The reviewer findings are now represented by hardware-free tests; no
production source, generated image, catalog, validation ledger, compiler,
package manager, hardware command, or formatter was run or edited.

| Finding | RED contract |
|---|---|
| Runtime substrate was accepted whenever nonempty | `test_lane_map_conformance_rejects_non_admitted_runtime_substrate` mutates the synthetic request to `cpu`, `other-loader`, and a different TinyGPU interface, and requires fail-closed input validation. Only `TinyGPU.app/APLRemotePCIDevice/PCIIface` is admitted. |
| Observed JSON could self-attest `pack_sha256` | `test_lane_map_identity_uses_computed_pack_digest_not_observed_claim` requires the comparator seam to compute the digest when the observed record omits it and to ignore or reject forged observed values. `test_lane_map_identity_changes_when_immutable_pack_metadata_is_tampered` copies the image, mutates manifest raw-word order, and requires a changed computed identity. |
| The documented host runner was absent | `test_wmma_lane_map_runner.py` requires `native_r9700/wmma_lane_map_runner.cpp`, compiles it with the current AMDev/runtime source closure into `build/f2-wmma/wmma_lane_map_gfx1201`, verifies the exact `--asset-root`/`--log` CLI, checks the three tagged cases and 2048-byte readbacks, checks the request-bound observed schema, and exercises hardware-free help/invalid-argument paths with an unreachable TinyGPU socket. |

The diagnostic pack digest test freezes an independent UTF-8 RFC8785-style
canonical JSON preimage:

```text
{
  "domain": "r9700-wmma-lane-map-diagnostic-pack-v1",
  "pack": {
    "schema_version": 1,
    "target": "gfx1201",
    "source_path": "...",
    "source_sha256": "...",
    "image_path": "...",
    "image_sha256": "...",
    "manifest_path": "...",
    "manifest_sha256": "...",
    "abi": <exact four uint64 pointer fields>,
    "geometry": {
      "wave_size": 32,
      "workgroup": [32, 1, 1],
      "global": [32, 1, 1],
      "readback_bytes": 2048,
      "raw_words_per_lane": 16,
      "observation_cases": ["a_map", "b_map", "d_map"]
    },
    "instruction": "v_wmma_f32_16x16x16_f16",
    "raw_word_order": ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"],
    "numerical_policy": "F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1"
  }
}
```

The exact focused RED command for supervisor execution is:

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_wmma_lane_map_asset.py \
  tests/native_r9700/test_wmma_lane_map_runner.py -v
```

Before production fixes, the expected RED is the non-admitted substrate
mutations, observed-pack fallback/tamper cases, and ordinary assertion
failures (`RED: host runner missing`) for the missing host-runner
source/compile contract. No verification command was run in this lane.

# F2 contract freeze — task set 1

**Status:** Needs review  
**Owner:** `F2Contract`  
**Phase row:** `docs/tasks/r9700-products/phase-f2-gfx1201-wmma-foundation.md`, task set 1  
**Scope:** source/ABI/validation freeze only. No production WMMA source, generated image, catalog migration, hardware run, or shared validation-ledger edit was performed.

## 1. Grounded decisions

### 1.1 Immutable upstream pins and allowed reuse

The following are the complete F2 P0 source/tool set from `docs/REFERENCES.md` and `docs/upstream-reference-manifest.yaml`; the layout-only AITER reference is recorded immediately below and is not an F2 runtime dependency. A revision is not a license to copy: any compiled, translated, vendored, or generated local artifact still requires the file-level review and source/image digests required by the manifest.

| Manifest ID | Immutable source and exact path(s) | License recorded by manifest/source | F2 role and allowed reuse |
|---|---|---|---|
| `llvm-amdgpu-usage` | `https://github.com/llvm/llvm-project` at `8dba93818258d95c46fa2c17e902a8256e4d91b5`; `llvm/docs/AMDGPUUsage.rst` | Apache-2.0 WITH LLVM-exception; `license_review: reference-only` | **Normative.** Use target/code-object, AMDHSA descriptor, kernarg size/alignment, segment, wave, and dispatch semantics. No LLVM runtime/toolchain dependency in the product path. |
| `amd-isa-spec-manager` | `https://github.com/GPUOpen-Tools/isa_spec_manager` at `452645535ac05f466b06a13e5eafeb5a86d3ad11`; `include/amdisa`, `source/examples/basic_decoder.cpp`, `source/examples/multi_arch_decoder.cpp`, `documentation/spec_documentation.md` | MIT; `license_review: required_before_vendoring_tool_code` | **Tool and normative decoder.** Use the pinned RDNA4 XML/`IsaDecoder` for admitted-image disassembly, WMMA-instruction presence, and unsupported-instruction rejection. No runtime dependency; no vendoring without per-file review. |
| `radeon-gpu-analyzer` | `https://github.com/GPUOpen-Tools/radeon_gpu_analyzer` at `39688b004af6993f7146dd8e26b52994ec020fe6`; `source`, `README.md` | MIT; `license_review: required_before_vendoring_tool_code` | **Offline tool.** Produce ISA, SGPR/VGPR, LDS/scratch, control-flow, and source-correlation evidence for the pack. RGA output is admission/review evidence, not runtime correctness. |
| `amd-matrix-instruction-calculator` | `https://github.com/ROCm/amd_matrix_instruction_calculator` at `2ef91896bcdc4d26624f952e5c905c787cd9bc9e`; `matrix_calculator.py`, `README.md` | MIT; `license_review: required_before_vendoring_tool_code` | **Expected-layout tool.** Generate the gfx1201 register/lane record below. The generated record is not hardware acceptance; task set 2 must independently prove it through the local loader/readback path. Do not hard-code a narrative lane map. |
| `rocm-libraries-rocwmma-hipblaslt` / rocWMMA | `https://github.com/ROCm/rocm-libraries` at `f7f2aee8e764e612f49f2dc030b7e1639fb30d34`; `projects/rocwmma/samples/simple_hgemm.cpp`, `projects/rocwmma/samples/perf_hgemm.cpp`, `projects/rocwmma/library/include/rocwmma` | Manifest correctly records component-specific licensing; the pinned sample files and `library/include/rocwmma/rocwmma.hpp` carry an MIT header. No repository-wide license assertion. | **Normative plus Port/Adapt.** Use the 16x16x16 fragment/type/layout, wave-size, GEMM decomposition, LDS synchronization, and tile geometry as source patterns. Copy/translate only an individually reviewed file, preserving notices and recording modifications. Do not link rocWMMA/HIP/ROCm host runtime. |
| `rocm-libraries-rocwmma-hipblaslt` / hipBLASLt | Same repository/revision; `projects/hipblaslt` (no path outside this manifest entry is admitted). | Component-specific; no repository-wide assertion. | **Pattern and narrow Port/Adapt only.** Study problem classification, weight layouts, and epilogue/tuning patterns. Exact file/component license review is required before any reuse. Never link or port the host runtime wholesale. |
For the layout proof only, the pinned AITER reference is `https://github.com/ROCm/aiter` at `35c652ed3bd34e5d5828954e1545babc9255a69a`, `aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py`. It is a Pattern/Port/Adapt source with file-level license review required; its attention-specific layout is not a linear WMMA weight-pack contract and no AITER runtime is admitted.

The local authority is not an upstream substitute: `native_r9700/hsa_code_image_asset.*`, `kernel_assets.*`, `kernel_catalog.*`, `experiments/native-r9700-runtime/generate_hsa_code_image.py`, `native_r9700/primitives.py`, `native_r9700/model_weight_binder.cpp`, and the checked-in scalar Llama kernels/tests define the current loader, admission, shape, and correctness-control seams.

### 1.2 Local contracts inspected

* `native_r9700/hsa_code_image_asset.h:12-24` defines the attested image record (`image_sha256`, descriptor/entry offsets, `rsrc1/2/3`, `wave32`, source path/digest, schema). `hsa_code_image_asset.cpp:205-527,529-575` rejects unknown manifest fields and binds exact target, image layout, descriptors, source/image digests, and resource values. `image_is_wave32()` reads the AMDHSA descriptor `kernel_code_properties` at descriptor offset +56, bit `0x400`; the bit is expected metadata, not a lane-map proof.
* `native_r9700/kernel_catalog.h:12-38` defines `KernelDescriptor` and requires target-bound code, `rsrc1/2/3`, workgroup/global dimensions, and kernarg bytes. `kernel_catalog.cpp:140-177` rejects duplicate names, non-lowercase/nonmatching SHA-256, empty code, zero resources, zero geometry, and zero kernarg bytes; `kCatalog` is intentionally empty until a reviewed asset is integrated.
* `native_r9700/kernel_assets.h:13-50` defines `KernelAssetLocation`, `LlamaKernelAsset`, and the verified direct-child code loader. `kernel_assets.cpp:159-278` requires `gfx1201`, `source_amdgpu_metadata`, matching descriptor/location digests, a non-symlink asset root, a safe direct-child code path, and a complete digest-validated descriptor.
* `experiments/native-r9700-runtime/generate_hsa_code_image.py:317-458` is a closed `REVIEWED_ASSETS` source/ABI allowlist. `validate_source_profile()` (`:572-637`) rejects preprocessor/host/runtime content and requires one exact C-linkage device ABI. `_descriptor()` (`:1010-1071`) requires a 64-byte descriptor, exact compiler kernarg bytes, exact group/private/preload/properties fields, a 256-byte-aligned entry, and positive descriptor resources; `MAX_IMAGE_BYTES` is 4 MiB and `PM4_PROGRAM_ENTRY_ALIGNMENT` is 256. `_publish_output()` (`:1251-1321`) publishes the image/manifest pair atomically.
* The current gate/up scalar asset is concrete evidence of the existing admission shape, not a WMMA resource prediction: `native_r9700/kernels/llama_gate_up_projection_f16.cpp:1-67` uses hidden size 2048, intermediate size 8192, FP32 accumulators, and one final FP16 cast; its generated manifest `native_r9700/kernels/llama-gate-up-projection-hsa-assets/llama_gate_up_projection_f16.json:24-142` records `group_segment_bytes=4100`, `private_segment_bytes=0`, `kernarg_bytes=56`, `kernarg_preload_bytes=0`, `kernel_code_properties=1032 (0x408)`, `rsrc1=3222208515`, `rsrc2=295044`, `rsrc3=320`, target `gfx1201`, and source/image digests. Those values must not be copied to a WMMA image without a generated-image/ISA review.
* `native_r9700/config.py:36-40,177-182` fixes the current model at hidden size 2048, intermediate size 8192, eight KV heads, and head dimension 64. `native_r9700/model_weight_binder.cpp:592-618` and `:660-711` bind gate/up weights as `[8192,2048]`, O/Q as `[2048,2048]`, and K/V as `[512,2048]`, all F16. `native_r9700/primitives.py:107-143` is the scalar reference contract: `(M,K) x (K,N) -> (M,N)`, F16 operands, FP32 contraction, one final F16 rounding.
* The existing in-page kernarg contract (`tests/native_r9700/test_kernarg_slot_contract.py:1-8,33-78,102-118`) gives a 4 KiB page, ten 256-byte slots, and 8-byte-aligned stage schemas. LLVM AMDGPU Usage additionally requires the backing kernarg allocation to be at least 16-byte aligned (`AMDGPUUsage.rst` pinned revision, dispatch procedure around lines 5883-5886).

## 2. Expected gfx1201 WMMA lane/register record

This is **expected calculator output only**. It is not accepted R9700 hardware evidence and must not be labeled `r9700_native` until task set 2 has a fresh request-bound hardware log.

Pinned calculator facts:

* calculator source revision: `2ef91896bcdc4d26624f952e5c905c787cd9bc9e` (`VERSION = 1.3.2`);
* calculator source symbols: `dict_insts['rdna4']['v_wmma_f32_16x16x16_f16']` supplies the instruction record; `InstCalcGfx12` emits the wave32 register/lane equations and point/layout queries.
* architecture alias: `gfx1201 -> rdna4`;
* instruction: `v_wmma_f32_16x16x16_f16`, VOP3P opcode `0x40`;
* atom: `M=16, N=16, K=16`, one block, 8192 FLOPs, 16 execution cycles;
* wave32 register use: A=4 GPRs, B=4 GPRs, C=8 GPRs, D=8 GPRs, GPR alignment=4 bytes;
* source/destination types: Src0 FP16, Src1 FP16, Src2 FP32, Vdst FP32; no OPSEL/CBSZ/ABID/BLGP modifiers; NEG is supported but the expected record uses the unmodified/default mapping;
* wave64 is explicitly not admitted by this freeze.

With lane in `[0,31]`, matrix coordinates in `[0,15]`, and no instruction modifiers, the calculator's wave32 equations are:

```text
A[i][k] GPR  = 2*floor(k/8) + (floor(k/2) mod 2)
A[i][k] bits = [16*(k mod 2) + 15 : 16*(k mod 2)]
A[i][k] lane = 16*(floor(k/4) mod 2) + i

B[k][j] GPR  = 2*floor(k/8) + (floor(k/2) mod 2)
B[k][j] bits = [16*(k mod 2) + 15 : 16*(k mod 2)]
B[k][j] lane = 16*(floor(k/4) mod 2) + j

C[i][j] / D[i][j] GPR  = i mod 8
C[i][j] / D[i][j] lane = 16*floor(i/8) + j
```

Calculator point records that the lane-map probe must independently reproduce:

```text
A[0][0] = v0{0}.[15:0]       A[0][1] = v0{0}.[31:16]
A[0][4] = v0{16}.[15:0]      A[0][8] = v2{0}.[15:0]
B[0][0] = v0{0}.[15:0]       B[1][0] = v0{0}.[31:16]
B[4][0] = v0{16}.[15:0]      B[8][0] = v2{0}.[15:0]
D[0][0] = v0{0}              D[8][0] = v0{16}
D[15][15] = v7{31}
```

Expected-layout command (the supervisor must materialize exactly the pinned checkout; a missing checkout is a blocker, not permission to use another revision):

```sh
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
CALC=<tools-root>/amd_matrix_instruction_calculator-2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py
$PY "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --detail-instruction
$PY "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --A-matrix --csv
$PY "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --B-matrix --csv
$PY "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --D-matrix --csv
```

The expected output must be retained with the G0 record as calculator provenance. The lane-map hardware proof must compare observed register/lane/bit ownership to this record and report a mismatch rather than adding a transpose or compensating permutation in the production kernel.

### 2.1 Diagnostic lane-map probe ABI

The diagnostic probe has four pointer-only kernargs: `a_matrix` (`uint64`, offset 0), `b_matrix` (offset 8), `c_matrix` (offset 16), and `observations` (offset 24). The descriptor-reported kernarg segment is exactly 32 bytes with no tail padding or preload. One wave32 launches with workgroup/global geometry `(32,1,1)` and executes exactly one `v_wmma_f32_16x16x16_f16`. Each lane writes raw A (4 VGPR), B (4 VGPR), and D (8 VGPR) words to a fixed lane-major observation record: 32 lanes × 16 `uint32` words = 2048 bytes, ordered A0–A3, B0–B3, D0–D7.

The hardware proof runs three independent tagged cases. `A-map` uses A element `(row,col) = (row*16+col+1)/256` as exactly representable FP16, with B and C zero. `B-map` uses the same FP16 tag formula for B, with A and C zero. `D-map` uses C element `(row,col) = row*16+col+1` as exactly representable FP32, with A and B zero, so WMMA returns the tagged C fragment unchanged. The kernel uses ordinary source-grounded fragment loads; it never hard-codes the expected lane/register mapping. Each case emits a separate 2048-byte observation record. The probe has no model weights, graph selector, compensating transpose, or production-kernel catalog entry.

Task set 2 owns the hardware-free comparator `native_r9700.wmma_lane_map.validate_lane_map_conformance(expected_records, observed_records, asset_identity)` and its CLI. `expected_records` is the parsed pinned calculator detail/A/B/D mapping; `observed_records` contains the three request-bound hardware cases. The comparator derives expected raw tags from the public formulas above, requires exact lane/register/bit equality, and emits the separate `target_conformance/conformance` EvidenceRef. Hardware evidence never mutates or embeds into the asset manifest.

## 3. Frozen first family and ABI

### 3.1 Family identity and dimensions

The first admitted family is a **single linear operator instantiated for the gate and up projection weights**, not two independently accepted shape families and not a fused gate/up epilogue:

```text
family: f2-linear-gate-up-f16-v1
source symbol: linear_wmma_f16
source path: native_r9700/kernels/linear_wmma_f16.cpp
asset root: native_r9700/kernels/linear-wmma-f16-hsa-assets
image: linear_wmma_f16.image
manifest: linear_wmma_f16.json
producer target: gfx1201
wave: 32
instruction: v_wmma_f32_16x16x16_f16
canonical shape: A[M=128,K=2048] x B[K=2048,N=8192] -> D[M=128,N=8192]
```

The packet's suggested square `M=128,K=2048,N=2048` is **not** retained for this first family. It matches the scalar O projection (`[2048,2048]`) but contradicts the first profile-ordered gate/up source: `config.py` fixes `intermediate_size=8192`, `model_weight_binder.cpp` requires gate/up `[8192,2048]`, and `llama_gate_up_projection_f16.cpp` computes `kIntermediateSize=8192`. The canonical family therefore uses `N=8192`; a square O-projection family is downstream and requires its own source/image/evidence record.

`K=2048` and `N=8192` are both exact multiples of the pinned 16x16x16 atom (`K/16=128`, `N/16=512`); no K or N tail is admitted in this first family. The family is not a generic arbitrary-K/N interface.

### 3.2 One bounded-M tail policy

* Canonical full tile: `M=128`.
* The single family `f2-linear-gate-up-f16-v1` accepts the bounded runtime dimension `1 <= M <= 128` under one named `tail_policy: masked/padded` and one closed `geometry_rule: f2-wmma-64x64-m-tail-v1`. M values are requests within this family, not separate shape families and not 128 enumerated compatibility records.
* Pad activation rows to `ceil(M/16)*16` (and to the selected 64-row output supertile as needed) with zero F16 rows. Every WMMA tile runs the same FP32 accumulation; D stores are masked to `row < M`. No bytes outside the valid `M x N` output may be written. The tail output is compared only over valid rows, and padding/sentinel rows must remain unchanged.
* `M=13` is the required non-multiple tail case in the standalone command. The numerical policy is `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1` below.

The closed `geometry_rule: f2-wmma-64x64-m-tail-v1` is keyed by the actual request `M`, with fixed `K=2048` and `N=8192`: one wave computes one 16x16 output atom; `workgroup=(4*32,4,1)=(128,4,1)`; one workgroup computes a 64x64 output supertile; `grid=(ceil(M/64),ceil(8192/64),1)`; and the AQL global size is `(grid.x*128,grid.y*4,1)`. This is one rule, not one geometry case per M. It is an expected launch contract to be checked by task sets 2–4, not hardware evidence. The source must retain element-level M masking even when a tile's starting row is in bounds.
The source symbols grounding this expected decomposition are `projects/rocwmma/samples/simple_hgemm.cpp:hgemm_rocwmma_d`, `ROCWMMA_M/N/K`, `T_BLOCK_X`, `T_BLOCK_Y`, and its `gridDim` calculation. The sample's comments state that each wave computes one output block and that `K` is stepped by the 16-element atom.

### 3.3 Source tensor versus physical WMMA weight layout

`source_tensor_layout_version: f16-row-major-nk-source-v1` is the pinned source contract:

* activation A: F16 row-major `[M,K]`, contiguous row stride `K=2048`;
* source model weights: F16 row-major `[N,K]=[8192,2048]`, with source element `source_weight[n*K+k]`, exactly as the local `ModelWeightBinder` and scalar gate/up source bind and index the safetensors tensor;
* logical WMMA B: `B[k,n] = source_weight[n,k]`;
* output D: F16 row-major `[M,N]`, contiguous row stride `N=8192`;
* source persistence has no quantization, sparsity, or implicit source-tensor padding.

The physical WMMA weight representation is a different contract. `weight_packing_version: f2-wmma-physical-tile-v1` is reserved but **not admitted or frozen until its source proof exists**; `f16-row-major-nk-v1` MUST NOT be used as the physical pack merely because it names the source tensor layout. The first image must consume the proved physical representation, not the safetensors source rows directly.

The pinned sources bound what may be reused but do not yet establish this linear-family physical mapping: the rocWMMA `simple_hgemm.cpp` sample proves `matrix_a` row-major, `matrix_b` col-major, `ldb=k`, 16x16x16 fragments, and the 4-wave-by-4-row-block geometry. The pinned rocWMMA API/header symbols `projects/rocwmma/library/include/rocwmma/rocwmma.hpp::{matrix_b,col_major,fragment,load_matrix_sync}`, `internal/io_config.hpp::{IOConfig}`, `internal/io_layout.hpp`, `internal/mapping_util.hpp`, `internal/accessors_impl.hpp::{GetMappingUtil}`, and `internal/layout/matrix_layout_traits_impl.hpp` are mandatory layout-proof inputs; they define fragment/load/layout machinery but explicitly leave packed fragment element order and user prepacking to the implementation. The pinned AITER `flash_attn_func_gfx1201.py` proves an attention-specific wave32 operand/register mapping and padded LDS K/V (`HEAD_DIM + 4`), not a gate/up linear B-tile layout; the matrix calculator proves register/lane ownership, not global weight-byte or LDS-tile ordering. The local scalar source proves the source row indexing only. No transpose, interleave, swizzle, tile stride, or padding may be inferred from those analogies.

Task set 3's proof command is an acceptance harness, not a layout oracle: it must consume either a direct mapping established by those pinned header symbols or an explicitly reviewed local `f2-wmma-physical-tile-v1` layout specification. A local format is admissible only when the spec names every source-element-to-byte/tile/LDS rule, carries its own digest/version, round-trips the inverse fixture, and is later bound to task-set-5 request-bound hardware numerical evidence; the harness MUST reject a missing/hand-waved spec rather than inventing a mapping.
Task set 4 is therefore blocked until task set 3 produces and accepts the `offline_review` artifact with the exact `source_weight[n*K+k] -> physical byte offset -> 16x16 B tile/LDS offset` mapping, alignment/stride/padding/swizzle (if any), inverse/conformance fixture, layout-spec/version digest, and source/tool digests. A missing or contradictory proof keeps task sets 4–6 and G0 fail closed; it cannot be repaired by an image-side compensating permutation.
There is no epilogue, bias, activation, residual, or graph selection in the F2 image. F3 may invoke the same family for gate and up only after this standalone record and the physical layout proof are accepted.

### 3.4 Kernarg order and alignment

The standalone source ABI is frozen as the smallest fixed-shape ABI; K and N are family constants, not runtime fields:

```cpp
extern "C" __attribute__((global)) void linear_wmma_f16(
    const unsigned short* activation,  // A[M,2048], offset 0
    const unsigned short* weight_nk,   // physical f2-wmma-physical-tile-v1 pack for logical [8192,2048], offset 8
    unsigned short* output,            // D[M,8192], offset 16
    unsigned int m);                   // valid rows, offset 24
```


The ABI field remains named `weight_nk` for the logical N-by-K operator, but its pointer MUST reference the versioned physical WMMA pack after model-load prepacking; it MUST NOT point at the raw safetensors source tensor. Source-layout and physical-pack EvidenceRefs are separate.
The exact schema is:

```json
{"name":"f2-linear-wmma-f16-v1","bytes":32,"tail_padding_bytes":4,"fields":[
  {"name":"activation","offset":0,"size":8,"alignment":8,"type":"uint64"},
  {"name":"weight_nk","offset":8,"size":8,"alignment":8,"type":"uint64"},
  {"name":"output","offset":16,"size":8,"alignment":8,"type":"uint64"},
  {"name":"m","offset":24,"size":4,"alignment":4,"type":"uint32"}
]}
```

Kernarg validation requires fields to be aligned, non-overlapping, and fully contained in the descriptor-reported segment. The descriptor segment may exceed the final field end only by the recorded `tail_padding_bytes`; here offsets 28–31 are exactly four tail-padding bytes and MUST be zero in the submitted kernarg. The existing local in-page binder must place this schema in one zero-padded 256-byte slot; no field may be relocated or preloaded. Pointer fields are naturally 8-byte aligned; `m` is 4-byte aligned; the segment's maximum argument alignment is 8 bytes; and the backing allocation must be at least 16-byte aligned per pinned LLVM AMDGPU Usage/HSA dispatch semantics. `m=0` or `m>128` is rejected before submission.

### 3.5 Numerical policy

`F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1` is the one numerical policy name carried by every later F2 task:

1. Read F16 inputs; multiply/accumulate in FP32; cast once to F16 at the D store.
2. Require finite values for every valid output element and unchanged padding/sentinel rows.
3. Build one canonical input record for each request (`M=128` and `M=13`) and bind its `input_digest` to all comparisons. Compare the WMMA output independently with:
   * the NumPy FP64 diagnostic/oracle (`producer_kind: cpu_reference`, `record_kind: offline_oracle`, `evidence_slot: numpy_oracle`), retained for analysis and not target evidence; and
   * the accepted scalar/native projection request (`producer_kind: r9700_native`, `record_kind: target_conformance`, `evidence_slot: scalar_native_projection`) on the identical input bytes, with its own request-bound output/evidence record.
   `native_r9700.primitives.matmul` remains the retained scalar CPU control used to construct/check the oracle; it is not a substitute for the request-bound native projection record.
4. Record max absolute, mean absolute, finite count, and distribution/tail rows for full `M=128` and the required `M=13` tail, with separate NumPy and scalar/native comparison outcomes.
5. A reviewed max/mean tolerance is a required G0 field. No WMMA tolerance is guessed here: the existing `fp32_abs<=2e-6_or_ulp<=64` records in `runtime.h` are scoped K/V/Q primitive chains, while the existing gate/up trace has different measured ULP behavior. Until task set 5 records and review accepts the gate/up-family threshold, missing/unknown tolerance rejects production admission.

Neither a `cpu_reference` output, the calculator output, nor a request-unbound native log can promote the image. The native projection comparison is a separate acceptance gate even when the NumPy oracle passes.

### 3.6 Concrete evidence references

F2 and P3 use this exact closed `EvidenceRef` shape; an opaque nonempty string is not an evidence record:

```text
EvidenceRef {
  record_path,
  record_kind,
  evidence_slot,
  record_id,
  record_sha256,
  subject_target,
  image_sha256,
  pack_sha256,
  producer_kind,
  tool_digest,
  input_digest,
  output_digest
}
```

`record_kind` is exactly these five values: `offline_oracle`, `offline_review`, `target_conformance`, `native_run`, and `benchmark`. `evidence_slot` is exactly these nine values: `numpy_oracle`, `source_review`, `isa_review`, `resource_review`, `layout_proof`, `scalar_native_projection`, `conformance`, `native_run`, and `benchmark`. Every reference contains the common fields `record_path`, `record_kind`, `evidence_slot`, `record_id`, and canonical `record_sha256`; omitted or extra fields, an unknown kind or slot, and every other kind/slot combination reject.

The closed five-kind/evidence-slot matrix is:

| `record_kind` | `evidence_slot` | Required fields and exact values | Fields exactly empty |
|---|---|---|---|
| `offline_oracle` | `numpy_oracle` | `record_path`, `record_id`, `record_sha256`, `input_digest`, `output_digest`; `producer_kind=cpu_reference` | `subject_target`, `image_sha256`, `pack_sha256`, `tool_digest` |
| `offline_review` | `source_review`, `isa_review`, `resource_review`, or `layout_proof` | `record_path`, `record_id`, `record_sha256`, `subject_target`, `image_sha256`, `pack_sha256`, `tool_digest`, `input_digest`, `output_digest`; `producer_kind=""` | none |
| `target_conformance` | `scalar_native_projection` or `conformance` | `record_path`, `record_id`, `record_sha256`, `subject_target`, `image_sha256`, `pack_sha256`, `producer_kind`, `input_digest`, `output_digest`; `producer_kind=r9700_native` | `tool_digest` |
| `native_run` | `native_run` | `record_path`, `record_id`, `record_sha256`, `subject_target`, `image_sha256`, `pack_sha256`, `producer_kind`, `input_digest`, `output_digest`; `producer_kind=r9700_native`; the path and ID identify the request-bound native evidence record | `tool_digest` |
| `benchmark` | `benchmark` | `record_path`, `record_id`, `record_sha256`, `subject_target`, `image_sha256`, `pack_sha256`, `producer_kind`, `input_digest`, `output_digest`, `tool_digest`; promoted performance uses `producer_kind=r9700_native` | none |

`offline_review` always has nonempty target/image/pack/tool/input/output digests and an exactly empty `producer_kind`. Its `tool_digest` identifies the exact review script/tool plus version or the signed manual-review record digest; it is never optional. `target_conformance` and `native_run` always have nonempty target/image/pack/producer/input/output digests, exactly `producer_kind=r9700_native`, and an exactly empty `tool_digest`. `native_run` is a distinct record kind with `evidence_slot=native_run`; it is not rejected, aliased, resolved, or collapsed into `target_conformance`. Its path and ID are request-bound and its input/output digests bind that exact native request. A promoted `benchmark` record has a nonempty tool digest and `producer_kind=r9700_native`. Correctness-control packs omit the benchmark reference and use a nonempty `benchmark_not_applicable_reason`.

For every `M` request, the NumPy and native projection references are separate records with the same canonical `input_digest`:

* `numpy_oracle_record_id`: stable ID in `logs/f2/numpy-oracle.json`, `record_kind: offline_oracle`, `evidence_slot: numpy_oracle`, `producer_kind: cpu_reference`; the matrix requires its record, input, and output digests and exactly empty target/image/pack/tool fields.
* `native_projection_record_id`: stable ID in `logs/f2/native-projection.json`, `record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`; the matrix requires its subject target, image/pack, record, input, and output digests and exactly empty `tool_digest`. A native projection record with only a path or nonempty ID is invalid.

The task-set-3 physical-layout record is `record_kind: offline_review` with `evidence_slot: layout_proof`. Its target/image/pack/tool/input/output digests and exact source-to-byte/tile/LDS mapping are required by the matrix, its `producer_kind` is exactly empty, and its `tool_digest` identifies the exact review tool/version or signed manual-review digest. Generated runtime records embed the closed evidence digests/keys at generation time; runtime never resolves an arbitrary filesystem path, YAML, or documentation string.

### Canonical `pack_sha256` preimage

`pack_sha256` is exactly the SHA-256 of the UTF-8 RFC8785 JCS for `{ "domain":"r9700-kernel-pack-identity-v1", "pack": <the normalized complete pack record with the top-level `evidence` object and every `pack_sha256` and `record_sha256` field removed> }`. The normalized complete pack record includes all identity, provenance, license, image, build, entry, kernarg, resource, geometry, compatibility, and numerical fields, including declared paths and semantic evidence IDs/input/output digests. Remove the top-level `evidence` object and recursively remove every field named `pack_sha256` or `record_sha256` before RFC8785 JCS serialization. Non-finite numbers reject. Evidence references bind to this result without a recursive file-digest cycle.

### 3.7 Resource and descriptor limits

The following limits are frozen because they are explicit in the current local loader/generator contract:

* image and direct code-file size: at most 4 MiB (`MAX_IMAGE_BYTES`, `load_verified_kernel_code`);
* manifest read size: at most 64 KiB (`kMaximumManifestBytes`);
* code entry offset: positive and 256-byte aligned (`PM4_PROGRAM_ENTRY_ALIGNMENT` and resident-stage preflight);
* AMDHSA code properties: exact generated descriptor value, with wave32 bit required; no wave64 fallback;
* kernarg segment: exact 32 bytes for this family, at most one existing 256-byte slot, no preload;
* private segment: exactly 0 bytes; dynamic LDS: 0 (static LDS only); kernarg preload: exactly 0 bytes. These are existing `_descriptor()` rejection rules, not inferred performance values;
* static LDS (`group_segment_bytes`), `rsrc1`, `rsrc2`, `rsrc3`, SGPR count, and VGPR count: **no numerical WMMA value is accepted yet**. Task set 3 must bind each value from the generated image plus pinned IsaDecoder/RGA output and the source AMDGPU metadata. Missing, contradictory, or filename-only values reject before allocation/submission.

The current scalar gate/up values (`group_segment_bytes=4100`, `private=0`, `rsrc1=3222208515`, `rsrc2=295044`, `rsrc3=320`, `kernarg=56`) remain a scalar baseline/provenance example only. They are not WMMA limits and must not be copied into `linear_wmma_f16.json` without evidence. This is an explicit unresolved external blocker for task set 3/G0, not a guessed value.

## 4. Ownership matrix and integration boundary

The shared boundary is intentionally one slot, named once for every phase report:

> **F2 owns WMMA-specific source/image/evidence contracts. P3 owns generic Kernel Pack records/tooling. A supervisor-selected single integration owner serializes `native_r9700/kernel_assets.cpp`, `native_r9700/kernel_catalog.cpp`, and all generated catalogs.** F2 and P3 must not edit those shared files concurrently; the selected owner migrates the exact G0 identity without regeneration or reinterpretation.

| Later task | Concrete owner/files/symbols | Required policy identity |
|---|---|---|
| F2 task set 2 — independent lane-map proof | F2 owns `native_r9700/kernels/wmma_lane_map_gfx1201.cpp`, `native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets/`, and `tests/native_r9700/test_wmma_lane_map_asset.py`; the probe must expose `v_wmma_f32_16x16x16_f16` A/B/D register/lane/bit ownership through the existing verified-loader/readback seam. | `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1` for the probe's FP16 operand/result finite control; exact lane-map equality is a separate non-negotiable proof. |
| F2 task set 3 — offline ISA/resource and physical-layout admission | F2 owns WMMA-specific extensions to `experiments/native-r9700-runtime/generate_hsa_code_image.py`, `native_r9700/hsa_code_image_asset.*`, the focused HSA tests, and the new offline `tools/f2-wmma-layout-proof` command plus its `offline_review` EvidenceRef (fixed layout review scope). Its proof inputs are the pinned rocWMMA `simple_hgemm.cpp`, `library/include/rocwmma/rocwmma.hpp`, `internal/io_config.hpp`, `internal/io_layout.hpp`, `internal/mapping_util.hpp`, `internal/accessors_impl.hpp`, `internal/layout/matrix_layout_traits_impl.hpp`, pinned AITER, calculator, local scalar source, task-3-owned layout spec, and inverse fixture. The supervisor-selected shared owner alone touches `kernel_assets.cpp`/`kernel_catalog.cpp`; task set 4 waits for accepted task sets 2 and 3. | `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`; all WMMA resource/ISA values and the physical mapping remain fail-closed until concrete records are accepted. |
| F2 task set 4 — first linear source/image | F2 owns `native_r9700/kernels/linear_wmma_f16.cpp`, `native_r9700/kernels/linear-wmma-f16-hsa-assets/`, and `tests/native_r9700/test_linear_wmma_f16_asset.py`; no gate/up/down/QKV graph selection and no edits to service lifecycle files. Task set 4 is blocked until accepted task sets 2 and 3, including the task-set-3 `offline_review` physical-layout record; no source/image may consume an unproved layout. | `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`; only the fixed `K=2048,N=8192` family with runtime `1<=M<=128` under `tail_policy: masked/padded` and `geometry_rule: f2-wmma-64x64-m-tail-v1`. |
| F2 task set 5 — numerics/tails/performance | F2 owns focused extensions to `tests/native_r9700/test_linear_wmma_f16_asset.py`, the standalone comparison harness, and `.superpowers/swarm/reports/f2-wmma-numerics.md`; output scope is GPU compute/standalone kernel, never warm product throughput. The harness must produce separate NumPy and request-bound native projection EvidenceRefs over identical inputs. | `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`; this task owns the reviewed max/mean tolerance closure and native-vs-NumPy comparison. |
| F2 task set 6 — G0 publication | F2 owns `.superpowers/swarm/reports/g0-wmma-conformance.md` and the exact F2/G0 evidence identity; the report binds calculator expectation, hardware lane map, source/image digests, descriptor/resources, shape/tail corpus, numerics, ISA, and performance. G0 publication uses only F2/HSA/asset/catalog/numerical/evidence gates; P3 pack types/tests are not a prerequisite. | `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`; `g0_status: pass` is impossible while the policy threshold, physical layout proof, or resource values are unresolved. |
| P3 task set 2 — runtime pack identity | P3 owns new `native_r9700/kernel_pack.h`, `native_r9700/kernel_pack.cpp`, and `tests/native_r9700/test_kernel_pack_contract.py`; reuse HSA admission, do not duplicate it. | Carry the exact F2 policy identity, bounded-M rule, tail-padding/evidence names, and later immutable G0 result; do not define a second WMMA tolerance. |
| P3 task set 3 — offline pack manifest | P3 owns new `native_r9700/kernel_pack_manifest.py`, `tests/native_r9700/test_kernel_pack_manifest.py`, and offline ISA/RGA linkage. Runtime never parses `docs/upstream-reference-manifest.yaml`. | Carry the exact F2 policy identity, `EvidenceRef` fields, and exact G0 values/digests once F2 publishes them; absent layout/evidence rejects. |
| P3 task sets 4–6 — scalar/G0 migration and promotion | P3 owns migration logic and tests, but the supervisor-selected integration owner alone serializes `kernel_assets.cpp`, `kernel_catalog.cpp`, and generated catalogs. P3 later consumes the immutable exact `.superpowers/swarm/reports/g0-wmma-conformance.md` record and may not regenerate lane-map/GEMM/layout evidence. | Carry the exact F2 policy identity and G0 result; no changed shape, tail rule, physical pack, tolerance, or evidence digest. |
| F1 service lifecycle | F1 owns `model_service.py`, `service_protocol.py`, `native_worker.py`, and persistent model lifecycle. F2 does not edit or assign these files. | No F2 numerical policy is added to F1 service semantics. |
| P1 device owner | P1 owns TinyGPU DEXT/user-client ABI and local conformance clients; it consumes G0 and does not change WMMA source/catalog files. | G0 policy is consumed, not redefined. |

G0 is independently publishable by F2. P3 task-set-2/3 implementation and tests may be absent when F2 publishes; P3's later migration command must validate consumption of the immutable exact G0 record rather than gate or regenerate it.

## Active validation ledger insertion

The shared file `docs/tasks/native-r9700-producer/validation-commands.md` was intentionally not edited in this task set. Insert the following four named sections verbatim after the ledger's command-discovery policy. F2 task set 3 owns the offline physical-layout proof command and `offline_review` EvidenceRef; task set 2 owns the lane-map probe; task set 5 owns the NumPy/scalar-native/WMMA comparison interfaces; task set 6 owns G0 publication. Before an interface and its evidence record exist, a missing command is a blocker, not a reason to substitute `native_r9700_runner --native-prefill-proof`, a CPU/tinygrad path, or an unproved source-to-pack permutation.

**F2 physical WMMA layout proof (task set 3-owned; required before task set 4):**

```sh
/bin/bash -o pipefail -c '
  set -u
  : "${ROCWMMA_CHECKOUT:?set ROCWMMA_CHECKOUT to the exact f7f2aee8e764e612f49f2dc030b7e1639fb30d34 checkout}"
  : "${AITER_CHECKOUT:?set AITER_CHECKOUT to the exact 35c652ed3bd34e5d5828954e1545babc9255a69a checkout}"
  mkdir -p build/f2-wmma logs/f2
  layout_spec=build/f2-wmma/f2-wmma-physical-layout-spec.json
  inverse_fixture=build/f2-wmma/f2-wmma-physical-layout-inverse.npz
  log=logs/f2/wmma-physical-layout-proof.log
  {
    printf "%s\n" "command: tools/f2-wmma-layout-proof --source-layout-version f16-row-major-nk-source-v1 --physical-layout-version f2-wmma-physical-tile-v1 --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/samples/simple_hgemm.cpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/rocwmma.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_config.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_layout.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/mapping_util.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/accessors_impl.hpp --rocwmma-source \$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/layout/matrix_layout_traits_impl.hpp --rocwmma-symbols matrix_b,col_major,fragment,load_matrix_sync,IOConfig,GetMappingUtil --aiter-source \$AITER_CHECKOUT/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py --calculator-source <tools-root>/amd_matrix_instruction_calculator-2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py --local-source native_r9700/kernels/llama_gate_up_projection_f16.cpp --layout-spec build/f2-wmma/f2-wmma-physical-layout-spec.json --inverse-fixture build/f2-wmma/f2-wmma-physical-layout-inverse.npz --output logs/f2/wmma-physical-layout-proof.json"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    tools/f2-wmma-layout-proof \
      --source-layout-version f16-row-major-nk-source-v1 \
      --physical-layout-version f2-wmma-physical-tile-v1 \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/samples/simple_hgemm.cpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/rocwmma.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_config.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/io_layout.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/mapping_util.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/accessors_impl.hpp" \
      --rocwmma-source "$ROCWMMA_CHECKOUT/projects/rocwmma/library/include/rocwmma/internal/layout/matrix_layout_traits_impl.hpp" \
      --rocwmma-symbols matrix_b,col_major,fragment,load_matrix_sync,IOConfig,GetMappingUtil \
      --aiter-source "$AITER_CHECKOUT/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py" \
      --calculator-source <tools-root>/amd_matrix_instruction_calculator-2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py \
      --local-source native_r9700/kernels/llama_gate_up_projection_f16.cpp \
      --layout-spec "$layout_spec" \
      --inverse-fixture "$inverse_fixture" \
      --output logs/f2/wmma-physical-layout-proof.json
    status=$?
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

The required output is a concrete `EvidenceRef` with `record_kind: offline_review`, `evidence_slot: layout_proof`, `record_id: f2-wmma-physical-layout-proof-v1`, nonempty record/source/tool/spec/fixture/target/image/pack/input/output digests, and exactly empty `producer_kind`. It also requires nonempty `source_tensor_layout_version`, `physical_layout_version`, `layout_spec_path`, `layout_spec_sha256`, `inverse_fixture_path`, and `inverse_fixture_sha256`; exact source-element-to-physical-byte and 16x16 B-tile/LDS mapping; strides/alignment/padding/swizzle; `layout_origin: pinned_header|reviewed_local_v1`; inverse/conformance fixture input/output digests; and `layout_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. Its `tool_digest` identifies the exact review tool/version or signed manual-review record digest. A `reviewed_local_v1` origin carries the task-set-5 hardware numerical record key/digest before G0; a missing source checkout, absent or unreviewed spec, or failed inverse fixture rejects task set 3 and keeps task set 4 blocked.


### F2 lane-map proof

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p build/f2-wmma logs/f2
  PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
  CALC=<tools-root>/amd_matrix_instruction_calculator-2ef91896bcdc4d26624f952e5c905c787cd9bc9e/matrix_calculator.py
  detail=logs/f2/wmma-calculator-detail.txt
  a_map=logs/f2/wmma-calculator-a.csv
  b_map=logs/f2/wmma-calculator-b.csv
  d_map=logs/f2/wmma-calculator-d.csv
  observed=logs/f2/wmma-lane-map-proof.json
  conformance=logs/f2/wmma-lane-map-conformance.json
  log=logs/f2/wmma-lane-map-proof.log
  {
    printf "%s\n" "command: pinned calculator outputs; tools/native-r9700-hardware-run build/f2-wmma/wmma_lane_map_gfx1201; python -m native_r9700.wmma_lane_map"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --detail-instruction > "$detail"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --A-matrix --csv > "$a_map"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --B-matrix --csv > "$b_map"
    "$PY" "$CALC" --architecture gfx1201 --instruction v_wmma_f32_16x16x16_f16 --register-layout --D-matrix --csv > "$d_map"
    tools/native-r9700-hardware-run \
      build/f2-wmma/wmma_lane_map_gfx1201 \
      --asset-root native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets \
      --log "$observed"
    status=$?
    if [ "$status" -eq 0 ]; then
      "$PY" -m native_r9700.wmma_lane_map \
        --calculator-detail "$detail" --calculator-a "$a_map" \
        --calculator-b "$b_map" --calculator-d "$d_map" \
        --observed "$observed" \
        --asset-root native_r9700/kernels/wmma-lane-map-gfx1201-hsa-assets \
        --out "$conformance"
      status=$?
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

Expected log observations, all required: `runtime_substrate: TinyGPU.app/APLRemotePCIDevice/PCIIface`, `pci_id: 1002:7551`, `arch: gfx1201`, `wave_size: 32`, `instruction: v_wmma_f32_16x16x16_f16`, the pinned calculator source revision and layout digest, exact source/image/manifest paths and SHA-256 values, observed A/B/D register/lane/bit records matching the equations in §2, `record_kind: target_conformance`, `evidence_slot: conformance`, nonempty target/image/pack/producer/input/output digests, exactly `producer_kind: r9700_native`, exactly empty `tool_digest`, `lane_map_status: pass`, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. A mismatch is a failed proof and must update this contract before task set 4; no compensating transpose is allowed.

### F2 standalone WMMA
This named section includes the mandatory NumPy oracle, accepted scalar/native projection, and WMMA comparisons over identical inputs; they are separate EvidenceRefs and acceptance outcomes.

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p build/f2-wmma logs/f2
  input_record=build/f2-wmma/f2-projection-inputs-fp16.npz
  layout_record=logs/f2/wmma-physical-layout-proof.json
  numpy_record=logs/f2/numpy-oracle.json
  native_record=logs/f2/native-projection.json
  native_log=logs/f2/native-projection.log
  log=logs/f2/standalone-wmma.log
  {
    printf "%s\n" "command: tools/f2-wmma-numpy-oracle and scalar_native_projection_gfx1201 and standalone_wmma_gfx1201 --input-record build/f2-wmma/f2-projection-inputs-fp16.npz --m 128 --k 2048 --n 8192 --tail-m 13 --tail-policy masked/padded --geometry-rule f2-wmma-64x64-m-tail-v1 --source-layout-version f16-row-major-nk-source-v1 --packing-version f2-wmma-physical-tile-v1 --layout-record logs/f2/wmma-physical-layout-proof.json --numpy-oracle-record logs/f2/numpy-oracle.json --native-projection-record logs/f2/native-projection.json"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    tools/f2-wmma-numpy-oracle \
      --input-record "$input_record" \
      --m 128 --k 2048 --n 8192 --tail-m 13 \
      --tail-policy masked/padded \
      --geometry-rule f2-wmma-64x64-m-tail-v1 \
      --source-layout-version f16-row-major-nk-source-v1 \
      --record "$numpy_record"
    numpy_status=$?
    if [ "$numpy_status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/scalar_native_projection_gfx1201 \
        --input-record "$input_record" \
        --m 128 --k 2048 --n 8192 --tail-m 13 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --source-layout-version f16-row-major-nk-source-v1 \
        --evidence-record "$native_record" \
        --log "$native_log"
      native_status=$?
    else
      native_status=$numpy_status
    fi
    if [ "$native_status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/standalone_wmma_gfx1201 \
        --asset-root native_r9700/kernels/linear-wmma-f16-hsa-assets \
        --input-record "$input_record" \
        --m 128 --k 2048 --n 8192 --tail-m 13 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --source-layout-version f16-row-major-nk-source-v1 \
        --packing-version f2-wmma-physical-tile-v1 \
        --layout-record "$layout_record" \
        --numpy-oracle-record "$numpy_record" \
        --native-projection-record "$native_record" \
        --numerical-policy F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1 \
        --log logs/f2/standalone-wmma.json
      status=$?
    else
      status=$native_status
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

Expected observations, all required: both requests use the same canonical input record and emit the same `input_digest` in the NumPy, scalar/native, and WMMA evidence; request/shape `M=128,K=2048,N=8192`; explicit bounded tail `M=13`; `input_dtype: fp16`, `accumulator_dtype: fp32`, `output_dtype: fp16`; `wave_size: 32`; `instruction: v_wmma_f32_16x16x16_f16`; `source_tensor_layout_version: f16-row-major-nk-source-v1`; `packing_version: f2-wmma-physical-tile-v1` bound to the passing `record_kind: offline_review`, `evidence_slot: layout_proof` EvidenceRef; separate `numpy_oracle_record_id` (`record_kind: offline_oracle`, `evidence_slot: numpy_oracle`) and `native_projection_record_id` (`record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`); native subject target/image/pack/record/input/output digests; finite valid outputs; unchanged padding/sentinel rows; separate NumPy and scalar/native max/mean error fields within the reviewed policy; exact manifest resource/descriptor fields; `standalone_wmma_status: pass`; `failure_stage: none`; `exit_status: 0`; and `wrapper_exit_status: 0`. The record must identify `benchmark_scope: gpu_compute`/standalone, not warm service throughput. A CPU/scalar-only run, mismatched input digest, missing native record, or request-unbound log is failure.

### F2 G0 publication — canonical full invocation (P3 copies verbatim)
P3's G0 migration MUST copy the complete hardware invocation in the block below, including every argument after `--g0`, and may append only P3 pack-consumption outputs/observations after that exact call. It MUST NOT omit an argument, add a default, or reinterpret an EvidenceRef; this block is the sole F2 G0 CLI authority.

```sh
/bin/bash -o pipefail -c '
  set -u
  mkdir -p logs/f2
  log=logs/f2/g0-publication.log
  {
    printf "%s\n" "command: F2 G0 HSA/asset/catalog/numerical/evidence gates and hardware publication"
    date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"
    status=0
    if [ "$status" -eq 0 ]; then
      test -s .superpowers/swarm/reports/g0-wmma-conformance.md || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      ${PY} -m pytest \
        tests/native_r9700/test_wmma_lane_map_asset.py \
        tests/native_r9700/test_linear_wmma_f16_asset.py \
        tests/native_r9700/test_hsa_code_image_generator.py \
        tests/native_r9700/test_hsa_code_image_loader.py \
        tests/native_r9700/test_kernel_assets.py \
        tests/native_r9700/test_kernel_catalog.py -v || status=$?
    fi
    if [ "$status" -eq 0 ]; then
      tools/native-r9700-hardware-run \
        build/f2-wmma/standalone_wmma_gfx1201 \
        --g0 \
        --asset-root native_r9700/kernels/linear-wmma-f16-hsa-assets \
        --source-layout-version f16-row-major-nk-source-v1 \
        --tail-policy masked/padded \
        --geometry-rule f2-wmma-64x64-m-tail-v1 \
        --packing-version f2-wmma-physical-tile-v1 \
        --layout-record logs/f2/wmma-physical-layout-proof.json \
        --numpy-oracle-record logs/f2/numpy-oracle.json \
        --native-projection-record logs/f2/native-projection.json \
        --g0-report .superpowers/swarm/reports/g0-wmma-conformance.md \
        --log logs/f2/g0-wmma-dispatch.log || status=$?
    fi
    printf "wrapper_exit_status: %d\n" "$status"
    exit "$status"
  } 2>&1 | tee "$log"
'
```

Expected observations, all required: F2 HSA/asset/catalog/numerical/evidence gates exit 0 (no P3 pack contract or pack-manifest test is a G0 prerequisite); the report contains `g0_status: pass`, the exact calculator expectation and independent hardware lane-map result, target `1002:7551`/`gfx1201`, source/image/manifest SHA-256 values, the passing `record_kind: offline_review`, `evidence_slot: layout_proof` physical-layout EvidenceRef, descriptor/kernarg/wave/resource/ISA records, canonical fixed `K=2048,N=8192` family with runtime `1<=M<=128` under `tail_policy: masked/padded` and `geometry_rule: f2-wmma-64x64-m-tail-v1`, `M=13` tail result, `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`, `source_tensor_layout_version: f16-row-major-nk-source-v1`, `weight_packing_version: f2-wmma-physical-tile-v1`, separate NumPy (`record_kind: offline_oracle`, `evidence_slot: numpy_oracle`) and request-bound native projection (`record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`) EvidenceRefs sharing each `input_digest`, standalone GPU-compute performance evidence, reviewer result, explicit replacement/supersession rules, and `pack_sha256` equal to the canonical RFC8785 JCS preimage digest. The dispatch log must contain matching G0 image/source/pack/layout/evidence digests, `standalone_wmma_status: pass`, separate NumPy/native comparison passes, `failure_stage: none`, `exit_status: 0`, and `wrapper_exit_status: 0`. No G0 record may be published from `cpu_reference`, a stale image, a source-layout-as-pack alias, or a request-unbound log. P3 later validates exact immutable G0 consumption; it does not gate this command or regenerate any evidence.

## 6. Unresolved external blockers (fail closed)

1. This task-set-1 report remains `Needs review`. After the supervisor accepts it, task sets 2 and 3 are ready to proceed in parallel: task set 2 proves the lane map and task set 3 owns the offline ISA/resource plus physical-layout admission. Their outputs remain independent blockers for task set 4/G0; no task set 4 image may start before **accepted task sets 2 and 3 in full**—the lane-map result plus task-set-3 ISA/resource and physical-layout `offline_review` evidence.
2. The pinned calculator provides the expected map but no hardware evidence. Task set 2 must add and execute the independent lane-map image/probe; this report does not promote the expected equations.
3. No WMMA source/image exists in the current checkout, and the current generator allowlist does not yet admit `linear_wmma_f16.cpp` or `wmma_lane_map_gfx1201.cpp`. Task sets 2–4, through the selected shared integration owner where required, must add exact source/image digests and manifests.
4. The current scalar gate/up resource numbers are not WMMA values. Task set 3 must obtain WMMA `rsrc1/2/3`, SGPR/VGPR, static LDS/private, code-object, and ISA values from the generated image plus pinned offline tools. Any unknown value rejects admission.
5. The reviewed max/mean WMMA numerical tolerance is intentionally unresolved. Task set 5 owns measurement and review; G0 cannot pass with a missing or guessed threshold.
6. No pinned source currently proves the first linear family's physical WMMA weight representation. F2 task set 3 owns the exact `tools/f2-wmma-layout-proof` command, pinned rocWMMA header/symbol inputs, task-3 layout spec/inverse fixture, and `offline_review` EvidenceRef above; it must run/accept a direct header-grounded or explicitly reviewed-local mapping before task set 4. A reviewed-local mapping additionally needs task-set-5 request-bound hardware numerical binding before G0; `f16-row-major-nk-source-v1` is source layout only and `f2-wmma-physical-tile-v1` remains unadmitted until these proofs pass.
7. The accepted scalar/native projection comparator and its request-bound `native_projection_record_id` are not present yet. Task set 5 must provide the exact comparator interface and separate NumPy/native EvidenceRefs over identical inputs; a CPU/scalar-only result or missing `input_digest` match blocks G0.
8. Component-specific hipBLASLt licensing remains file-level review work. No hipBLASLt file may be copied or compiled into the F2 path before that review.
9. The physical-layout proof, NumPy oracle, lane-map, scalar/native comparator, and standalone command binaries named under `## Active validation ledger insertion` are not present yet. They are frozen interfaces for task sets 2–5; the supervisor must not substitute the existing scalar/native-prefill runner.

## 7. Supervisor review checklist

* Review this report and the task-set-1 row, then serialize the exact sections under `## Active validation ledger insertion` into the shared validation ledger.
* Verify the pinned calculator checkout/revision before retaining its expected output.
* Require task set 2's fresh request-bound lane-map log before allowing task set 4's production image.
* Require task set 3's `tools/f2-wmma-layout-proof`/`offline_review` acceptance plus exact WMMA descriptor/resource/ISA fields and task set 5's reviewed numerical threshold before G0 publication.
* Require the passing `offline_review` layout EvidenceRef and `f2-wmma-physical-tile-v1` mapping before task set 4 source/image work or any G0 claim.
* Require separate `numpy_oracle_record_id` and request-bound `native_projection_record_id` with identical input digests before G0 publication; a CPU/scalar-only comparison is insufficient.
* Confirm that F2 G0 publication runs only F2/HSA/asset/catalog/numerical/evidence gates and that P3 later imports the immutable exact G0 record without regeneration.
* Keep B0, `S-1`/final-token semantics, producer-owned KV truth, fallback-before-cache-acceptance-only, and `cpu_reference` versus request-bound `r9700_native` evidence unchanged.

### Supervisor-only active-ledger reconciliation

The shared `docs/tasks/native-r9700-producer/validation-commands.md` ledger is intentionally not edited here. Preserve the required headings `F2 physical WMMA layout proof`, `F2 lane-map proof`, `F2 standalone WMMA`, and `F2 G0 publication`. If the ledger integrator copied pre-freeze evidence prose, the supervisor must replace that prose with the exact `EvidenceRef` five-kind/evidence-slot matrix and canonical `pack_sha256` preimage in §3.6 above: restore `native_run` as its own `record_kind`/`evidence_slot`, add `evidence_slot` to every reference, replace all conditional field wording with the unconditional matrix rows, require `offline_review` target/image/pack/tool/input/output digests and exact-empty producer, and require `benchmark_not_applicable_reason` for correctness controls. Do not add aliases or edit the ledger from this task.

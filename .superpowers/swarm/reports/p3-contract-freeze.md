# P3 task set 1 — Kernel Pack contract freeze

## Status and scope

- **Task:** P3 task set 1, schema/API/ownership/command freeze.
- **Status:** Needs review.
- **Owner:** `P3Contract`.
- **Report:** `.superpowers/swarm/reports/p3-contract-freeze.md`.
- **Worktree boundary:** `${HOME}/Development/ml/tools/egpu/.worktrees/r9700-products-wave-a`.
- **Verification policy:** no tests, compilers, formatters, package managers, git commands, or hardware commands were run for this report. The commands in [Active validation ledger insertion](#active-validation-ledger-insertion) are ready for supervisor execution.
- **Non-goals honored:** no C++/Python implementation, asset migration, runtime YAML parser, F2 WMMA implementation, or shared validation-ledger edit.

## Source-grounded current boundary

| Source | Grounded contract | P3 consequence |
|---|---|---|
| `docs/DESIGN.md:104-122`, **Kernel Pack contract** | Production executables require concrete identity, provenance, image, entry/resource/geometry, compatibility, numerics, and conformance/benchmark records. Unknown or implicit values are not admissible; selection is by declared compatibility and measured record; no generic plugin registry or unconstrained autotuner. | Freeze one closed record shape. Unknown fields, extension maps, plugin callbacks, runtime discovery, and implicit defaults are rejected. |
| `docs/DESIGN.md:278-294`, **Executable / request lifecycles** | Executables move `unseen → validating → admitted|rejected → loaded → retired`; rejected images never reach a queue. Serving fallback is legal only before cache `accepted`. | Pack validation/admission is before allocation/submission and never silently falls back to another pack or a CPU/reference execution. Existing service fallback semantics remain unchanged. |
| `docs/DESIGN.md:242-252`, **Numerical acceptance** | Standalone, graph, and product correctness are separate. CPU/NumPy/scalar paths remain correctness controls, not performance acceptance; optimized intermediates use named per-pack policies. | An `offline_oracle` produced by `cpu_reference` may be retained as the numerical reference, but cannot satisfy native conformance/benchmark evidence. Native evidence is request-bound `r9700_native`. |
| `docs/DESIGN.md:254-268`, **Benchmark contract** | A performance record identifies scope, device/runtime, model and pack digests, geometry, timings, samples, correctness, and failure state. | A resolved `benchmark_record` `EvidenceRef` is mandatory for a promoted performance pack and binds the exact pack/image digest; a correctness-control pack instead carries a nonempty `benchmark_not_applicable_reason`. Cold, warm, and GPU-compute scope must not be conflated. |
| `docs/REFERENCES.md:7-19`, **Usage policy** | Exact manifest revision/path and file/component license must be reviewed; local modifications, ASIC/IP scope, source/image hashes, numerical policy, and conformance linkage are recorded. Current code/hardware evidence outranks analogies. | Provenance and license coverage are required per component/file. No branch/HEAD substitution, inferred hardware field, or unreviewed copied source is admissible. |
| `docs/REFERENCES.md:208-219`, **Source promotion and refresh** | Refresh updates immutable revision/paths, rechecks license, summarizes relevant changes, records translated/generated source/image hashes, reruns conformance/numerics, and updates design/ADR if ownership changes. | Refresh creates a reviewed record update/new pack identity; it is offline and explicit, never an automatic runtime update. |
| `docs/upstream-reference-manifest.yaml:1-10` | The manifest is a reference-use authority; copying, translation, vendoring, or compiling requires file-level review, local modifications, source/image hashes, ASIC/IP scope, and conformance. | `docs/upstream-reference-manifest.yaml` is policy input to offline validation only. Runtime consumes generated records and never opens this documentation YAML. |
| `docs/upstream-reference-manifest.yaml:103-154` | P3 pins `llvm-amdgpu-usage`, `amd-isa-spec-manager`, and `radeon-gpu-analyzer`, with required code-object/kernarg/wave, RDNA4 ISA, and offline ISA/resource evidence. | Every pack has concrete ISA/source and resource-review `EvidenceRef` records. A B0 scalar may bind its immutable reviewed source/byte record; WMMA/generated/external code must bind the task-1-named IsaDecoder/RGA artifacts with their tool/input/output digests. |
| `docs/upstream-reference-manifest.yaml:173-191`, `:193-211` | ROCm super-repository and AITER licenses are component/path-specific; repository-wide license assertions are prohibited. | Every imported source or generated asset gets its own accepted license review. `unknown`, absent, or repository-wide-only status rejects production selection. |
| `docs/IMPLEMENTATION_PLAN.md:341-365`, **P3** | P3 wraps `hsa_code_image_asset.*` and catalog structures, emits concrete compile-time records, adds offline ISA/resource validation, and rejects unknown/contradictory metadata before allocation/submission. | Generic records live in `native_r9700/kernel_pack.h/.cpp`; generated catalog integration remains in the existing catalog/asset boundary. |
| `docs/tasks/r9700-products/integration-gates.md:42-67`, **G0** | G0 is one immutable F2 record binding target/image/descriptors/resources/shapes/tails/numerics/ISA/performance; P1/P2/P3 consume the exact record and do not duplicate proof. | P3 imports the exact G0 record and digests. A mismatch returns to F2 for reviewed replacement; P3 does not regenerate lane maps or WMMA evidence. |

### Existing local symbols that this contract extends

- `native_r9700/kernel_catalog.h:12-38` defines `KernelDescriptor` with `name`, code SHA-256, code bytes, `rsrc1/2/3`, workgroup/global dimensions, and `kernarg_bytes`, plus `validate_kernel_descriptors(...)` and `find_kernel(...)`.
- `native_r9700/kernel_catalog.cpp:140-177` currently rejects empty/duplicate names, noncanonical or mismatched code digests, empty code, zero resource registers, zero geometry, and zero kernarg bytes; unknown names return `nullptr`. These remain the low-level descriptor checks, not a substitute for pack identity/provenance/evidence.
- `native_r9700/kernel_assets.h:13-50` defines `KernelAssetLocation`, `LlamaKernelAsset`, `find_llama_kernel_asset(...)`, `find_qwen_kernel_asset(...)`, and `load_verified_kernel_code(...)`.
- `native_r9700/kernel_assets.cpp:172-277` currently requires target `gfx1201`, `resource_metadata_provenance == "source_amdgpu_metadata"`, matching location/descriptor digests, a code-free manifest descriptor, safe direct-child paths, a non-symlink regular file under 4 MiB, and final `validate_kernel_descriptors(...)` admission. P3 reuses this admission path and does not duplicate ELF/resource parsing.
- `native_r9700/hsa_code_image_asset.h:10-44` defines the attested image boundary (`image`, image digest, descriptor/entry offsets, `rsrc1/2/3`, wave32, kernarg schema, source path/digest). `native_r9700/hsa_code_image_asset.cpp:312-372` is a deliberately narrow exact-field JSON reader for the existing `llama_embed_row_f16` asset; it is not a generic documentation/manifest parser. The new pack path may invoke the existing image admission after selecting a generated record, but must not generalize this parser to YAML or use it as a runtime registry.
- `tests/native_r9700/test_kernel_catalog.py:12-124`, `test_kernel_assets.py:13-213`, and `test_hsa_code_image_loader.py:49-273` establish no-hardware compile probes, output-preserving rejection, digest/path/symlink/descriptor checks, and the current image-loader boundary. The future pack tests extend these observable behaviors rather than replacing them.

## Frozen Kernel Pack schema v1

The schema is closed. In schema v1, `entries` contains exactly one entry. The only top-level fields are `schema_version`, `name`, `version`, `target`, `required_features`, `provenance`, `image`, `entries`, `compatibility`, `numerics`, and `evidence`. Unknown keys are rejected offline; missing required values are rejected both offline and before runtime allocation.

### Identity

| Field | Type/semantics |
|---|---|
| `schema_version` | Unsigned integer, exactly `1` for this freeze. This is the record schema, not the pack upgrade/version. |
| `name` | Nonempty stable pack identity, matching the existing catalog name vocabulary. It is not a runtime-discovered symbol or plugin key. |
| `version` | Canonical `MAJOR.MINOR.PATCH` string. It is part of identity/provenance only; it is never used for an implicit upgrade, downgrade, ranking, or fallback. |
| `target` | Exact target string from admitted code-object metadata (the current product target is `gfx1201`; no value is inferred from a filename). |
| `required_features` | Required sorted, unique feature list; an explicit empty list means the target needs no additional feature beyond the target identity. Runtime compares the declared set with queried device features; it never invents features or treats a missing field as empty. |

The following declarations are the runtime-facing C++ view schema. They contain only `std::string_view`, POD scalars, explicit `{pointer,size}` spans, and explicit `{present,value}` optionals; they never own or allocate. The offline JSON/Python parser may temporarily own strings, lists, and dictionaries while validating a record, but those owning representations are not runtime headers and are not emitted into generated records.

```cpp
template <typename T>
struct KernelPackSpan {
  const T* data;
  std::size_t size;
};

template <typename T>
struct KernelPackOptional {
  bool present;
  T value;
};

struct KernelPackIdentity {
  uint32_t schema_version;
  std::string_view name;
  std::string_view version;
  std::string_view target;
  KernelPackSpan<std::string_view> required_features;
};
```

### Provenance, modifications, and license

```cpp
struct KernelPackSource {
  std::string_view path;    // exact checkout-relative local source path
  std::string_view sha256;  // lowercase SHA-256 of that file
};

struct KernelPackLicenseReview {
  std::string_view component;       // exact upstream/local file or generated component
  std::string_view spdx_expression; // reviewed expression, never "unknown"
  std::string_view review_id;       // durable review/evidence identifier
  std::string_view status;          // exactly "accepted" for a production pack
};

struct KernelPackModification {
  std::string_view component; // exact file/component changed locally
  std::string_view summary;   // explicit translation/generation/local-change summary
};

struct KernelPackProvenance {
  std::string_view upstream_repository; // exact URL, or literal "local" for first-party source
  std::string_view upstream_revision;   // immutable full revision; "local" only for first-party source
  KernelPackSpan<std::string_view> upstream_paths;
  KernelPackSpan<KernelPackSource> local_sources;
  KernelPackSpan<KernelPackLicenseReview> license_reviews;
  KernelPackSpan<KernelPackModification> modifications;
};
```

Rules:

1. An imported source has the exact URL, immutable revision, exact upstream paths, local translated/generated source paths and SHA-256 values, and a review entry covering every copied/translated/generated component. First-party source uses the explicit `local` marker, still lists exact local source paths/digests, and is not an excuse to omit review or modification state.
2. `license_reviews` must cover every path/component named by `upstream_paths`, `local_sources`, `image.image_path`, and each generated/translated component. Every covered component must have `status == "accepted"`, a nonempty SPDX expression, and a nonempty review ID. `unknown`, empty, pending, repository-wide-only, or unreviewed status is rejection.
3. `modifications` is explicit. An unchanged source uses an empty list; it must not use an omitted field or an implicit “unchanged” value. A changed source records each changed component and summary.
4. Paths are exact, relative, and safe (no absolute path, root name, `.`/`..` component, or symlink escape). Source/image SHA-256 strings are 64 lowercase hexadecimal characters.

### Image, code-object, and build identity

```cpp
struct KernelPackBuildIdentity {
  std::string_view toolchain_id;       // compiler/toolchain identity
  std::string_view toolchain_revision; // immutable compiler/toolchain revision
  std::string_view generator_id;       // generator name and version
  std::string_view generator_revision; // immutable generator/source revision
  std::string_view command_sha256;     // digest of the reviewed build command/configuration
};

struct KernelPackImage {
  std::string_view image_path;          // generated image relative to the asset root
  std::string_view image_sha256;
  uint64_t image_size;
  std::string_view code_object_version; // exact admitted AMDHSA/code-object version
  KernelPackBuildIdentity build;
};
```

`image_path`, `image_sha256`, image size, code-object version, and all build-identity fields are required. Branch names and release labels alone are not revisions. The image digest must match the bytes admitted by the existing HSA/image boundary; the source digest(s) remain in `provenance.local_sources`. Descriptor and entry offsets are entry fields below, not inferred from a path or symbol.

### Entries, kernargs, resources, and geometry

```cpp
struct KernelPackKernargField {
  std::string_view name;
  std::string_view type;
  uint32_t offset;
  uint32_t size;
  uint32_t alignment;
};

struct KernelPackKernargs {
  uint32_t bytes;                       // descriptor-reported segment size, exact
  KernelPackSpan<KernelPackKernargField> fields; // declaration order
  uint32_t tail_padding_bytes;          // explicit zeroed suffix after the final field
};

struct KernelPackResources {
  uint32_t rsrc1;
  uint32_t rsrc2;
  uint32_t rsrc3;
  uint32_t wave_size;
  uint32_t sgpr_count;
  uint32_t vgpr_count;
  uint64_t lds_bytes;
  uint64_t private_segment_bytes;
  std::string_view metadata_provenance; // e.g. source AMDGPU metadata, exactly cited
};

struct KernelPackGeometryCase {
  std::string_view shape_family;   // exact compatibility shape-family name
  std::string_view geometry_rule;  // closed named rule, never an arbitrary formula
  uint32_t workgroup_x;
  uint32_t workgroup_y;
  uint32_t workgroup_z;
  uint32_t global_x;          // positive for exact-global-v1, zero for a bounded rule
  uint32_t global_y;
  uint32_t global_z;
  uint32_t grid_tile_m;       // positive for a bounded runtime rule, zero for exact-global-v1
  uint32_t grid_tile_n;
  bool dynamic_lds_allowed;
  uint64_t dynamic_lds_max_bytes;
};

struct KernelPackGeometry {
  KernelPackSpan<KernelPackGeometryCase> cases; // one rule per compatible family/entry
};

struct KernelPackEntry {
  std::string_view symbol;       // exact code-object symbol
  uint64_t descriptor_offset;
  uint64_t entry_offset;
  KernelPackKernargs kernargs;
  KernelPackResources resources;
  KernelPackGeometry geometry;
};
```

There must be exactly one entry, and its symbol is unique within the pack. That entry has one geometry rule for each compatible shape family, and each rule names that family exactly. `geometry_rule` is a closed v1 value: `exact-global-v1` uses positive `global_x/global_y/global_z` and zero `grid_tile_m/grid_tile_n`; `f2-wmma-64x64-m-tail-v1` uses `workgroup=(128,4,1)`, `grid_tile_m=64`, `grid_tile_n=64`, and computes the global size from the actual bounded runtime `M` (its global fields are explicitly zero). Workgroup/tile values and computed global dimensions must be positive and aligned; dynamic-LDS limits are checked against the image/entry. No arbitrary geometry formula, extension map, or unlisted rule is accepted.

`kernargs.bytes` must equal the descriptor-reported kernarg segment size exactly. For ordered fields, declaration order must have strictly increasing offsets, every `offset` is aligned to its declared alignment, every field is contained in `[0, bytes)`, fields do not overlap, and `last_field_end = max(offset + size)`. The only permitted segment suffix is explicit: `tail_padding_bytes == bytes - last_field_end`; a zero value is required when the final field reaches the descriptor end. The submission path must zero the recorded suffix `[last_field_end, bytes)` and reject any nonzero tail byte before allocation/submission. No descriptor bytes, leading/interior padding, argument, or field extent may be inferred. The F2 `linear_wmma_f16` ABI therefore records a 32-byte descriptor segment, fields `activation`/`weight_nk`/`output`/`m` at offsets `0/8/16/24` with sizes `8/8/8/4` and alignments `8/8/8/4`, and `tail_padding_bytes: 4`.

Resource counts may be zero only where the admitted source/analysis explicitly reports zero (for example private memory); the record must still carry the value and provenance. The schema carries the existing descriptor register names and dimensions rather than introducing a second descriptor format.

### Dtypes, shapes, and packing compatibility

```cpp
struct KernelPackShapeDimension {
  std::string_view name; // stable ordered dimension name, e.g. M/K/N or sequence/head
  uint32_t value;
};

struct KernelPackRuntimeDimension {
  std::string_view name;       // bounded runtime dimension, e.g. M
  uint32_t min_value;
  uint32_t max_value;
  uint32_t full_value;    // full-tile boundary; tails are values below this
};

struct KernelPackShapeFamily {
  std::string_view name;
  KernelPackSpan<KernelPackShapeDimension> fixed_dimensions; // exact fixed dimensions only
  KernelPackOptional<KernelPackRuntimeDimension> runtime_dimension; // `present == false` for fixed-only
  std::string_view tail_policy;    // one exact named policy; no implicit tails
  std::string_view geometry_rule;  // exact rule name shared with the entry geometry
};

struct KernelPackCompatibility {
  std::string_view input_dtype;
  std::string_view weight_dtype;
  std::string_view output_dtype;
  std::string_view source_tensor_layout_version; // exact source tensor layout/version
  KernelPackSpan<KernelPackShapeFamily> shape_families;
  std::string_view weight_packing_version;       // exact physical packing version
};
```

Dtypes use the existing canonical vocabulary (`fp16`, `bf16`, `fp32`, `int8`, `int4`, and explicit integer pointer/scalar types where kernargs require them); unknown dtype strings reject. `fixed_dimensions` contains dimensions fixed for the family, while `runtime_dimension` is either explicit `null` for a fixed-shape family or one named dimension with inclusive `min_value`, `max_value`, and `full_value`. A request supplies the actual runtime value separately; it must be within the recorded bound and uses the one named tail/geometry rule, not a new family.

The F2 family is exactly `f2-linear-gate-up-f16-v1`: `fixed_dimensions` are `K=2048` and `N=8192`; `runtime_dimension` is `{"name":"M","min_value":1,"max_value":128,"full_value":128}`; `tail_policy` is the named `masked/padded` M-tail policy; and `geometry_rule` is `f2-wmma-64x64-m-tail-v1`. That rule computes `grid=(ceil(M/64),ceil(N/64),1)` and AQL global `(grid.x*128,grid.y*4,1)` with `workgroup=(128,4,1)`, masks stores to `row < M`, and never writes outside the valid `M x N` output. `M=128` is the full tile and every `1 <= M < 128` value is a tail of this same family; no M values are enumerated as shape families.

A B0 scalar fixed-shape family remains representable by setting `runtime_dimension: null`, recording its exact fixed dimensions, using its reviewed `tail_policy` (normally `none` when the source has no tail), and naming `exact-global-v1`; this does not add a compatibility alias or a second selection path. `source_tensor_layout_version` and `weight_packing_version` are separate required values. A source-equivalent B0 pack binds both through its reviewed `record_kind: offline_review`, `evidence_slot: source_review` source/byte evidence; a distinct physical layout binds a resolved `record_kind: offline_review`, `evidence_slot: layout_proof` layout-review ID. For F2, `f16-row-major-nk-source-v1` is source layout only; P3 must not emit or admit the reserved `f2-wmma-physical-tile-v1` (or any guessed physical version) until the pinned source-to-tile/LDS proof exists.

### Evidence references

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

### Numerics and retained references

```cpp
struct KernelPackCastPoint {
  std::string_view stage;
  std::string_view from_dtype;
  std::string_view to_dtype;
};

struct KernelPackNumerics {
  std::string_view input_dtype;
  std::string_view accumulation_dtype;
  std::string_view output_dtype;
  KernelPackSpan<KernelPackCastPoint> cast_points;
  std::string_view finite_value_rule;
  std::string_view tolerance_policy;
  std::string_view reference_set_kind;              // b0_scalar_control or f2_wmma_dual
  KernelPackOptional<EvidenceRef> retained_reference; // b0_scalar_control: present=true; f2_wmma_dual: present=false
  KernelPackOptional<EvidenceRef> numpy_oracle;     // b0_scalar_control: present=false; f2_wmma_dual: present=true
  KernelPackOptional<EvidenceRef> scalar_native_projection; // b0_scalar_control: present=false; f2_wmma_dual: present=true
};
```

The input/output dtype values must equal the compatibility values; `accumulation_dtype`, cast points, finite-value rule, and tolerance policy are required. `reference_set_kind` is a closed per-pack reference set:

* `b0_scalar_control`: `retained_reference.present == true` with `record_kind: offline_oracle`, `evidence_slot: numpy_oracle`, and `producer_kind: cpu_reference`; `numpy_oracle.present == false` and `scalar_native_projection.present == false`.
* `f2_wmma_dual`: `numpy_oracle.present == true` with `record_kind: offline_oracle`, `evidence_slot: numpy_oracle`, and `producer_kind: cpu_reference`; `scalar_native_projection.present == true` with `record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, and `producer_kind: r9700_native`; `retained_reference.present == false`.

The dual input digests must be identical and bind the exact activation/weight/runtime-value request, while output digests remain separate. For F2/G0 these references resolve the separate `numpy_oracle_record_id` and `native_projection_record_id` records. NumPy/CPU output remains a diagnostic control only; no other reference-set combination is accepted.

### Pack evidence

```cpp
struct KernelPackEvidence {
  EvidenceRef source_review;                   // offline_review/source_review, producer_kind=""
  EvidenceRef conformance;                    // target_conformance/conformance, producer_kind=r9700_native
  EvidenceRef native_run;                     // native_run/native_run, producer_kind=r9700_native
  EvidenceRef resource_review;                // offline_review/resource_review, producer_kind=""
  EvidenceRef isa_review;                     // offline_review/isa_review, producer_kind=""
  KernelPackOptional<EvidenceRef> layout_proof;    // distinct physical pack: present=true; source-equivalent B0: present=false
  KernelPackOptional<EvidenceRef> benchmark_record; // promoted performance: present=true; correctness control: present=false
  std::string_view benchmark_not_applicable_reason;  // promoted performance: empty; correctness control: nonempty
};

```
Every pack has resolved `source_review`, `conformance`, `native_run`, `resource_review`, and `isa_review` references. `source_review` is `record_kind: offline_review` with `evidence_slot: source_review`, exactly empty producer, and concrete target/image/pack/tool/input/output digests; it binds the reviewed source/byte identity for every pack, including source-equivalent B0. `conformance` is `record_kind: target_conformance` with `evidence_slot: conformance` and `producer_kind: r9700_native`; `native_run` is `record_kind: native_run` with `evidence_slot: native_run` and `producer_kind: r9700_native`; `resource_review` and `isa_review` are `record_kind: offline_review` with `evidence_slot: resource_review` or `isa_review`, exactly empty producer, and concrete target/image/pack/tool/input/output digests. A source-equivalent B0 pack sets `layout_proof.present == false`; a distinct physical `weight_packing_version` sets `layout_proof.present == true` with `record_kind: offline_review` and `evidence_slot: layout_proof` for its source-to-byte/tile/LDS mapping. A `cpu_reference` record is never substituted for target/native evidence. Promoted performance sets `benchmark_record.present == true` with `record_kind: benchmark`, `evidence_slot: benchmark`, a nonempty tool digest, and `benchmark_not_applicable_reason == ""`; correctness-control packs set `benchmark_record.present == false` and use a nonempty `benchmark_not_applicable_reason`. Every non-null `pack_sha256` equals the canonical preimage digest above.

## Exact C++ interfaces and catalog boundary

P3 task set 2 creates `native_r9700/kernel_pack.h` and `native_r9700/kernel_pack.cpp` with these concrete public, allocation-free view types and functions:

```cpp
struct KernelPackRecord {
  KernelPackIdentity identity;
  KernelPackProvenance provenance;
  KernelPackImage image;
  KernelPackSpan<KernelPackEntry> entries;
  KernelPackCompatibility compatibility;
  KernelPackNumerics numerics;
  KernelPackEvidence evidence;
};

struct KernelPackCompatibilityKey {
  std::string_view target;
  KernelPackSpan<std::string_view> required_features;
  std::string_view input_dtype;
  std::string_view weight_dtype;
  std::string_view output_dtype;
  std::string_view source_tensor_layout_version;
  std::string_view shape_family_name; // caller-supplied family identity; it carries tail/geometry semantics
  KernelPackSpan<KernelPackShapeDimension> fixed_dimensions; // caller-supplied exact fixed dims
  KernelPackOptional<KernelPackShapeDimension> runtime_value; // `present == false` for fixed-only; actual M otherwise
  std::string_view weight_packing_version;
  std::string_view tolerance_policy;
};

struct KernelPackErrorBuffer {
  char* data;
  std::size_t size;
};

bool validate_kernel_pack(const KernelPackRecord& record,
                          KernelPackErrorBuffer error_text);
bool kernel_pack_matches_key(const KernelPackRecord& record,
                             const KernelPackCompatibilityKey& key);

// Exact identity lookup over an explicit generated-record view. `version` is
// mandatory; no hidden global catalog and no implicit upgrade/downgrade.
const KernelPackRecord* find_kernel_pack(
    KernelPackSpan<KernelPackRecord> records,
    std::string_view name,
    std::string_view version,
    KernelPackErrorBuffer error_text);

// Exact compatibility-key lookup over the same explicit view. Zero matches and
// more than one match reject; insertion order, version ranking, and feature
// scoring are not selection rules.
const KernelPackRecord* find_kernel_pack_for_key(
    KernelPackSpan<KernelPackRecord> records,
    const KernelPackCompatibilityKey& key,
    KernelPackErrorBuffer error_text);

// Validate the selected record against the exact compatibility key and delegate
// image/code admission to the existing HSA/KernelAsset boundary before any device
// allocation or submission.
bool admit_kernel_pack(const KernelPackRecord& record,
                       const KernelPackCompatibilityKey& selected_key,
                       std::string_view entry_symbol,
                       std::string_view asset_root,
                       KernelDescriptor* out_descriptor,
                       KernelPackErrorBuffer error_text);
```

The compatibility key is request-owned: the caller supplies the exact family name, fixed dimensions, and actual runtime dimension/value (for example `M`), but never supplies the family’s `min_value`, `max_value`, `full_value`, tail policy, or geometry rule. The family name resolves one named tail/geometry contract in the selected record. `kernel_pack_matches_key` first compares the request-owned fields, then validates the actual runtime value against the selected record’s own bounded runtime dimension and validates its named tail/geometry rule; a fixed family requires `runtime_value.present == false`. This keeps pack-owned admission metadata out of lookup while still rejecting an out-of-range tail before allocation/submission.

The preceding C++ snippets are the complete runtime view schema. Task set 2 emits `std::string_view` values and `KernelPackSpan` views over static string and record arrays, and uses `KernelPackOptional<T>` for every nullable nested value. No generated `KernelPackRecord`, nested field, compatibility key, or catalog entry owns memory, allocates, copies, ranks, or parses files. The offline parser may own temporary Python strings/lists/dictionaries only while validating and then emits these views.

`kernel_pack.cpp` owns record validation and key equality. The generated catalog remains the existing `native_r9700/kernel_catalog.cpp` boundary: one supervisor-selected integration owner defines static nested-record arrays and exposes them through `KernelPackSpan` views plus the exact lookup wrappers. `native_r9700/kernel_assets.cpp` remains the one owner for generated source/image locations and bridges selected entries to `load_verified_kernel_code(...)`; it must not grow an independent catalog. Existing `KernelDescriptor`, `find_kernel`, `find_llama_kernel_asset`, and HSA image checks remain the low-level compatibility/control boundary until task sets 4–5 perform the clean migration.

## Offline manifest records and runtime/offline boundary
- **Record files:** one canonical JSON pack record at `native_r9700/kernels/<pack-name>-hsa-assets/<pack-name>.pack.json`, beside (but distinct from) the existing per-image `<pack-name>.json` sidecar. The current sidecars retain their narrow HSA image-loader contract; a `.pack.json` is the closed schema above and is not opened by runtime code.
- **Offline tool:** `native_r9700/kernel_pack_manifest.py`, with `tests/native_r9700/test_kernel_pack_manifest.py`. It validates canonical JSON, exact schema keys/types, source/image digests, path safety, file/component license coverage, target/features, code-object/descriptor/kernarg/resource/geometry fields, shape/dtype/layout/packing, numerics/reference, resolved `EvidenceRef` records, and the pinned `docs/upstream-reference-manifest.yaml` policy input. It emits deterministic concrete C++ initializers for the supervisor-selected catalog owner; it does not download sources/tools or execute GPU work.
- **Runtime input:** generated `const` C++ records from `kernel_catalog.cpp`/`kernel_assets.cpp` plus the selected image/code bytes through the existing HSA/code-asset admission. Runtime does not parse `.pack.json`, `docs/upstream-reference-manifest.yaml`, YAML, or an arbitrary JSON registry. The runtime path validates the selected generated record, then performs existing image/descriptor admission before allocation/submission.
- **No docs-as-runtime-config:** changing a documentation manifest cannot change a running catalog. A source refresh must be offline-validated, regenerated, reviewed, and compiled into a new record before runtime can see it.

The JSON record has no additional keys. Its top-level shape is:

```json
{
  "schema_version": 1,
  "name": "<stable-pack-name>",
  "version": "<MAJOR.MINOR.PATCH>",
  "target": "<admitted-target>",
  "required_features": ["<sorted-feature>"],
  "provenance": {
    "upstream_repository": "<exact-url-or-local>",
    "upstream_revision": "<immutable-revision-or-local>",
    "upstream_paths": ["<exact-path>"],
    "local_sources": [{"path": "<path>", "sha256": "<64-lowercase-hex>"}],
    "license_reviews": [{"component": "<exact-component>", "spdx_expression": "<reviewed-expression>", "review_id": "<id>", "status": "accepted"}],
    "modifications": [{"component": "<path>", "summary": "<explicit-summary>"}]
  },
  "image": {
    "image_path": "<relative-image>",
    "image_sha256": "<64-lowercase-hex>",
    "image_size": 1,
    "code_object_version": "<exact-version>",
    "build": {
      "toolchain_id": "<id>",
      "toolchain_revision": "<immutable-revision>",
      "generator_id": "<id>",
      "generator_revision": "<immutable-revision>",
      "command_sha256": "<64-lowercase-hex>"
    }
  },
  "entries": [{
    "symbol": "<exact-symbol>",
    "descriptor_offset": 1,
    "entry_offset": 1,
    "kernargs": {"bytes": 1, "fields": [{"name": "<name>", "type": "<type>", "offset": 0, "size": 1, "alignment": 1}], "tail_padding_bytes": 0},
    "resources": {"rsrc1": 1, "rsrc2": 1, "rsrc3": 1, "wave_size": 1, "sgpr_count": 0, "vgpr_count": 0, "lds_bytes": 0, "private_segment_bytes": 0, "metadata_provenance": "<cited-source>"},
    "geometry": {"cases": [{"shape_family": "<family>", "geometry_rule": "<closed-rule>", "workgroup_x": 1, "workgroup_y": 1, "workgroup_z": 1, "global_x": 1, "global_y": 1, "global_z": 1, "grid_tile_m": 0, "grid_tile_n": 0, "dynamic_lds_allowed": false, "dynamic_lds_max_bytes": 0}]}
  }],
  "compatibility": {
    "input_dtype": "<dtype>",
    "weight_dtype": "<dtype>",
    "output_dtype": "<dtype>",
    "source_tensor_layout_version": "<exact-source-layout-version>",
    "shape_families": [{"name": "<family>", "fixed_dimensions": [{"name": "<dimension>", "value": 1}], "runtime_dimension": null, "tail_policy": "<named-policy>", "geometry_rule": "<closed-rule>"}],
    "weight_packing_version": "<exact-physical-packing-version>"
  },
  "numerics": {
    "input_dtype": "<dtype>",
    "accumulation_dtype": "<dtype>",
    "output_dtype": "<dtype>",
    "cast_points": [{"stage": "<stage>", "from_dtype": "<dtype>", "to_dtype": "<dtype>"}],
    "finite_value_rule": "<named-rule>",
    "tolerance_policy": "<named-policy>",
    "reference_set_kind": "b0_scalar_control",
    "retained_reference": {"record_path": "<path>", "record_kind": "offline_oracle", "evidence_slot": "numpy_oracle", "record_id": "<id>", "record_sha256": "<64-lowercase-hex>", "subject_target": "", "image_sha256": "", "pack_sha256": "", "producer_kind": "cpu_reference", "tool_digest": "", "input_digest": "<input-digest>", "output_digest": "<control-output-digest>"},
    "numpy_oracle": null,
    "scalar_native_projection": null
  },
  "evidence": {
    "conformance": {"record_path": "<path>", "record_kind": "target_conformance", "evidence_slot": "conformance", "record_id": "<id>", "record_sha256": "<64-lowercase-hex>", "subject_target": "<target>", "image_sha256": "<image-sha256>", "pack_sha256": "<pack-sha256>", "producer_kind": "r9700_native", "tool_digest": "", "input_digest": "<input-digest>", "output_digest": "<output-digest>"},
    "source_review": {"record_path": "<path>", "record_kind": "offline_review", "evidence_slot": "source_review", "record_id": "<id>", "record_sha256": "<64-lowercase-hex>", "subject_target": "<target>", "image_sha256": "<image-sha256>", "pack_sha256": "<pack-sha256>", "producer_kind": "", "tool_digest": "<tool-digest>", "input_digest": "<input-digest>", "output_digest": "<output-digest>"},
    "native_run": {"record_path": "<path>", "record_kind": "native_run", "evidence_slot": "native_run", "record_id": "<request-bound-r9700-native-id>", "record_sha256": "<64-lowercase-hex>", "subject_target": "<target>", "image_sha256": "<image-sha256>", "pack_sha256": "<pack-sha256>", "producer_kind": "r9700_native", "tool_digest": "", "input_digest": "<input-digest>", "output_digest": "<output-digest>"},
    "resource_review": {"record_path": "<path>", "record_kind": "offline_review", "evidence_slot": "resource_review", "record_id": "<id>", "record_sha256": "<64-lowercase-hex>", "subject_target": "<target>", "image_sha256": "<image-sha256>", "pack_sha256": "<pack-sha256>", "producer_kind": "", "tool_digest": "<tool-digest>", "input_digest": "<input-digest>", "output_digest": "<output-digest>"},
    "isa_review": {"record_path": "<path>", "record_kind": "offline_review", "evidence_slot": "isa_review", "record_id": "<id>", "record_sha256": "<64-lowercase-hex>", "subject_target": "<target>", "image_sha256": "<image-sha256>", "pack_sha256": "<pack-sha256>", "producer_kind": "", "tool_digest": "<tool-digest>", "input_digest": "<input-digest>", "output_digest": "<output-digest>"},
    "layout_proof": null,
    "benchmark_record": null,
    "benchmark_not_applicable_reason": "<nonempty-reason-for-correctness-control>"
  }
}
```

The angle-bracket values in this illustration are field descriptions, not accepted production values. A real record must contain concrete values; empty/unknown placeholders reject.

## Rejection, exact lookup, and refresh precedence

### Rejection precedence

Validation is fail-closed and deterministic in this order:

1. **Schema/shape of record:** parse the closed JSON offline or read the generated C++ record; reject unknown keys, duplicate keys, wrong types, missing required groups, schema version other than `1`, malformed identity/version, and unsafe paths.
2. **Provenance/license:** reject missing/contradictory source revision/path/digest, missing modification declaration, any uncovered component, or any license status other than accepted. `unknown` is rejection, not a pending production value.
3. **Image/build:** reject image-size/digest drift, missing code-object version, incomplete build identity, target mismatch, or a generated record that does not bind the admitted image/source bytes.
4. **Entry ABI:** reject duplicate/missing symbols, descriptor/entry offsets outside the image, kernarg fields that are out of bounds, misaligned, overlapping, or inconsistent with the descriptor-reported segment; reject `tail_padding_bytes` when it is not exactly `bytes - last_field_end`, or when the recorded suffix is not zero before submission; reject resource metadata without cited provenance, wave/resource mismatch, invalid geometry-rule values, and dynamic-LDS over the declared limit.
5. **Compatibility/numerics:** reject missing/unknown dtypes, empty/duplicate fixed dimensions, malformed or out-of-range runtime bounds/values, a missing or mismatched tail/geometry rule, packing mismatch, missing source tensor layout, a distinct physical packing without `layout_proof` (`record_kind: offline_review`, `evidence_slot: layout_proof`), dtype disagreement between compatibility and numerics, missing finite/tolerance policy, an unknown `reference_set_kind`, an invalid B0 retained-reference set, an invalid F2 dual-reference set or non-identical dual input digests, or a scalar/native reference not labeled/bound as `record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, and `producer_kind: r9700_native`.
6. **Evidence:** reject missing or unresolved `EvidenceRef` fields, unsafe paths, noncanonical record digests, nonmatching record ID/kind/slot/target/image/pack/producer/tool/input/output fields, a nonmatching `pack_sha256` preimage digest, missing task-1-named IsaDecoder/RGA analysis for WMMA/generated/external code, a missing `record_kind: offline_review`, `evidence_slot: layout_proof` for a distinct physical pack, an invalid benchmark outcome pair, an empty correctness-control `benchmark_not_applicable_reason`, or a native run not labeled/bound as `record_kind: native_run`, `evidence_slot: native_run`, and `producer_kind: r9700_native`.
7. **Runtime image admission:** only after all record checks pass, reuse `load_verified_kernel_code(...)`/`HsaCodeImageAsset` admission. Failure leaves output state unchanged and performs no allocation/submission.

A malformed record is not skipped in favor of an older record. The caller receives a rejection and may not continue with an unreviewed alternative. Service-level fallback remains exactly the existing pre-cache-acceptance behavior; after cache acceptance, a decode or pack failure is terminal for that request.

### Exact lookup precedence

1. Caller supplies an exact `name + version`, or supplies one complete `KernelPackCompatibilityKey`.
2. Catalog entries are already offline-validated concrete records. Filter by exact equality of every request-owned field (target, required features, input/weight/output dtypes, source tensor layout version, shape family name, fixed dimensions, and actual runtime dimension/value, weight-packing version, and tolerance policy); then validate the matched record’s own bounds, tail policy, and geometry rule against that request.
3. **Zero matches:** reject with a stable no-match error.
4. **More than one match:** reject as an ambiguous catalog; do not use insertion order, semver order, filename order, feature scoring, or measured-speed ranking.
5. **One match:** validate/admit that exact record, then load its requested entry symbol. Any admission failure is terminal for that selection; there is no automatic version upgrade, downgrade, CPU fallback, or plugin lookup.

`version` is an identity/provenance field and is never an implicit upgrade mechanism.

### Upstream refresh workflow

1. Offline reviewer edits `docs/upstream-reference-manifest.yaml` and the `.pack.json` record together, retaining immutable revision/path and the exact manifest IDs (`llvm-amdgpu-usage`, `amd-isa-spec-manager`, `radeon-gpu-analyzer`, plus any source-specific ID).
2. Recheck every source/image/generated component license; unknown or changed license state blocks the refresh.
3. Recompute local source and generated image SHA-256 values; update the explicit modification list and build identity; never accept a branch name or release label as a pin.
4. Re-run offline code-object/descriptor/kernarg/resource/ISA/layout checks and the pack numerical/conformance evidence. A promoted performance pack has the `benchmark/benchmark` record; a correctness-control pack has no benchmark reference and a nonempty `benchmark_not_applicable_reason`. Native evidence and scalar-native projection remain request-bound `r9700_native`; NumPy evidence remains a separate `offline_oracle/numpy_oracle` diagnostic reference in `reference_set_kind: f2_wmma_dual`. A distinct physical layout remains blocked until its `offline_review/layout_proof` record binds the source-to-tile/LDS mapping.
5. Generate deterministic C++ records into the supervisor-selected `kernel_catalog.cpp`/`kernel_assets.cpp` boundary, review the diff and ownership, and publish a new pack version when identity/image/build evidence changes. Keep the prior accepted record available for audit until a reviewed cutover.
6. Only after review and supervisor validation does a runtime build consume the new generated record. Runtime never downloads, refreshes, or parses the documentation manifest.

## Ownership matrix and handoffs

Task sequencing is explicit: task sets 2 and 3 may start only after this task-set-1 report receives final re-review acceptance and its active validation-ledger insertion is present; they then run in parallel. Task set 4 (B0 scalar-control migration) may start only after accepted task sets 2 and 3 have produced and validated the concrete generated records; task set 5 additionally waits for the accepted immutable F2/G0 record. Neither migration task edits the generic schema/tooling owned by task sets 2–3.

| Later task set | Owns | Must not own/edit | Handoff/acceptance |
|---|---|---|---|
| 2. Runtime identity/compatibility | `native_r9700/kernel_pack.h`, `native_r9700/kernel_pack.cpp`, `tests/native_r9700/test_kernel_pack_contract.py`; immutable record validation, exact identity/key lookup, selected-entry admission bridge. | Offline manifest parser, source/image migration, F2 WMMA source/image, service/model lifecycle, shared generated catalog files. | Consumes this schema; reuses `KernelDescriptor`/HSA admission; rejects before allocation/submission; no runtime YAML/JSON docs parsing. |
| 3. Offline manifest/ISA/resource validation | `native_r9700/kernel_pack_manifest.py`, `tests/native_r9700/test_kernel_pack_manifest.py`; canonical JSON validation, policy/pin/license/evidence linkage, deterministic generated-record input. | Runtime pack selection, device submission, service/model lifecycle, F2 WMMA source/image implementation. | Produces concrete records for task 2; links source/tool version/input/output digests according to the pack type; malformed/unlicensed/wrong-target/numerical/evidence records fail offline. |
| 4. Scalar-control migration | Scalar pack records and tests through the one integration owner; preserves B0 image bytes, descriptors, dispatch, outputs, and evidence. | F1 service/model files; F2 WMMA source/image; independent catalog path or compatibility alias. | Starts only after accepted task sets 2–3; uses their exact generated records; hardware observation must be `r9700_native`, not `cpu_reference`; no behavior change. |
| 5. Exact G0 WMMA migration | Imports the exact F2/G0 record, image/source digests, descriptors/resources, shapes/tails, numerics, ISA, and hardware evidence through the one integration owner. | Regenerating/redefining lane map/image/tolerances; independent G0 proof; F1 lifecycle files. | Any mismatch blocks migration and returns to F2; P3 consumes the immutable G0 record without duplicate proof. |
| 6. Selection/refresh/review/promotion | Final exact-lookup review, explicit refresh evidence, license review, selection/rejection tests, and final report. | New schema fields, automatic version ranking, runtime docs parsing, service/model lifecycle. | Promotion requires all production records concrete, native evidence bound, and zero Critical/Important findings. |

**Shared integration boundary:** F2 owns WMMA-specific source/image contracts; P3 owns generic Kernel Pack records/tooling. `native_r9700/kernel_assets.cpp`, `native_r9700/kernel_catalog.cpp`, and all generated catalogs/record initializers have exactly one **supervisor-selected integration owner**. Tasks 4 and 5 provide reviewed inputs to that owner and do not edit those files concurrently. This is the same boundary nominated by F2 task set 1. F1 owns service/model lifecycle files; Q1 does not change or assign them. P1 owns executable/device-owner handles and carries pack digest/entry identity but does not allocate P3 selectors or handles.

## Active validation ledger insertion

The following sections are ready to insert under the shared ledger. They are commands only; the supervisor runs them from the worktree root. No command below was run by this agent.

### P3 schema

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
mkdir -p build/native-r9700-runtime
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/prefill_npz.cpp native_r9700/vram_layout.cpp \
  native_r9700/vram_allocator.cpp native_r9700/dynamic_page_table.cpp \
  native_r9700/resident_memory.cpp native_r9700/vram_smoke_asset.cpp \
  native_r9700/hsa_code_image_asset.cpp native_r9700/model_weight_binder.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_pack.cpp \
  native_r9700/kernel_catalog.cpp native_r9700/device_memory.cpp \
  native_r9700/hardware_lock.cpp native_r9700/llama_stage_layout.cpp \
  native_r9700/llama_layer_executor.cpp native_r9700/kernel_assets.cpp \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
build/native-r9700-runtime/native_r9700_runner --lifecycle-dry-run
"$PY" -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py -v
```

Expected observations:

- The direct C++ build succeeds with `kernel_pack.cpp` linked into the existing runner shape; `--lifecycle-dry-run` remains hardware-free and passes.
- Pack contract tests prove exact identity/key lookup, no implicit version upgrade, no plugin/runtime-document path, output-preserving rejection, and reuse of existing descriptor/image admission.
- Manifest tests prove canonical schema validation and deterministic generated-record output. Malformed-pack cases (unknown key, missing/unknown license, missing source/image digest, wrong target, duplicate symbol, malformed kernarg/resource/geometry/tail padding, contradictory dtype/shape/layout/packing/numerics/reference-set kind, missing or unresolved `EvidenceRef`, unknown or mismatched kind/slot, missing physical-layout proof, invalid `pack_sha256` preimage digest, missing required matrix fields or nonempty exact-empty fields, both/neither benchmark outcome fields, `native_run` collapsed into `target_conformance`, and `cpu_reference` substituted for native evidence) reject with a nonempty reason before any generated output is published; the input/output sentinel remains unchanged.
- No runtime process opens `docs/upstream-reference-manifest.yaml` or a `.pack.json` file.

### P3 malformed-pack rejection (focused observation)

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
"$PY" -m pytest tests/native_r9700/test_kernel_pack_manifest.py -v
```

Expected: every malformed record exits through offline validation with a named rejection; no malformed record produces a generated C++ initializer or becomes visible to the runtime catalog. In particular, a component license of `unknown` is a hard rejection, not a warning or pending state.

### P3 scalar migration

```sh
PY=${HOME}/.pyenv/versions/3.12.8/bin/python3
"$PY" -m pytest \
  tests/native_r9700/test_kernel_pack_contract.py \
  tests/native_r9700/test_kernel_pack_manifest.py \
  tests/native_r9700/test_kernel_assets.py \
  tests/native_r9700/test_kernel_catalog.py \
  tests/native_r9700/test_runtime_contract.py -v
mkdir -p build/native-r9700-runtime logs
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  native_r9700/amdev_packets.cpp native_r9700/runtime_contract.cpp \
  native_r9700/prefill_npz.cpp native_r9700/vram_layout.cpp \
  native_r9700/vram_allocator.cpp native_r9700/dynamic_page_table.cpp \
  native_r9700/resident_memory.cpp native_r9700/vram_smoke_asset.cpp \
  native_r9700/hsa_code_image_asset.cpp native_r9700/model_weight_binder.cpp \
  native_r9700/amdev_session.cpp native_r9700/kernel_pack.cpp \
  native_r9700/kernel_catalog.cpp native_r9700/device_memory.cpp \
  native_r9700/hardware_lock.cpp native_r9700/llama_stage_layout.cpp \
  native_r9700/llama_layer_executor.cpp native_r9700/kernel_assets.cpp \
  native_r9700/runtime.cpp native_r9700/runner.cpp -I native_r9700 \
  -o build/native-r9700-runtime/native_r9700_runner
/bin/bash -o pipefail -c 'mkdir -p logs; log=logs/p3-scalar-migration-native-prefill.log; { printf "%s\\n" "command: tools/native-r9700-hardware-run env APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock build/native-r9700-runtime/native_r9700_runner --native-prefill-proof --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json [128000,128001] --out logs/p3-scalar-migration.npz --log logs/p3-scalar-migration-runner.log"; date -u "+timestamp_utc: %Y-%m-%dT%H:%M:%SZ"; tools/native-r9700-hardware-run env APL_REMOTE_SOCK=${TMPDIR}/tinygpu.sock build/native-r9700-runtime/native_r9700_runner --native-prefill-proof --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --token-ids-json "[128000,128001]" --out logs/p3-scalar-migration.npz --log logs/p3-scalar-migration-runner.log; status=$?; printf "wrapper_exit_status: %d\\n" "$status"; exit "$status"; } >"$log" 2>&1'
```

- The request-bound hardware log identifies R9700 `1002:7551`, `gfx1201`, the selected pack/image/pack-preimage digests, entry/dispatch identity, and resolved `EvidenceRef` IDs and digests: `conformance` is `target_conformance/conformance`, `native_run` is `native_run/native_run`, and `resource_review`/`isa_review` are `offline_review/resource_review` and `offline_review/isa_review`. A promoted performance control carries its `benchmark/benchmark` record; a correctness-control pack carries a nonempty `benchmark_not_applicable_reason`. The hardware record is never relabeled `cpu_reference` and never invokes a CPU fallback after cache acceptance.
Expected real-hardware observations:

- The selected records are the exact scalar pack name/version and entry symbols requested by the migrated graph; no older/newer/other compatibility record is chosen.
- The runner log has `record_kind: native_run`, `evidence_slot: native_run`, `producer_kind: r9700_native`, `native_prefill_acceptance: pass`, `native_prefill_full_layer_loop_status: pass`, `failure_stage: none`, and `exit_status: 0`; the wrapper log has `wrapper_exit_status: 0`.
- The request-bound hardware log identifies R9700 `1002:7551`, `gfx1201`, the selected pack/image/pack-preimage digests, entry/dispatch identity, and the resolved `EvidenceRef` IDs/digests. It is `record_kind: native_run`, `evidence_slot: native_run`, `producer_kind: r9700_native`; it must not be relabeled `cpu_reference` and must not invoke a CPU fallback after cache acceptance.
- B0 scalar image bytes, descriptors, S-1/final-token behavior, producer-owned KV truth, and accepted-cache/fallback semantics remain unchanged; the migrated scalar record keeps its exact reviewed kernarg tail-padding value and zeroes that suffix before submission.

### P3 G0 migration

The F2 G0 publication command is copied verbatim below. P3 adds no CLI defaults, substitutions, or flags; the only P3 additions are the pack-consumption observations after the command.

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
      ${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
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

P3 pack-consumption observations (the only additions to the copied F2 command contract):

- Offline validation consumes the accepted task-set-2/3 generated record and the immutable `.superpowers/swarm/reports/g0-wmma-conformance.md` record. It records the exact F2 names `f2-linear-wmma-f16-v1`, `f2-linear-gate-up-f16-v1`, `linear_wmma_f16`, `linear_wmma_f16.image`, `linear_wmma_f16.json`, `F2_WMMA_FP16_FP32_ACC_SINGLE_CAST_V1`, and `f2-wmma-64x64-m-tail-v1`, together with the exact image/source/digest, descriptor/resource, shape/tail, numerical, ISA, and hardware evidence; the physical layout record is `record_kind: offline_review`, `evidence_slot: layout_proof`; it does not regenerate the lane map/image or alter tolerance/results, and it does not emit/admit the reserved `f2-wmma-physical-tile-v1` without that resolved layout record.
- The P3 pack-consumption output identifies the selected pack name/version, canonical `pack_sha256` preimage digest, imported image SHA-256, exact G0 record ID, `target: gfx1201`, entry symbol, and the resolved `EvidenceRef` IDs/digests. It records `pack_validation: pass`, `pack_consumption: pass`, `load_status: pass`, `dispatch_status: pass`, finite output/tail comparison within the immutable G0 policy, and process `exit_status: 0`.
- The hardware run is request-bound native evidence (`record_kind: native_run`, `evidence_slot: native_run`, `producer_kind: r9700_native`) with the resolved scalar-native/NumPy input binding; the scalar projection is `record_kind: target_conformance`, `evidence_slot: scalar_native_projection`, `producer_kind: r9700_native`; the NumPy reference is `record_kind: offline_oracle`, `evidence_slot: numpy_oracle`, `producer_kind: cpu_reference`, with target/image/pack/tool fields exactly empty. Any pack/image/G0 mismatch, missing evidence, or nonzero load/dispatch status blocks migration and returns to F2; it is not repaired by selecting another version.

The copied F2 invocation retains the complete argument set: `--asset-root`, `--source-layout-version`, `--tail-policy`, `--geometry-rule`, `--packing-version`, `--layout-record`, `--numpy-oracle-record`, `--native-projection-record`, `--g0-report`, and `--log`. P3 adds no replacement defaults or alternate runner mode.

## Unresolved external blockers

1. **G0 publication:** `.superpowers/swarm/reports/g0-wmma-conformance.md` and its immutable `g0_record_id`/image digest are not available in this task set. Task set 5 remains blocked until F2 publishes an accepted `g0_status: pass` record with the exact `f2-linear-wmma-f16-v1`/`f2-linear-gate-up-f16-v1` identity, a resolved `offline_review` layout record, and resolved numerical/native evidence; P3 must import its exact values without inference.
2. **Single shared integration owner:** the supervisor must name one owner before task sets 4–5 edit `kernel_assets.cpp`, `kernel_catalog.cpp`, or generated catalogs. This report deliberately does not assign a second owner or edit those files.
3. **Concrete source/build/evidence:** existing scalar sidecars/catalog entries do not yet carry the complete P3 license, build identity, source/physical-layout proof, ISA/source review, resource review, numerical `EvidenceRef`, conformance, native-run, benchmark outcome, and canonical pack-preimage digest. Migration must add resolved values; this report does not guess them.
4. **Hardware evidence:** no P3 scalar or G0 load/dispatch command has been run here. Supervisor must record fresh logs with target/device identity, exact pack/image/entry identity, resolved request-bound `r9700_native` refs, failure stage, and exit status before promotion.

## Supervisor review checklist

- Confirm the C++/JSON field names above match the task-set-2/3 implementation tests exactly; any discrepancy returns to this task-set-1 contract rather than adding an alias.
- Confirm one and only one integration owner for `kernel_assets.cpp`, `kernel_catalog.cpp`, and generated catalogs; F2 and P3 must retain the same boundary.
- Confirm component/file license reviews cover every imported/generated path and that no `unknown` value survives.
- Confirm exact-name/exact-key lookup has zero/>1-match rejection and no semver/feature ranking or fallback.
- Confirm runtime has no documentation YAML/pack JSON parser and uses only compiled/generated concrete records plus existing HSA/image admission.
- Confirm malformed-pack, scalar native, and exact G0 hardware observations are present before changing task-set-1 status from `Needs review`.

### Supervisor-only active-ledger reconciliation

The shared `docs/tasks/native-r9700-producer/validation-commands.md` ledger is intentionally not edited here. Preserve the required headings `P3 schema`, `P3 malformed-pack rejection (focused observation)`, `P3 scalar migration`, and `P3 G0 migration`. If the ledger integrator copied pre-freeze evidence prose, the supervisor must replace that prose with the exact `EvidenceRef` five-kind/evidence-slot matrix and canonical `pack_sha256` preimage in §Evidence references above: restore `native_run` as its own `record_kind`/`evidence_slot`, add `evidence_slot` to every reference, replace all conditional field wording with the unconditional matrix rows, require `offline_review` target/image/pack/tool/input/output digests and exact-empty producer, and require `benchmark_not_applicable_reason` for correctness controls. Do not add aliases or edit the ledger from this task.

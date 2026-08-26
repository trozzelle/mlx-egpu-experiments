# P3 contract freeze — current correction

## Scope

This correction fixes only the authoritative P3 freeze contradictions named for the current review. No production source, test, task-ledger, or shared integration files were edited. The freeze remains a documentation contract pending supervisor review.

## Exact changed sections

1. **`§Frozen Kernel Pack schema v1` opening schema statement**
   - States that schema v1 `entries` contains exactly one entry.

2. **`§Entries, kernargs, resources, and geometry` cardinality paragraph**
   - Replaces the lower bound (“at least one entry”) with exactly one entry while retaining the existing symbol, geometry-rule, ABI, resource, and geometry requirements.

3. **`§Pack evidence` C++ view and normative prose**
   - Adds the required `EvidenceRef source_review` member with the closed `offline_review/source_review` role and empty `producer_kind`.
   - Requires every pack’s source-review reference to carry complete target/image/pack/tool/input/output bindings.
   - Makes the source-equivalent-v1 versus physical-layout distinction explicit: source-equivalent B0 sets `layout_proof.present == false`; a distinct physical packing additionally requires `layout_proof.present == true` for the source-to-byte/tile/LDS mapping.

4. **`§Exact C++ interfaces and catalog boundary` selected-admission declaration**
   - Changes the frozen declaration to:

     ```cpp
     bool admit_kernel_pack(const KernelPackRecord& record,
                            const KernelPackCompatibilityKey& selected_key,
                            std::string_view entry_symbol,
                            std::string_view asset_root,
                            KernelDescriptor* out_descriptor,
                            KernelPackErrorBuffer error_text);
     ```

   - The adjacent comment now requires admission to validate the selected record against the exact compatibility key before delegating to the existing HSA/KernelAsset boundary.

5. **`§Offline manifest records and runtime/offline boundary` canonical JSON shape**
   - Adds the required top-level `evidence.source_review` object, using the exact closed `EvidenceRef` field set and `offline_review/source_review` binding.
   - The example continues to show one `entries` element and keeps all existing field names and evidence roles.

## Preserved invariants

- `gfx1201`, the closed schema-v1 field names, the five-kind/nine-slot evidence matrix, and the exact source-equivalent-v1/physical-layout semantics remain frozen.
- The canonical RFC8785/JCS `pack_sha256` preimage and its recursion exclusions are unchanged: remove the top-level `evidence` object and recursively remove every field named `pack_sha256` or `record_sha256` before serialization.
- No runtime YAML/JSON documentation parsing, fallback, plugin path, or additional schema key was introduced.

## Supervisor verification (not run here)

```sh
git diff --check -- .superpowers/swarm/reports/p3-contract-freeze.md .superpowers/swarm/reports/p3-contract-current-correction.md
```

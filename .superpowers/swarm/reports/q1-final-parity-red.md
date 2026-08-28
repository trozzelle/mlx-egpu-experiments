# Q1 final parity RED review

**Status:** RED contracts added; supervisor validation pending  
**Owner:** `Q1ParityFinalRed`  
**Scope:** Fail-closed admission of the complete schema-v2 inventory and bounded Qwen oracle fixture package.  
**Non-goals:** production implementation, fixture artifact changes, source-text assertions, model loading, numerical computation, and hardware execution.

## Changed test surface

Only `tests/native_r9700/test_qwen_parity.py` was extended. No production module or committed fixture artifact was changed. The tests reuse the committed Qwen fixture package and canonical tensor-inventory report, using temporary copies only for the forged cases.

## New RED contracts

### Frozen scalar identity cannot bless incomplete inventory contents

`test_qwen_parity_rejects_inventory_structure_drift_with_frozen_scalar_identity` is parameterized over four independent inventory drifts:

- remove `tensors`;
- replace `shards` with an empty list;
- remove `affine_classification`; and
- replace the CPU/oracle provenance labels with `producer_kind="r9700_native"` and `native_evidence=true`.

Each forged mapping retains `schema_version=2`, the canonical `model_fingerprint`, and its self-declared canonical `inventory_sha256`, then must be rejected by `compare_qwen_fixtures`. The test passes the mapping directly, so it checks the public parity contract without asserting implementation text.

The current comparator checks only those top-level scalar identity fields when an inventory is supplied. It does not validate the complete tensor records, shard records, affine table, or inventory provenance; these cases therefore remain intentionally RED until the full schema-v2 inventory is bound.

### Self-consistent empty NPZs cannot satisfy the oracle package

`test_qwen_parity_rejects_self_consistent_empty_required_fixture_arrays` copies the committed fixture directory into `tmp_path`, writes empty NPZ archives for the affine-window, hybrid-state-sample, and oracle-trace artifacts, replaces each corresponding schema `arrays` map with `{}`, and recomputes every changed artifact SHA-256 plus the determinism digest from the resulting preimage. The schema retains its top-level identities, shard/source records, prompt identity, state components, and boundary declarations. The forged package must still be rejected by `compare_qwen_fixtures`.

This pins the final-review requirement that the package contain the exact six affine windows, four state samples, all required trace arrays, and their source/component/boundary metadata; matching self-declared artifact and determinism hashes are not sufficient. The current comparator accepts an empty metadata map paired with an empty NPZ, so this case is intentionally RED.

## Supervisor focused command

Run exactly:

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_parity.py -v
```

This lane did not run tests, builds, linters, formatters, package managers, model loads, hardware commands, or git commands.

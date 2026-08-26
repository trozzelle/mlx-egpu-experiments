# Q1 final parity GREEN handoff

**Status:** Implementation complete; supervisor validation pending  
**Scope:** Close the final schema-v2 inventory and bounded Qwen oracle fixture parity gaps.  
**Owner:** `Q1ParityFinalGreen`

## Changed production surface

- `native_r9700/qwen_text_adapter.py`
  - Promoted strict schema-v2 inventory admission to the public `validate_qwen_tensor_inventory` identity owner.
  - The owner accepts a JSON path or mapping, verifies exact CPU-reference/non-native provenance, frozen scalar counts, shard/header identity, six-field tensor records, affine classification, bounds/overlap, and the canonical tensor/affine digest.
  - Returned private `_tensor_by_name` and `_shard_by_name` indexes remain consumer-only derived fields; they are not part of the canonical preimage.
- `native_r9700/ref_fixtures.py`
  - Fixture generation now calls the shared inventory owner directly. The former private entry points are compatibility wrappers only and contain no second validation path.
- `native_r9700/qwen_parity.py`
  - Supplied inventories are admitted through the shared owner and all owner failures are normalized to `QwenParityError`.
  - Added closed immutable contracts for the six affine-window records, four hybrid-state sample records/components, and all six layer0/layer3/final trace records and boundaries. Exact keys and metadata are checked before any self-declared artifact/determinism hash is trusted.


The state contract builder separates variable-rank sample slices from stored
shape and digest fields, so both recurrent and full-attention samples retain
their exact source/component metadata.
## Final RED cases covered

1. Inventory tensor records removed while scalar identity/digest labels remain.
2. Inventory shard records replaced with an empty list.
3. Inventory affine classification removed while scalar identity/digest labels remain.
4. Inventory provenance changed to native evidence.
5. All three required NPZs replaced with self-consistent empty archives and recomputed file/determinism hashes.

The committed package identity, CPU-reference labels, fixture bytes, and optional `model_dir=None` behavior remain unchanged. No tests, model loads, hardware commands, linters, formatters, package commands, or git commands were run in this lane.

## Supervisor focused command

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_qwen_parity.py -v
```

# Q1 task-set 4: oracle fixtures RED contract

**Status:** RED contract added; supervisor validation pending
**Owner:** `Q1OracleRed`
**Scope:** deterministic CPU/MLX Qwen3.8-27B text-only fixture package, fixture catalog binding, and parity integration.
**Non-goals:** production implementation, committed fixture generation, model loading, native execution, hardware evidence, cache/spill ownership, or changes to the existing Llama fixture package.

## Changed test surface

Only these task-set-4 test files were extended:

- `tests/native_r9700/test_ref_fixtures.py`
- `tests/native_r9700/test_fixture_catalog.py`
- `tests/native_r9700/test_qwen_parity.py`

No production module or fixture artifact was changed. The RED tests intentionally require the Qwen package and task-set-4 APIs, so missing generator/fixtures/parity integration fails as a focused contract failure rather than being skipped by the legacy Llama fixture setup.

## Frozen fixture package

The committed Qwen fixture directory must contain exactly these five `qwen_*` files:

1. `qwen_prompts.json`
2. `qwen_affine_windows.npz`
3. `qwen_hybrid_state_samples.npz`
4. `qwen_oracle_trace.npz`
5. `qwen_fixtures_schema.json`

The schema contract freezes:

- `schema_version=1`, `kind=qwen3.8_text_oracle`.
- Model fingerprint `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`.
- `base_model_revision=unavailable_in_pinned_conversion_metadata`.
- `inventory_schema_version=2` and inventory digest `508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`.
- Exact converted model revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, MLX-VLM revision `2b31570bdee86e2cdeea049761885aeed524a98c`, and mlx-lm revision `e2f2fb2aef987f86878d17638446183cffe21fe4`.
- Exact metadata sidecar digests and all three pinned shard names, sizes, and SHA-256 values from the task-set-1 identity freeze.
- `producer_kind=cpu_reference`, `native_evidence=false`, `text_only=true`, and the sensitive-data policy `minimal text-only token IDs; no image/video bytes or full model dump`.
- Per-artifact byte SHA-256 records and a deterministic digest over the frozen identity, source revisions, shard records, and artifact digests.

`test_ref_fixtures.py` requires every artifact digest to match its bytes. It requires `qwen_prompts.json` to contain the exact text-only probe token IDs `[760,6511,314,9338,369]`, `S=5`, prefix `[760,6511,314,9338]`, `prefix_length=4`, `final_token_id=369`, and rejected multimodal IDs `[248053,248054,248056,248057]`.

## State and tensor boundary contracts

The state sample schema freezes the explicit 64-layer runtime order: `ArraysCache` on every layer except `KVCache` at `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`, with `committed_position=4` and final token `369`. It requires the four selected component IDs in captured order:

- `layer.0.arrays.conv_state`: `ArraysCache`, `(1,3,10240)`, `bfloat16`, owner `Qwen3_5GatedDeltaNet`, update `retain_last_3_mixed_qkv_rows`, position `committed_position`, not trimmable.
- `layer.0.arrays.delta_state`: `ArraysCache`, `(1,48,128,128)`, `float32`, owner `gated_delta_update called by Qwen3_5GatedDeltaNet`, update `recurrent_delta_update`, position `committed_position`, not trimmable.
- `layer.3.full_attention.keys` and `.values`: `KVCache`, `(1,4,4,256)`, `bfloat16`, owner `Qwen3_5Attention/KVCache`, update `KVCache.update_and_fetch`, position `offset=N`, trim `KVCache.trim`.

`qwen_affine_windows.npz` is required to contain only bounded `uint8` windows and metadata for the six minimum layer-local affine tensors: layer 0 `linear_attn.in_proj_qkv.{weight,scales,biases}` with source shapes `[10240,640]`, `[10240,80]`, `[10240,80]`, and layer 3 `self_attn.q_proj.{weight,scales,biases}` with source shapes `[12288,640]`, `[12288,80]`, `[12288,80]`. Each record carries source tensor name/shard/absolute offset/byte count, source dtype/shape, affine mode/bits/group size, model fingerprint, inventory digest, and array digest.

`qwen_hybrid_state_samples.npz` and `qwen_oracle_trace.npz` are schema-bound by array key, stored shape/dtype, source dtype, token range, component/boundary, tolerance policy, and array digest. Trace coverage must include recurrent layer 0, full-attention layer 3, and final-token output boundaries.

## Generator and parity integration contracts

`test_ref_fixtures.py::test_qwen_fixture_generation_exposes_deterministic_owner` requires a callable `native_r9700.ref_fixtures.generate_qwen_fixtures` owned by the `--generate-qwen` path and checks the persisted deterministic digest preimage. The generator must fail closed on mismatched model/inventory/source/shard identity and must emit reproducible bytes.

`test_qwen_parity.py::test_qwen_parity_uses_task3_mlx_restore_and_final_token_only` patches the task-set-3 `restore_qwen_hybrid_cache_into_mlx` boundary and requires parity to call it with the captured state before invoking `generate_step([369], model, prompt_cache=restored_cache, ...)`. Assigning opaque `QwenStateLeaf` objects directly is not accepted.

`test_qwen_parity.py::test_qwen_parity_exposes_fixture_comparison_and_rejects_native_evidence` requires a callable `compare_qwen_fixtures` for `--compare-fixtures` and a `validate_qwen_fixture_evidence` guard. The guard must accept only `producer_kind=cpu_reference` with `native_evidence=false` and reject `r9700_native` or any true native-evidence claim with `QwenParityError`.

## Expected RED cause

In the current pre-task-set-4 source, the Qwen fixture files/schema are absent, `generate_qwen_fixtures` is absent, `fixture_catalog` has no Qwen archive entries, `qwen_parity` has no fixture comparison/evidence validator, and parity still calls the opaque `restore_qwen_hybrid_cache` path. These are the intended RED failures. The legacy Llama fixture directory remains present, so collection is not blocked by a missing setup directory and existing C1 tests retain their independent behavior.

## Supervisor validation


```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py \
  tests/native_r9700/test_qwen_parity.py -v
```

This RED lane did not run tests, model loads, builds, formatters, package managers, hardware commands, or git commands. The supervisor owns RED execution and later fixture generation/GREEN verification.

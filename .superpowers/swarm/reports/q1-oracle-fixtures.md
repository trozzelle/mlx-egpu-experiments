# Q1 task-set 4: oracle fixtures GREEN handoff

**Status:** Implemented; supervisor generation, parity, and GREEN verification complete; final package review pending  
**Owner:** `Q1OracleFixtures`  
**Scope:** deterministic CPU/MLX Qwen3.8-27B text-only fixture generation, immutable fixture catalog binding, and parity comparison/evidence integration.  
**Production files:** `native_r9700/ref_fixtures.py`, `native_r9700/fixture_catalog.py`, `native_r9700/qwen_parity.py`

## Delivered contract

The `--generate-qwen` owner consumes the task-set-2 schema-v2 tensor inventory and the task-set-3 cache/capture/MLX-restore seams. It validates the frozen model fingerprint, source revisions, inventory digest, four metadata sidecar digests, three shard names/sizes/digests, exact text-only token IDs `[760, 6511, 314, 9338, 369]`, and the `S=5`, `N=S-1=4`, final-token `369` boundary before loading MLX.

Generation emits exactly these five `qwen_*` files under the requested fixture directory:

1. `qwen_prompts.json`
2. `qwen_affine_windows.npz`
3. `qwen_hybrid_state_samples.npz`
4. `qwen_oracle_trace.npz`
5. `qwen_fixtures_schema.json`

The affine archive stores only bounded `uint8` windows (maximum 65,536 bytes) for the six required layer-0/layer-3 affine tensors, with source shard/absolute offset/shape/dtype/quantization metadata and per-window digests. State samples store bounded float32 views of the layer-0 recurrent and layer-3 full-attention leaves while preserving complete source shape/dtype and ownership/update/position/trim metadata. The trace stores the same failure-localizing boundaries and final-token input/output IDs; it does not dump full weights, images, videos, prompts, or model activations.

NPZ archives use sorted keys and fixed ZIP metadata. JSON and NPZ bytes are staged, flushed, and fsynced before a rollback-capable publication transaction replaces the five managed paths. Existing artifacts and stale `qwen_*` files are restored if any replacement fails; staging files are removed on success or failure. The optional generation report is written atomically after the package commit.

`fixture_catalog.fixture_specs()` now loads the Qwen schema in addition to the unchanged legacy schema, preserving immutable `FixtureSpec` lookup and one declaration per NPZ array.

All parity restoration uses `native_r9700.qwen_hybrid_cache.restore_qwen_hybrid_cache_into_mlx`; no opaque spill leaves are assigned by `qwen_parity`. `generate_qwen_from_hybrid_state` passes only `[369]` to `generate_step` after the accepted prefix restore. `compare_qwen_fixtures` validates artifact bytes, array metadata/digests, identity, determinism preimage, runtime layer order, selected component contracts, and `S-1` text-only semantics. `validate_qwen_fixture_evidence` accepts only `producer_kind="cpu_reference"` with `native_evidence=false` and rejects native evidence labels/claims.

## Frozen identity and evidence boundary

- `model_fingerprint`: `4304f20a69213c8f0620ab7388163dd58b324278679d94c5915f279438d1b371`
- `base_model_revision`: `unavailable_in_pinned_conversion_metadata`
- `model_revision`: `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
- `mlx_vlm_revision`: `2b31570bdee86e2cdeea049761885aeed524a98c`
- `mlx_lm_revision`: `e2f2fb2aef987f86878d17638446183cffe21fe4`
- `inventory_schema_version`: `2`
- `inventory_sha256`: `508567ed00f7d65283fcb7f5ecba55e9a9904a9f2f41e8724bf1ef37589725e4`
- `producer_kind`: `cpu_reference`
- `native_evidence`: `false`
- `sensitive_data_policy`: `minimal text-only token IDs; no image/video bytes or full model dump`

The unresolved base-model revision remains an explicit promotion blocker. Q1 fixtures are CPU/MLX oracle artifacts only and cannot satisfy `r9700_native` acceptance.

## Supervisor generation command

The supervisor reran this command after the source-pin provenance and full-shard verification corrections:

```sh
PY="${PY:?set PY to the pinned Python 3.12.8 interpreter}"
QWEN_MODEL=<model-hub>/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff

"$PY" -m native_r9700.ref_fixtures \
  --generate-qwen \
  --model "$QWEN_MODEL" \
  --token-ids-json '[760,6511,314,9338,369]' \
  --fixtures-dir tests/native_r9700/fixtures \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --report logs/q1-qwen-oracle-fixtures.json
```

Observed package output was exactly the five files above. The generated schema's per-file SHA-256 values and deterministic digest are the authority consumed by the catalog and parity comparison.

## Supervisor GREEN commands

```sh
"$PY" -m pytest \
  tests/native_r9700/test_ref_fixtures.py \
  tests/native_r9700/test_fixture_catalog.py \
  tests/native_r9700/test_qwen_parity.py -v

"$PY" -m native_r9700.qwen_parity \
  --compare-fixtures \
  --model "$QWEN_MODEL" \
  --fixtures-dir tests/native_r9700/fixtures \
  --inventory logs/q1-qwen-tensor-inventory.json \
  --token-ids-json '[760,6511,314,9338,369]' \
  --out logs/q1-qwen-parity.json
```

Observed parity report fields are `status="pass"`, the frozen model/inventory identities, `producer_kind="cpu_reference"`, `native_evidence=false`, `prefix_length=4`, and `final_token_input=[369]`.

## Expected resource needs

The exact pinned snapshot requires the four sidecars and three safetensors shards from the Q1 identity report. The three shard files total 16,054,541,349 bytes on disk; generation additionally needs temporary space for the five staged artifacts and the task-set-2 inventory/report. MLX must have enough unified memory to load the pinned 27B affine-4bit reference and its 64-layer hybrid cache; no R9700/native device or native evidence is required. The generator's persisted data remains bounded to six affine byte windows, four selected cache-leaf samples, and a minimal prompt/final-token trace.

Supervisor evidence after final Q1 fixes: source-pin and schema-v2 inventory refresh passed; pinned mlx-lm `0.32.0` / MLX `0.32.1` fixture generation wrote exactly five files; model-bound parity passed after full metadata/shard hashing; and the active Q1 package gate passed `259` tests with two MLX dependency deprecation warnings. MLX-VLM remains explicitly reference-only. No native R9700 claim is made by this CPU/MLX oracle gate.

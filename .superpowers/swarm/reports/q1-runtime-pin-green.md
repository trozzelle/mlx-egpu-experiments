# Q1 executed oracle-runtime pin GREEN handoff

**Status:** Implementation complete; supervisor validation and fixture regeneration pending  
**Scope:** Bind the Qwen CPU/MLX oracle producer to one verified, exact `mlx-lm`/MLX runtime.  
**Owner:** `Q1RuntimePinGreen`

## Changed production surface

- `native_r9700/ref_fixtures.py`
  - Added the exact `mlx-lm` 0.32.0 and MLX 0.32.1 pins plus the five frozen `mlx_lm` source SHA-256 digests.
  - Added bounded chunked source hashing through `_qwen_hash_runtime_source(path)`.
  - Added strict explicit-root `METADATA`/`dist-info` parsing and `direct_url.json` commit provenance validation for `mlx-lm` revision `e2f2fb2aef987f86878d17638446183cffe21fe4`.
  - Added default-runtime resolution from the already imported `mlx_lm`/`mlx` package roots and their matching distributions; a distribution metadata root that differs from the imported package is rejected.
  - Added `validate_qwen_oracle_runtime(runtime_root=None)`, returning the canonical runtime record below and rejecting missing modules, wrong versions, wrong provenance, or any source digest drift.
  - Added `runtime_root=` to `generate_qwen_fixtures` and `--runtime-root` to `--generate-qwen`; runtime validation occurs directly after inventory validation and before model metadata/shard hashing, affine-window reads, or MLX loading.
  - `_load_mlx` receives the verified root for Qwen generation, prepends that root only for import resolution, and rejects preloaded/imported `mlx_lm` or `mlx` modules outside it.
  - Bound `oracle_runtime` into the generated schema, JSON report, deterministic input order, and canonical determinism preimage.

## Canonical executed-runtime record

```json
{
  "kind": "qwen_oracle_runtime",
  "loader": "mlx_lm.utils.load",
  "model_module": "mlx_lm.models.qwen3_5",
  "mlx_lm": {
    "revision": "e2f2fb2aef987f86878d17638446183cffe21fe4",
    "version": "0.32.0",
    "source_sha256": {
      "mlx_lm/generate.py": "2eee82cdcca2a3c4637643efbdaacc23d6524a1e1093b0953d6860ed8fb5196f",
      "mlx_lm/models/cache.py": "bec0ef0f869cb0ab34c9155e3a03f00c2c505c9cfb48b353daab8cd1d32cde27",
      "mlx_lm/models/gated_delta.py": "a9f1f68d83acdccb0866377d588dde1e50900f810da9ba2deeb67b7df847faff",
      "mlx_lm/models/qwen3_5.py": "51c823c054872f852ba6762daeae202b81a9847dc45702788937950cfcbefda8",
      "mlx_lm/utils.py": "88208cb75b592853f549299af239eb7c6d1049c4c8d6ea73acf4b0c8f3fabada"
    }
  },
  "mlx": {"version": "0.32.1"},
  "mlx_vlm": {
    "revision": "2b31570bdee86e2cdeea049761885aeed524a98c",
    "role": "reference_only"
  }
}
```

The executed loader is `mlx_lm.utils.load`; MLX-VLM is not loaded by this producer and is recorded only as the explicit `reference_only` conversion/reference revision.

## Determinism contract

The generated schema uses this exact input order:

```text
model_fingerprint
inventory_sha256
oracle_runtime
source_revisions
shards
fixture_file_sha256
```

`oracle_runtime` is included in the canonical JSON preimage in the same position, and the report copies the exact same object.

## Supervisor commands

Focused runtime-pin command (not run in this lane):

```sh
${PY} -m pytest \
  tests/native_r9700/test_ref_fixtures.py -k \
  'oracle_runtime or generation_verifies_runtime_before_model_oracle_reads' -v
```

Pinned-runtime fixture regeneration (supervisor-owned; command shape):

```sh
PYTHONPATH=build/q1-oracle-runtime:$PYTHONPATH \
${PY} -m native_r9700.ref_fixtures \
  --generate-qwen \
  --runtime-root build/q1-oracle-runtime \
  --model <qwen-model-dir> \
  --fixtures-dir tests/native_r9700/fixtures \
  --token-ids-json '[760,6511,314,9338,369]' \
  --inventory <qwen-inventory.json> \
  --report logs/q1-qwen-oracle-fixtures.json
```

No tests, model loads, fixture regeneration, package installation, linters, formatters, or git commands were run in this lane.

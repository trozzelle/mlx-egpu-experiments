# Q1 executed oracle-runtime pin RED contract

**Status:** RED tests added; focused validation intentionally not run
**Owner:** `Q1RuntimePinRed`
**Scope:** Bind Q1 fixture generation to the exact executed `mlx-lm`/MLX runtime rather than ambient packages.
**Non-goals:** production implementation, fixture regeneration, package installation, model loading, and validation execution.

## Changed test surface

Only these requested files were changed:

- `tests/native_r9700/test_ref_fixtures.py`
- `.superpowers/swarm/reports/q1-runtime-pin-red.md`

No production module or committed fixture artifact was edited. The tests create tiny package trees under `tmp_path`; they do not install or import an actual MLX runtime. The tiny-tree tests patch only the narrow `_qwen_hash_runtime_source(path)` I/O seam so exact frozen digests can be exercised without embedding source-file preimages. A changed source byte returns its real tiny-file digest and therefore still exercises fail-closed digest rejection.

## Frozen executed-runtime contract

The public/internal verifier is:

```python
validate_qwen_oracle_runtime(runtime_root: str | Path | None = None) -> dict[str, object]
```

The canonical record required by tests is:

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

The `mlx-lm` commit is read from the explicit package provenance and the exact `mlx-lm-0.32.0.dist-info/METADATA` and `mlx-0.32.1.dist-info/METADATA` trees are checked. Package version alone is not an identity proof.

## RED behavior coverage

- Exact versions plus all five source digests return the canonical runtime record and freeze the five pinned digest constants/paths.
- Wrong `mlx-lm` or MLX version rejects.
- Missing source module rejects.
- One changed source byte rejects by digest.
- `generate_qwen_fixtures(..., runtime_root=...)` rejects a runtime mismatch before model metadata/shard hashing, affine-window reads, or `_load_mlx`; the test guards all three later stages.
- Generated schema requires exact `oracle_runtime`; its `mlx_vlm` entry is explicitly `role="reference_only"`, while the executed loader remains `mlx_lm.utils.load`.
- The deterministic input list and canonical preimage include `oracle_runtime`.
- The generated Q1 JSON report is required to carry the same exact canonical runtime record.

## Expected RED cause

The current ambient-unverified generator has no executed-runtime verifier/root boundary, does not bind package metadata and source digests to the Q1 schema/report, and does not include `oracle_runtime` in the deterministic preimage. Consequently the new verifier, ordering, schema, determinism, and report assertions fail until the runtime pin is implemented and the Q1 package/report is regenerated from that verified runtime.

## Focused validation command (not run)

```sh
${PY} -m pytest \
  tests/native_r9700/test_ref_fixtures.py -k \
  'oracle_runtime or generation_verifies_runtime_before_model_oracle_reads' -v
```

Per assignment, no tests, linters, formatters, package commands, model loads, or builds were run by this RED lane.

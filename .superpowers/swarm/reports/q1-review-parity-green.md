# Q1 parity identity GREEN handoff

**Status:** Implementation complete; supervisor validation pending  
**Scope:** Qwen parity admission of the model directory against the frozen fixture/source identity.  
**Owner:** `Q1ParityGreen`

## Changed production surface

- `native_r9700/qwen_parity.py`
  - `compare_qwen_fixtures(..., model_dir=...)` now admits a model only when the path is an existing directory whose basename is the frozen revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
  - The fixture schema's exact `metadata_sha256` mapping is frozen and every required sidecar (`config.json`, `model.safetensors.index.json`, `tokenizer.json`, and `tokenizer_config.json`) is hashed before acceptance.
  - The model directory's `.safetensors` file names must exactly match the three frozen shard names. Each shard's complete file size and SHA-256 digest is checked.
  - Model metadata and shard hashes stream through bounded 1 MiB reads; no multi-gigabyte shard is loaded with `Path.read_bytes`.
  - Missing paths, stat/open failures, basename drift, sidecar drift, shard-name drift, size drift, and digest drift are normalized to `QwenParityError`. `model_dir=None` keeps the existing fixture-only behavior.
  - The validated recurrent delta-state owner is exactly `gated_delta_update`; the report no longer admits the stale descriptive owner string.

## Preserved contract

The accepted report remains text-only CPU/MLX oracle evidence with `producer_kind=cpu_reference`, `native_evidence=false`, the frozen inventory digest, and the S-1 boundary (`prefix_length=4`, `final_token_input=[369]`). No native evidence, model fallback, or alternate revision is admitted.

## Supervisor focused command

```sh
${PY} -m pytest \
  tests/native_r9700/test_qwen_parity.py -v
```

This worker did not run tests, builds, linters, formatters, package managers, model loads, hardware commands, or git commands.

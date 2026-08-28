# C1K Task 2 — Weight/config container decision + narrow loader (Lane B)

Status: **COMPLETE** — container decided (MLX safetensors), narrow loader
built and smoke-verified against the real model config; tests + validation
commands recorded for the supervisor to run.

## Container decision

**Selected first native producer weight container: the MLX safetensors
directory (`mlx_models/meta-Llama-3.2-1B-Instruct`).**

Rationale (exact config parity wins):

- The MLX safetensors dir is the **single self-contained source** carrying
  BOTH the fp16 weights AND the complete `config.json` sidecar (geometry +
  Llama-3 `rope_scaling`).
- The F16 GGUF (`meta-Llama-3.2-1B-Instruct.F16.gguf`) is fp16 but its archive
  KV metadata records `rope.freq_base`/`rope.dimension_count` and **not** the
  Llama-3 `rope_scaling` fields. The Phase 0 harness already documents this
  (`tinygrad_kv_worker/harness.py` `_load_mlx_rope_config`, and
  `docs/path-a-validation-results.md` note) and patches tinygrad from the MLX
  sidecar for exactly that reason. Choosing the GGUF alone would force a
  dual-source RoPE patch and cannot provide exact consumer parity.
- The alternative clause of the task ("MLX safetensors if a narrow loader is
  simpler and exact config parity is preserved") is satisfied: a stdlib
  `json` parse of the sidecar replaces any GGUF binary parsing, and the
  loader's provenance is the same on-disk config the Phase 0 MLX consumer
  reads, so geometry/RoPE parity is by construction.

**Dtype note (fp16 contract):** the MLX `config.json` `torch_dtype` is
`bfloat16`, which describes the HF-original repo. The on-disk weights are
**fp16**: safetensors header inspection of
`mlx_models/meta-Llama-3.2-1B-Instruct/model.safetensors` reports dtype `F16`
for every tensor (verified with the `safetensors` library and the loader's
own header reader). The loader treats actual on-disk weight dtype as
authoritative for the fp16 gate and reports `config_torch_dtype` as advisory
provenance.

## Files created

- `native_r9700/__init__.py` — package marker (Python helpers import
  explicitly; no eager submodule imports).
- `native_r9700/config.py` — `Llama32Config` dataclass + `load_config_from_json`
  with strict validation and loud `ConfigError`/`UnsupportedModelError`/
  `GeometryMismatchError`/`UnsupportedDtypeError`. Configures exact geometry
  (16/8/64/2048), RoPE theta 500000, and the Llama-3 `rope_scaling` sidecar.
- `native_r9700/loader.py` — `load_model_metadata` + `format_report` + `main`
  CLI. Reads only `config.json` + safetensors header records (never weights),
  validates fp16 weight dtype from each shard header, and prints geometry +
  provenance.
- `tests/native_r9700/test_loader.py` — focused unit tests using a small
  on-disk config fixture + tiny synthetic safetensors header records; no model
  weights and no download required.
- `docs/tasks/native-r9700-producer/validation-commands.md` — appended
  "### C1 loader (Lane B — weight/config, task set 2)" section with the exact
  loader command, expected report lines, and the focused pytest command;
  updated the C1 loader row in the command-discovery table.

## Geometry the loader reports (from the real MLX `config.json`)

```
model: Llama-3.2-1B-Instruct (official Meta, mlx safetensors consumer)
model_type: llama
architectures: LlamaForCausalLM
num_layers: 16
n_kv_heads: 8
head_dim: 64
hidden_size: 2048
intermediate_size: 8192
vocab_size: 128256
max_position_embeddings: 131072
rope_theta: 500000.0
rope_scaling: rope_type=llama3 factor=32.0 high_freq_factor=4.0 low_freq_factor=1.0 original_max_position_embeddings=8192
rms_norm_eps: 1e-05
weight_dtype: F16 (F16 from safetensors header)
config_torch_dtype (advisory HF-original): bfloat16
config_source: <abs path>/config.json
weight_index_source: <abs path>/model.safetensors.index.json
weight_shard_count: 1
provenance: official fp16 meta-llama/Llama-3.2-1B-Instruct weights (mlx safetensors consumer dir; same on-disk config the Phase 0 MLX consumer reads for geometry and Llama-3 RoPE parity)
exit_status: 0
```

Geometry is read from the same `config.json` the Phase 0 MLX consumer reads
(`tinygrad_kv_worker.harness._load_mlx_rope_config`), so parity is by
construction (documented source).

## Failure behavior (verified by smoke)

- Missing model dir / missing `config.json` → descriptive error, exit 1.
- Geometry mismatch (e.g. `num_hidden_layers: 32`, `head_dim: 128`,
  `rope_theta: 10000`) → descriptive `geometry mismatch` error, exit 1.
- Unsupported model (`model_type: gpt2`) → `unsupported model_type` error.
- Unsupported dtype (safetensors header dtype not `F16`, or config
  `torch_dtype` outside the fp16/bf16 advisory set) → `unsupported` error.
- None fall through silently.

## Exact commands for the supervisor

Loader validation (reads only config + headers; needs the model dir):

```sh
cd <former-native-r9700-worktree>
${PY} -m native_r9700.loader \
  --model mlx_models/meta-Llama-3.2-1B-Instruct
```

Note: the model files live in the `tinygrad-kv-worker-phase0` worktree
(`.worktrees/tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct`)
in this environment; the same command with that path was smoke-verified to
exit 0 and print geometry + provenance (above). If the model dir is not
present under `mlx_models/` in this worktree, point `--model` at wherever the
MLX safetensors dir lives.

Focused loader tests (no model weights required):

```sh
cd <former-native-r9700-worktree>
${PY} -m pytest tests/native_r9700 -v
```

Regression guards the supervisor already runs (unchanged by this work):
`tests/test_native_amdev_transfer_contract.py` (23 passed) and the baseline
`tests -v` (17 passed).

## Notes / constraints honored

- No model weights or logs staged — `loader.py` reads only headers; `logs/`
  and `mlx_models/` stay git-ignored.
- Does not import or call tinygrad (Path C producer constraint).
- The C0 committed probe `experiments/native-r9700-runtime/native_amdev_transfer_probe.cpp`
  was not touched.
- Shared `native_r9700/` and `tests/native_r9700/` dirs were created in
  coordination with Lane A (`C1RunnerScaffold` owns `runtime.h`/`runtime.cpp`;
  this lane owns the Python loader files).

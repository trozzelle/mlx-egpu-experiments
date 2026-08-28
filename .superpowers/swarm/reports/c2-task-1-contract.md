# C2 task set 1 — integration contract and validation discovery

Status: implemented for supervisor review.

## Source evidence

- C2 goal/dependencies: `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md:14-24` requires real mlx-lm serving integration after C1, through the stable C1 producer contract.
- C2 task set 1 acceptance: `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md:56-83` requires source paths, threshold behavior, fallback semantics, and exact commands.
- Stable producer invocation: `.superpowers/swarm/reports/c1-task-10-review-handoff.md:29-40` requires request token ids via `native_r9700.prefill --token-ids-json` followed by `native_r9700.kv_cache`.
- Imported-cache rule: `.superpowers/swarm/reports/c1-task-10-review-handoff.md:43-47`, `docs/DESIGN.md:138-144`, and `docs/pinned-upstream-interfaces.md:78-84` require an imported `S-1` cache plus only the final prompt token to `mlx_lm.generate.generate_step`.
- Fallback boundary: `docs/DESIGN.md:178-181` permits consumer fallback only before accepting an imported cache; ADR 0002 `docs/adr/0002-producer-owns-kv-truth.md:7-12` says the consumer must not recompute producer-owned KV as a per-request verification path.
- No network exposure: `docs/DESIGN.md:183-186`; C2 task set 1 non-goal `phase-c2-serving-integration.md:62`.
- mlx-lm API re-check: local `mlx_lm/generate.py:307-322` defines `generate_step(prompt, model, ..., prompt_cache=None)`, `generate.py:430-453` processes all supplied prompt tokens and calls `_step(prompt[-1])`; local `mlx_lm/models/cache.py:62-85` loads safetensors prompt caches and returns metadata when requested; `KVCache.state` setter reconstructs `offset` from `keys.shape[2]` at `cache.py:370-373`.
- C1 producer source: `native_r9700/prefill.py:381-412` accepts request token ids and splits S-1/final; `native_r9700/prefill.py:418-431` logs command/model/config/final token/prefix/output/status; `native_r9700/kv_cache.py:256-305` validates log/output behavior and writes prompt cache; `native_r9700/parity.py:126-141` and `native_r9700/parity.py:246-258` show final-token decode and metadata validation seams to reuse.

## Frozen C2 contract

### Source and test paths

- Wrapper source: `native_r9700/serving.py`.
- Focused tests: `tests/native_r9700/test_serving.py`.
- Report/ledger updates: `docs/path-a-validation-results.md`, `docs/archive/tasks/native-r9700-producer/phase-c2-serving-integration.md`, `docs/tasks/native-r9700-producer/validation-commands.md`, and `.superpowers/swarm/progress.md`.

### Public API shape for task set 2

Task set 2 should implement a thin, testable wrapper without changing mlx-lm internals:

- `NativePrefillConfig`: `producer_model_dir`, Python executable, `threshold_tokens`, `producer_timeout_s`, artifacts dir, and optional request id. `producer_model_dir` defaults to the consumer `--model` path but can be overridden for producer-unavailable fallback tests.
- `generate_with_native_prefill(model, tokenizer, prompt, *, native: NativePrefillConfig, max_tokens=256, log_path=None, generate_step_fn=None, load_prompt_cache_fn=None, **generate_kwargs) -> dict`: reusable serving seam for a resident mlx-lm model/tokenizer. It tokenizes string prompts with mlx-lm semantics or validates supplied token ids.
- CLI: `python -m native_r9700.serving --model <consumer-mlx-model-dir> [--producer-model <producer-model-dir>] (--prompt <text> | --token-ids-json '[...]' | --fixtures-dir <dir> [--prompt-name <name> ...]) --max-new-tokens <N> --threshold-tokens <S> --producer-timeout-s <seconds> --artifacts-dir <dir> --json <result.json> --log <run.log> [--report <path>]`. With `--fixtures-dir` and no `--prompt-name`, the CLI runs all committed fixture prompts.

Return/result JSON should include at least: `schema_version`, `status`, `gate_result` for integration runs, `route` (`native_producer` or `native_mlx_fallback`), `fallback_reason`, `model_dir`, `producer_model_dir`, `prompt_count` or `prompt_length`, `n_prefix`, `threshold_tokens`, `producer_timeout_s`, `accepted_cache`, `producer_prefill_npz`, `producer_prompt_cache`, `prefill_log`, `kv_cache_log`, `metadata`, `decoded_tokens`, `duration_ms`, and `exit_status`.

### Threshold

Default threshold: **128 total prompt tokens** (`S >= 128` uses the native producer when available). Rationale: it keeps `prompt-0` (`S=6`) on the native mlx-lm fallback path while routing Phase 0/C1 large prompts (`prompt-1 S=222`, `prompt-2 S=661`) through the producer; the threshold is above the minimum `S=2` required by the `S-1` cache split.

### Producer invocation and transport

Use a local subprocess/file handoff for C2, not TCP and not an in-process hidden API:

1. Tokenize/validate prompt; if `S < threshold_tokens`, call `generate_step(full_prompt, model, max_tokens=...)` with no imported cache.
2. If `S >= threshold_tokens`, split `prefix_token_ids = prompt_token_ids[:-1]`, `final_token_id = prompt_token_ids[-1]`.
3. Run the exact C1 request-token producer command:
   - `python -m native_r9700.prefill --model <model-dir> --token-ids-json '<full-prompt-token-json>' --out <artifacts>/<request>.prefill.npz --log <artifacts>/<request>.prefill.log`
   - `python -m native_r9700.kv_cache --prefill-npz <...prefill.npz> --out <artifacts>/<request>.prompt-cache.safetensors --log <artifacts>/<request>.kv-cache.log`
4. Before accepting the cache, load with `load_prompt_cache(..., return_metadata=True)` and require full C1 ABI: metadata `offset == str(S-1)`, `num_layers == '16'`, `n_kv_heads == '8'`, `head_dim == '64'`; exactly 16 loaded cache layers; every layer type name `KVCache`; every layer exposes K/V state shaped `(1, 8, S-1, 64)`; and every layer offset/size equals `S-1`.
5. Accepted path calls `generate_step(mx.array([final_token_id]), model, max_tokens=..., prompt_cache=cache)`. Do not recompute or repair the offloaded prefix after this point.

### Fallback/error policy

- Allowed fallback cases: below-threshold prompts; producer subprocess timeout/nonzero exit; missing output files; prompt-cache load/metadata/schema failure before acceptance.
- Disallowed fallback cases: any `generate_step`/decode failure after the imported cache is accepted. Those propagate as `status=error`/nonzero CLI exit; no native full-prompt retry.
- Timeout default: **300 seconds** per producer command for the current Python/NumPy native-producer path; task set 2 must expose `NativePrefillConfig.producer_timeout_s` and CLI `--producer-timeout-s`.
- Malformed prompts (empty, non-integer ids, tokenizer failure) fail loudly, not fallback.
- A supplied `prompt_cache` is mutated by mlx-lm generation; do not reuse one imported cache object across independent requests.
- All artifacts remain local under `logs/c2-serving/`; logs and model files are not committed.

### Log metadata

C2 wrapper logs must include: command line; timestamp; model/config path; producer model path; prompt source/name; prompt length `S`; `n_prefix`; `threshold_tokens`; `producer_timeout_s`; route; fallback reason; `accepted_cache`; producer command statuses; prefill NPZ path; prompt-cache path; prefill log path; kv-cache log path; loaded prompt-cache metadata; decoded token ids; duration; exit status; exception type/message/traceback on error.

## Commands recorded for downstream task sets

- Focused wrapper/behavior tests: `${PY} -m pytest tests/native_r9700/test_serving.py -v`.
- Full native suite after wrapper/test changes: `${PY} -m pytest tests/native_r9700 -v`.
- Full Python suite after Python changes: `${PY} -m pytest tests -v`.
- C2 full prompt-suite integration CLI shape: `${PY} -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --max-new-tokens 4 --threshold-tokens 128 --producer-timeout-s 300 --artifacts-dir logs/c2-serving --json logs/c2-serving/result.json --log logs/c2-serving/run.log --report docs/path-a-validation-results.md`.
- C2 producer-unavailable fallback CLI shape: `${PY} -m native_r9700.serving --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --producer-model /tmp/native-r9700-missing-producer-model --fixtures-dir tests/native_r9700/fixtures --prompt-name prompt-1 --max-new-tokens 4 --threshold-tokens 128 --producer-timeout-s 5 --artifacts-dir logs/c2-serving-unavailable --json logs/c2-serving-unavailable/result.json --log logs/c2-serving-unavailable/run.log`.

## Decisions recorded

- C2 starts with mlx-lm only. oMLX remains a task set 5 scope decision; do not build it before that row is explicitly decided.
- Default threshold is `threshold_tokens=128` total prompt tokens; default producer timeout is `producer_timeout_s=300`.
- Wrapper source/test paths are `native_r9700/serving.py` and `tests/native_r9700/test_serving.py`.

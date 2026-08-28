# F1 serving evidence GREEN implementation

**Scope:** `native_r9700/serving.py` and this report only.

**Status:** Implemented. No tests, builds, linters, formatters, package commands, hardware commands, or git commands were run in this lane, per assignment.

## Native identity gate

For `--producer-kind r9700_native`, `main` now performs both identity checks before either consumer `load_model` or native `build_registry`:

1. `verify_model_identity(--model)` returns a non-empty canonical URI and an exact `sha256:` plus 64 lowercase-hex digest.
2. `verify_model_identity(--producer-model or --model)` is checked with the same rules.
3. The canonical consumer and producer digests must be byte-for-byte equal. A mismatch raises a bounded blocked result (`status: blocked`, `exit_status: 2`) and no consumer load, registry construction, or persistent session construction occurs.
4. The persistent session is loaded only after this equality gate and is bound to the verified producer canonical URI/digest.

CPU/reference serving retains its existing model-load and one-shot diagnostic control path.

## Reusable persistent evidence validator

`_native_prefill_evidence_problems` remains the single evidence rule implementation. Its persistent call supplies the request prefix length and verified model URI, enabling the complete native gate before metadata or cache acceptance:

- `producer_kind == r9700_native` and non-empty `producer_fingerprint`;
- `native_prefill_acceptance == pass`;
- `native_prefill_full_layer_loop_status == pass`;
- `runtime_substrate == TinyGPU.app/APLRemotePCIDevice/PCIIface`;
- `compute_completion_policy == terminal`;
- `compute_barrier_policy == full`;
- `failure_stage == ""`, `failure_text == ""`, and exact integer `exit_status == 0`;
- request-bound `prefill_npz_path` and `hardware_log_path` resolve to the service-requested paths;
- the hardware log is a bounded, readable UTF-8 file;
- the NPZ is an existing, request-bound strict native prefill archive validated against the request model/prefix;
- for `N > 0`, `kernel_count`, `transfer_bytes`, `block_tokens`, and `block_count` are positive, with `block_count == ceil(N / block_tokens)`;
- for declared `N == 0`, `kernel_count == transfer_bytes == block_tokens == block_count == 0` is accepted.

Persistent `Prefill` invokes this validator before cache metadata validation or `load_prompt_cache`. Any problem returns a bounded pre-acceptance `cache_validation_failed` route rejection; full-prompt fallback is the only repairable native path, and no imported cache reaches decode. The existing one-shot minimum diagnostic checks remain available without introducing a second semantic rule set.

## Focused supervisor command (not run here)

```sh
${PY} -m pytest tests/native_r9700/test_serving.py -k 'digest_mismatch_before_load_or_session or semantically_invalid_evidence or declared_zero_prefix' -v
```

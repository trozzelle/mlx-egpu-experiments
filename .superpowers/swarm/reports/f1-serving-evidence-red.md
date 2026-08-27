# F1 serving evidence RED contracts

**Scope:** `r9700_native` serving model identity and persistent prefill evidence gates.

**Targets:** `tests/native_r9700/test_serving.py`; this report.

**Status:** RED contracts authored; supervisor validation pending. No production source was changed.

## Contracts added

- `test_r9700_native_main_rejects_consumer_producer_model_digest_mismatch_before_load_or_session` supplies distinct canonical consumer and producer digests and requires native `main` to verify both identities, reject the mismatch with `status: blocked` / `exit_status: 2`, and perform neither consumer model loading nor registry/session construction.
- `test_persistent_prefill_rejects_semantically_invalid_evidence_before_cache_acceptance` is parameterized over blocked native acceptance, failed full-layer status, wrong runtime substrate, wrong completion/barrier policy, nonempty failure stage/text, nonzero exit, missing/unreadable/unbound hardware log, missing/invalid request-bound NPZ, and zero kernel/transfer evidence for `N>0`. Each mutation must remain pre-acceptance: `accepted_cache` is false, no prompt cache reaches decode, and only the legal blocked full-prompt fallback or an explicitly retained pre-acceptance error is allowed.
- `test_persistent_prefill_accepts_declared_zero_prefix_without_positive_work_counters` supplies a strict empty `(1, 8, 0, 64)` cache/evidence projection with `block_tokens: 0`, `block_count: 0`, `kernel_count: 0`, and `transfer_bytes: 0`; it requires accepted native serving and final-token decode for the frozen `N=0` no-work semantics.

The mutation dispatcher writes an otherwise valid request-bound strict NPZ and readable log before applying exactly one semantic mutation, so each RED case isolates the serving acceptance boundary rather than relying on malformed test setup.

## Expected current RED causes

1. `serving.main` verifies only `--producer-model` and binds that digest to the persistent session; it does not verify the consumer `--model` or compare producer/consumer canonical digests before loading the consumer model.
2. `PersistentPrefillSession.prefill` checks only producer fingerprint/kind before cache import. A status-pass response can therefore carry failed acceptance/full-layer/policy/failure/exit evidence, bad log readability, invalid NPZ bytes, or zero positive-prefix work and still reach cache acceptance.
3. Persistent serving has no accepted N=0 evidence exception for the worker-declared empty-prefix/no-work counters.

## Focused supervisor command (not run by this worker)

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -k 'digest_mismatch_before_load_or_session or semantically_invalid_evidence or declared_zero_prefix' -v
```

Per the swarm boundary, this worker did not run tests, builds, linters, formatters, package managers, hardware commands, or git operations.

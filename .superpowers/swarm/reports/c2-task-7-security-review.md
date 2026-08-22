# C2 task set 7 — security/review handoff

Status: **Done**.

## Initial security review

Reviewer: `agent://C2SecurityReview`.

Verdict: **CHANGES_REQUIRED**; Critical 0, Important 2, Minor 0.

Findings:

1. `C2-SEC-001`: untrusted `request_id`/fixture prompt names could escape `artifacts_dir` when used to build producer artifact paths.
2. `C2-SEC-002`: C2 result JSON/run logs and C2-created prefill logs persisted full prompt token sequences through `--token-ids-json` command serialization.

## Fixes

- `native_r9700/serving.py` now validates artifact request IDs before path construction: nonempty, no NUL, no `/` or `\`, not `.`/`..`, and only `[A-Za-z0-9._-]+`.
- `tests/native_r9700/test_serving.py` proves `../outside/request` is rejected before producer subprocess execution or outside writes.
- `native_r9700/serving.py` now redacts `--prompt` and `--token-ids-json` values in persisted top-level CLI commands and producer command statuses.
- `native_r9700/prefill.py` now redacts `--token-ids-json` in producer-created prefill logs and logs `prompt: token-ids-json` instead of raw token JSON on failures.
- C2 artifacts were regenerated after the fixes.

## Verification

- Security RED: serving focused suite exited 1 with three expected failures: unsafe request ID reached subprocess, producer command logged raw `--token-ids-json`, and top-level CLI command logged raw `--token-ids-json`.
- Prefill RED: `test_prefill_cli_accepts_token_ids_json_without_fixture_name` exited 1 because the prefill log contained raw `[11, 22, 33]`.
- GREEN focused C2 serving suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_serving.py -v` → `16 passed in 0.08s`.
- GREEN focused prefill suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py -v` → `8 passed in 2.11s`.
- Native Python slice: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_prefill.py tests/native_r9700/test_kv_cache.py tests/native_r9700/test_parity.py tests/native_r9700/test_serving.py -v` → `53 passed, 2 warnings in 3.99s`.
- Full native package suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v` → `119 passed, 2 warnings in 9.76s`.
- Full Python suite: `${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v` → `159 passed, 2 warnings in 42.77s`.
- Producer-unavailable smoke regenerated: `C2 serving status=pass prompts=1`.
- Full C2 fixture integration/report regenerated: `C2 serving status=pass prompts=3`.
- Raw-token leakage check across `logs/c2-serving/result.json`, `logs/c2-serving/run.log`, `logs/c2-serving-unavailable/result.json`, `logs/c2-serving-unavailable/run.log`, and C2-created prefill logs found no `--token-ids-json '[...]'` or `[128000, 791` matches; redacted markers are present.

## Final security review

Reviewer: `agent://C2SecurityReReview`.

Verdict: **APPROVE**; Critical 0, Important 0, Minor 0.

Disposition: C2-7 can be marked Done from a security-review perspective. C3 is not blocked for security reasons by C2; remaining C3 gates are the documented non-security scope/performance/backend decision process.

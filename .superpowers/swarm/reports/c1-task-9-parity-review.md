# C1 task set 9 — parity harness review

Reviewer: `C1ParityReview` then `C1ParityReReview`.

Verdict: **APPROVE** after fixes.

## Initial findings

- Important: exception/blocked path wrote only a short log and could leave stale Path C PASS evidence.
- Important: Path C report missed required `log_path`, `json_path`, weight provenance, and RoPE/config grounding fields.

## Fixes applied

- `native_r9700.parity.main` now builds schema-compatible `blocked` results on exceptions, writes JSON, replaces Path C with blocked evidence, and writes a log with exit status `2`.
- Path C rendering now includes `log_path`, `json_path`, `config_path`, `weight_provenance`, and `rope_config_note`.
- Tests now cover both required fields and stale-PASS replacement on blocked CLI errors.

## Final review result

`C1ParityReReview` returned APPROVE with 0 Critical, 0 Important, and 0 Minor findings.

Evidence cited by reviewer:

- Blocked exceptions produce structured BLOCKED JSON/report/log and remove stale PASS evidence: `native_r9700/parity.py`, `tests/native_r9700/test_parity.py`.
- Normal PASS path preserves Path A while writing JSON/log/report: `tests/native_r9700/test_parity.py`.
- Current artifacts include log/json/config/weight/RoPE metadata, prompt results, and per-layer deltas: `docs/path-a-validation-results.md`, `logs/c1-parity/result.json`, `logs/c1-parity/run.log`.

## Supervisor verification after fixes

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700/test_parity.py -v
# pytest: 16 passed in 0.08s
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m native_r9700.parity --model ../tinygrad-kv-worker-phase0/mlx_models/meta-Llama-3.2-1B-Instruct --fixtures-dir tests/native_r9700/fixtures --r-source both --max-new-tokens 4 --artifacts-dir logs/c1-parity --json logs/c1-parity/result.json --log logs/c1-parity/run.log --report docs/path-a-validation-results.md
# C1 parity gate_result=pass prompts=3
```

```text
Blocked CLI smoke with missing model: returncode 2; JSON gate_result blocked; report contains Status BLOCKED and no stale gate_result pass; log contains exit_status: 2.
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests/native_r9700 -v
# pytest: 100 passed, 2 warnings in 9.46s
```

```sh
${HOME}/.pyenv/versions/3.12.8/bin/python3 -m pytest tests -v
# pytest: 140 passed, 2 warnings in 42.61s
```

```sh
git diff --check
# no output
```

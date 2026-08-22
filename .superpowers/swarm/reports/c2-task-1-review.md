# C2 task set 1 — review result

Initial reviewer: `C2ContractReview`.
Re-reviewer: `C2ContractReReview`.

Verdict: **APPROVE** after fixes.

## Initial important findings

1. Require the full C1 prompt-cache ABI before cache acceptance: metadata `offset`, `num_layers`, `n_kv_heads`, `head_dim`; 16 `KVCache` layers; per-layer K/V shape/offset.
2. Record concrete C2 integration commands for the full fixture suite and producer-unavailable fallback, rather than leaving task set 4 to invent a prompt-suite flag.
3. Name the producer timeout override in both API and CLI.

## Fixes landed

- `.superpowers/swarm/reports/c2-task-1-contract.md` now freezes `NativePrefillConfig.producer_model_dir`, `threshold_tokens`, `producer_timeout_s`, CLI `--producer-model`, `--threshold-tokens`, and `--producer-timeout-s`; requires full C1 ABI before `accepted_cache=true`; records full fixture-suite and unavailable fallback command shapes.
- `docs/tasks/native-r9700-producer/phase-c2-serving-integration.md` now records the same source/test paths, full ABI acceptance, timeout override, mutation/no-reuse rule, and required commands.
- `docs/tasks/native-r9700-producer/validation-commands.md` now records exact C2 focused test, full fixture-suite CLI, producer-unavailable fallback CLI, and full-suite commands.

## Re-review

`C2ContractReReview` approved with 0 Critical, 0 Important, and 0 Minor findings. Supervisor may mark C2-1 Done and proceed to RED tests.

## Supervisor verification

```sh
git diff --check docs/tasks/native-r9700-producer/phase-c2-serving-integration.md docs/tasks/native-r9700-producer/validation-commands.md .superpowers/swarm/reports/c2-task-1-contract.md .superpowers/swarm/native-r9700-producer-supervisor.md .superpowers/swarm/progress.md
# no output
```
